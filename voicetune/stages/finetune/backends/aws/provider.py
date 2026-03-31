"""AWS EC2-based Fish Speech S2 Pro LoRA fine-tuning.

Uploads training data to S3, launches a g6e.2xlarge GPU instance with the
Deep Learning AMI, monitors progress via an S3 status file, and downloads
the finetuned model.
"""

import base64
import logging
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

BUCKET_NAME = "voicetune-finetune-cdcunha"
INSTANCE_NAME_BASE = "voicetune-finetune"
INSTANCE_TYPE = "g6e.2xlarge"
STARTUP_SCRIPT = Path(__file__).parent / "startup.sh"
TRAIN_SCRIPT = Path(__file__).parent / "train.py"

DLAMI_SSM_PARAM = "/aws/service/deep-learning/ami/pytorch/2.6/ubuntu-22.04/x86_64/latest"

REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]


def ensure_bucket(s3, region: str) -> None:
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
    except ClientError:
        log.info(f"Creating S3 bucket {BUCKET_NAME}...")
        kwargs = {"Bucket": BUCKET_NAME}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)


def upload_data(s3, data_dir: Path) -> None:
    files = sorted(f for f in data_dir.rglob("*") if f.is_file())
    log.info(f"Uploading {len(files)} files to s3://{BUCKET_NAME}/data/...")
    for f in files:
        key = f"data/{f.relative_to(data_dir)}"
        s3.upload_file(str(f), BUCKET_NAME, key)


def upload_train_script(s3) -> None:
    log.info("Uploading training script...")
    s3.upload_file(str(TRAIN_SCRIPT), BUCKET_NAME, "train.py")


def get_ami_id(region: str) -> str:
    ssm = boto3.client("ssm", region_name=region)
    resp = ssm.get_parameter(Name=DLAMI_SSM_PARAM)
    return resp["Parameter"]["Value"]


def cleanup_existing(ec2, instance_name: str) -> None:
    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [instance_name]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    instance_ids = [
        i["InstanceId"]
        for r in resp["Reservations"]
        for i in r["Instances"]
    ]
    if instance_ids:
        log.info(f"Terminating existing instances: {instance_ids}")
        ec2.terminate_instances(InstanceIds=instance_ids)
        waiter = ec2.get_waiter("instance_terminated")
        waiter.wait(InstanceIds=instance_ids)


def build_user_data(bucket: str, max_steps: int) -> str:
    startup_content = STARTUP_SCRIPT.read_text()
    header = f"export BUCKET='{bucket}'\nexport MAX_STEPS='{max_steps}'\n"
    script = f"#!/bin/bash\n{header}{startup_content}"
    return base64.b64encode(script.encode()).decode()


def create_instance(
    region: str, instance_name: str, max_steps: int, spot: bool,
    instance_profile: str,
) -> tuple[str, str]:
    for r in REGIONS:
        ec2 = boto3.client("ec2", region_name=r)
        try:
            ami_id = get_ami_id(r)
        except Exception:
            log.info(f"  Region {r}: couldn't resolve AMI, skipping")
            continue

        log.info(f"Trying region {r} (AMI: {ami_id})...")
        run_kwargs = {
            "ImageId": ami_id,
            "InstanceType": INSTANCE_TYPE,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": build_user_data(BUCKET_NAME, max_steps),
            "TagSpecifications": [{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": instance_name}],
            }],
            "IamInstanceProfile": {"Name": instance_profile},
            "BlockDeviceMappings": [{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"},
            }],
        }
        if spot:
            run_kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            }

        try:
            resp = ec2.run_instances(**run_kwargs)
            instance_id = resp["Instances"][0]["InstanceId"]
            return r, instance_id
        except ClientError as e:
            log.info(f"  Region {r} unavailable: {e.response['Error']['Message']}")

    raise RuntimeError(
        f"No region had {INSTANCE_TYPE} capacity across {REGIONS}. Try again later."
    )


def poll_status(s3) -> str:
    try:
        resp = s3.get_object(Bucket=BUCKET_NAME, Key="status.txt")
        return resp["Body"].read().decode().strip()
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return "PENDING"
        raise


def check_instance_alive(ec2, instance_id: str) -> bool:
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
    return state in ("pending", "running")


def terminate_instance(ec2, instance_id: str) -> None:
    try:
        ec2.terminate_instances(InstanceIds=[instance_id])
        log.info(f"Terminated instance {instance_id}")
    except ClientError:
        pass


def download_model(s3, output_dir: Path) -> Path:
    model_dir = output_dir / "s2-pro-finetuned"
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading finetuned model to {model_dir}...")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="model/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key.removeprefix("model/")
            local_path = model_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET_NAME, key, str(local_path))
    return model_dir


def run(
    data_dir: Path,
    max_steps: int,
    test: bool,
    output_dir: Path,
) -> dict:
    region = os.environ.get("AWS_REGION", "us-east-1")
    instance_profile = os.environ.get("AWS_INSTANCE_PROFILE", "voicetune-finetune")
    instance_name = f"{INSTANCE_NAME_BASE}-test" if test else INSTANCE_NAME_BASE

    s3 = boto3.client("s3", region_name=region)
    ec2 = boto3.client("ec2", region_name=region)

    if test:
        max_steps = 1
        log.info("=== TEST MODE: SPOT g6e.2xlarge, 1 step ===")

    log.info(f"AWS finetune — region: {region}, instance: {INSTANCE_TYPE}, steps: {max_steps}")
    log.info(f"Training data: {data_dir}")

    ensure_bucket(s3, region)
    upload_data(s3, data_dir)
    upload_train_script(s3)
    cleanup_existing(ec2, instance_name)

    # Clear any previous status
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key="status.txt")
    except ClientError:
        pass

    launch_region, instance_id = create_instance(
        region, instance_name, max_steps, spot=test,
        instance_profile=instance_profile,
    )
    ec2 = boto3.client("ec2", region_name=launch_region)
    s3 = boto3.client("s3", region_name=launch_region)
    log.info(f"Instance {instance_id} created in {launch_region}. Monitoring status...")

    prev_status = ""
    while True:
        status = poll_status(s3)
        if status != prev_status:
            log.info(f"Status: {status}")
            prev_status = status

        if status == "COMPLETE":
            log.info("Training complete!")
            model_dir = download_model(s3, output_dir)
            terminate_instance(ec2, instance_id)
            return {"status": "complete", "model": str(model_dir)}

        if status.startswith("FAILED"):
            raise RuntimeError(f"Training failed: {status}")

        if not check_instance_alive(ec2, instance_id):
            raise RuntimeError(
                "EC2 instance terminated unexpectedly (spot reclaim?). "
                "Check S3 for partial results or retry."
            )

        time.sleep(30)
