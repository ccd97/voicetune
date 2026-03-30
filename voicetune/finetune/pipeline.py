"""GCP-based Fish Speech S2 Pro LoRA fine-tuning.

Uploads training data to GCS, launches an A100 VM that runs the full
Fish Speech pipeline (VQ extraction -> dataset build -> LoRA train -> merge),
monitors progress via guest attributes, and downloads the finetuned model.
"""

import logging
import os
import time
from pathlib import Path

from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import compute_v1, storage

log = logging.getLogger(__name__)

GCP_PROJECT = os.environ["GCP_PROJECT"]
BUCKET_NAME = "voicetune-finetune-cdcunha"
BUCKET_URI = f"gs://{BUCKET_NAME}"
INSTANCE_BASE = "voicetune-finetune"
MACHINE_TYPE = "a2-highgpu-1g"
ACCELERATOR = "nvidia-tesla-a100"
IMAGE_FAMILY = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
IMAGE_PROJECT = "deeplearning-platform-release"
STARTUP_SCRIPT = Path(__file__).parent / "startup_gcp.sh"
TRAIN_SCRIPT = Path(__file__).parent / "train_gcp.py"

ZONES = [
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-east1-b", "us-east1-c",
    "us-west1-a", "us-west1-b",
    "us-west4-a", "us-west4-b",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
    "asia-southeast1-b", "asia-southeast1-c",
]


def validate_data(data_dir: Path) -> tuple[int, int]:
    me_dir = data_dir / "me"
    if not me_dir.is_dir():
        raise FileNotFoundError(f"No training data directory: {me_dir}")
    wavs = list(me_dir.glob("*.wav"))
    labs = list(me_dir.glob("*.lab"))
    if not wavs or not labs:
        raise FileNotFoundError(f"No wav+lab pairs in {me_dir}. Run the export step first.")
    return len(wavs), len(labs)


def ensure_bucket(bucket: storage.Bucket) -> None:
    if bucket.exists():
        return
    log.info(f"Creating bucket {BUCKET_NAME}...")
    bucket.client.create_bucket(bucket, location="us")


def upload_data(bucket: storage.Bucket, data_dir: Path) -> None:
    files = sorted(f for f in data_dir.rglob("*") if f.is_file())
    log.info(f"Uploading {len(files)} files to gs://{BUCKET_NAME}/data/...")
    for f in files:
        blob_name = f"data/{f.relative_to(data_dir)}"
        bucket.blob(blob_name).upload_from_filename(str(f))


def upload_train_script(bucket: storage.Bucket) -> None:
    log.info("Uploading training script...")
    bucket.blob("train_gcp.py").upload_from_filename(str(TRAIN_SCRIPT))


def cleanup_existing(compute: compute_v1.InstancesClient, instance: str) -> None:
    for zone in ZONES:
        try:
            compute.get(project=GCP_PROJECT, zone=zone, instance=instance)
        except NotFound:
            continue
        log.info(f"Deleting existing instance {instance} in {zone}...")
        compute.delete(project=GCP_PROJECT, zone=zone, instance=instance).result()


def _build_instance(
    name: str, zone: str, max_steps: int, spot: bool,
) -> compute_v1.Instance:
    startup_content = STARTUP_SCRIPT.read_text()

    scheduling = compute_v1.Scheduling(on_host_maintenance="TERMINATE")
    if spot:
        scheduling.provisioning_model = "SPOT"
        scheduling.instance_termination_action = "DELETE"

    return compute_v1.Instance(
        name=name,
        machine_type=f"zones/{zone}/machineTypes/{MACHINE_TYPE}",
        disks=[
            compute_v1.AttachedDisk(
                auto_delete=True,
                boot=True,
                initialize_params=compute_v1.AttachedDiskInitializeParams(
                    source_image=f"projects/{IMAGE_PROJECT}/global/images/family/{IMAGE_FAMILY}",
                    disk_size_gb=100,
                ),
            ),
        ],
        network_interfaces=[
            compute_v1.NetworkInterface(
                access_configs=[compute_v1.AccessConfig(name="External NAT")],
            ),
        ],
        guest_accelerators=[
            compute_v1.AcceleratorConfig(
                accelerator_type=f"zones/{zone}/acceleratorTypes/{ACCELERATOR}",
                accelerator_count=1,
            ),
        ],
        scheduling=scheduling,
        metadata=compute_v1.Metadata(items=[
            compute_v1.Items(key="MAX_STEPS", value=str(max_steps)),
            compute_v1.Items(key="BUCKET", value=BUCKET_URI),
            compute_v1.Items(key="enable-guest-attributes", value="TRUE"),
            compute_v1.Items(key="startup-script", value=startup_content),
        ]),
        service_accounts=[
            compute_v1.ServiceAccount(scopes=[
                "https://www.googleapis.com/auth/devstorage.full_control",
                "https://www.googleapis.com/auth/compute",
            ]),
        ],
    )


def create_instance(
    compute: compute_v1.InstancesClient,
    instance: str,
    max_steps: int,
    spot: bool,
) -> str:
    for zone in ZONES:
        log.info(f"Trying zone {zone}...")
        inst = _build_instance(instance, zone, max_steps, spot)
        try:
            compute.insert(
                project=GCP_PROJECT, zone=zone, instance_resource=inst,
            ).result()
            return zone
        except GoogleAPICallError as e:
            log.info(f"  Zone {zone} unavailable: {e.message}")

    raise RuntimeError(
        f"No zone had {ACCELERATOR} capacity across {len(ZONES)} zones. Try again later."
    )


def poll_status(
    compute: compute_v1.InstancesClient, instance: str, zone: str,
) -> str:
    try:
        result = compute.get_guest_attributes(
            project=GCP_PROJECT, zone=zone, instance=instance,
            query_path="voicetune/status",
        )
        if result.query_value and result.query_value.items:
            return result.query_value.items[0].value
        return "PENDING"
    except NotFound:
        try:
            compute.get(project=GCP_PROJECT, zone=zone, instance=instance)
            return "PENDING"
        except NotFound:
            return "VM_GONE"
    except Exception:
        return "PENDING"


def delete_instance(
    compute: compute_v1.InstancesClient, instance: str, zone: str,
) -> None:
    try:
        compute.delete(project=GCP_PROJECT, zone=zone, instance=instance).result()
        log.info(f"Deleted instance {instance} in {zone}")
    except NotFound:
        pass


def download_model(bucket: storage.Bucket, output_dir: Path) -> Path:
    model_dir = output_dir / "s2-pro-finetuned"
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading finetuned model to {model_dir}...")
    for blob in bucket.client.list_blobs(bucket, prefix="model/"):
        if blob.name.endswith("/"):
            continue
        rel = blob.name.removeprefix("model/")
        local_path = model_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
    return model_dir


def run_finetune(
    data_dir: Path,
    max_steps: int,
    test: bool,
    output_dir: Path,
) -> dict:
    instance = f"{INSTANCE_BASE}-test" if test else INSTANCE_BASE
    bucket = storage.Client(project=GCP_PROJECT).bucket(BUCKET_NAME)
    compute = compute_v1.InstancesClient()

    if test:
        max_steps = 1
        log.info("=== TEST MODE: A100 SPOT, 1 step ===")

    log.info(f"GCP finetune — project: {GCP_PROJECT}, machine: {MACHINE_TYPE} ({ACCELERATOR}), steps: {max_steps}")

    wav_count, lab_count = validate_data(data_dir)
    log.info(f"Training data: {wav_count} wav, {lab_count} lab files")

    ensure_bucket(bucket)
    upload_data(bucket, data_dir)
    upload_train_script(bucket)
    cleanup_existing(compute, instance)

    zone = create_instance(compute, instance, max_steps, spot=test)
    log.info(f"Instance created in {zone}. Monitoring status...")
    log.info(f"  Serial log: gcloud compute instances get-serial-port-output {instance} --zone={zone} --project={GCP_PROJECT}")

    prev_status = ""
    while True:
        status = poll_status(compute, instance, zone)
        if status != prev_status:
            log.info(f"Status: {status}")
            prev_status = status

        if status == "COMPLETE":
            log.info("Training complete!")
            model_dir = download_model(bucket, output_dir)
            delete_instance(compute, instance, zone)
            return {"status": "complete", "model": str(model_dir)}

        if status.startswith("FAILED"):
            raise RuntimeError(f"Training failed: {status}")

        if status == "VM_GONE":
            raise RuntimeError(
                "VM was deleted unexpectedly (spot preemption?). "
                "Check GCS for partial results or retry."
            )

        time.sleep(30)
