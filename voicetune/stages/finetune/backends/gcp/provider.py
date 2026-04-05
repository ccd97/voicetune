"""GCP-based VoxCPM2 LoRA fine-tuning.

Uploads training data to GCS, launches an A100 VM that downloads
openbmb/VoxCPM2 and runs scripts/train_voxcpm_finetune.py against a
JSONL manifest, monitors progress via guest attributes, and downloads
the LoRA adapter + checkpoints when complete.
"""

import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path

from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.cloud import compute_v1, storage

log = logging.getLogger(__name__)

BUCKET_NAME = "voicetune-finetune-cdcunha"
BUCKET_URI = f"gs://{BUCKET_NAME}"
INSTANCE_BASE = "voicetune-finetune"
MACHINE_TYPE = "a2-highgpu-1g"
ACCELERATOR = "nvidia-tesla-a100"
IMAGE_FAMILY = "pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
IMAGE_PROJECT = "deeplearning-platform-release"
STARTUP_SCRIPT = Path(__file__).parent / "startup.sh"
TRAIN_SCRIPT = Path(__file__).parent / "train.py"

ZONES = [
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-east1-b", "us-east1-c",
    "us-west1-a", "us-west1-b",
    "us-west4-a", "us-west4-b",
    "europe-west4-a", "europe-west4-b", "europe-west4-c",
    "asia-southeast1-b", "asia-southeast1-c",
]


def ensure_bucket(bucket: storage.Bucket) -> None:
    if bucket.exists():
        return
    log.info(f"Creating bucket {BUCKET_NAME}...")
    bucket.client.create_bucket(bucket, location="us")


def cleanup_previous_results(bucket: storage.Bucket) -> None:
    """Wipe results/ from the prior run so stale checkpoints don't leak into the download."""
    blobs = list(bucket.client.list_blobs(bucket, prefix="results/"))
    if not blobs:
        log.info("No previous results/ in gs://%s to clean up", BUCKET_NAME)
        return
    log.info(f"Deleting {len(blobs)} previous result blobs from gs://{BUCKET_NAME}/results/...")
    with bucket.client.batch():
        for blob in blobs:
            blob.delete()


def upload_data(bucket: storage.Bucket, data_dir: Path) -> None:
    files = sorted(f for f in data_dir.rglob("*") if f.is_file())
    log.info(f"Zipping {len(files)} files...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f.relative_to(data_dir))
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        log.info(f"Uploading training-data.zip ({size_mb:.1f} MB) to gs://{BUCKET_NAME}/...")
        bucket.blob("training-data.zip").upload_from_filename(str(zip_path))
    finally:
        zip_path.unlink(missing_ok=True)


def upload_train_script(bucket: storage.Bucket) -> None:
    log.info("Uploading training script...")
    bucket.blob("train_gcp.py").upload_from_filename(str(TRAIN_SCRIPT))


def cleanup_existing(compute: compute_v1.InstancesClient, project: str, instance: str) -> None:
    for zone in ZONES:
        try:
            compute.get(project=project, zone=zone, instance=instance)
        except NotFound:
            continue
        log.info(f"Deleting existing instance {instance} in {zone}...")
        compute.delete(project=project, zone=zone, instance=instance).result()


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
            compute_v1.Items(key="HF_TOKEN", value=os.environ.get("HF_TOKEN", "")),
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
    project: str,
    instance: str,
    max_steps: int,
    spot: bool,
) -> str:
    for zone in ZONES:
        log.info(f"Trying zone {zone}...")
        inst = _build_instance(instance, zone, max_steps, spot)
        try:
            compute.insert(
                project=project, zone=zone, instance_resource=inst,
            ).result()
            return zone
        except GoogleAPICallError as e:
            log.info(f"  Zone {zone} unavailable: {e.message}")

    raise RuntimeError(
        f"No zone had {ACCELERATOR} capacity across {len(ZONES)} zones. Try again later."
    )


def poll_status(
    compute: compute_v1.InstancesClient, project: str, instance: str, zone: str,
) -> str:
    # query_path isn't in the flattened kwargs for get_guest_attributes in
    # google-cloud-compute >=1.20; pass it via a request object instead.
    request = compute_v1.GetGuestAttributesInstanceRequest(
        project=project, zone=zone, instance=instance,
        query_path="voicetune/status",
    )
    try:
        result = compute.get_guest_attributes(request=request)
        if result.query_value and result.query_value.items:
            return result.query_value.items[0].value
        return "PENDING"
    except NotFound:
        try:
            compute.get(project=project, zone=zone, instance=instance)
            return "PENDING"
        except NotFound:
            return "VM_GONE"
    except Exception as e:
        log.warning(f"Guest attribute poll failed: {type(e).__name__}: {e}")
        return "PENDING"


def delete_instance(
    compute: compute_v1.InstancesClient, project: str, instance: str, zone: str,
) -> None:
    try:
        compute.delete(project=project, zone=zone, instance=instance).result()
        log.info(f"Deleted instance {instance} in {zone}")
    except NotFound:
        pass


def _download_prefix(bucket: storage.Bucket, prefix: str, local_dir: Path) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in bucket.client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix):]
        local_path = local_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        count += 1
    return count


def download_results(bucket: storage.Bucket, output_dir: Path) -> Path:
    """Download LoRA adapter checkpoints and tensorboard logs.

    VoxCPM saves each checkpoint as a directory (step_NNNNNNN/ containing
    lora_weights.safetensors + lora_config.json), not a single file. We mirror
    everything under results/lora/ to {output_dir}/voxcpm2-lora/.
    """
    lora_dir = output_dir / "voxcpm2-lora"
    log.info(f"Downloading LoRA adapter checkpoints to {lora_dir}...")
    n = _download_prefix(bucket, "results/lora/", lora_dir)
    log.info(f"  Downloaded {n} LoRA files")

    tb_dir = output_dir / "tensorboard"
    n = _download_prefix(bucket, "results/tensorboard/", tb_dir)
    if n:
        log.info(f"Downloaded {n} tensorboard files to {tb_dir}/")

    return lora_dir


def run(
    data_dir: Path,
    max_steps: int,
    test: bool,
    output_dir: Path,
) -> dict:
    project = os.environ["GCP_PROJECT_ID"]
    instance = f"{INSTANCE_BASE}-test" if test else INSTANCE_BASE
    bucket = storage.Client(project=project).bucket(BUCKET_NAME)
    compute = compute_v1.InstancesClient()

    if test:
        max_steps = 1
        log.info("Test mode: A100 spot, 1 step")

    log.info(f"GCP finetune — project: {project}, machine: {MACHINE_TYPE} ({ACCELERATOR}), steps: {max_steps}")
    log.info(f"Training data: {data_dir}")

    ensure_bucket(bucket)
    cleanup_previous_results(bucket)
    upload_data(bucket, data_dir)
    upload_train_script(bucket)
    cleanup_existing(compute, project, instance)

    zone = create_instance(compute, project, instance, max_steps, spot=test)
    log.info(f"Instance created in {zone}. Monitoring status...")
    log.info(f"  Serial log: gcloud compute instances get-serial-port-output {instance} --zone={zone} --project={project}")

    prev_status = ""
    status_since = time.monotonic()
    while True:
        status = poll_status(compute, project, instance, zone)
        now = time.monotonic()
        if status != prev_status:
            log.info(f"Status: {status}")
            prev_status = status
            status_since = now
        else:
            elapsed = int(now - status_since)
            mins, secs = divmod(elapsed, 60)
            log.info(f"Status: {status} ({mins}m{secs:02d}s)")

        if status == "COMPLETE":
            log.info("Training complete!")
            lora_dir = download_results(bucket, output_dir)
            delete_instance(compute, project, instance, zone)
            return {"status": "complete", "lora": str(lora_dir)}

        if status.startswith("FAILED"):
            raise RuntimeError(f"Training failed: {status}")

        if status == "VM_GONE":
            raise RuntimeError(
                "VM was deleted unexpectedly (spot preemption?). "
                "Check GCS for partial results or retry."
            )

        time.sleep(30)
