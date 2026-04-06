"""Training orchestrator for GCP VM (VoxCPM2 LoRA).

Called by startup.sh after environment setup. Downloads openbmb/VoxCPM2
(GCS cache first, HuggingFace fallback), pulls the JSONL training manifest
and wavs from GCS, renders a LoRA config YAML, runs VoxCPM's
scripts/train_voxcpm_finetune.py, and uploads step checkpoints +
tensorboard logs back to GCS. Reports status via GCE guest attributes.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

import yaml
from google.cloud import storage

log = logging.getLogger(__name__)

MODEL_REPO = "openbmb/VoxCPM2"
MODEL_DIR_NAME = "VoxCPM2"
MODEL_CACHE_PREFIX = "voxcpm2-base/"

PROJECT = "me-lora"
LORA_RANK = 64
LORA_ALPHA = 128
LEARNING_RATE = 5.0e-4

GUEST_ATTR_URL = (
    "http://metadata.google.internal/computeMetadata/v1"
    "/instance/guest-attributes/voicetune/status"
)


def _download_gcs_dir(bucket: storage.Bucket, prefix: str, local_dir: Path) -> int:
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


def _upload_gcs_dir(bucket: storage.Bucket, local_dir: Path, prefix: str) -> int:
    count = 0
    for f in sorted(local_dir.rglob("*")):
        if not f.is_file():
            continue
        blob_name = f"{prefix}{f.relative_to(local_dir)}"
        bucket.blob(blob_name).upload_from_filename(str(f))
        count += 1
    return count


def set_status(status: str) -> None:
    req = urllib.request.Request(
        GUEST_ATTR_URL, data=status.encode(), method="PUT",
        headers={"Metadata-Flavor": "Google"},
    )
    urllib.request.urlopen(req)


def download_model(bucket: storage.Bucket, vox_dir: Path) -> Path:
    model_dir = vox_dir / "models" / MODEL_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    if bucket.blob(f"{MODEL_CACHE_PREFIX}config.json").exists():
        log.info(f"Pulling {MODEL_DIR_NAME} from GCS cache...")
        _download_gcs_dir(bucket, MODEL_CACHE_PREFIX, model_dir)
        return model_dir

    log.info(f"First run — downloading {MODEL_REPO} from HuggingFace...")
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or None
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(model_dir), token=token)

    log.info(f"Caching model to gs://{bucket.name}/{MODEL_CACHE_PREFIX}...")
    _upload_gcs_dir(bucket, model_dir, MODEL_CACHE_PREFIX)
    return model_dir


def pull_training_data(bucket: storage.Bucket, vox_dir: Path) -> Path:
    log.info("Pulling training data from GCS...")
    data_dir = vox_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = vox_dir / "training-data.zip"
    bucket.blob("training-data.zip").download_to_filename(str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    count = sum(1 for _ in data_dir.rglob("*") if _.is_file())
    log.info(f"  Extracted {count} files to {data_dir}")
    return data_dir


def rewrite_manifest(manifest: Path, data_dir: Path) -> None:
    """Rewrite absolute audio paths (from local prepare step) to VM paths.

    prepare_dataset stamps absolute paths from the local machine into the
    JSONL. On the VM those don't resolve, so replace them with paths under
    data_dir based on the filename.
    """
    if not manifest.exists():
        return
    me_dir = data_dir / "me"
    out = []
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            name = Path(entry["audio"]).name
            entry["audio"] = str(me_dir / name)
            if "ref_audio" in entry:
                entry["ref_audio"] = str(me_dir / Path(entry["ref_audio"]).name)
            out.append(entry)
    with manifest.open("w", encoding="utf-8") as f:
        for entry in out:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"  Rewrote {len(out)} manifest entries in {manifest.name}")


def render_config(vox_dir: Path, model_dir: Path, data_dir: Path, max_steps: int) -> Path:
    save_path = vox_dir / "results" / PROJECT / "lora"
    tb_path = vox_dir / "results" / PROJECT / "tensorboard"
    val_manifest = data_dir / "val.jsonl"

    cfg = {
        "pretrained_path": str(model_dir),
        "train_manifest": str(data_dir / "train.jsonl"),
        "val_manifest": str(val_manifest) if val_manifest.exists() else "",
        "sample_rate": 16000,
        "out_sample_rate": 48000,
        "batch_size": 1,
        "grad_accum_steps": 16,
        "num_workers": 4,
        "num_iters": max_steps,
        "max_steps": max_steps,
        "log_interval": 10,
        "valid_interval": max(1, min(50, max_steps)),
        "save_interval": max(1, min(100, max_steps)),
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "warmup_steps": max(1, max_steps // 10),
        "max_batch_tokens": 8192,
        "max_grad_norm": 1.0,
        "save_path": str(save_path),
        "tensorboard": str(tb_path),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": True,
            "r": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": 0.0,
        },
        # Portable adapter: stamp the HF repo id (not the VM-local path) into lora_config.json.
        "hf_model_id": MODEL_REPO,
        "distribute": True,
    }
    conf_dir = vox_dir / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    path = conf_dir / "me_lora.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log.info(f"Rendered config to {path}")
    return path


def _find_step_dirs(save_path: Path) -> list[Path]:
    if not save_path.exists():
        return []
    return sorted(
        (d for d in save_path.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: int(re.search(r"step_(\d+)", d.name).group(1)),
    )


def _step_dir_complete(step_dir: Path) -> bool:
    weights_present = (
        (step_dir / "lora_weights.safetensors").exists()
        or (step_dir / "lora_weights.ckpt").exists()
    )
    return weights_present and (step_dir / "lora_config.json").exists()


def _watch_and_upload_checkpoints(
    bucket: storage.Bucket,
    save_path: Path,
    prefix: str,
    stop_event: threading.Event,
    interval: float = 15.0,
) -> None:
    """Mirror each completed step_* dir to GCS. latest/ is handled in the final sweep."""
    uploaded: set[str] = set()
    while not stop_event.is_set():
        try:
            for step_dir in _find_step_dirs(save_path):
                if step_dir.name in uploaded or not _step_dir_complete(step_dir):
                    continue
                step_prefix = f"{prefix}{step_dir.name}/"
                try:
                    n = _upload_gcs_dir(bucket, step_dir, step_prefix)
                    uploaded.add(step_dir.name)
                    log.info(f"  [ckpt-sync] uploaded {step_dir.name} ({n} files)")
                except Exception as e:
                    log.warning(f"  [ckpt-sync] failed to upload {step_dir.name}: {e}")
        except Exception as e:
            log.warning(f"  [ckpt-sync] watcher error: {e}")
        stop_event.wait(interval)


def find_latest_step(save_path: Path) -> Path | None:
    dirs = _find_step_dirs(save_path)
    return dirs[-1] if dirs else None


VAL_LOSS_TAG = "val/loss/total"


def find_best_step(save_path: Path, tb_dir: Path) -> Path | None:
    """Pick the step dir with the lowest val/loss/total; fall back to the latest step."""
    dirs = _find_step_dirs(save_path)
    if not dirs:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = sorted(tb_dir.rglob("events.out.tfevents.*"))
        step_to_loss: dict[int, float] = {}
        for ev_path in events:
            ea = EventAccumulator(str(ev_path.parent), size_guidance={"scalars": 0})
            ea.Reload()
            if VAL_LOSS_TAG not in ea.Tags().get("scalars", []):
                continue
            for ev in ea.Scalars(VAL_LOSS_TAG):
                step_to_loss[ev.step] = ev.value
    except Exception as e:
        log.warning(f"Could not read TensorBoard {VAL_LOSS_TAG} ({e}); using latest step")
        return dirs[-1]

    if not step_to_loss:
        return dirs[-1]

    best: tuple[float, Path] | None = None
    for d in dirs:
        step = int(re.search(r"step_(\d+)", d.name).group(1))
        loss = step_to_loss.get(step)
        if loss is None:
            nearby = [(abs(s - step), l) for s, l in step_to_loss.items() if abs(s - step) <= 10]
            if not nearby:
                continue
            loss = min(nearby)[1]
        if best is None or loss < best[0]:
            best = (loss, d)

    if best is None:
        return dirs[-1]
    log.info(f"Best step by {VAL_LOSS_TAG}: {best[1].name} (loss={best[0]:.4f})")
    return best[1]


def train(vox_dir: Path, python: str, config_path: Path, bucket: storage.Bucket) -> Path:
    save_path = vox_dir / "results" / PROJECT / "lora"

    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_and_upload_checkpoints,
        args=(bucket, save_path, "results/lora/", stop_event),
        daemon=True,
    )
    watcher.start()
    log.info("Started background checkpoint sync to GCS (results/lora/)")

    train_env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled",
    }
    cmd = [python, "scripts/train_voxcpm_finetune.py", "--config_path", str(config_path)]
    log.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(vox_dir), env=train_env)

    stop_event.set()
    watcher.join(timeout=60)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit {result.returncode})")

    if save_path.exists():
        for step_dir in _find_step_dirs(save_path):
            _upload_gcs_dir(bucket, step_dir, f"results/lora/{step_dir.name}/")
        latest = save_path / "latest"
        if latest.exists():
            _upload_gcs_dir(bucket, latest, "results/lora/latest/")
            log.info("  [ckpt-sync] final latest/ upload")

    tb_dir = vox_dir / "results" / PROJECT / "tensorboard"
    best = find_best_step(save_path, tb_dir) or find_latest_step(save_path)
    if not best:
        raise RuntimeError(f"No step checkpoints found in {save_path}")
    return best


def upload_results(bucket: storage.Bucket, vox_dir: Path) -> None:
    log.info("Uploading tensorboard logs to GCS...")
    tb_dir = vox_dir / "results" / PROJECT / "tensorboard"
    if tb_dir.exists():
        n = _upload_gcs_dir(bucket, tb_dir, "results/tensorboard/")
        log.info(f"  Uploaded {n} tensorboard files")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--vox-dir", type=Path, default=Path("/opt/voxcpm"))
    args = parser.parse_args()

    bucket_name = args.bucket.removeprefix("gs://")
    bucket = storage.Client().bucket(bucket_name)
    python = str(args.vox_dir / ".venv" / "bin" / "python")

    try:
        set_status("MODEL_DOWNLOADING")
        model_dir = download_model(bucket, args.vox_dir)
        set_status("MODEL_DOWNLOADED")

        data_dir = pull_training_data(bucket, args.vox_dir)
        rewrite_manifest(data_dir / "train.jsonl", data_dir)
        rewrite_manifest(data_dir / "val.jsonl", data_dir)
        set_status("DATA_READY")

        config_path = render_config(args.vox_dir, model_dir, data_dir, args.max_steps)
        set_status("CONFIG_READY")

        set_status("TRAINING")
        log.info(f"Starting VoxCPM2 LoRA training (max_steps={args.max_steps})...")
        best_step = train(args.vox_dir, python, config_path, bucket=bucket)
        log.info(f"Training done. Best step: {best_step.name}")
        set_status("TRAINING_DONE")

        upload_results(bucket, args.vox_dir)
        set_status("COMPLETE")
    except Exception as e:
        log.exception("Training failed")
        set_status(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
