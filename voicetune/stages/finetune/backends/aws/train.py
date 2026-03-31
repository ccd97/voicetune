"""Training orchestrator for AWS EC2 instance.

Called by startup.sh after environment setup. Handles model download,
data pull, VQ extraction, dataset build, LoRA training with retry, merge,
and result upload. Reports status via S3.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

HF_REPO = "https://huggingface.co/fishaudio/s2-pro/resolve/main"
HF_SMALL_FILES = [
    "config.json", "chat_template.jinja", "LICENSE.md", "README.md",
    "model.safetensors.index.json", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json", "overview.png",
]
HF_LARGE_FILES = [
    "codec.pth",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]

PROJECT = "my-voice"
LORA_RANK = 8
LORA_ALPHA = 16
MAX_RETRIES = 3


def _run(cmd: list[str], cwd: Path | str | None = None, env: dict | None = None) -> None:
    log.info(f"  Running: {' '.join(cmd)}")
    run_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=run_env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd[:3])}...")


def set_status(s3, bucket: str, status: str) -> None:
    s3.put_object(Bucket=bucket, Key="status.txt", Body=status.encode())


def _download_s3_prefix(s3, bucket: str, prefix: str, local_dir: Path) -> int:
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            count += 1
    return count


def _upload_s3_dir(s3, bucket: str, local_dir: Path, prefix: str) -> int:
    count = 0
    for f in sorted(local_dir.rglob("*")):
        if not f.is_file():
            continue
        key = f"{prefix}{f.relative_to(local_dir)}"
        s3.upload_file(str(f), bucket, key)
        count += 1
    return count


def download_model(s3, bucket: str, fish_dir: Path) -> None:
    ckpt_dir = fish_dir / "checkpoints" / "s2-pro"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        s3.head_object(Bucket=bucket, Key="s2-pro-base/codec.pth")
        log.info("Pulling model from S3 cache...")
        _download_s3_prefix(s3, bucket, "s2-pro-base/", ckpt_dir)
        return
    except Exception:
        pass

    log.info("First run — downloading from HuggingFace via wget...")
    for f in HF_SMALL_FILES:
        log.info(f"  {f}")
        _run(["wget", "-q", f"{HF_REPO}/{f}", "-O", f], cwd=ckpt_dir)

    for f in HF_LARGE_FILES:
        log.info(f"  {f} (large file)...")
        _run(["wget", "-q", f"{HF_REPO}/{f}", "-O", f], cwd=ckpt_dir)

    log.info("Caching model to S3...")
    _upload_s3_dir(s3, bucket, ckpt_dir, "s2-pro-base/")


def pull_training_data(s3, bucket: str, fish_dir: Path) -> None:
    log.info("Pulling training data from S3...")
    data_dir = fish_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    count = _download_s3_prefix(s3, bucket, "data/", data_dir)
    log.info(f"  Downloaded {count} files")


def extract_vq(fish_dir: Path, python: str) -> None:
    log.info("Extracting semantic tokens...")
    _run([
        python, "tools/vqgan/extract_vq.py", "data",
        "--num-workers", "1", "--batch-size", "16",
        "--config-name", "modded_dac_vq",
        "--checkpoint-path", "checkpoints/s2-pro/codec.pth",
    ], cwd=fish_dir)


def build_dataset(fish_dir: Path, python: str) -> None:
    log.info("Building dataset...")
    _run([
        python, "tools/llama/build_dataset.py",
        "--input", "data", "--output", "data/protos",
        "--text-extension", ".lab", "--num-workers", "1",
    ], cwd=fish_dir)


def find_latest_checkpoint(results_dir: Path) -> Path | None:
    ckpts = sorted(
        results_dir.glob("step_*.ckpt"),
        key=lambda p: int(re.search(r"step_(\d+)", p.stem).group(1)),
    )
    return ckpts[-1] if ckpts else None


def train_lora(fish_dir: Path, python: str, max_steps: int) -> Path:
    lora_config = f"r_{LORA_RANK}_alpha_{LORA_ALPHA}"
    cmd = [
        python, "fish_speech/train.py",
        "--config-name", "text2semantic_finetune",
        f"project={PROJECT}",
        "pretrained_ckpt_path=checkpoints/s2-pro",
        "tokenizer.model_path=checkpoints/s2-pro",
        f"+lora@model.model.lora_config={lora_config}",
        f"trainer.max_steps={max_steps}",
        "data.batch_size=8",
        "max_length=1024",
    ]
    if max_steps < 100:
        cmd.append(f"callbacks.model_checkpoint.every_n_train_steps={max_steps}")

    train_env = {
        "WANDB_MODE": "disabled",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HYDRA_FULL_ERROR": "1",
    }
    results_dir = fish_dir / "results" / PROJECT / "checkpoints"

    for attempt in range(1, MAX_RETRIES + 1):
        run_cmd = list(cmd)
        ckpt = find_latest_checkpoint(results_dir) if results_dir.exists() else None
        if ckpt:
            log.info(f"Resuming from {ckpt.name} (attempt {attempt}/{MAX_RETRIES})")
            run_cmd.append(f"ckpt_path={ckpt}")

        run_env = {**os.environ, **train_env}
        result = subprocess.run(run_cmd, cwd=str(fish_dir), env=run_env)
        if result.returncode == 0:
            break
        log.error(f"Training failed (exit {result.returncode}), attempt {attempt}/{MAX_RETRIES}")
        if attempt == MAX_RETRIES:
            raise RuntimeError(f"Training failed after {MAX_RETRIES} attempts")

    ckpt = find_latest_checkpoint(results_dir)
    if not ckpt:
        raise RuntimeError(f"No checkpoint found in {results_dir}")
    return ckpt


def merge_lora(fish_dir: Path, python: str, checkpoint: Path) -> None:
    log.info(f"Merging LoRA weights from {checkpoint.name}...")
    _run([
        python, "tools/llama/merge_lora.py",
        "--lora-config", f"r_{LORA_RANK}_alpha_{LORA_ALPHA}",
        "--base-weight", "checkpoints/s2-pro",
        "--lora-weight", str(checkpoint),
        "--output", "checkpoints/s2-pro-finetuned/",
    ], cwd=fish_dir)


def upload_results(s3, bucket: str, fish_dir: Path) -> None:
    log.info("Uploading finetuned model to S3...")
    n = _upload_s3_dir(s3, bucket, fish_dir / "checkpoints" / "s2-pro-finetuned", "model/")
    log.info(f"  Uploaded {n} model files")
    n = _upload_s3_dir(s3, bucket, fish_dir / "results" / PROJECT, "results/")
    log.info(f"  Uploaded {n} result files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fish-dir", type=Path, default=Path("/opt/fish-speech"))
    args = parser.parse_args()

    s3 = boto3.client("s3")
    python = str(args.fish_dir / ".venv" / "bin" / "python")

    try:
        set_status(s3, args.bucket, "MODEL_DOWNLOADING")
        download_model(s3, args.bucket, args.fish_dir)
        set_status(s3, args.bucket, "MODEL_DOWNLOADED")

        pull_training_data(s3, args.bucket, args.fish_dir)
        set_status(s3, args.bucket, "DATA_READY")

        extract_vq(args.fish_dir, python)
        set_status(s3, args.bucket, "VQ_DONE")

        build_dataset(args.fish_dir, python)
        set_status(s3, args.bucket, "DATASET_BUILT")

        set_status(s3, args.bucket, "TRAINING")
        log.info(f"Starting LoRA training (max_steps={args.max_steps})...")
        checkpoint = train_lora(args.fish_dir, python, args.max_steps)
        set_status(s3, args.bucket, "TRAINING_DONE")

        merge_lora(args.fish_dir, python, checkpoint)
        set_status(s3, args.bucket, "MERGE_DONE")

        upload_results(s3, args.bucket, args.fish_dir)
        set_status(s3, args.bucket, "COMPLETE")
    except Exception as e:
        log.exception("Training failed")
        set_status(s3, args.bucket, f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
