"""Fish Speech S2 Pro LoRA fine-tuning pipeline.

Orchestrates the Fish Speech training pipeline:
1. Download S2 Pro model weights (if not present)
2. Extract semantic tokens using S2 Pro codec
3. Pack training data into protobuf format
4. Run LoRA fine-tuning
5. Merge LoRA weights into base model

All commands run inside the fish-speech repo directory.
"""

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_REPO = "fishaudio/s2-pro"
CODEC_FILENAME = "codec.pth"


def _run_cmd(cmd: list[str], cwd: Path, description: str) -> None:
    """Run a subprocess command with logging and error handling."""
    log.info(f"  Running: {' '.join(cmd)}")
    log.info(f"  cwd: {cwd}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {result.returncode}")


def prepare_data_link(fish_speech_dir: Path, data_dir: Path) -> dict:
    """Symlink or verify the exported data is accessible from fish-speech/data/."""
    target = fish_speech_dir / "data"
    source = data_dir.resolve()

    if target.is_symlink():
        existing = target.resolve()
        if existing == source:
            log.info(f"  Data symlink already correct: {target} -> {source}")
            return {"status": "exists", "path": str(target)}
        else:
            log.warning(f"  Removing stale symlink: {target} -> {existing}")
            target.unlink()
    elif target.exists():
        raise RuntimeError(
            f"{target} exists and is not a symlink. "
            f"Please remove or rename it, then retry."
        )

    target.symlink_to(source)
    log.info(f"  Created symlink: {target} -> {source}")
    return {"status": "created", "path": str(target)}


def download_model(fish_speech_dir: Path, skip: bool = False) -> dict:
    """Download S2 Pro model weights from HuggingFace."""
    checkpoint_dir = fish_speech_dir / "checkpoints" / "s2-pro"
    codec_path = checkpoint_dir / CODEC_FILENAME

    if skip:
        log.info("  Skipping model download (--skip-download)")
        return {"status": "skipped", "path": str(checkpoint_dir)}

    if codec_path.exists():
        log.info(f"  Model already downloaded: {codec_path}")
        return {"status": "exists", "path": str(checkpoint_dir)}

    _run_cmd(
        ["huggingface-cli", "download", MODEL_REPO, "--local-dir", str(checkpoint_dir)],
        cwd=fish_speech_dir,
        description="Model download",
    )
    return {"status": "downloaded", "path": str(checkpoint_dir)}


def extract_semantic_tokens(
    fish_speech_dir: Path, num_workers: int, batch_size: int
) -> dict:
    """Extract semantic tokens using S2 Pro codec (VQ extraction)."""
    data_me = fish_speech_dir / "data" / "me"
    wav_count = len(list(data_me.glob("*.wav")))
    npy_count = len(list(data_me.glob("*.npy")))

    if npy_count >= wav_count and wav_count > 0:
        log.info(f"  VQ tokens already extracted ({npy_count} .npy for {wav_count} .wav)")
        return {"status": "skipped", "wav_count": wav_count, "npy_count": npy_count}

    _run_cmd(
        [
            sys.executable, "tools/vqgan/extract_vq.py", "data",
            "--num-workers", str(num_workers),
            "--batch-size", str(batch_size),
            "--config-name", "modded_dac_vq",
            "--checkpoint-path", "checkpoints/s2-pro/codec.pth",
        ],
        cwd=fish_speech_dir,
        description="VQ extraction",
    )

    npy_count = len(list(data_me.glob("*.npy")))
    return {"status": "completed", "wav_count": wav_count, "npy_count": npy_count}


def build_dataset(fish_speech_dir: Path, dataset_workers: int) -> dict:
    """Pack wav+lab pairs into protobuf format for training."""
    protos_dir = fish_speech_dir / "data" / "protos"

    if protos_dir.exists() and any(protos_dir.iterdir()):
        count = len(list(protos_dir.iterdir()))
        log.info(f"  Protobuf dataset already exists ({count} files in {protos_dir})")
        return {"status": "skipped", "protos_count": count}

    _run_cmd(
        [
            sys.executable, "tools/llama/build_dataset.py",
            "--input", "data",
            "--output", "data/protos",
            "--text-extension", ".lab",
            "--num-workers", str(dataset_workers),
        ],
        cwd=fish_speech_dir,
        description="Dataset build",
    )

    count = len(list(protos_dir.iterdir())) if protos_dir.exists() else 0
    return {"status": "completed", "protos_count": count}


def find_latest_checkpoint(results_dir: Path) -> Path | None:
    """Find the latest checkpoint file in training results."""
    ckpts = sorted(
        results_dir.glob("step_*.ckpt"),
        key=lambda p: int(re.search(r"step_(\d+)", p.stem).group(1)),
    )
    return ckpts[-1] if ckpts else None


def train_lora(
    fish_speech_dir: Path,
    project: str,
    lora_rank: int,
    lora_alpha: int,
    max_steps: int | None,
) -> dict:
    """Run LoRA fine-tuning on S2 Pro."""
    lora_config = f"r_{lora_rank}_alpha_{lora_alpha}"

    cmd = [
        sys.executable, "fish_speech/train.py",
        "--config-name", "text2semantic_finetune",
        f"project={project}",
        f"model.pretrained_checkpoint=checkpoints/s2-pro",
        f"+lora@model.model.lora_config={lora_config}",
    ]
    if max_steps is not None:
        cmd.append(f"trainer.max_steps={max_steps}")

    _run_cmd(cmd, cwd=fish_speech_dir, description="LoRA training")

    # Find the checkpoint produced
    results_dir = fish_speech_dir / "results" / project / "checkpoints"
    checkpoint = find_latest_checkpoint(results_dir)

    return {
        "status": "completed",
        "lora_config": lora_config,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def merge_lora(
    fish_speech_dir: Path,
    project: str,
    lora_rank: int,
    lora_alpha: int,
    checkpoint_step: int | None,
    skip: bool = False,
) -> dict:
    """Merge LoRA weights back into the base model."""
    if skip:
        log.info("  Skipping LoRA merge (--skip-merge)")
        return {"status": "skipped"}

    lora_config = f"r_{lora_rank}_alpha_{lora_alpha}"
    results_dir = fish_speech_dir / "results" / project / "checkpoints"

    if checkpoint_step is not None:
        ckpt = results_dir / f"step_{checkpoint_step:05d}.ckpt"
        if not ckpt.exists():
            # Try without zero-padding
            ckpt = results_dir / f"step_{checkpoint_step}.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: step_{checkpoint_step} in {results_dir}")
    else:
        ckpt = find_latest_checkpoint(results_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoints found in {results_dir}")

    log.info(f"  Using checkpoint: {ckpt.name}")

    output_dir = fish_speech_dir / "checkpoints" / "s2-pro-finetuned"

    _run_cmd(
        [
            sys.executable, "tools/llama/merge_lora.py",
            "--lora-config", lora_config,
            "--base-weight", "checkpoints/s2-pro",
            "--lora-weight", str(ckpt),
            "--output", str(output_dir),
        ],
        cwd=fish_speech_dir,
        description="LoRA merge",
    )

    return {
        "status": "completed",
        "checkpoint": str(ckpt),
        "output": str(output_dir),
    }


def run_finetune(
    fish_speech_dir: Path,
    data_dir: Path,
    project: str,
    lora_rank: int,
    lora_alpha: int,
    batch_size: int,
    num_workers: int,
    dataset_workers: int,
    max_steps: int | None,
    skip_download: bool,
    skip_merge: bool,
    checkpoint_step: int | None,
    output_dir: Path,
) -> dict:
    """Run the complete fine-tuning pipeline."""
    timings = {}
    results = {}

    steps = [
        ("data_link", lambda: prepare_data_link(fish_speech_dir, data_dir)),
        ("download", lambda: download_model(fish_speech_dir, skip=skip_download)),
        ("extract_vq", lambda: extract_semantic_tokens(fish_speech_dir, num_workers, batch_size)),
        ("build_dataset", lambda: build_dataset(fish_speech_dir, dataset_workers)),
        ("train", lambda: train_lora(fish_speech_dir, project, lora_rank, lora_alpha, max_steps)),
        ("merge", lambda: merge_lora(fish_speech_dir, project, lora_rank, lora_alpha, checkpoint_step, skip=skip_merge)),
    ]

    for name, fn in steps:
        log.info(f"{'=' * 50}")
        log.info(f"FINETUNE SUB-STEP: {name}")
        log.info(f"{'=' * 50}")
        t0 = time.time()
        results[name] = fn()
        elapsed = time.time() - t0
        timings[name] = round(elapsed, 1)
        log.info(f"  {name} completed in {elapsed:.1f}s — {results[name].get('status', 'done')}")

    total = sum(timings.values())

    summary = {
        "project": project,
        "fish_speech_dir": str(fish_speech_dir),
        "lora_config": f"r_{lora_rank}_alpha_{lora_alpha}",
        "steps": {name: {**results[name], "elapsed": timings[name]} for name in timings},
        "total_elapsed": round(total, 1),
        "finetuned_model": results.get("merge", {}).get("output"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "finetune_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"{'=' * 50}")
    log.info("FINETUNE COMPLETE")
    log.info(f"{'=' * 50}")
    for name, elapsed in timings.items():
        log.info(f"  {name:20s} {elapsed:8.1f}s")
    log.info(f"  {'TOTAL':20s} {total:8.1f}s")
    log.info(f"Summary: {summary_path}")

    return summary
