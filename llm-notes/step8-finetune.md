# Step 8: Finetune (Cloud GPU)

## Purpose

Launch a GCP A100 VM that runs the full Fish Speech S2 Pro LoRA fine-tuning pipeline: download model, extract semantic tokens, build protobuf dataset, LoRA train, merge weights. Training data is uploaded to GCS, status is polled from the local machine, and the finetuned model is downloaded when complete.

## Module

`voicetune/finetune/` — run via `python -m voicetune.finetune`

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--labeled-dir` | `./output/labeled` | Labeled dialogue.json directory (from label step) |
| `--segmented-dir` | `./output/segmented` | Segmented audio directory |
| `--data-dir` | `./output/fish-speech/data` | Where to write/read prepared dataset |
| `--min-duration` | `1.0` | Skip turns shorter than this (seconds) |
| `--max-duration` | `60.0` | Skip turns longer than this (seconds) |
| `--no-prepare` | `False` | Skip dataset preparation (use existing data) |
| `--max-steps` | `4000` | Training steps |
| `--test` | `False` | Spot A100, 1 step, auto-delete |
| `--output-dir` | `./output/finetune` | Where to download finetuned model |
| `--provider` | `gcp` | Cloud provider for fine-tuning |

## What It Does

### Local Side (pipeline.py)

1. Prepares dataset: reads labeled `dialogue.json` files from `output/labeled/`, copies "me" turns as `.wav` + `.lab` pairs to `output/fish-speech/data/me/`, filtering by duration bounds
2. Validates `{data-dir}/me/` has `.wav` + `.lab` pairs
3. Creates GCS bucket if needed, grants compute SA access
3. Zips training data and uploads `training-data.zip` to `gs://voicetune-finetune-cdcunha/`
4. Cleans up any existing instance across all candidate zones
5. Tries ~15 zones until one has A100 capacity
6. Polls `{bucket}/status.txt` every 30s until COMPLETE or FAILED
7. Downloads finetuned model to `{output-dir}/s2-pro-finetuned/`

### VM Side

Two files run on the VM:

**`startup_gcp.sh`** — bash bootstrap (runs as GCE startup script):
1. Install system deps (`python3.12-venv`, `git`, `portaudio19-dev`)
2. Install NVIDIA drivers if needed
3. Clone fish-speech, create venv, install deps
4. Pull `train_gcp.py` from GCS and hand off to Python

**`train_gcp.py`** — Python training orchestrator (pulled from GCS):
1. Download S2 Pro model (GCS cache first, HuggingFace fallback)
2. Pull training data from GCS
3. Extract VQ semantic tokens
4. Build protobuf dataset
5. LoRA training (up to 3 retries with checkpoint resume)
6. Merge LoRA weights into base model
7. Upload finetuned model + training results to GCS

On success, the bash script self-deletes the VM. On failure, `train_gcp.py` uploads a FAILED status with the error message and the VM stays up for debugging.

Status is reported to `{bucket}/status.txt` at each phase: STARTING → DRIVERS_READY → DEPS_INSTALLED → MODEL_DOWNLOADING → MODEL_DOWNLOADED → DATA_READY → VQ_DONE → DATASET_BUILT → TRAINING → TRAINING_DONE → MERGE_DONE → COMPLETE.

## Input/Output

**Input:**
- `output/labeled/{call_id}/dialogue.json` (from label step)
- `output/segmented/{call_id}/turns/*.wav` (audio files)

**Output:** `output/finetune/s2-pro-finetuned/` — merged Fish Speech model weights

## GCP Configuration

| Setting | Value |
|---------|-------|
| Project | `ehc-cdcunha-1a374e` |
| Bucket | `gs://voicetune-finetune-cdcunha` |
| Machine | `a2-highgpu-1g` (A100 40GB) |
| Image | `pytorch-2-9-cu129-ubuntu-2404-nvidia-580` |
| Instance | `voicetune-finetune` (or `voicetune-finetune-test`) |

These are constants in `pipeline.py`. The bucket is passed to the VM via instance metadata.

## Test Mode

`python -m voicetune.finetune --test` uses a SPOT A100 with 1 training step for validating the pipeline end-to-end without burning GPU hours.

## Training Tips

- Start with low LoRA rank (r=8) — S2 Pro is RL-trained, aggressive fine-tuning degrades quality
- 500–2000 steps usually enough for voice adaptation
- Monitor via serial log: `gcloud compute instances get-serial-port-output voicetune-finetune --zone=ZONE --project=ehc-cdcunha-1a374e`
- The VM caches the base model in GCS after first download, so subsequent runs skip the ~10GB HuggingFace download

## Dependencies

Requires `gcloud` CLI authenticated with the GCP project. No Python GPU deps needed locally — all heavy computation happens on the VM.
