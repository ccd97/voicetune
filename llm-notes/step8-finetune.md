# Step 8: Finetune (Cloud GPU)

## Purpose

Run VoxCPM2 LoRA fine-tuning on a GCP A100 VM. Uploads training data to GCS, polls status from the local machine, downloads the LoRA adapter when complete.

## Module

`voicetune/stages/finetune/` — run via `python -m voicetune.stages.finetune`

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--run-dir` | `./output` | Base output directory |
| `--labeled-dir` | `<run-dir>/labeled` | Labeled dialogue.json directory (from label step) |
| `--filtered-dir` | `<run-dir>/filtered` | Filtered per-call audio directory (from the filter step) |
| `--data-dir` | `<run-dir>/voxcpm/data` | Where to write/read prepared dataset |
| `--no-prepare` | `False` | Skip dataset preparation (use existing data) |
| `--max-steps` | `200` | Training steps (effective batch size 16; ~3 epochs on ~1k clips) |
| `--max-turns-per-call` | `50` | Per-call turn cap |
| `--keep-languages` | _(all)_ | Comma-separated list of language buckets to keep (e.g. `english,hindi`). Valid: `english`, `hindi`, `marathi`, `mixed`, `devanagari_unknown`, `other` |
| `--test` | `False` | Spot A100, 1 step, auto-delete |
| `--output-dir` | `<run-dir>/finetune` | Where to download the LoRA adapter |

## What It Does

### Local Side (pipeline.py)

1. Prepares dataset: reads labeled `dialogue.json` from `output/labeled/`, copies matching "me" turn WAVs from `output/filtered/` to `output/voxcpm/data/me/*.wav`, and emits `train.jsonl` + `val.jsonl` (random 6% turn-level holdout, seed=0; skipped when <20 turns). Calls with more than `max_turns_per_call` turns (default 50) are downsampled with a stable seed. When `keep_languages` is set, clips are classified by `_classify_language` and only allowed buckets are kept (e.g. `--keep-languages english,hindi` drops Marathi).
2. Validates `{data-dir}/train.jsonl` is present and well-formed.
3. Creates GCS bucket if needed, grants compute SA access.
4. Zips the dataset directory and uploads `training-data.zip` to `gs://voicetune-finetune-cdcunha/`.
5. Cleans up any existing instance across all candidate zones.
6. Tries ~15 zones until one has A100 capacity.
7. Polls guest attribute `voicetune/status` every 30s until COMPLETE or FAILED.
8. Downloads the LoRA adapter (`step_*/` + `latest/`) to `{output-dir}/voxcpm2-lora/` and tensorboard logs to `{output-dir}/tensorboard/`.

### VM Side

Two files run on the VM:

**`startup.sh`** — bash bootstrap (GCE startup script):
1. Install system deps (`python3.12-venv`, `git`)
2. Install NVIDIA drivers if needed
3. Clone `https://github.com/OpenBMB/VoxCPM` into `/opt/voxcpm`, create venv, `pip install -e .` plus training extras (`tensorboardX`, `pyyaml`, `google-cloud-storage`, `huggingface_hub`, cu129 torch/torchaudio)
4. Pull `train_gcp.py` from GCS and hand off to Python

**`train_gcp.py`** — training orchestrator (pulled from GCS):
1. Download `openbmb/VoxCPM2` (GCS cache first, HuggingFace fallback via `huggingface_hub.snapshot_download`; token is optional since the repo is public)
2. Pull training data from GCS and rewrite manifest paths to VM-local absolute paths
3. Render `conf/me_lora.yaml` (bf16 autocast; r=64 alpha=128 → scaling 2.0; enable_lm+enable_dit+enable_proj; bs=1 × grad_accum=16 → effective bs 16; lr 5e-4, cosine + warmup; save/valid every 50 steps)
4. Run `scripts/train_voxcpm_finetune.py --config_path conf/me_lora.yaml` (a background thread mirrors each `step_*/` directory to GCS as soon as it appears)
5. Pick best step by TensorBoard `val/loss/total` (falls back to latest step if no val manifest)
6. Upload tensorboard logs to GCS

On success the bash script's parent `startup.sh` returns and the provider deletes the VM. On failure `train_gcp.py` sets a FAILED status with the error message and the VM stays up for debugging.

Status is reported via guest attribute `voicetune/status` at each phase: `STARTING → DRIVERS_READY → DEPS_INSTALLED → MODEL_DOWNLOADING → MODEL_DOWNLOADED → DATA_READY → CONFIG_READY → TRAINING → TRAINING_DONE → COMPLETE`.

## Input/Output

**Input:**
- `output/labeled/{call_id}/dialogue.json` (from label step)
- `output/filtered/{call_id}/turns/*.wav` (from the filter step)

**Prepared dataset layout** (`output/voxcpm/data/`):
```
me/{call_id}_turn_NNN.wav
train.jsonl   # {"audio": "...", "text": "...", "duration": N.N} per line
val.jsonl     # same schema, ~6% random turn-level holdout (skipped when <20 turns)
export_summary.json
```

**Output:** `output/finetune/voxcpm2-lora/` — LoRA adapter directory tree:
```
step_0000050/
  lora_weights.safetensors
  lora_config.json
  optimizer.pth
  scheduler.pth
  training_state.json
step_0000100/
  ...
latest/       # copy of most recent step (shutil.copytree, not a symlink)
```

Plus `output/finetune/tensorboard/` for training curves.

## Inference with the LoRA adapter

```python
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_weights_path="output/finetune/voxcpm2-lora/latest",
)
wav = model.generate(text="...", cfg_value=2.0, inference_timesteps=10)
```

VoxCPM loads base + adapter side-by-side at runtime; hot-swap is supported via `model.set_lora_enabled(False/True)` and `model.load_lora(path)`. No merge step is needed. If a merged monolithic model is ever required, switch to the full SFT config (`conf/voxcpm_v2/voxcpm_finetune_all.yaml`) which saves a standalone model directory.

## GCP Configuration

| Setting | Value |
|---------|-------|
| Project | `ehc-cdcunha-1a374e` |
| Bucket | `gs://voicetune-finetune-cdcunha` |
| Machine | `a2-highgpu-1g` (A100 40GB) |
| Image | `pytorch-2-9-cu129-ubuntu-2404-nvidia-580` |
| Instance | `voicetune-finetune` (or `voicetune-finetune-test`) |

These are constants in `backends/gcp/provider.py`. The bucket is passed to the VM via instance metadata.

Monitor a running job via serial console:

```
gcloud compute instances get-serial-port-output voicetune-finetune --zone=ZONE --project=ehc-cdcunha-1a374e
```

## Test Mode

`python -m voicetune.stages.finetune --test` uses a SPOT A100 with 1 training step to validate the pipeline end-to-end.

## Training Tips

- LoRA config: `r=64, alpha=128, enable_lm=true, enable_dit=true, enable_proj=true`. VoxCPM2's doc default is r=32; A100 40 GB at bs=1 fits r=64. `alpha=2*r` (scaling 2.0): an earlier `alpha=r` 800-step run gave DiT ΔW ≤ 0.012, which wasn't enough to move voice identity out of the training-language distribution. Disabling `enable_dit` tanks audio quality.
- A100 40 GB: `batch_size=1, grad_accum_steps=16, max_batch_tokens=8192`. VoxCPM docs cite ~20 GB at bs=2 but that OOMs on CUDA 12.9; bs=1 fits.
- Learning rate 5e-4, 5× VoxCPM's LoRA default. 1e-4 for 200 steps moved val/loss 1.7% (effective ΔW 7.5 vs 16.9 at 800 steps); grad norms sit at 0.1–0.2 so the 1.0 clip never fires.
- 1–3 epochs for single-speaker cloning. 300–900 clips at effective bs 16 is ~20–170 steps; `--max-steps 200` gives ~3 epochs on ~1k clips. An 800-step run regressed `val/loss/total` from 0.924 → 1.105.
- Best-step selection reads `val/loss/total` from TensorBoard (VoxCPM splits val loss into `total`/`diff`/`stop`; there's no bare `val/loss` tag). Falls back to the latest step when no val manifest exists.
- Language mix: when one language dominates, the LoRA anchors voice identity to it and other-language inference reverts to the base prior. `prepare_dataset` classifies by script (Latin → English, Devanagari + Latin → mixed) plus the call-level `language` (`mr-IN` → Marathi, `hi` → Hindi). `--keep-languages english,hindi` drops the rest. Oversampling 65 Hindi clips 7–8× made it worse.
- Per-call cap `--max-turns-per-call 50` stops one long call from dominating: in the first run a single 177-turn call was 16% of the data.
- Duration bounds (`--min-duration 3.0 --max-duration 30.0`) live in the filter stage; finetune consumes `output/filtered/` as-is.
- Trailing silence >0.5 s causes "run-away" inference. The filter stage drops decayed clips via `TAIL_DECAY_CODE`; if run-away still shows up, add `librosa.effects.trim(top_db=40)` in `prepare_dataset`.
- VoxCPM's pretraining is sentence-cased; ALL-CAPS transcripts degrade text adherence.

## HuggingFace Auth

`openbmb/VoxCPM2` is not gated (Apache-2.0, public), so `HF_TOKEN` can be empty. Set it in `.env` if HF ever flips the gate or you need a private fork.

## Dependencies

Requires `gcloud` CLI authenticated with the GCP project and `GCP_PROJECT_ID` in `.env`. No Python GPU deps needed locally — all heavy computation happens on the VM.
