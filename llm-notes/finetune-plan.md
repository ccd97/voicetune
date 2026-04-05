# Fine-Tuning VoxCPM2 on Personal Call Recordings

## Goal

Fine-tune VoxCPM2 (2B tokenizer-free diffusion-autoregressive TTS) on personal call recordings so the model learns my voice, speech patterns, and conversational style. The end result is a speech-to-speech system that responds in my voice.

## Model: VoxCPM2

- **Architecture**: Tokenizer-free diffusion autoregressive — LocEnc → TSLM → RALM → LocDiT operating in AudioVAE V2 latent space, on a MiniCPM-4 backbone
- **Total params**: 2B
- **HuggingFace**: `openbmb/VoxCPM2` (public, Apache-2.0, not gated)
- **Fine-tuning method**: LoRA on LM + DiT + projection (`enable_lm=true, enable_dit=true, enable_proj=true, r=64, alpha=64`)
- **Repo**: [https://github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **Audio**: 16 kHz encoder input (matches pipeline output), 48 kHz decoder output via AudioVAE V2 super-resolution
- **Inference**: Base + LoRA side-by-side, hot-swappable. No merge step required.

## Pipeline Overview

```
input/*.m4a
    │
    ▼
[1. preprocess]   → output/preprocessed/   (16kHz 16-bit mono WAV)
    │
    ▼
[2. diarize]      → output/diarized/       (raw speaker-labeled transcript JSON)
    │
    ▼
[3. scrub]        → output/scrubbed/       (PII scrubbed from transcripts)
    │
    ▼
[4. validation]   → output/validated/      (LLM-graded transcripts with file/turn issue codes)
    │
    ▼
[5. segment]      → output/segmented/{call_id}/  (per-turn WAV + dialogue JSON; validation codes applied)
    │
    ▼
[6. filter]       → output/filtered/{call_id}/   (audio-quality drops, multi-speaker checks)
    ▼
[7. label]        → output/labeled/{call_id}/    (speaker_label: "me" / "other" added)
    │
    ▼
[8. finetune]     → VoxCPM2 LoRA training (GCP A100 VM)
```

## Current Status

| Step             | Module                                             | Status                        |
| ---------------- | -------------------------------------------------- | ----------------------------- |
| 1. Preprocess    | `python -m voicetune.stages.preprocess`            | Done                          |
| 2. Diarize       | `python -m voicetune.stages.diarize`               | Done                          |
| 3. Scrub         | `python -m voicetune.stages.scrub`                 | Done                          |
| 4. Validation    | `python -m voicetune.stages.validation`            | Done                          |
| 5. Segment       | `python -m voicetune.stages.segment`               | Done                          |
| 6. Filter        | `python -m voicetune.stages.filter`                | Done (includes clip cleaning) |
| 7. Label         | `python -m voicetune.stages.label enroll` / `label`| Done (resemblyzer voiceprint) |
| 8. Fine-tune     | `python -m voicetune.stages.finetune`              | Migrated from Fish Speech     |


## Step 8: Export to VoxCPM Format

VoxCPM expects a **JSONL manifest**, one JSON object per line:

```jsonl
{"audio": "/abs/path/to/me/call123_turn_005.wav", "text": "Hello, how are you?", "duration": 2.8}
{"audio": "/abs/path/to/me/call123_turn_009.wav", "text": "I think that sounds good.", "duration": 3.1}
```

Required fields: `audio`, `text`. Optional: `duration` (speeds up length filtering), `ref_audio` (same-speaker reference clip for voice-clone conditioning), `dataset_id` (multi-dataset mixing).

**What step 8 exports**:

- All "me" turns from labeled output as per-turn WAV files under `output/voxcpm/data/me/`
- A `train.jsonl` manifest + ~5% `val.jsonl` holdout, stratified by `call_id` so no call is split across train/val (skipped for small datasets)
- No `.lab` sidecar files — text lives inside the manifest

**Audio requirements**:

- Already 16kHz 16-bit mono from the preprocess step (matches VoxCPM2 encoder)
- Clips between ~3–30 seconds (filter step enforces this)

## Step 8: Fine-Tune VoxCPM2

### Prerequisites

- GPU: 24GB+ VRAM (A100 40GB used; LoRA fits in ~20GB)
- `gcloud` authenticated against `ehc-cdcunha-1a374e`
- `GCP_PROJECT_ID` in `.env`

### Commands (what runs on the VM)

```bash
# 1. Clone VoxCPM
git clone https://github.com/OpenBMB/VoxCPM.git /opt/voxcpm
cd /opt/voxcpm
python3 -m venv .venv
.venv/bin/pip install -e . tensorboardX pyyaml

# 2. Download base model (one time, then cached in GCS)
python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2', local_dir='models/VoxCPM2')"

# 3. LoRA fine-tune (VM orchestrator renders conf/me_lora.yaml at runtime)
python scripts/train_voxcpm_finetune.py --config_path conf/me_lora.yaml
```

### Training config (rendered by `backends/gcp/train.py`)

```yaml
pretrained_path: /opt/voxcpm/models/VoxCPM2
train_manifest: /opt/voxcpm/data/train.jsonl
val_manifest: /opt/voxcpm/data/val.jsonl
sample_rate: 16000
out_sample_rate: 48000
batch_size: 1
grad_accum_steps: 16
num_workers: 4
num_iters: 200
max_steps: 200
save_interval: 50
valid_interval: 50
learning_rate: 0.0005
weight_decay: 0.01
warmup_steps: 20
max_batch_tokens: 8192
max_grad_norm: 1.0
save_path: /opt/voxcpm/results/me-lora/lora
tensorboard: /opt/voxcpm/results/me-lora/tensorboard
lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
lora:
  enable_lm: true
  enable_dit: true
  enable_proj: true
  r: 64
  alpha: 128
  dropout: 0.0
```

### Training Tips

- `enable_dit: true` is required for audio quality.
- Target 1–3 epochs for single-speaker cloning. With 300–900 clips at effective bs 16 that's ~20–170 optimizer steps; `--max-steps 200` lands just past 3 epochs on ~1k clips.
- Learning rate 5e-4 (5× VoxCPM default). At 1e-4 the 200-step run produced effective ΔW 7.5 vs 16.9 at 800 steps, too small to shift voice identity.
- Best-step selection uses `val/loss/total` (VoxCPM splits val loss into `total`/`diff`/`stop`; there is no bare `val/loss` tag). Falls back to the latest step when no val manifest exists.
- Language mix: if one language dominates, the LoRA anchors voice identity to it and other-language inference reverts to the base prior. `prepare_dataset` classifies by script (Latin → English, Devanagari + Latin → mixed) plus the call-level `language` (`mr-IN` → Marathi, `hi` → Hindi). Use `--keep-languages english,hindi` to drop the rest. Oversampling 65 Hindi clips 7–8× made this worse, not better.
- Per-call cap: `--max-turns-per-call 50` (default) prevents a single long recording from dominating; one 177-turn call in the first run was 16% of the data.
- Duration bounds live in the filter stage (`--min-duration 3.0 --max-duration 30.0`); finetune consumes `output/filtered/` as-is.
- Trailing silence: clips with >0.5 s of trailing silence cause "run-away" inference. The filter stage rejects decayed clips via `TAIL_DECAY_CODE`; if run-away still appears, add `librosa.effects.trim(top_db=40)` in `prepare_dataset`.
- VoxCPM's pretraining is sentence-cased; ALL-CAPS transcripts degrade text adherence.

### Inference

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_weights_path="output/finetune/voxcpm2-lora/latest",
    load_denoiser=False,
)
wav = model.generate(text="Hello!", cfg_value=2.0, inference_timesteps=10)
sf.write("out.wav", wav, model.tts_model.sample_rate)
```

## Data Budget

From 1 call recording (~11 min): 50 turns total, 25 "me" turns, ~24 training pairs after filtering.

Target 30–60 min of clean "me" audio for voice cloning; 2–5 h of conversational data for speech-pattern learning.
