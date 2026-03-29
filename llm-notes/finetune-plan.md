# Fine-Tuning Fish Audio S2 Pro on Personal Call Recordings

## Goal

Fine-tune Fish Audio S2 Pro (5B Dual-AR TTS) on personal call recordings so the model learns my voice, speech patterns, and conversational style. The end result is a speech-to-speech system that responds in my voice.

## Model: Fish Audio S2 Pro

- **Architecture**: Dual-AR (Slow AR 4B Qwen3-4B backbone + Fast AR 400M + DAC codec 446M)
- **Total params**: ~5B
- **HuggingFace**: `fishaudio/s2-pro`
- **License**: Fish Audio Research License (non-commercial free, commercial requires license)
- **Fine-tuning method**: LoRA on the Slow AR transformer
- **Repo**: [https://github.com/fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
- **Note**: S2 Pro was trained with GRPO RL — fine-tuning may shift distribution. Use conservative LoRA rank and monitor quality.

## Pipeline Overview

```
input/*.m4a
    │
    ▼
[1. preprocess]     → output/preprocessed/   (16kHz 16-bit mono WAV)
    │
    ▼
[2. diarize]        → output/diarized/       (speaker-labeled transcript JSON)
    │
    ▼
[3. segment]        → output/segmented/      (per-turn WAV + dialogue JSON)
    │
    ▼
[4. label]          → output/segmented/      (speaker_label: "me" / "other" added)
    │
    ▼
[5. pairformat]     → output/pairs/          (input/prompt/response audio pairs)
    │
    ▼
[6. export]         → output/fish-speech/    (*.wav + *.lab in Fish Speech format)  ← TODO
    │
    ▼
[7. quality filter]                          (PII scrub, min duration, noise check)  ← TODO
    │
    ▼
[8. fine-tune]      → Fish Speech LoRA training on S2 Pro                           ← TODO
```

## Current Status


| Step                             | Module                                             | Status                        |
| -------------------------------- | -------------------------------------------------- | ----------------------------- |
| 1. Preprocess                    | `python -m preprocess`                             | Done                          |
| 2. Diarize                       | `python -m diarize --mode aws                      | gcp`                          |
| 3. Segment                       | `python -m segment`                                | Done                          |
| 4. Label speakers                | `python -m label enroll` / `python -m label label` | Done (resemblyzer voiceprint) |
| 5. Pair format                   | `python -m pairformat`                             | Done                          |
| 6. Export to Fish Speech format  | `python -m export`                                 | **TODO**                      |
| 7. Quality filtering + PII scrub | `python -m filter`                                 | **TODO**                      |
| 8. Fine-tune S2 Pro              | Fish Speech training pipeline                      | **TODO**                      |


## Step 6: Export to Fish Speech Format

Fish Speech expects this directory structure:

```
data/
  SPEAKER_ID/
    utterance1.wav      # audio clip
    utterance1.lab      # plain text transcription (no timestamps, no markup)
    utterance2.wav
    utterance2.lab
```

**What to export**:

- All "me" turns from segmented output as individual WAV + LAB pairs
- Speaker ID = "me" (single speaker for voice cloning)
- LAB files contain just the transcript text
- Apply loudness normalization: `fap loudness-norm data-raw data --clean`

**Audio requirements**:

- Already 16kHz 16-bit mono from preprocess step
- Recommend clips between 3-30 seconds (skip very short turns <1s)

## Step 7: Quality Filtering

Before training, clean the data:

1. **PII scrubbing** (critical)
  - Names, phone numbers, account numbers, financial info
  - Use regex + NER (spaCy/presidio) to redact from .lab transcripts
  - Silence/bleep PII segments in audio
2. **Duration filtering**
  - Drop turns < 1 second (too short for meaningful speech patterns)
  - Drop turns > 60 seconds (may cause training instability)
3. **Quality checks**
  - Flag turns with excessive background noise
  - Flag turns where transcript doesn't match audio (diarization errors)

## Step 8: Fine-Tune S2 Pro

### Prerequisites

- GPU: 24GB+ VRAM (A100/L40S recommended, or cloud: RunPod, Lambda, Colab Pro)
- Clone fish-speech repo
- Download S2 Pro weights from HuggingFace

### Commands

```bash
# 1. Download model
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro

# 2. Extract semantic tokens using S2 Pro codec
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"

# 3. Pack into protobuf
python tools/llama/build_dataset.py \
    --input "data" --output "data/protos" \
    --text-extension .lab --num-workers 16

# 4. LoRA fine-tune (conservative settings for RL-trained model)
python fish_speech/train.py --config-name text2semantic_finetune \
    project=my-voice \
    model.pretrained_checkpoint=checkpoints/s2-pro \
    +lora@model.model.lora_config=r_8_alpha_16

# 5. Merge LoRA weights
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/s2-pro \
    --lora-weight results/my-voice/checkpoints/step_XXXXX.ckpt \
    --output checkpoints/s2-pro-finetuned/
```

### Training Tips

- **Start with low LoRA rank** (r=8) since S2 Pro is RL-trained — aggressive fine-tuning degrades quality
- **Short training** — 500-2000 steps is usually enough for voice adaptation
- **Monitor validation loss** — if it diverges, reduce learning rate or LoRA rank
- The model learns speech patterns by default, not just timbre
- Use reference audio prompts at inference time for timbre consistency

### Inference

```bash
python -m tools.api_server \
    --checkpoint-path checkpoints/s2-pro-finetuned \
    --listen 0.0.0.0:8080

# Then call the API with a reference audio clip of your voice
```

## Data Budget

From 1 call recording (~11 min):

- 50 turns total, 25 "me" turns
- ~24 training pairs after filtering

**For good voice cloning**: 30-60 minutes of clean "me" audio recommended.
**For speech pattern learning**: 2-5 hours of conversational data recommended.

More call recordings → better results. The pipeline scales to batch processing.