# voicetune — design

## Goal

Fine-tune VoxCPM2 on personal call recordings so it clones my voice and speech patterns. The current deliverable is a voice-cloned TTS model; the longer-term target is a speech-to-speech system that replies in my voice.

## Why VoxCPM2

- Tokenizer-free diffusion-autoregressive TTS: LocEnc → TSLM → RALM → LocDiT in AudioVAE V2 latent space, on a MiniCPM-4 backbone.
- 2B params, public on HuggingFace as `openbmb/VoxCPM2` (Apache-2.0, not gated).
- LoRA fits on a single A100 40 GB (`r=64, alpha=128, enable_lm+enable_dit+enable_proj`; ~20 GB at bs=1).
- 16 kHz encoder input matches the pipeline's preprocess output; 48 kHz decoder output via AudioVAE V2 super-resolution.
- Base + LoRA load side-by-side at inference, hot-swappable, no merge step.
- Repo: <https://github.com/OpenBMB/VoxCPM>

## Pipeline

```
input/*.m4a
    │
    ▼
[1. preprocess]   → output/preprocessed/       16 kHz 16-bit mono WAV
    │
    ▼
[2. diarize]      → output/diarized/           speaker-labeled transcript JSON
    │
    ▼
[3. scrub]        → output/scrubbed/           PII-scrubbed transcripts
    │
    ▼
[4. validation]   → output/validated/          LLM-graded; file/turn issue codes
    │
    ▼
[5. segment]      → output/segmented/{call}/   per-turn WAV + dialogue JSON
    │
    ▼
[6. filter]       → output/filtered/{call}/    audio-quality + multi-speaker drops
    │
    ▼
[7. label]        → output/labeled/{call}/     speaker_label: "me" / "other"
    │
    ▼
[8. finetune]     → output/finetune/           VoxCPM2 LoRA (GCP A100 VM)
    │
    ▼
[9. infer]        → generated speech via base + LoRA
```

Per-step detail lives in `step{N}-<name>.md`.
