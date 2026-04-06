"""voicetune — audio processing pipeline for voice fine-tuning.

Pipeline steps:
    1. preprocess  — normalize audio to 16kHz 16-bit WAV
    2. diarize     — speaker diarization + transcription
    3. scrub       — PII redaction via local LLM (optional, SKIP_STEPS=scrub)
    4. validation  — fix speaker labels via LLM (optional, SKIP_STEPS=validation)
    5. filter      — remove flagged turns, reject files with file-level issues
    6. segment     — merge turns, cut per-turn audio
    7. label       — identify 'me' vs 'other' via voiceprint + prepare dataset
    8. finetune    — LoRA fine-tune VoxCPM2
    9. infer       — local Gradio UI for base + LoRA inference
"""
