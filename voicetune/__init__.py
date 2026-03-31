"""voicetune — audio processing pipeline for voice fine-tuning.

Pipeline steps:
    1. preprocess  — normalize audio to 16kHz 16-bit WAV
    2. diarize     — speaker diarization + transcription
    3. scrub       — PII redaction via local LLM (optional, SKIP_STEPS=scrub)
    4. translate   — translate non-English turns via Claude (optional, SKIP_STEPS=translate)
    5. validation  — fix speaker labels via Claude (optional, SKIP_STEPS=validation)
    6. segment     — merge turns, cut per-turn audio
    7. label       — identify 'me' vs 'other' via voiceprint
    8. export      — export to Fish Speech format
    9. finetune    — LoRA fine-tune Fish Speech S2 Pro
"""

from voicetune.common import TARGET_SR, merge_segments_to_turns, write_wav
from voicetune.run import STEPS, main, parse_steps, run_step

__all__ = [
    "STEPS",
    "TARGET_SR",
    "main",
    "merge_segments_to_turns",
    "parse_steps",
    "run_step",
    "write_wav",
]
