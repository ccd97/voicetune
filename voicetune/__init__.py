"""voicetune — audio processing pipeline for voice fine-tuning.

Pipeline steps:
    1. preprocess  — normalize audio to 16kHz 16-bit WAV
    2. diarize     — speaker diarization + transcription
    3. translate   — translate non-English turns via Claude
    4. correction  — fix speaker labels via Claude
    5. segment     — merge turns, cut per-turn audio
    6. label       — identify 'me' vs 'other' via voiceprint
    7. pairformat  — build (prompt, response) training pairs
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
