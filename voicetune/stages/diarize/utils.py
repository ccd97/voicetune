"""Shared utilities for diarization backends."""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def get_call_id(audio_path: Path) -> str:
    """Derive a call ID from the audio path.

    If the file is .../call_recording/full_normalized.wav, returns 'call_recording'.
    Otherwise falls back to the file stem.
    """
    if audio_path.name == "full_normalized.wav":
        return audio_path.parent.name
    return audio_path.stem


def find_preprocessed_wavs(input_dir: Path) -> list[Path]:
    """Find all full_normalized.wav files from preprocessing output."""
    wavs = sorted(input_dir.glob("*/full_normalized.wav"))
    if not wavs:
        wavs = sorted(input_dir.glob("*.wav"))
    return wavs


def save_result(result: dict, output_dir: Path) -> Path:
    """Write diarization result JSON and return the output path."""
    out_path = output_dir / f"{result['call_id']}_diarized.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved {len(result['turns'])} turns to {out_path}")
    return out_path


def join_words(words: list[str]) -> str:
    """Join words, attaching punctuation to the preceding word."""
    if not words:
        return ""
    result = words[0]
    for w in words[1:]:
        if w in ".,!?;:'\")-":
            result += w
        else:
            result += " " + w
    return result
