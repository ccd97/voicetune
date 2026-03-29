"""Shared utilities used across pipeline steps."""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def setup_logging() -> None:
    """Configure logging format for CLI entry points."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def write_wav(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write audio as 16-bit PCM WAV, clipping to [-1, 1]."""
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def merge_segments_to_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments from the same speaker into turns.

    Each segment must have 'speaker', 'start', 'end', and 'text' keys.
    """
    turns = []
    current_speaker = None
    current_texts: list[str] = []
    current_start = None
    current_end = None

    for seg in segments:
        speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        if not text:
            continue

        if speaker != current_speaker and current_texts:
            turns.append({
                "speaker": current_speaker,
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "text": " ".join(current_texts),
            })
            current_texts = []
            current_start = None

        current_speaker = speaker
        if current_start is None:
            current_start = start
        current_end = end
        current_texts.append(text)

    if current_texts:
        turns.append({
            "speaker": current_speaker,
            "start": round(current_start, 2),
            "end": round(current_end, 2),
            "text": " ".join(current_texts),
        })

    return turns
