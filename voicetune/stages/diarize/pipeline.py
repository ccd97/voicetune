"""Diarization pipeline: backend dispatch and result persistence."""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


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


def process_file(audio_path: Path, output_dir: Path, mode: str,
                 num_speakers: int | None = None, language: str | None = None) -> dict:
    if mode == "whisperx":
        from .backends.whisperx_backend import diarize
    elif mode == "whispermlx":
        from .backends.whispermlx_backend import diarize
    elif mode == "llamacpp":
        from .backends.llamacpp_backend import diarize
    else:
        from .backends.mlx_backend import diarize

    result = diarize(audio_path, num_speakers, language)
    result["num_speakers"] = len(set(t["speaker"] for t in result["turns"]))
    save_result(result, output_dir)
    return result


def process_files_batch_aws(audio_paths: list[Path], output_dir: Path,
                            num_speakers: int | None = None,
                            language: str | None = None) -> list[tuple[Path, dict | str]]:
    """Submit all files to AWS Transcribe concurrently and collect results."""
    from .backends.aws import diarize_batch

    batch_results = diarize_batch(audio_paths, num_speakers, language)
    processed = []
    for audio_path, result in batch_results:
        if isinstance(result, dict):
            result["num_speakers"] = len(set(t["speaker"] for t in result["turns"]))
            save_result(result, output_dir)
        processed.append((audio_path, result))
    return processed
