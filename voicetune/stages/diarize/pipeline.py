"""Diarization pipeline: backend dispatch and result persistence."""

import logging
from pathlib import Path

from voicetune.common import unique_speakers, write_json

log = logging.getLogger(__name__)


def save_result(result: dict, output_dir: Path) -> Path:
    """Write diarization result JSON and return the output path."""
    out_path = output_dir / f"{result['call_id']}_diarized.json"
    write_json(out_path, result, ensure_ascii=True)
    log.info(f"Saved {len(result['turns'])} turns to {out_path}")
    return out_path


def process_file(audio_path: Path, output_dir: Path, mode: str,
                 num_speakers: int | None = None, language: str | None = None) -> dict:
    if mode == "whisperx":
        from .backends.whisperx import diarize
    elif mode == "llamacpp":
        from .backends.llamacpp import diarize
    else:
        from .backends.mlx import diarize

    result = diarize(audio_path, num_speakers, language)
    result["num_speakers"] = len(unique_speakers(result["turns"]))
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
            result["num_speakers"] = len(unique_speakers(result["turns"]))
            save_result(result, output_dir)
        processed.append((audio_path, result))
    return processed
