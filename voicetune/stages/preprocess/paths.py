"""Path conventions for preprocess output (`{call_id}/full_normalized.wav`)."""

from pathlib import Path


def get_call_id(audio_path: Path) -> str:
    """Call ID: parent dir for `.../{call}/full_normalized.wav`, else file stem."""
    if audio_path.name == "full_normalized.wav":
        return audio_path.parent.name
    return audio_path.stem


def find_preprocessed_wavs(input_dir: Path) -> list[Path]:
    """Find preprocessed WAVs: `*/full_normalized.wav` under `input_dir`, else bare `*.wav`."""
    wavs = sorted(input_dir.glob("*/full_normalized.wav"))
    if not wavs:
        wavs = sorted(input_dir.glob("*.wav"))
    return wavs


def find_audio_for_call(call_id: str, audio_dir: Path) -> Path | None:
    """Preprocessed WAV for a call; tries `{call}/full_normalized.wav` then `{call}.wav`."""
    candidate = audio_dir / call_id / "full_normalized.wav"
    if candidate.exists():
        return candidate

    candidate = audio_dir / f"{call_id}.wav"
    if candidate.exists():
        return candidate

    return None
