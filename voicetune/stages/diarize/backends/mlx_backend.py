"""MLX Whisper + pyannote diarization backend for Apple Silicon.

Uses mlx-whisper for GPU-accelerated transcription on Apple Silicon,
and pyannote for speaker diarization. Much faster than WhisperX on M-series Macs
since mlx-whisper runs natively on the Metal GPU.

Requires: pip install mlx-whisper pyannote.audio
Requires: HF_TOKEN env var with pyannote model access.
"""

import logging
import os
from pathlib import Path

import numpy as np

from voicetune.common import merge_segments_to_turns

from voicetune.common import get_call_id

log = logging.getLogger(__name__)


def diarize(audio_path: Path, num_speakers: int | None = None, language: str | None = None) -> dict:
    """Run transcription via mlx-whisper + diarization via pyannote."""
    import mlx_whisper
    import torch
    from pyannote.audio import Pipeline

    hf_token = os.environ["HF_TOKEN"]

    model_repo = "mlx-community/whisper-large-v3-turbo"
    log.info(f"Transcribing with mlx-whisper ({model_repo})...")

    transcribe_kwargs = {"word_timestamps": True}
    if language:
        transcribe_kwargs["language"] = language

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        **transcribe_kwargs,
    )

    detected_lang = result.get("language", "en")
    log.info(f"Language: {detected_lang}, {len(result['segments'])} segments")

    # Pass waveform directly to avoid torchcodec issues with pyannote
    import soundfile as sf

    log.info(f"Running speaker diarization (speakers: {'auto-detect' if num_speakers is None else num_speakers})...")
    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )

    if torch.backends.mps.is_available():
        log.info("Using MPS (Metal) for diarization")
        diarize_pipeline.to(torch.device("mps"))
    else:
        log.info("Using CPU for diarization")

    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim == 1:
        audio_data = audio_data[np.newaxis, :]  # (1, time)
    else:
        audio_data = audio_data.T  # (channels, time)
    waveform = torch.from_numpy(audio_data)

    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["min_speakers"] = num_speakers
        diarize_kwargs["max_speakers"] = num_speakers

    diarization = diarize_pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        **diarize_kwargs,
    )

    annotation = getattr(diarization, "speaker_diarization", diarization)
    speaker_timeline = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_timeline.append((turn.start, turn.end, speaker))

    for segment in result["segments"]:
        if "words" not in segment:
            segment["speaker"] = _find_speaker(
                (segment["start"] + segment["end"]) / 2, speaker_timeline
            )
            continue
        for word in segment["words"]:
            word["speaker"] = _find_speaker(
                (word["start"] + word["end"]) / 2, speaker_timeline
            )
        # Assign segment speaker by majority vote from words
        speakers = [w["speaker"] for w in segment["words"] if w.get("speaker")]
        segment["speaker"] = max(set(speakers), key=speakers.count) if speakers else "UNKNOWN"

    call_id = get_call_id(audio_path)
    turns = merge_segments_to_turns(result["segments"])

    output = {
        "call_id": call_id,
        "mode": "mlx",
        "language": detected_lang,
        "turns": turns,
    }

    return output


def _find_speaker(time: float, timeline: list[tuple]) -> str:
    """Find which speaker is active at a given time."""
    best = None
    best_overlap = 0
    for start, end, speaker in timeline:
        if start <= time <= end:
            return speaker
        # If no exact match, find closest
        dist = min(abs(time - start), abs(time - end))
        if best is None or dist < best_overlap:
            best = speaker
            best_overlap = dist
    return best or "UNKNOWN"
