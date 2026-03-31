"""WhisperMLX diarization backend — WhisperX pipeline on Apple Silicon via MLX.

Uses whispermlx (a WhisperX fork) with mlx-whisper for GPU-accelerated
transcription on Apple Silicon, plus wav2vec2 alignment and pyannote diarization.

Requires: pip install whispermlx
Requires: HF_TOKEN env var with pyannote model access.
"""

import logging
import os
from pathlib import Path

from voicetune.common import merge_segments_to_turns

from voicetune.common import get_call_id

log = logging.getLogger(__name__)


def diarize(audio_path: Path, num_speakers: int | None = None, language: str | None = None) -> dict:
    """Run diarization + transcription via WhisperMLX (Apple Silicon)."""
    import whispermlx

    hf_token = os.environ["HF_TOKEN"]

    log.info("Loading WhisperMLX model...")
    model = whispermlx.load_model("large-v3-turbo", device="cpu")

    log.info("Transcribing via MLX...")
    audio = whispermlx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=4)

    detected_lang = result.get("language", "en")
    log.info(f"Language: {detected_lang}")

    log.info("Aligning timestamps...")
    align_model, align_metadata = whispermlx.load_align_model(
        language_code=detected_lang, device="cpu"
    )
    result = whispermlx.align(
        result["segments"], align_model, align_metadata, audio, "cpu",
        return_char_alignments=False,
    )

    from whispermlx.diarize import DiarizationPipeline, assign_word_speakers

    log.info(f"Running speaker diarization (speakers: {'auto-detect' if num_speakers is None else num_speakers})...")
    diarize_model = DiarizationPipeline(token=hf_token, device="cpu")
    diarize_segments = diarize_model(
        audio,
        min_speakers=num_speakers,
        max_speakers=num_speakers,
    )

    result = assign_word_speakers(diarize_segments, result)

    call_id = get_call_id(audio_path)
    turns = merge_segments_to_turns(result["segments"])

    output = {
        "call_id": call_id,
        "mode": "whispermlx",
        "language": detected_lang,
        "turns": turns,
    }

    return output
