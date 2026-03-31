"""WhisperX local diarization backend."""

import logging
import os
from pathlib import Path

from voicetune.common import merge_segments_to_turns

from voicetune.common import get_call_id

log = logging.getLogger(__name__)


def diarize(audio_path: Path, num_speakers: int | None = None, language: str | None = None) -> dict:
    """Run diarization + transcription via WhisperX (local).

    Requires: pip install whisperx torch torchaudio
    Requires: HF_TOKEN env var with pyannote model access.
    Language is auto-detected by Whisper.
    """
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    hf_token = os.environ["HF_TOKEN"]

    log.info(f"WhisperX using device: {device}, compute_type: {compute_type}")

    # 1. Transcribe
    log.info("Loading Whisper model...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    log.info("Transcribing...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16 if device == "cuda" else 4)

    detected_lang = result.get("language", "en")
    log.info(f"Language: {detected_lang}")

    # 2. Align timestamps at word level
    log.info("Aligning timestamps...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_lang, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )

    # 3. Diarize with pyannote
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    log.info(f"Running speaker diarization (speakers: {'auto-detect' if num_speakers is None else num_speakers})...")
    diarize_model = DiarizationPipeline(
        token=hf_token, device=device
    )
    diarize_segments = diarize_model(
        audio,
        min_speakers=num_speakers,
        max_speakers=num_speakers,
    )

    # 4. Assign speaker labels to words
    result = assign_word_speakers(diarize_segments, result)

    # 5. Convert to dialogue format
    call_id = get_call_id(audio_path)
    turns = merge_segments_to_turns(result["segments"])

    output = {
        "call_id": call_id,
        "mode": "whisperx",
        "language": detected_lang,
        "turns": turns,
    }

    return output
