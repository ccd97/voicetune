"""WhisperX local diarization backend."""

import logging
import os
import warnings
from pathlib import Path

from voicetune.common import merge_segments_to_turns
from voicetune.common import get_call_id

warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")

log = logging.getLogger(__name__)

_whisper_model = None
_diarize_model = None

SUPPORTED_ALIGN_LANGS = set(
    os.environ.get("WHISPERX_ALIGN_LANGS", "en,hi,mr").split(",")
)


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    import torch
    import whisperx
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    log.info(f"WhisperX using device: {device}, compute_type: {compute_type}")
    log.info("Loading Whisper model...")
    _whisper_model = (whisperx.load_model("large-v3", device, compute_type=compute_type), device)
    return _whisper_model


def _get_diarize_model(device):
    global _diarize_model
    if _diarize_model is not None:
        return _diarize_model
    from whisperx.diarize import DiarizationPipeline
    hf_token = os.environ["HF_TOKEN"]
    log.info("Loading diarization model...")
    _diarize_model = DiarizationPipeline(token=hf_token, device=device)
    return _diarize_model


def diarize(audio_path: Path, num_speakers: int | None = None, language: str | None = None) -> dict:
    import whisperx
    from whisperx.diarize import assign_word_speakers

    model, device = _get_whisper_model()

    log.info("Transcribing...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16 if device == "cuda" else 4)

    detected_lang = result.get("language", "en")
    align_lang = detected_lang if detected_lang in SUPPORTED_ALIGN_LANGS else "en"
    if align_lang != detected_lang:
        log.warning(f"Detected '{detected_lang}', using '{align_lang}' for alignment")
    log.info(f"Language: {detected_lang} (align: {align_lang})")

    log.info("Aligning timestamps...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=align_lang, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )

    log.info(f"Running speaker diarization (speakers: {'auto-detect' if num_speakers is None else num_speakers})...")
    diarize_model = _get_diarize_model(device)
    diarize_segments = diarize_model(
        audio,
        min_speakers=num_speakers,
        max_speakers=num_speakers,
    )

    result = assign_word_speakers(diarize_segments, result)

    for seg in result["segments"]:
        if "speaker" not in seg:
            seg["speaker"] = "spk_0"

    call_id = get_call_id(audio_path)
    turns = merge_segments_to_turns(result["segments"])

    return {
        "call_id": call_id,
        "mode": "whisperx",
        "language": detected_lang,
        "turns": turns,
    }
