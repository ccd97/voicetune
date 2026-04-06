"""Speaker labeling pipeline.

Uses pyannote WeSpeakerResNet34 (bundled inside
`pyannote/speaker-diarization-community-1`) to extract speaker embeddings and
match against a reference voiceprint to label speakers as 'me' vs 'other'.
"""

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from voicetune.common import read_json, read_mono_wav, write_json

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "pyannote/speaker-diarization-community-1"
EMBEDDING_SUBFOLDER = "embedding"
EMBEDDING_SAMPLE_RATE = 16000
MIN_EMBED_SECONDS = 0.5

_encoder: "_PyannoteEncoder | None" = None


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _PyannoteEncoder:
    def __init__(self, device: torch.device):
        # Silence pyannote's torchcodec/FFmpeg import warnings; we pass tensors directly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyannote.audio import Model

        self._model = Model.from_pretrained(
            EMBEDDING_MODEL,
            subfolder=EMBEDDING_SUBFOLDER,
            token=os.environ.get("HF_TOKEN"),
        )
        self._model.eval()
        self._model.to(device)
        self.device = device

    def embed(self, waveform: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            emb = self._model(waveform.to(self.device))
        return emb.detach().cpu().numpy()


def get_encoder() -> _PyannoteEncoder:
    global _encoder
    if _encoder is None:
        device = _select_device()
        log.info(
            f"Loading speaker encoder ({EMBEDDING_MODEL}#{EMBEDDING_SUBFOLDER}) on {device.type}..."
        )
        _encoder = _PyannoteEncoder(device)
    return _encoder


def extract_embedding(audio_paths: list[Path]) -> np.ndarray:
    """Extract a single averaged embedding from one or more audio files."""
    encoder = get_encoder()
    embeddings = []

    for path in audio_paths:
        audio, sr = read_mono_wav(path)
        if sr != EMBEDDING_SAMPLE_RATE:
            raise ValueError(f"{path}: expected {EMBEDDING_SAMPLE_RATE} Hz, got {sr} Hz")
        if len(audio) < int(MIN_EMBED_SECONDS * sr):
            continue
        waveform = torch.from_numpy(audio).view(1, 1, -1)
        emb = encoder.embed(waveform).squeeze(0)
        embeddings.append(emb)

    if not embeddings:
        raise ValueError("No valid audio segments to extract embedding from")

    return np.mean(embeddings, axis=0)


def _save_voiceprint(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        embedding=embedding.astype(np.float32),
        model=EMBEDDING_MODEL,
        subfolder=EMBEDDING_SUBFOLDER,
    )


def _load_voiceprint(path: Path) -> np.ndarray:
    """Load a voiceprint and verify it was produced by the current encoder."""
    if path.suffix != ".npz":
        raise ValueError(
            f"{path}: expected a .npz voiceprint produced by the current encoder. "
            "Re-run `python -m voicetune.stages.label enroll`."
        )
    data = np.load(str(path), allow_pickle=False)
    if "embedding" not in data.files:
        raise ValueError(f"{path}: missing 'embedding' key; re-run enroll.")
    stored_model = str(data["model"]) if "model" in data.files else "<unknown>"
    if stored_model != EMBEDDING_MODEL:
        raise ValueError(
            f"{path}: voiceprint was created with {stored_model!r}, "
            f"expected {EMBEDDING_MODEL!r}. Re-enroll to match the current encoder."
        )
    return data["embedding"]


def enroll(segmented_dir: Path, call_id: str, my_speaker_label: str, output_path: Path) -> None:
    """Create a voiceprint file from a known call where the user identifies themselves."""
    call_dir = segmented_dir / call_id
    dialogue = read_json(call_dir / "dialogue.json")

    my_turns = [t for t in dialogue["turns"] if t["speaker"] == my_speaker_label]
    if not my_turns:
        raise ValueError(f"No turns found for speaker '{my_speaker_label}' in {call_id}")

    my_turns.sort(key=lambda t: t["duration"], reverse=True)
    audio_paths = [call_dir / t["audio_path"] for t in my_turns[:10]]

    log.info(f"Enrolling from {len(audio_paths)} turns of '{my_speaker_label}' in {call_id}")
    embedding = extract_embedding(audio_paths)

    _save_voiceprint(output_path, embedding)
    log.info(
        f"Voiceprint saved to {output_path} "
        f"(dim={embedding.shape[0]}, model={EMBEDDING_MODEL}#{EMBEDDING_SUBFOLDER})"
    )


MIN_SIMILARITY = 0.70
MIN_MARGIN = 0.15
MIN_USABLE_TURNS = 3


def analyze_speakers(segmented_dir: Path, call_id: str, voiceprint_path: Path) -> dict:
    """Compute speaker similarities and quality flags without writing anything."""
    call_dir = segmented_dir / call_id
    dialogue = read_json(call_dir / "dialogue.json")

    ref_embedding = _load_voiceprint(voiceprint_path)
    ref_norm = float(np.linalg.norm(ref_embedding)) or 1.0

    speakers = dialogue["speakers"]
    speaker_embeddings = {}
    speaker_usable_counts = {}

    for speaker in speakers:
        speaker_turns = [t for t in dialogue["turns"] if t["speaker"] == speaker]
        speaker_turns.sort(key=lambda t: t["duration"], reverse=True)
        audio_paths = [call_dir / t["audio_path"] for t in speaker_turns[:10]]

        try:
            speaker_embeddings[speaker] = extract_embedding(audio_paths)
            speaker_usable_counts[speaker] = len(audio_paths)
        except ValueError:
            log.warning(f"  Could not extract embedding for {speaker} — too little audio")
            speaker_embeddings[speaker] = None
            speaker_usable_counts[speaker] = 0

    similarities = {}
    for speaker, emb in speaker_embeddings.items():
        if emb is None:
            similarities[speaker] = -1.0
            continue
        emb_norm = float(np.linalg.norm(emb)) or 1.0
        sim = float(np.dot(ref_embedding, emb) / (ref_norm * emb_norm))
        similarities[speaker] = sim
        log.info(f"  {speaker}: similarity = {sim:.3f}")

    if not similarities:
        log.warning(f"  No speakers found in {call_id}, skipping")
        return {
            "call_id": call_id,
            "dialogue": dialogue,
            "similarities": {},
            "best_match": None,
            "quality_flags": ["no_speakers"],
            "speaker_samples": {},
        }

    best_match = max(similarities, key=similarities.get)

    quality_flags = []
    sorted_sims = sorted(similarities.values(), reverse=True)
    best_sim = sorted_sims[0]

    if best_sim < MIN_SIMILARITY:
        quality_flags.append("low_similarity")
        log.warning(f"  Low similarity: best match {best_match} scored {best_sim:.3f} (threshold {MIN_SIMILARITY})")

    if len(sorted_sims) >= 2 and sorted_sims[0] - sorted_sims[1] < MIN_MARGIN:
        quality_flags.append("ambiguous_match")
        log.warning(f"  Ambiguous match: margin between top two speakers is {sorted_sims[0] - sorted_sims[1]:.3f} (threshold {MIN_MARGIN})")

    for speaker, count in speaker_usable_counts.items():
        if 0 < count < MIN_USABLE_TURNS:
            quality_flags.append(f"insufficient_audio:{speaker}")
            log.warning(f"  Insufficient audio for {speaker}: only {count} usable turn(s)")

    speaker_samples = {}
    for speaker in speakers:
        turns = [t for t in dialogue["turns"] if t["speaker"] == speaker]
        speaker_samples[speaker] = [t["text"] for t in turns[:3]]

    return {
        "call_id": call_id,
        "dialogue": dialogue,
        "similarities": similarities,
        "best_match": best_match,
        "quality_flags": quality_flags,
        "speaker_samples": speaker_samples,
    }


def apply_labels(analysis: dict, me_speaker: str, output_dir: Path) -> None:
    """Apply speaker labels and write labelled dialogue.json to output_dir."""
    call_id = analysis["call_id"]
    dialogue = analysis["dialogue"]
    speakers = dialogue["speakers"]

    label_map = {s: ("me" if s == me_speaker else "other") for s in speakers}
    log.info(f"  Label map: {label_map}")

    for turn in dialogue["turns"]:
        turn["speaker_label"] = label_map[turn["speaker"]]

    dialogue["speaker_labels"] = label_map
    dialogue["speaker_similarities"] = analysis["similarities"]
    dialogue["label_quality_flags"] = analysis["quality_flags"]
    dialogue["speaker_embedding_model"] = f"{EMBEDDING_MODEL}#{EMBEDDING_SUBFOLDER}"

    dialogue_path = output_dir / call_id / "dialogue.json"
    write_json(dialogue_path, dialogue)

    log.info(f"  Wrote {dialogue_path}")
