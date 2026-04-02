"""Speaker labeling pipeline.

Uses resemblyzer to extract speaker embeddings and match against a
reference voiceprint to label speakers as 'me' vs 'other'.
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav

log = logging.getLogger(__name__)

_encoder: VoiceEncoder | None = None


def get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        log.info("Loading speaker encoder model...")
        _encoder = VoiceEncoder()
    return _encoder


def extract_embedding(audio_paths: list[Path]) -> np.ndarray:
    """Extract a single averaged embedding from one or more audio files."""
    encoder = get_encoder()
    embeddings = []

    for path in audio_paths:
        audio, sr = sf.read(str(path), dtype="float32")
        wav = preprocess_wav(audio, source_sr=sr)
        if len(wav) < 1600:
            continue
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

    if not embeddings:
        raise ValueError("No valid audio segments to extract embedding from")

    return np.mean(embeddings, axis=0)


def enroll(segmented_dir: Path, call_id: str, my_speaker_label: str, output_path: Path) -> None:
    """Create a voiceprint file from a known call where the user identifies themselves."""
    call_dir = segmented_dir / call_id
    dialogue_path = call_dir / "dialogue.json"

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    my_turns = [t for t in dialogue["turns"] if t["speaker"] == my_speaker_label]
    if not my_turns:
        raise ValueError(f"No turns found for speaker '{my_speaker_label}' in {call_id}")

    my_turns.sort(key=lambda t: t["duration"], reverse=True)
    audio_paths = [call_dir / t["audio_path"] for t in my_turns[:10]]

    log.info(f"Enrolling from {len(audio_paths)} turns of '{my_speaker_label}' in {call_id}")
    embedding = extract_embedding(audio_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), embedding)
    log.info(f"Voiceprint saved to {output_path}")


MIN_SIMILARITY = 0.60
MIN_MARGIN = 0.10
MIN_USABLE_TURNS = 3


def analyze_speakers(segmented_dir: Path, call_id: str, voiceprint_path: Path) -> dict:
    """Compute speaker similarities and quality flags without writing anything."""
    call_dir = segmented_dir / call_id
    dialogue_path = call_dir / "dialogue.json"

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    ref_embedding = np.load(str(voiceprint_path))

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
        if emb is not None:
            sim = np.dot(ref_embedding, emb) / (np.linalg.norm(ref_embedding) * np.linalg.norm(emb))
            similarities[speaker] = float(sim)
            log.info(f"  {speaker}: similarity = {sim:.3f}")
        else:
            similarities[speaker] = -1.0

    if not similarities:
        log.warning(f"  No speakers found in {call_id}, skipping")
        return {
            "call_id": call_id,
            "dialogue": dialogue,
            "similarities": {},
            "best_match": None,
            "quality_flags": ["no_speakers"],
            "speaker_samples": {},
            "needs_review": False,
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
        "needs_review": "low_similarity" in quality_flags or "ambiguous_match" in quality_flags,
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

    call_dir = output_dir / call_id
    call_dir.mkdir(parents=True, exist_ok=True)
    dialogue_path = call_dir / "dialogue.json"
    with open(dialogue_path, "w") as f:
        json.dump(dialogue, f, indent=2, ensure_ascii=False)

    log.info(f"  Wrote {dialogue_path}")
