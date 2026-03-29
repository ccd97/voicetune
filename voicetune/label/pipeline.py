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

# Singleton encoder — loads model once
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
        if len(wav) < 1600:  # too short for a meaningful embedding
            continue
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

    if not embeddings:
        raise ValueError("No valid audio segments to extract embedding from")

    return np.mean(embeddings, axis=0)


def enroll(segmented_dir: Path, call_id: str, my_speaker_label: str, output_path: Path) -> None:
    """Create a voiceprint file from a known call where the user identifies themselves.

    Args:
        segmented_dir: Path to segmented output (contains call_id/dialogue.json + turns/)
        call_id: Which call to use as reference
        my_speaker_label: The speaker label (e.g. 'spk_0') that is 'me'
        output_path: Where to save the voiceprint .npy file
    """
    call_dir = segmented_dir / call_id
    dialogue_path = call_dir / "dialogue.json"

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    # Collect audio paths for the user's turns
    my_turns = [t for t in dialogue["turns"] if t["speaker"] == my_speaker_label]
    if not my_turns:
        raise ValueError(f"No turns found for speaker '{my_speaker_label}' in {call_id}")

    # Use up to 10 longest turns for a robust embedding
    my_turns.sort(key=lambda t: t["duration"], reverse=True)
    audio_paths = [call_dir / t["audio_path"] for t in my_turns[:10]]

    log.info(f"Enrolling from {len(audio_paths)} turns of '{my_speaker_label}' in {call_id}")
    embedding = extract_embedding(audio_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), embedding)
    log.info(f"Voiceprint saved to {output_path}")


def label_call(segmented_dir: Path, call_id: str, voiceprint_path: Path) -> dict:
    """Label speakers in a segmented call using a reference voiceprint.

    Compares each speaker's embedding against the voiceprint and assigns
    'me' to the closest match, 'other' to the rest.
    """
    call_dir = segmented_dir / call_id
    dialogue_path = call_dir / "dialogue.json"

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    ref_embedding = np.load(str(voiceprint_path))

    # Group turns by speaker and extract embeddings
    speakers = dialogue["speakers"]
    speaker_embeddings = {}

    for speaker in speakers:
        speaker_turns = [t for t in dialogue["turns"] if t["speaker"] == speaker]
        # Use longest turns for embedding
        speaker_turns.sort(key=lambda t: t["duration"], reverse=True)
        audio_paths = [call_dir / t["audio_path"] for t in speaker_turns[:10]]

        try:
            speaker_embeddings[speaker] = extract_embedding(audio_paths)
        except ValueError:
            log.warning(f"  Could not extract embedding for {speaker} — too little audio")
            speaker_embeddings[speaker] = None

    # Compute similarity to reference voiceprint
    similarities = {}
    for speaker, emb in speaker_embeddings.items():
        if emb is not None:
            sim = np.dot(ref_embedding, emb) / (np.linalg.norm(ref_embedding) * np.linalg.norm(emb))
            similarities[speaker] = float(sim)
            log.info(f"  {speaker}: similarity = {sim:.3f}")
        else:
            similarities[speaker] = -1.0

    # Assign 'me' to the most similar speaker
    best_match = max(similarities, key=similarities.get)
    label_map = {s: ("other" if s != best_match else "me") for s in speakers}
    log.info(f"  Label map: {label_map}")

    # Update dialogue
    for turn in dialogue["turns"]:
        turn["speaker_label"] = label_map[turn["speaker"]]

    dialogue["speaker_labels"] = label_map
    dialogue["speaker_similarities"] = similarities

    # Write updated dialogue
    with open(dialogue_path, "w") as f:
        json.dump(dialogue, f, indent=2, ensure_ascii=False)

    log.info(f"  Updated {dialogue_path}")
    return dialogue
