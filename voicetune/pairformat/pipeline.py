"""Pair formatting pipeline for speech-to-speech training.

Converts segmented dialogues into (input, output) pairs where:
  input  = context turns + other person's prompt (audio + text)
  output = 'me' response turn (audio + text)

Each pair also includes concatenated context audio for models that
consume full conversation history.
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from voicetune.common import TARGET_SR, write_wav

log = logging.getLogger(__name__)

CONTEXT_GAP_SECONDS = 0.3


def read_turn_audio(call_dir: Path, turn: dict) -> np.ndarray:
    """Read a turn's audio file."""
    audio, sr = sf.read(str(call_dir / turn["audio_path"]), dtype="float32")
    if sr != TARGET_SR:
        raise ValueError(f"Unexpected sample rate {sr} in {turn['audio_path']}")
    return audio


def concat_audio(segments: list[np.ndarray], gap_seconds: float = CONTEXT_GAP_SECONDS) -> np.ndarray:
    """Concatenate audio segments with silence gaps between them."""
    if not segments:
        return np.array([], dtype=np.float32)

    gap = np.zeros(int(gap_seconds * TARGET_SR), dtype=np.float32)
    parts = []
    for i, seg in enumerate(segments):
        if i > 0:
            parts.append(gap)
        parts.append(seg)

    return np.concatenate(parts)


def process_file(
    dialogue_path: Path,
    output_dir: Path,
    context_turns: int = 4,
    min_duration: float = 0.5,
) -> dict:
    """Build training pairs from a segmented dialogue.

    Args:
        dialogue_path: Path to dialogue.json
        output_dir: Where to write pairs
        context_turns: Max number of previous turns to include as context
        min_duration: Skip 'me' turns shorter than this (seconds)
    """
    call_dir = dialogue_path.parent

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    call_id = dialogue["call_id"]
    turns = dialogue["turns"]

    log.info(f"Processing {call_id}: {len(turns)} turns")

    # Create output structure
    call_output = output_dir / call_id
    audio_out = call_output / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    pairs = []
    pair_idx = 0

    for i, turn in enumerate(turns):
        # We only create a pair when 'me' responds
        if turn.get("speaker_label") != "me":
            continue

        # Skip very short responses
        if turn["duration"] < min_duration:
            log.debug(f"  Skipping turn {turn['turn']}: too short ({turn['duration']:.2f}s)")
            continue

        # Find the preceding 'other' turn (the prompt)
        prompt_turn = None
        for j in range(i - 1, -1, -1):
            if turns[j].get("speaker_label") == "other":
                prompt_turn = turns[j]
                break

        if prompt_turn is None:
            # 'me' speaks first with no prompt — skip
            continue

        pair_idx += 1
        pair_id = f"{call_id}_pair_{pair_idx:03d}"

        # Gather context: up to N turns before the prompt turn
        prompt_idx = turns.index(prompt_turn)
        context_start = max(0, prompt_idx - context_turns)
        context = turns[context_start:prompt_idx]

        context_audio_segments = []
        for ctx_turn in context:
            try:
                context_audio_segments.append(read_turn_audio(call_dir, ctx_turn))
            except Exception:
                log.warning(f"  Could not read context turn {ctx_turn['turn']}")

        prompt_audio = read_turn_audio(call_dir, prompt_turn)
        response_audio = read_turn_audio(call_dir, turn)

        input_audio = concat_audio(context_audio_segments + [prompt_audio])
        write_wav(audio_out / f"{pair_id}_input.wav", input_audio)
        write_wav(audio_out / f"{pair_id}_prompt.wav", prompt_audio)
        write_wav(audio_out / f"{pair_id}_response.wav", response_audio)

        context_text = [
            {"speaker": ct.get("speaker_label", ct["speaker"]), "text": ct["text"]}
            for ct in context
        ]

        pair = {
            "pair_id": pair_id,
            "call_id": call_id,
            "input": {
                "audio_path": f"audio/{pair_id}_input.wav",
                "prompt_audio_path": f"audio/{pair_id}_prompt.wav",
                "prompt_text": prompt_turn["text"],
                "prompt_speaker": prompt_turn.get("speaker_label", prompt_turn["speaker"]),
                "prompt_duration": round(prompt_turn["duration"], 3),
                "context": context_text,
            },
            "output": {
                "audio_path": f"audio/{pair_id}_response.wav",
                "text": turn["text"],
                "speaker": turn.get("speaker_label", turn["speaker"]),
                "duration": round(turn["duration"], 3),
            },
        }
        pairs.append(pair)

    # Write pairs manifest
    manifest = {
        "call_id": call_id,
        "language": dialogue.get("language", "unknown"),
        "num_pairs": len(pairs),
        "context_turns": context_turns,
        "pairs": pairs,
    }

    manifest_path = call_output / "pairs.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info(f"  Generated {len(pairs)} training pairs -> {call_output.name}/")
    return manifest
