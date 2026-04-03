"""Turn segmentation: merge same-speaker turns, cut per-turn WAVs.

Does not drop turns — validation metadata is propagated to the filter step.
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from voicetune.common import write_wav

log = logging.getLogger(__name__)


def merge_turns(turns: list[dict], max_gap: float) -> list[dict]:
    """Merge consecutive turns from the same speaker when gap is <= max_gap seconds."""
    if not turns:
        return []

    merged = [dict(turns[0])]
    for turn in turns[1:]:
        prev = merged[-1]
        gap = turn["start"] - prev["end"]
        if turn["speaker"] == prev["speaker"] and gap <= max_gap:
            prev["end"] = turn["end"]
            prev["text"] = prev["text"] + " " + turn["text"]
            combined = list(dict.fromkeys([*prev.get("issues", []), *turn.get("issues", [])]))
            if combined:
                prev["issues"] = combined
        else:
            merged.append(dict(turn))

    return merged


def cut_turn_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Extract a segment of audio between start and end times."""
    start_sample = max(0, int(start * sr))
    end_sample = min(len(audio), int(end * sr))
    return audio[start_sample:end_sample]


def find_audio_for_call(call_id: str, audio_dir: Path) -> Path | None:
    """Locate the preprocessed WAV for a given call ID."""
    candidate = audio_dir / call_id / "full_normalized.wav"
    if candidate.exists():
        return candidate

    candidate = audio_dir / f"{call_id}.wav"
    if candidate.exists():
        return candidate

    return None


def process_file(
    validated_path: Path,
    audio_dir: Path,
    output_dir: Path,
    merge_gap: float,
) -> dict:
    """Cut per-turn audio from a validated JSON. Does not drop turns."""
    with open(validated_path) as f:
        data = json.load(f)

    call_id = data["call_id"]
    turns = data.get("turns", [])

    log.info(f"Processing {call_id}: {len(turns)} turns")

    merged = merge_turns(turns, merge_gap)
    if len(merged) != len(turns):
        log.info(f"  Merged: {len(turns)} -> {len(merged)} turns (gap threshold: {merge_gap}s)")

    audio_path = find_audio_for_call(call_id, audio_dir)
    if not audio_path:
        raise FileNotFoundError(f"No preprocessed audio found for {call_id} in {audio_dir}")

    audio, sr = sf.read(str(audio_path), dtype="float32")
    log.info(f"  Audio: {audio_path.name} ({len(audio)/sr:.1f}s, {sr}Hz)")

    call_output_dir = output_dir / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)
    turns_dir = call_output_dir / "turns"
    turns_dir.mkdir(exist_ok=True)

    output_turns = []
    for idx, turn in enumerate(merged, start=1):
        turn_audio = cut_turn_audio(audio, sr, turn["start"], turn["end"])
        turn_filename = f"turn_{idx:03d}_{turn['speaker']}.wav"
        write_wav(turns_dir / turn_filename, turn_audio, sr)

        out_turn = {
            "turn": idx,
            "speaker": turn["speaker"],
            "start": round(turn["start"], 3),
            "end": round(turn["end"], 3),
            "duration": round(turn["end"] - turn["start"], 3),
            "audio_path": f"turns/{turn_filename}",
            "text": turn["text"],
        }
        if turn.get("issues"):
            out_turn["issues"] = list(turn["issues"])
        output_turns.append(out_turn)

    speakers = sorted({t["speaker"] for t in output_turns})
    result = {
        "call_id": call_id,
        "language": data.get("language", "unknown"),
        "speakers": speakers,
        "num_turns": len(output_turns),
        "total_duration": round(
            max(t["end"] for t in output_turns) - min(t["start"] for t in output_turns), 3
        ) if output_turns else 0,
        "turns": output_turns,
    }
    if data.get("rejected"):
        result["rejected"] = True
    if data.get("reject_reasons"):
        result["reject_reasons"] = list(data["reject_reasons"])
    if "validation_confidence" in data:
        result["validation_confidence"] = data["validation_confidence"]

    with open(call_output_dir / "dialogue.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"  Output: {len(output_turns)} turns, {len(speakers)} speakers -> {call_output_dir.name}/")
    return result
