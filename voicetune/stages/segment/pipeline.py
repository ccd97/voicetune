"""Turn segmentation: merge same-speaker turns, cut per-turn WAVs.

Does not drop turns — validation metadata is propagated to the filter step.
"""

import logging
from pathlib import Path

from voicetune.common import (
    cut_turn_audio,
    read_json,
    read_mono_wav,
    turns_total_duration,
    unique_speakers,
    write_json,
    write_wav,
)
from voicetune.stages.preprocess.paths import find_audio_for_call

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


def process_file(
    validated_path: Path,
    audio_dir: Path,
    output_dir: Path,
    merge_gap: float,
) -> dict:
    """Cut per-turn audio from a validated JSON. Does not drop turns."""
    data = read_json(validated_path)

    call_id = data["call_id"]
    turns = data.get("turns", [])

    log.info(f"Processing {call_id}: {len(turns)} turns")

    merged = merge_turns(turns, merge_gap)
    if len(merged) != len(turns):
        log.info(f"  Merged: {len(turns)} -> {len(merged)} turns (gap threshold: {merge_gap}s)")

    audio_path = find_audio_for_call(call_id, audio_dir)
    if not audio_path:
        raise FileNotFoundError(f"No preprocessed audio found for {call_id} in {audio_dir}")

    audio, sr = read_mono_wav(audio_path)
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

    speakers = unique_speakers(output_turns)
    result = {
        "call_id": call_id,
        "language": data.get("language", "unknown"),
        "speakers": speakers,
        "num_turns": len(output_turns),
        "total_duration": round(turns_total_duration(output_turns), 3) if output_turns else 0,
        "turns": output_turns,
    }
    if data.get("rejected"):
        result["rejected"] = True
    if data.get("reject_reasons"):
        result["reject_reasons"] = list(data["reject_reasons"])
    if "validation_confidence" in data:
        result["validation_confidence"] = data["validation_confidence"]

    write_json(call_output_dir / "dialogue.json", result)

    log.info(f"  Output: {len(output_turns)} turns, {len(speakers)} speakers -> {call_output_dir.name}/")
    return result
