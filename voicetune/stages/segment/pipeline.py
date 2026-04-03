"""Turn segmentation pipeline.

Reads validated transcripts, merges consecutive same-speaker turns, cuts
per-turn WAV files from the preprocessed audio, and outputs a structured
dialogue JSON. File-level and per-turn rejection codes from validation
are applied here: rejected files are skipped and flagged turns are not cut.
Audio-quality filtering and clip cleaning happen later in the filter step.
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from voicetune.common import write_wav
from voicetune.stages.validation.pipeline import FILE_ISSUES, TURN_ISSUES, RejectReason

log = logging.getLogger(__name__)

FILE_REJECT_CODES = {r.value for r in FILE_ISSUES} | {
    RejectReason.LANGUAGE_NOT_ALLOWED.value,
    RejectReason.MONO_SPEAKER.value,
    RejectReason.LOW_CONFIDENCE.value,
}
TURN_REMOVE_CODES = {r.value for r in TURN_ISSUES}


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
        else:
            merged.append(dict(turn))

    return merged


def cut_turn_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Extract a segment of audio between start and end times."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
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
    """Process a validated JSON: drop rejected files/turns, merge, cut audio."""
    with open(validated_path) as f:
        data = json.load(f)

    call_id = data["call_id"]

    if data.get("rejected"):
        reject_reasons = set(data.get("reject_reasons", []))
        if reject_reasons & FILE_REJECT_CODES:
            log.info(f"  {call_id}: skipped (file-level rejection: {', '.join(reject_reasons)})")
            return {"call_id": call_id, "rejected": True, "reasons": sorted(reject_reasons)}

    original_turns = data["turns"]
    kept_turns: list[dict] = []
    removed_counts: dict[str, int] = {}
    for turn in original_turns:
        issues = turn.get("issues", [])
        removable = [i for i in issues if i in TURN_REMOVE_CODES]
        if removable:
            for code in removable:
                removed_counts[code] = removed_counts.get(code, 0) + 1
            continue
        kept_turns.append(turn)

    if not kept_turns:
        log.warning(f"  {call_id}: all {len(original_turns)} turns removed by validation codes")
        return {"call_id": call_id, "rejected": True, "reasons": ["all_turns_removed"]}

    log.info(f"Processing {call_id}: {len(original_turns)} raw turns, {len(kept_turns)} after validation codes")

    merged = merge_turns(kept_turns, merge_gap)
    log.info(f"  Merged: {len(kept_turns)} -> {len(merged)} turns (gap threshold: {merge_gap}s)")

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

        output_turns.append({
            "turn": idx,
            "speaker": turn["speaker"],
            "start": round(turn["start"], 3),
            "end": round(turn["end"], 3),
            "duration": round(turn["end"] - turn["start"], 3),
            "audio_path": f"turns/{turn_filename}",
            "text": turn["text"],
        })

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
    if removed_counts:
        result["validation_removed"] = removed_counts

    with open(call_output_dir / "dialogue.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    validation_note = f", skipped {sum(removed_counts.values())} by validation codes" if removed_counts else ""
    log.info(f"  Output: {len(output_turns)} turns, {len(speakers)} speakers{validation_note} -> {call_output_dir.name}/")
    return result
