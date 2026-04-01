"""Turn segmentation pipeline.

Merges consecutive same-speaker turns, cuts per-turn WAV files from
the preprocessed audio, and outputs a structured dialogue JSON.
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from voicetune.common import TARGET_SR, write_wav

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


def audio_quality_check(turn_audio: np.ndarray) -> str | None:
    """Return a quality issue if audio fails checks, or None if OK."""
    rms = np.sqrt(np.mean(turn_audio ** 2)) + 1e-10
    rms_db = 20 * np.log10(rms)
    if rms_db < -40.0:
        return "mostly_silence"
    if rms_db < -30.0:
        return "low_energy"
    return None


def find_audio_for_call(call_id: str, audio_dir: Path) -> Path | None:
    """Locate the preprocessed WAV for a given call ID."""
    # Try preprocessed directory structure: audio_dir/call_id/full_normalized.wav
    candidate = audio_dir / call_id / "full_normalized.wav"
    if candidate.exists():
        return candidate

    # Try flat structure: audio_dir/call_id.wav
    candidate = audio_dir / f"{call_id}.wav"
    if candidate.exists():
        return candidate

    return None


def process_file(diarized_path: Path, audio_dir: Path, output_dir: Path, merge_gap: float) -> dict:
    """Process a single diarized JSON: merge turns, cut audio, write output."""
    with open(diarized_path) as f:
        data = json.load(f)

    call_id = data["call_id"]
    turns = data["turns"]

    log.info(f"Processing {call_id}: {len(turns)} raw turns")

    # Merge consecutive same-speaker turns
    merged = merge_turns(turns, merge_gap)
    log.info(f"  Merged: {len(turns)} -> {len(merged)} turns (gap threshold: {merge_gap}s)")

    # Find source audio
    audio_path = find_audio_for_call(call_id, audio_dir)
    if not audio_path:
        raise FileNotFoundError(f"No preprocessed audio found for {call_id} in {audio_dir}")

    audio, sr = sf.read(str(audio_path), dtype="float32")
    log.info(f"  Audio: {audio_path.name} ({len(audio)/sr:.1f}s, {sr}Hz)")

    # Create output directory for this call
    call_output_dir = output_dir / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)
    turns_dir = call_output_dir / "turns"
    turns_dir.mkdir(exist_ok=True)

    output_turns = []
    dropped_count = 0
    turn_num = 0
    for turn in merged:
        turn_audio = cut_turn_audio(audio, sr, turn["start"], turn["end"])

        issue = audio_quality_check(turn_audio)
        if issue:
            dropped_count += 1
            log.debug(f"  Dropping turn at {turn['start']:.1f}s-{turn['end']:.1f}s ({issue})")
            continue

        turn_num += 1
        turn_filename = f"turn_{turn_num:03d}_{turn['speaker']}.wav"
        turn_path = turns_dir / turn_filename
        write_wav(turn_path, turn_audio, sr)

        output_turns.append({
            "turn": turn_num,
            "speaker": turn["speaker"],
            "start": round(turn["start"], 3),
            "end": round(turn["end"], 3),
            "duration": round(turn["end"] - turn["start"], 3),
            "audio_path": f"turns/{turn_filename}",
            "text": turn["text"],
        })

    speakers = sorted(set(t["speaker"] for t in output_turns))

    result = {
        "call_id": call_id,
        "language": data.get("language", "unknown"),
        "speakers": speakers,
        "num_turns": len(output_turns),
        "total_duration": round(max(t["end"] for t in output_turns) - min(t["start"] for t in output_turns), 3) if output_turns else 0,
        "turns": output_turns,
    }

    # Write structured JSON
    out_path = call_output_dir / "dialogue.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"  Output: {len(output_turns)} turns, {len(speakers)} speakers, {dropped_count} dropped -> {call_output_dir.name}/")
    return result
