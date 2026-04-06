"""Filter pipeline: drop unusable turns and files before labeling."""

import logging
import shutil
from pathlib import Path

import numpy as np

from voicetune.common import (
    audio_issue,
    read_json,
    read_mono_wav,
    turn_duration,
    unique_speakers,
    write_json,
)
from voicetune.stages.validation.pipeline import FILE_ISSUES, TURN_ISSUES, RejectReason

log = logging.getLogger(__name__)

FILE_REJECT_CODES = {r.value for r in FILE_ISSUES} | {
    RejectReason.LANGUAGE_NOT_ALLOWED.value,
    RejectReason.MONO_SPEAKER.value,
    RejectReason.LOW_CONFIDENCE.value,
}
VALIDATION_TURN_CODES = {r.value for r in TURN_ISSUES}

TOO_SHORT_CODE = RejectReason.TOO_SHORT.value
TOO_LONG_CODE = RejectReason.TOO_LONG.value

MIN_TURN_DURATION = 3.0
MAX_TURN_DURATION = 45.0


def _load_audio(wav_path: Path) -> tuple[np.ndarray, int] | None:
    if not wav_path.exists():
        log.warning(f"    missing audio: {wav_path}")
        return None
    return read_mono_wav(wav_path)


def _reject_file(call_id: str, reasons: list[str], original_count: int,
                 output_dir: Path) -> dict:
    """Write a rejected dialogue.json and return the result dict."""
    rejection = {
        "call_id": call_id,
        "rejected": True,
        "reject_reasons": sorted(set(reasons)),
        "original_turn_count": original_count,
        "filtered_turn_count": 0,
        "turns": [],
    }
    write_json(output_dir / call_id / "dialogue.json", rejection)
    return {"call_id": call_id, "rejected": True, "reasons": rejection["reject_reasons"]}


def process_call(
    segmented_call_dir: Path,
    output_dir: Path,
    min_duration: float = MIN_TURN_DURATION,
    max_duration: float = MAX_TURN_DURATION,
) -> dict:
    """Apply all turn/file filtering on a segmented call and write a clean output."""
    dialogue = read_json(segmented_call_dir / "dialogue.json")

    call_id = dialogue["call_id"]
    turns = dialogue.get("turns", [])
    original_count = len(turns)

    if dialogue.get("rejected"):
        file_reasons = [r for r in dialogue.get("reject_reasons", []) if r in FILE_REJECT_CODES]
        if file_reasons:
            log.info(f"  {call_id}: rejected by validation ({', '.join(file_reasons)})")
            return _reject_file(call_id, file_reasons, original_count, output_dir)

    removed_counts: dict[str, int] = {}
    kept: list[tuple[dict, Path]] = []

    for turn in turns:
        validation_codes = [c for c in turn.get("issues", []) if c in VALIDATION_TURN_CODES]
        if validation_codes:
            for code in validation_codes:
                removed_counts[code] = removed_counts.get(code, 0) + 1
            continue

        duration = turn_duration(turn)
        if duration < min_duration:
            removed_counts[TOO_SHORT_CODE] = removed_counts.get(TOO_SHORT_CODE, 0) + 1
            continue
        if duration > max_duration:
            removed_counts[TOO_LONG_CODE] = removed_counts.get(TOO_LONG_CODE, 0) + 1
            continue

        src_wav = segmented_call_dir / turn["audio_path"]
        loaded = _load_audio(src_wav)
        if loaded is None:
            continue
        audio, sr = loaded

        issue = audio_issue(audio, sr)
        if issue:
            removed_counts[issue] = removed_counts.get(issue, 0) + 1
            continue

        kept.append((turn, src_wav))

    if not kept:
        log.warning(f"  {call_id}: all {original_count} turns removed")
        return _reject_file(call_id, ["all_turns_filtered"], original_count, output_dir)

    if len(kept) < 2:
        log.warning(f"  {call_id}: only {len(kept)} turn kept, dropping call")
        return _reject_file(call_id, ["single_turn"], original_count, output_dir)

    call_output_dir = output_dir / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)
    (call_output_dir / "turns").mkdir(exist_ok=True)

    out_turns: list[dict] = []
    for turn, src_wav in kept:
        dst_wav = call_output_dir / turn["audio_path"]
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_wav), str(dst_wav))
        out_turn = {k: v for k, v in turn.items() if k != "issues"}
        out_turns.append(out_turn)

    out_dialogue = {k: v for k, v in dialogue.items()
                    if k not in ("turns", "num_turns", "speakers", "rejected", "reject_reasons")}
    out_dialogue["speakers"] = unique_speakers(out_turns)
    out_dialogue["num_turns"] = len(out_turns)
    out_dialogue["turns"] = out_turns
    out_dialogue["original_turn_count"] = original_count
    out_dialogue["filtered_turn_count"] = len(out_turns)
    if removed_counts:
        out_dialogue["filter_removed"] = removed_counts

    write_json(call_output_dir / "dialogue.json", out_dialogue)

    removed_total = original_count - len(out_turns)
    if removed_total:
        breakdown = ", ".join(f"{v} {k}" for k, v in sorted(removed_counts.items()))
        log.info(f"  {call_id}: kept {len(out_turns)}/{original_count} turns ({breakdown})")
    else:
        log.info(f"  {call_id}: kept all {original_count} turns")

    return {
        "call_id": call_id,
        "rejected": False,
        "original_count": original_count,
        "filtered_count": len(out_turns),
        "removed": removed_counts,
    }
