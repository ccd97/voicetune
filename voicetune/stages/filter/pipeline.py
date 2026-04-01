"""Filter pipeline: remove flagged turns or reject entire files based on validation codes."""

import json
import logging
from pathlib import Path

from voicetune.stages.validation.pipeline import FILE_ISSUES, TURN_ISSUES, RejectReason

log = logging.getLogger(__name__)

FILE_REJECT_CODES = {r.value for r in FILE_ISSUES} | {
    RejectReason.LANGUAGE_NOT_ALLOWED.value,
    RejectReason.MONO_SPEAKER.value,
    RejectReason.LOW_CONFIDENCE.value,
}

TURN_REMOVE_CODES = {r.value for r in TURN_ISSUES}


def process_file(input_path: Path, output_dir: Path) -> dict:
    """Filter a validated transcript: drop flagged turns, reject files with file-level issues."""
    with open(input_path) as f:
        data = json.load(f)

    call_id = data["call_id"]

    if data.get("rejected"):
        reject_reasons = set(data.get("reject_reasons", []))
        if reject_reasons & FILE_REJECT_CODES:
            log.info(f"  {call_id}: skipped (file-level rejection: {', '.join(reject_reasons)})")
            return {"call_id": call_id, "rejected": True, "reasons": sorted(reject_reasons)}

    original_turns = data["turns"]
    original_count = len(original_turns)
    kept = []
    removed_counts: dict[str, int] = {}

    for turn in original_turns:
        issues = turn.get("issues", [])
        removable = [i for i in issues if i in TURN_REMOVE_CODES]
        if removable:
            for code in removable:
                removed_counts[code] = removed_counts.get(code, 0) + 1
            continue
        kept.append(turn)

    if not kept:
        log.warning(f"  {call_id}: all {original_count} turns removed — rejecting file")
        return {"call_id": call_id, "rejected": True, "reasons": ["all_turns_filtered"]}

    removed_total = original_count - len(kept)
    if removed_total:
        breakdown = ", ".join(f"{v} {k}" for k, v in sorted(removed_counts.items()))
        log.info(f"  {call_id}: removed {removed_total}/{original_count} turns ({breakdown})")
    else:
        log.info(f"  {call_id}: {original_count} turns, nothing filtered")

    output = {k: v for k, v in data.items() if k not in ("rejected", "reject_reasons")}
    output["turns"] = kept
    output["original_turn_count"] = original_count
    output["filtered_turn_count"] = len(kept)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{call_id}_filtered.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return {
        "call_id": call_id,
        "rejected": False,
        "original_count": original_count,
        "filtered_count": len(kept),
        "removed": removed_counts,
    }
