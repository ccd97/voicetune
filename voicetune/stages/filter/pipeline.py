"""Filter pipeline: drop unusable turns and files before labeling."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from voicetune.stages.validation.pipeline import FILE_ISSUES, TURN_ISSUES, RejectReason

log = logging.getLogger(__name__)

FILE_REJECT_CODES = {r.value for r in FILE_ISSUES} | {
    RejectReason.LANGUAGE_NOT_ALLOWED.value,
    RejectReason.MONO_SPEAKER.value,
    RejectReason.LOW_CONFIDENCE.value,
}
VALIDATION_TURN_CODES = {r.value for r in TURN_ISSUES}

LOW_ENERGY_CODE = "low_energy"
MOSTLY_SILENCE_CODE = "mostly_silence"
TAIL_DECAY_CODE = "tail_decay"
TOO_SHORT_CODE = RejectReason.TOO_SHORT.value
TOO_LONG_CODE = RejectReason.TOO_LONG.value

MIN_TURN_DURATION = 3.0
MAX_TURN_DURATION = 30.0

SILENCE_RMS_DB = -40.0
LOW_ENERGY_RMS_DB = -30.0

MAX_TAIL_DECAY_DB = 5.0
DECAY_FRAME_MS = 30
SPEECH_FLOOR_DB = -30.0


def _frame_rms_db(audio: np.ndarray, sr: int) -> np.ndarray:
    frame = max(1, int(sr * DECAY_FRAME_MS / 1000))
    n = (len(audio) // frame) * frame
    if n < frame:
        return np.array([])
    frames = audio[:n].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def audio_issue(audio: np.ndarray, sr: int) -> str | None:
    """Return a drop-reason code if the clip is unusable for training, else None.

    Clips that pass are copied as-is; preprocess already normalized loudness.
    """
    if len(audio) == 0:
        return MOSTLY_SILENCE_CODE
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(audio ** 2) + 1e-10)) + 1e-12)
    if rms_db < SILENCE_RMS_DB:
        return MOSTLY_SILENCE_CODE
    if rms_db < LOW_ENERGY_RMS_DB:
        return LOW_ENERGY_CODE

    frames_db = _frame_rms_db(audio, sr)
    if len(frames_db) < 10:
        return MOSTLY_SILENCE_CODE
    speech = frames_db > SPEECH_FLOOR_DB
    if speech.sum() < 10:
        return MOSTLY_SILENCE_CODE
    t1 = len(frames_db) // 3
    t2 = 2 * len(frames_db) // 3
    first = frames_db[:t1][speech[:t1]]
    last = frames_db[t2:][speech[t2:]]
    if first.size == 0 or last.size == 0:
        return MOSTLY_SILENCE_CODE
    if float(first.mean() - last.mean()) > MAX_TAIL_DECAY_DB:
        return TAIL_DECAY_CODE
    return None


def _load_audio(wav_path: Path) -> tuple[np.ndarray, int] | None:
    if not wav_path.exists():
        log.warning(f"    missing audio: {wav_path}")
        return None
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _reject_file(call_id: str, reasons: list[str], original_count: int,
                 output_dir: Path) -> dict:
    """Write a rejected dialogue.json and return the result dict."""
    call_output_dir = output_dir / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)
    rejection = {
        "call_id": call_id,
        "rejected": True,
        "reject_reasons": sorted(set(reasons)),
        "original_turn_count": original_count,
        "filtered_turn_count": 0,
        "turns": [],
    }
    with open(call_output_dir / "dialogue.json", "w") as f:
        json.dump(rejection, f, indent=2, ensure_ascii=False)
    return {"call_id": call_id, "rejected": True, "reasons": rejection["reject_reasons"]}


def process_call(
    segmented_call_dir: Path,
    output_dir: Path,
    min_duration: float = MIN_TURN_DURATION,
    max_duration: float = MAX_TURN_DURATION,
) -> dict:
    """Apply all turn/file filtering on a segmented call and write a clean output."""
    with open(segmented_call_dir / "dialogue.json") as f:
        dialogue = json.load(f)

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

        duration = float(turn.get("duration", turn.get("end", 0) - turn.get("start", 0)))
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
    out_dialogue["speakers"] = sorted({t["speaker"] for t in out_turns})
    out_dialogue["num_turns"] = len(out_turns)
    out_dialogue["turns"] = out_turns
    out_dialogue["original_turn_count"] = original_count
    out_dialogue["filtered_turn_count"] = len(out_turns)
    if removed_counts:
        out_dialogue["filter_removed"] = removed_counts

    with open(call_output_dir / "dialogue.json", "w") as f:
        json.dump(out_dialogue, f, indent=2, ensure_ascii=False)

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
