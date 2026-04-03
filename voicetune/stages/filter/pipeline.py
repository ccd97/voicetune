"""Audio-quality filter: clean per-turn clips and drop bad audio.

Input is a segmented call directory (dialogue.json + turns/*.wav). Applies:
  - silence / low-energy drops
  - loudness normalization and tail-decay repair or drop

Output mirrors the input layout under the filter dir: a fresh dialogue.json
listing the kept turns and a turns/ folder with the cleaned WAVs.
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from voicetune.common import write_wav

log = logging.getLogger(__name__)

LOW_ENERGY_CODE = "low_energy"
MOSTLY_SILENCE_CODE = "mostly_silence"
TAIL_DECAY_CODE = "tail_decay"

SILENCE_RMS_DB = -40.0
LOW_ENERGY_RMS_DB = -30.0

TARGET_RMS_DBFS = -20.0
MAX_TAIL_DECAY_DB = 5.0
FADE_FRAME_MS = 30
SPEECH_FLOOR_DB = -30.0


def _frame_rms_db(audio: np.ndarray, sr: int, frame_ms: int = FADE_FRAME_MS) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    n = (len(audio) // frame) * frame
    if n < frame:
        return np.array([])
    frames = audio[:n].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def audio_quality_issue(audio: np.ndarray) -> str | None:
    if len(audio) == 0:
        return MOSTLY_SILENCE_CODE
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-10))
    rms_db = 20 * np.log10(rms + 1e-12)
    if rms_db < SILENCE_RMS_DB:
        return MOSTLY_SILENCE_CODE
    if rms_db < LOW_ENERGY_RMS_DB:
        return LOW_ENERGY_CODE
    return None


def clean_clip(audio: np.ndarray, sr: int) -> np.ndarray | None:
    # Phone-call turns often trail off in volume; leaving this in teaches the
    # fine-tune to generate fading audio. Measure first-third vs last-third
    # RMS among speech frames, drop the clip if the decay is unrepairable,
    # or apply a ramped tail boost to flatten the envelope before normalizing.
    x = audio.astype(np.float32, copy=True)
    rms_db = _frame_rms_db(x, sr)
    if len(rms_db) < 10:
        return None

    speech = rms_db > SPEECH_FLOOR_DB
    if speech.sum() < 10:
        return None

    t1 = len(rms_db) // 3
    t2 = 2 * len(rms_db) // 3
    first = rms_db[:t1][speech[:t1]]
    last = rms_db[t2:][speech[t2:]]
    if first.size == 0 or last.size == 0:
        return None
    decay_db = float(first.mean() - last.mean())
    if decay_db > MAX_TAIL_DECAY_DB:
        return None

    frame = max(1, int(sr * FADE_FRAME_MS / 1000))
    if decay_db > 1.0:
        gain_db = np.zeros(len(rms_db))
        gain_db[t1:] = np.linspace(0.0, decay_db, len(rms_db) - t1)
        gain = 10 ** (gain_db / 20)
        gain_samples = np.interp(
            np.arange(len(x)),
            np.arange(len(gain)) * frame + frame // 2,
            gain,
            left=gain[0], right=gain[-1],
        )
        x = x * gain_samples.astype(np.float32)

    rms_final = float(np.sqrt((x ** 2).mean() + 1e-12))
    target_linear = 10 ** (TARGET_RMS_DBFS / 20)
    x = x * (target_linear / rms_final)

    peak = float(np.abs(x).max())
    if peak > 0.99:
        x = x * (0.99 / peak)

    return x.astype(np.float32)


def _load_turn(segmented_call_dir: Path, turn: dict) -> tuple[np.ndarray, int] | None:
    wav_path = segmented_call_dir / turn["audio_path"]
    if not wav_path.exists():
        log.warning(f"    missing audio: {wav_path}")
        return None
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def process_call(segmented_call_dir: Path, output_dir: Path) -> dict:
    """Clean per-turn audio and write the filtered call directory."""
    with open(segmented_call_dir / "dialogue.json") as f:
        dialogue = json.load(f)

    call_id = dialogue["call_id"]
    turns = dialogue["turns"]
    original_count = len(turns)
    removed_counts: dict[str, int] = {}

    kept_final: list[tuple[dict, np.ndarray, int]] = []
    for turn in turns:
        got = _load_turn(segmented_call_dir, turn)
        if got is None:
            continue
        audio, sr = got
        issue = audio_quality_issue(audio)
        if issue:
            removed_counts[issue] = removed_counts.get(issue, 0) + 1
            continue
        cleaned = clean_clip(audio, sr)
        if cleaned is None:
            removed_counts[TAIL_DECAY_CODE] = removed_counts.get(TAIL_DECAY_CODE, 0) + 1
            continue
        kept_final.append((turn, cleaned, sr))

    if not kept_final:
        log.warning(f"  {call_id}: all {original_count} turns removed")
        return {"call_id": call_id, "rejected": True, "reasons": ["all_turns_filtered"]}

    call_output_dir = output_dir / call_id
    call_output_dir.mkdir(parents=True, exist_ok=True)
    (call_output_dir / "turns").mkdir(exist_ok=True)

    out_turns: list[dict] = []
    for turn, cleaned, sr in kept_final:
        out_wav_path = call_output_dir / turn["audio_path"]
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out_wav_path, cleaned, sr)
        out_turns.append(turn)

    out_dialogue = {k: v for k, v in dialogue.items() if k not in ("turns", "num_turns", "speakers")}
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
