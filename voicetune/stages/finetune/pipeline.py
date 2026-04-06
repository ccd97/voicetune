"""Fine-tuning pipeline: dataset preparation, validation, and GCP dispatch."""

import logging
import random
import shutil
from collections import Counter
from pathlib import Path

from voicetune.common import (
    audio_duration,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from voicetune.stages.segment.paths import turn_wav_name

log = logging.getLogger(__name__)


def _strip_private(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def prepare_dataset(
    labeled_dir: Path,
    filtered_dir: Path,
    output_dir: Path,
    val_split: float = 0.06,
    max_turns_per_call: int = 30,
    keep_languages: set[str] | None = None,
) -> dict:
    """Export 'me' turns as WAVs + train/val JSONL manifests for VoxCPM."""
    me_dir = output_dir / "me"
    me_dir.mkdir(parents=True, exist_ok=True)

    call_dirs = sorted(p.parent for p in labeled_dir.glob("*/dialogue.json"))
    if not call_dirs:
        raise FileNotFoundError(f"No labeled calls found in {labeled_dir}")

    rng = random.Random(0)
    entries_by_call: dict[str, list[dict]] = {}
    all_stats = []

    skipped_calls_by_language = 0

    for call_dir in call_dirs:
        call_id = call_dir.name
        dialogue = read_json(call_dir / "dialogue.json")

        call_lang = dialogue.get("language")

        if keep_languages is not None:
            lang_norm = (call_lang or "").lower()
            if not any(lang_norm.startswith(code) for code in keep_languages):
                skipped_calls_by_language += 1
                log.info(
                    f"  {call_id}: skipped, language {call_lang!r} not in "
                    f"{sorted(keep_languages)}"
                )
                all_stats.append({
                    "call_id": call_id,
                    "call_language": call_lang,
                    "exported": 0,
                    "skipped_other": 0,
                    "capped_from": 0,
                    "skipped_by_language": True,
                })
                continue

        audio_dir = filtered_dir / call_id
        call_entries: list[dict] = []
        skipped_other = 0

        for turn in dialogue["turns"]:
            if turn.get("speaker_label") != "me":
                skipped_other += 1
                continue

            src_audio = audio_dir / turn["audio_path"]
            if not src_audio.exists():
                log.warning(f"  Missing audio: {src_audio}")
                continue

            duration = round(audio_duration(src_audio), 2)

            text = turn["text"].strip()

            dst_audio = me_dir / turn_wav_name(call_id, turn["turn"])
            shutil.copy2(str(src_audio), str(dst_audio))

            call_entries.append({
                "audio": str(dst_audio.resolve()),
                "text": text,
                "duration": duration,
                "_lang": call_lang,
            })

        capped_from = len(call_entries)
        if len(call_entries) > max_turns_per_call:
            call_entries = rng.sample(call_entries, max_turns_per_call)

        if call_entries:
            entries_by_call[call_id] = call_entries

        stats = {
            "call_id": call_id,
            "call_language": call_lang,
            "exported": len(call_entries),
            "skipped_other": skipped_other,
            "capped_from": capped_from,
        }
        msg = f"  {call_id}: exported {len(call_entries)}, skipped {skipped_other} non-'me'"
        if capped_from > max_turns_per_call:
            msg += f", capped from {capped_from}"
        log.info(msg)
        all_stats.append(stats)

    all_entries = [e for cid in sorted(entries_by_call) for e in entries_by_call[cid]]
    total = len(all_entries)
    if total == 0:
        raise RuntimeError(f"No 'me' turns exported from {labeled_dir}")

    split_rng = random.Random(0)
    split_rng.shuffle(all_entries)

    if total <= 20:
        train_entries, val_entries = all_entries, []
    else:
        n_val = max(1, round(total * val_split))
        val_entries = all_entries[:n_val]
        train_entries = all_entries[n_val:]

    train_lang_counts = Counter(e["_lang"] for e in train_entries)
    val_lang_counts = Counter(e["_lang"] for e in val_entries)

    write_jsonl(output_dir / "train.jsonl", [_strip_private(e) for e in train_entries])
    if val_entries:
        write_jsonl(output_dir / "val.jsonl", [_strip_private(e) for e in val_entries])

    summary = {
        "total_exported": total,
        "train": len(train_entries),
        "val": len(val_entries),
        "output_dir": str(output_dir),
        "max_turns_per_call": max_turns_per_call,
        "keep_languages": sorted(keep_languages) if keep_languages else None,
        "skipped_calls_by_language": skipped_calls_by_language,
        "train_language_counts": dict(train_lang_counts),
        "val_language_counts": dict(val_lang_counts),
        "calls": all_stats,
    }
    write_json(output_dir / "export_summary.json", summary, ensure_ascii=True)

    log.info(
        f"Dataset: {total} utterances, "
        f"{len(train_entries)} train, {len(val_entries)} val → {output_dir}"
    )
    log.info(f"  train language mix: {dict(train_lang_counts)}")
    log.info(f"  val language mix:   {dict(val_lang_counts)}")
    return summary


def validate_data(data_dir: Path) -> int:
    train_manifest = data_dir / "train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(
            f"No training manifest at {train_manifest}. Run prepare_dataset first."
        )
    entries = read_jsonl(train_manifest)
    for entry in entries:
        if "audio" not in entry or "text" not in entry:
            raise ValueError(f"Manifest entry missing audio/text: {entry}")
    if not entries:
        raise ValueError(f"Training manifest is empty: {train_manifest}")
    return len(entries)


def run_finetune(
    data_dir: Path,
    max_steps: int,
    test: bool,
    output_dir: Path,
) -> dict:
    train_count = validate_data(data_dir)
    log.info(f"Training data: {train_count} manifest entries")

    from .backends.gcp import run
    return run(data_dir, max_steps, test, output_dir)
