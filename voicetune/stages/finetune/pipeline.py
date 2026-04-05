"""Fine-tuning pipeline: dataset preparation, validation, and GCP dispatch."""

import json
import logging
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import soundfile as sf

log = logging.getLogger(__name__)


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _classify_language(text: str, call_lang: str | None) -> str:
    """Bucket a turn by script; fall back to the call-level language for pure Devanagari."""
    has_dev = bool(_DEVANAGARI_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_dev and has_lat:
        return "mixed"
    if has_lat:
        return "english"
    if not has_dev:
        return "other"
    if call_lang and call_lang.lower().startswith("mr"):
        return "marathi"
    if call_lang and call_lang.lower().startswith("hi"):
        return "hindi"
    return "devanagari_unknown"


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def prepare_dataset(
    labeled_dir: Path,
    filtered_dir: Path,
    output_dir: Path,
    val_split: float = 0.05,
    max_turns_per_call: int = 50,
    keep_languages: set[str] | None = None,
) -> dict:
    """Export 'me' turns as WAVs + train/val JSONL manifests for VoxCPM."""
    me_dir = output_dir / "me"
    me_dir.mkdir(parents=True, exist_ok=True)

    call_dirs = sorted(
        d for d in labeled_dir.iterdir()
        if d.is_dir() and (d / "dialogue.json").exists()
    )
    if not call_dirs:
        raise FileNotFoundError(f"No labeled calls found in {labeled_dir}")

    rng = random.Random(0)
    entries_by_call: dict[str, list[dict]] = {}
    all_stats = []

    for call_dir in call_dirs:
        call_id = call_dir.name
        with open(call_dir / "dialogue.json") as f:
            dialogue = json.load(f)

        call_lang = dialogue.get("language")
        audio_dir = filtered_dir / call_id
        call_entries: list[dict] = []
        skipped_other = 0
        skipped_language = 0

        for turn in dialogue["turns"]:
            if turn.get("speaker_label") != "me":
                skipped_other += 1
                continue

            src_audio = audio_dir / turn["audio_path"]
            if not src_audio.exists():
                log.warning(f"  Missing audio: {src_audio}")
                continue

            info = sf.info(str(src_audio))
            duration = round(info.frames / info.samplerate, 2)

            text = turn["text"].strip()
            lang = _classify_language(text, call_lang)
            if keep_languages is not None and lang not in keep_languages:
                skipped_language += 1
                continue

            base_name = f"{call_id}_turn_{turn['turn']:03d}"
            dst_audio = me_dir / f"{base_name}.wav"
            shutil.copy2(str(src_audio), str(dst_audio))

            call_entries.append({
                "audio": str(dst_audio.resolve()),
                "text": text,
                "duration": duration,
                "_lang": lang,
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
            "skipped_language": skipped_language,
            "capped_from": capped_from,
        }
        msg = f"  {call_id}: exported {len(call_entries)}, skipped {skipped_other} non-'me'"
        if skipped_language:
            msg += f", {skipped_language} not in keep-languages"
        if capped_from > max_turns_per_call:
            msg += f", capped from {capped_from}"
        log.info(msg)
        all_stats.append(stats)

    total = sum(len(v) for v in entries_by_call.values())
    if total == 0:
        raise RuntimeError(f"No 'me' turns exported from {labeled_dir}")

    train_entries, val_entries, val_calls = _call_stratified_split(
        entries_by_call, val_split, seed=0
    )

    lang_counts = Counter(e["_lang"] for e in train_entries)

    _write_jsonl(output_dir / "train.jsonl", [_strip_private(e) for e in train_entries])
    if val_entries:
        _write_jsonl(output_dir / "val.jsonl", [_strip_private(e) for e in val_entries])

    summary = {
        "total_exported": total,
        "train": len(train_entries),
        "val": len(val_entries),
        "val_calls": val_calls,
        "output_dir": str(output_dir),
        "max_turns_per_call": max_turns_per_call,
        "keep_languages": sorted(keep_languages) if keep_languages else None,
        "language_counts": dict(lang_counts),
        "calls": all_stats,
    }
    with open(output_dir / "export_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        f"Dataset: {total} utterances, "
        f"{len(train_entries)} train, {len(val_entries)} val "
        f"across {len(val_calls)} calls → {output_dir}"
    )
    log.info(f"  train language mix: {dict(lang_counts)}")
    return summary


def _strip_private(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _call_stratified_split(
    entries_by_call: dict[str, list[dict]],
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[str]]:
    """Split so whole calls go to train or val, no call spans both.

    Greedy fill on a shuffled call order: add a call to val iff it fits under
    1.5× the target turn count, else send it to train. Falls back to the
    smallest call when every call exceeds that cap.
    """
    total = sum(len(v) for v in entries_by_call.values())
    rng = random.Random(seed)
    call_ids = sorted(entries_by_call.keys())
    rng.shuffle(call_ids)

    if total <= 20:
        train = [e for cid in call_ids for e in entries_by_call[cid]]
        rng.shuffle(train)
        return train, [], []

    target = max(1, int(total * val_split))
    cap = max(target, int(target * 1.5))

    val_entries: list[dict] = []
    train_entries: list[dict] = []
    val_calls: list[str] = []
    for cid in call_ids:
        turns = entries_by_call[cid]
        if len(val_entries) < target and len(val_entries) + len(turns) <= cap:
            val_entries.extend(turns)
            val_calls.append(cid)
        else:
            train_entries.extend(turns)

    if not val_entries:
        smallest = min(call_ids, key=lambda c: len(entries_by_call[c]))
        val_entries = list(entries_by_call[smallest])
        train_entries = [
            e for cid in call_ids if cid != smallest for e in entries_by_call[cid]
        ]
        val_calls = [smallest]
        log.warning(
            f"Call-stratified val fell back to smallest call ({smallest}, "
            f"{len(val_entries)} turns) — all other calls exceed the {cap}-turn cap."
        )

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries, val_calls


def validate_data(data_dir: Path) -> int:
    train_manifest = data_dir / "train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(
            f"No training manifest at {train_manifest}. Run prepare_dataset first."
        )
    count = 0
    with train_manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "audio" not in entry or "text" not in entry:
                raise ValueError(f"Manifest entry missing audio/text: {entry}")
            count += 1
    if count == 0:
        raise ValueError(f"Training manifest is empty: {train_manifest}")
    return count


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
