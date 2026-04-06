"""Summarize pipeline outputs across all stages."""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from voicetune.common import (
    audio_duration,
    read_json,
    read_jsonl,
    turn_duration,
    turn_text_chars,
    turns_total_duration,
)


def _normalize_lang(lang: str) -> str:
    return lang.split("-")[0].lower()


def _fmt_num(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _fmt_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    b = int(value)
    if b >= 1_000_000_000:
        return f"{b / 1_000_000_000:.2f} GB"
    if b >= 1_000_000:
        return f"{b / 1_000_000:.1f} MB"
    if b >= 1_000:
        return f"{b / 1_000:.1f} KB"
    return f"{b} B"


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    return f"{seconds:.1f}s"


def _pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "n/a"
    return f"{(part / whole) * 100:.1f}%"


def _pct_f(part: float, whole: float) -> str:
    if whole <= 0:
        return "n/a"
    return f"{(part / whole) * 100:.1f}%"


BOX_WIDTH = 60
INNER = BOX_WIDTH - 4


def _box_edge(left: str, right: str) -> None:
    print(left + "─" * (BOX_WIDTH - 2) + right)


def _box_line(text: str, visible_len: int | None = None) -> None:
    pad = INNER - (visible_len if visible_len is not None else len(text))
    print(f"│ {text}{' ' * max(0, pad)} │")


def _box_rows(rows: list[tuple[str, str]]) -> None:
    if not rows:
        return
    label_w = max(len(label) for label, _ in rows)
    for label, value in rows:
        _box_line(f"{label:<{label_w}}  {value}")


@dataclass
class StageSummary:
    name: str
    files: int = 0
    accepted: int = 0
    rejected: int = 0
    total_turns: int = 0
    total_duration: float = 0.0
    total_audio_bytes: int = 0
    total_turn_chars: int = 0
    extra: list[tuple[str, str]] = field(default_factory=list)
    languages: list[tuple[str, str]] = field(default_factory=list)
    languages_header: str = "languages"
    avg_denom: int = 0  # if > 0, used instead of `files` for per-file averages


def _add_language_rows(summary: StageSummary, lang_durations: dict, header: str = "languages") -> None:
    if not lang_durations:
        return
    summary.languages_header = header
    total = sum(lang_durations.values())
    ranked = sorted(lang_durations.items(), key=lambda x: -x[1])
    top = [(lang, dur) for lang, dur in ranked[:4] if dur >= 1.0]
    rest = total - sum(d for _, d in top)
    for lang, dur in top:
        summary.languages.append((lang, f"{_fmt_duration(dur)}  ({_pct_f(dur, total)})"))
    if rest >= 1.0:
        summary.languages.append(("other", f"{_fmt_duration(rest)}  ({_pct_f(rest, total)})"))


def _summarize_preprocess(root: Path) -> StageSummary:
    wavs = sorted(root.glob("*/full_normalized.wav"))
    summary = StageSummary("preprocess", files=len(wavs), accepted=len(wavs))
    if not wavs:
        return summary

    for wav in wavs:
        summary.total_duration += audio_duration(wav)
        summary.total_audio_bytes += wav.stat().st_size
    return summary


def _summarize_json_stage(root: Path, pattern: str, stage_name: str, audio_root: Path | None = None) -> StageSummary:
    files = sorted(root.glob(pattern))
    summary = StageSummary(stage_name, files=len(files))
    durations = []
    sizes = []
    lang_durations: dict = {}

    for path in files:
        data = read_json(path)
        turns = data.get("turns", [])
        summary.total_turns += len(turns)
        summary.total_turn_chars += turn_text_chars(turns)
        file_dur = 0.0
        if "total_duration" in data and isinstance(data["total_duration"], (int, float)):
            file_dur = float(data["total_duration"])
        elif "duration" in data and isinstance(data["duration"], (int, float)):
            file_dur = float(data["duration"])
        elif turns:
            file_dur = turns_total_duration(turns)
        if file_dur:
            durations.append(file_dur)
        lang = data.get("language")
        if lang:
            key = _normalize_lang(lang)
            lang_durations[key] = lang_durations.get(key, 0.0) + file_dur
        if audio_root is not None:
            audio = audio_root / data["call_id"] / "full_normalized.wav"
            if audio.exists():
                sizes.append(audio.stat().st_size)
        if data.get("rejected"):
            summary.rejected += 1
        else:
            summary.accepted += 1

    summary.total_duration = sum(durations)
    summary.total_audio_bytes = sum(sizes)
    _add_language_rows(summary, lang_durations)
    return summary


def _summarize_filter(root: Path) -> StageSummary:
    dialogue_files = sorted(root.glob("*/dialogue.json"))
    summary = StageSummary("filter", files=len(dialogue_files))
    removed_turns = 0
    removed_by_reason: Counter = Counter()
    reject_reasons_counter: Counter = Counter()
    wav_bytes = []
    durations = []
    lang_durations: dict = {}

    for path in dialogue_files:
        data = read_json(path)
        turns = data.get("turns", [])
        summary.total_turns += len(turns)
        summary.total_turn_chars += turn_text_chars(turns)

        rejected = bool(data.get("rejected"))
        if rejected:
            summary.rejected += 1
            for r in data.get("reject_reasons", []):
                reject_reasons_counter[r] += 1
        else:
            summary.accepted += 1

        file_dur = 0.0
        if "total_duration" in data and isinstance(data["total_duration"], (int, float)):
            file_dur = float(data["total_duration"])
        elif turns:
            file_dur = turns_total_duration(turns)
        if file_dur:
            durations.append(file_dur)

        lang = data.get("language")
        if lang:
            key = _normalize_lang(lang)
            lang_durations[key] = lang_durations.get(key, 0.0) + file_dur

        # whole-call drops are counted via reject:<reason>; skip them here
        if not rejected:
            orig = int(data.get("original_turn_count", len(turns)))
            kept = int(data.get("filtered_turn_count", len(turns)))
            removed_turns += max(0, orig - kept)
            for reason, count in (data.get("filter_removed") or {}).items():
                removed_by_reason[reason] += int(count)

        for turn in turns:
            wav_path = path.parent / turn.get("audio_path", "")
            if wav_path.exists():
                wav_bytes.append(wav_path.stat().st_size)

    summary.total_duration = sum(durations)
    summary.total_audio_bytes = sum(wav_bytes)
    summary.avg_denom = summary.accepted
    summary.extra.append(("removed turns", f"{removed_turns:,}"))
    if summary.total_turns and removed_turns:
        total = summary.total_turns + removed_turns
        summary.extra.append(("keep rate", _pct(summary.total_turns, total)))
    for reason, count in removed_by_reason.most_common():
        summary.extra.append((f"  {reason}", f"{count:,}"))
    for reason, count in reject_reasons_counter.most_common():
        summary.extra.append((f"reject:{reason}", f"{count:,}"))
    if wav_bytes:
        summary.extra.append(("avg turn wav", _fmt_bytes(mean(wav_bytes))))
    _add_language_rows(summary, lang_durations)
    return summary


def _summarize_segment(root: Path) -> StageSummary:
    dialogue_files = sorted(root.glob("*/dialogue.json"))
    summary = StageSummary("segment", files=len(dialogue_files), accepted=len(dialogue_files))
    durations = []
    wav_bytes = []
    lang_durations: dict = {}

    for path in dialogue_files:
        data = read_json(path)
        turns = data.get("turns", [])
        summary.total_turns += len(turns)
        summary.total_turn_chars += turn_text_chars(turns)
        file_dur = 0.0
        if "total_duration" in data and isinstance(data["total_duration"], (int, float)):
            file_dur = float(data["total_duration"])
            durations.append(file_dur)
        lang = data.get("language")
        if lang:
            key = _normalize_lang(lang)
            lang_durations[key] = lang_durations.get(key, 0.0) + file_dur
        for turn in turns:
            audio_path = path.parent / turn.get("audio_path", "")
            if audio_path.exists():
                wav_bytes.append(audio_path.stat().st_size)

    summary.total_duration = sum(durations)
    summary.total_audio_bytes = sum(wav_bytes)
    if wav_bytes:
        summary.extra.append(("avg turn wav", _fmt_bytes(mean(wav_bytes))))
    _add_language_rows(summary, lang_durations)
    return summary


def _summarize_label(labeled_root: Path, audio_root: Path) -> StageSummary:
    dialogue_files = sorted(labeled_root.glob("*/dialogue.json"))
    summary = StageSummary("label", files=len(dialogue_files))
    me_turns = 0
    other_turns = 0
    me_duration = 0.0
    lang_durations: dict = {}
    quality_flags_counter: Counter = Counter()
    wav_bytes = []

    for path in dialogue_files:
        data = read_json(path)
        call_id = data.get("call_id", path.parent.name)
        turns = data.get("turns", [])
        summary.total_turns += len(turns)
        summary.total_turn_chars += turn_text_chars(turns)

        file_dur = 0.0
        if "total_duration" in data and isinstance(data["total_duration"], (int, float)):
            file_dur = float(data["total_duration"])
            summary.total_duration += file_dur
        lang = data.get("language")

        call_me_dur = 0.0
        for turn in turns:
            dur = turn_duration(turn)
            if turn.get("speaker_label") == "me":
                me_turns += 1
                call_me_dur += dur
            elif "speaker_label" in turn:
                other_turns += 1
            audio_path = audio_root / call_id / turn.get("audio_path", "")
            if audio_path.exists():
                wav_bytes.append(audio_path.stat().st_size)

        me_duration += call_me_dur
        if lang:
            key = _normalize_lang(lang)
            lang_durations[key] = lang_durations.get(key, 0.0) + call_me_dur

        for flag in data.get("label_quality_flags", []):
            quality_flags_counter[flag] += 1

        if data.get("speaker_labels"):
            summary.accepted += 1

    summary.total_audio_bytes = sum(wav_bytes)
    summary.extra.append(("me turns", f"{me_turns:,}"))
    summary.extra.append(("other turns", f"{other_turns:,}"))
    summary.extra.append(("me duration", _fmt_duration(me_duration)))
    for flag, count in quality_flags_counter.most_common():
        summary.extra.append((f"flag:{flag}", f"{count:,}"))
    if wav_bytes:
        summary.extra.append(("avg turn wav", _fmt_bytes(mean(wav_bytes))))
    _add_language_rows(summary, lang_durations, header="me by language")
    return summary


def _summarize_finetune(data_root: Path) -> StageSummary:
    me_dir = data_root / "me"
    wavs = sorted(me_dir.glob("*.wav")) if me_dir.is_dir() else []

    train_manifest = data_root / "train.jsonl"
    val_manifest = data_root / "val.jsonl"
    train_entries = read_jsonl(train_manifest) if train_manifest.exists() else []
    val_entries = read_jsonl(val_manifest) if val_manifest.exists() else []
    manifest_entries = train_entries + val_entries

    if manifest_entries:
        summary = StageSummary(
            "finetune",
            files=len(manifest_entries),
            accepted=len(manifest_entries),
            total_turns=len(manifest_entries),
        )
        manifest_paths = [Path(e["audio"]) for e in manifest_entries if e.get("audio")]
        summary.total_audio_bytes = sum(p.stat().st_size for p in manifest_paths if p.exists())
        summary.total_turn_chars = sum(len(e.get("text", "")) for e in manifest_entries)
        summary.total_duration = sum(
            float(e["duration"]) for e in manifest_entries if isinstance(e.get("duration"), (int, float))
        )
        if summary.total_duration == 0.0 and manifest_paths:
            summary.total_duration = sum(audio_duration(p) for p in manifest_paths if p.exists())
    else:
        summary = StageSummary(
            "finetune",
            files=len(wavs),
            accepted=len(wavs),
        )
        summary.total_audio_bytes = sum(p.stat().st_size for p in wavs)
        if wavs:
            summary.total_duration = sum(audio_duration(wav) for wav in wavs)

    summary_path = data_root / "export_summary.json"
    if summary_path.exists():
        export = read_json(summary_path)
        calls = export.get("calls", [])
        summary.extra.append(("calls", f"{len(calls):,}"))
        summary.extra.append(("skipped other", f"{sum(c.get('skipped_other', 0) for c in calls):,}"))

    if train_entries or val_entries:
        summary.extra.append(("train/val", f"{len(train_entries):,} / {len(val_entries):,}"))

    if manifest_entries and len(wavs) != len(manifest_entries):
        summary.extra.append(("wavs on disk", f"{len(wavs):,}"))

    return summary


def _print_summary(summary: StageSummary) -> None:
    _box_edge("┌", "┐")
    name = summary.name.upper()
    _box_line(f"\033[1m{name}\033[0m", visible_len=len(name))

    if summary.files == 0:
        _box_edge("├", "┤")
        _box_line("no output found")
        _box_edge("└", "┘")
        print()
        return

    _box_edge("├", "┤")

    rows: list[tuple[str, str]] = [("files", f"{summary.files:,}")]

    if summary.accepted or summary.rejected:
        rows.append(("accepted", f"{summary.accepted:,}  ({_pct(summary.accepted, summary.files)})"))
        if summary.rejected:
            rows.append(("rejected", f"{summary.rejected:,}  ({_pct(summary.rejected, summary.files)})"))

    denom = summary.avg_denom or summary.files

    if summary.total_turns:
        rows.append(("turns", f"{summary.total_turns:,}"))
        if summary.total_turns != denom:
            rows.append(("avg turns/file", _fmt_num(summary.total_turns / denom)))

    if summary.total_duration:
        rows.append(("duration", _fmt_duration(summary.total_duration)))
        rows.append(("avg duration", _fmt_duration(summary.total_duration / denom)))

    if summary.total_audio_bytes:
        rows.append(("audio", _fmt_bytes(summary.total_audio_bytes)))
        rows.append(("avg audio/file", _fmt_bytes(summary.total_audio_bytes / denom)))

    if summary.total_turn_chars:
        rows.append(("text chars", f"{summary.total_turn_chars:,}"))
        rows.append(("avg chars/turn", _fmt_num(summary.total_turn_chars / max(summary.total_turns, 1))))

    _box_rows(rows)

    if summary.extra:
        _box_edge("├", "┤")
        _box_rows(summary.extra)

    if summary.languages:
        _box_edge("├", "┤")
        _box_line(summary.languages_header)
        _box_rows(summary.languages)

    _box_edge("└", "┘")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact stats across pipeline stages")
    parser.add_argument("--run-dir", type=Path, default=Path("./output"), help="Base pipeline output directory")
    parser.add_argument(
        "--stages",
        type=str,
        default="preprocess,diarize,validation,segment,filter,label,finetune",
        help="Comma-separated stages to include",
    )
    args = parser.parse_args()

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    collectors = {
        "preprocess": lambda: _summarize_preprocess(args.run_dir / "preprocessed"),
        "diarize": lambda: _summarize_json_stage(args.run_dir / "diarized", "*_diarized.json", "diarize", args.run_dir / "preprocessed"),
        "validation": lambda: _summarize_json_stage(args.run_dir / "validated", "*_validated.json", "validation", args.run_dir / "preprocessed"),
        "segment": lambda: _summarize_segment(args.run_dir / "segmented"),
        "filter": lambda: _summarize_filter(args.run_dir / "filtered"),
        "label": lambda: _summarize_label(args.run_dir / "labeled", args.run_dir / "filtered"),
        "finetune": lambda: _summarize_finetune(args.run_dir / "voxcpm" / "data"),
    }

    for stage in stages:
        collector = collectors.get(stage)
        if collector is None:
            raise SystemExit(f"Unknown stage: {stage}")
        _print_summary(collector())


if __name__ == "__main__":
    main()
