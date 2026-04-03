"""Fine-tuning pipeline: dataset preparation, validation, and provider dispatch."""

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

MIN_EXPORT_DURATION = 2.5
MAX_EXPORT_DURATION = 60.0


def prepare_dataset(
    labeled_dir: Path,
    filtered_dir: Path,
    output_dir: Path,
    min_duration: float = MIN_EXPORT_DURATION,
    max_duration: float = MAX_EXPORT_DURATION,
) -> dict:
    """Export 'me' turns from labeled calls as .wav + .lab pairs.

    WAVs are copied as-is from the filter step output; per-clip loudness
    normalization and amplitude-decay repair are handled there, not here.
    """
    me_dir = output_dir / "me"
    me_dir.mkdir(parents=True, exist_ok=True)

    call_dirs = sorted(
        d for d in labeled_dir.iterdir()
        if d.is_dir() and (d / "dialogue.json").exists()
    )
    if not call_dirs:
        raise FileNotFoundError(f"No labeled calls found in {labeled_dir}")

    total_exported = 0
    all_stats = []

    for call_dir in call_dirs:
        call_id = call_dir.name
        with open(call_dir / "dialogue.json") as f:
            dialogue = json.load(f)

        audio_dir = filtered_dir / call_id
        exported = 0
        skipped_short = 0
        skipped_long = 0
        skipped_other = 0

        for turn in dialogue["turns"]:
            if turn.get("speaker_label") != "me":
                skipped_other += 1
                continue

            duration = turn["duration"]
            if duration < min_duration:
                skipped_short += 1
                continue
            if duration > max_duration:
                skipped_long += 1
                continue

            src_audio = audio_dir / turn["audio_path"]
            if not src_audio.exists():
                log.warning(f"  Missing audio: {src_audio}")
                continue

            base_name = f"{call_id}_turn_{turn['turn']:03d}"
            shutil.copy2(str(src_audio), str(me_dir / f"{base_name}.wav"))
            (me_dir / f"{base_name}.lab").write_text(turn["text"].strip())
            exported += 1

        stats = {
            "call_id": call_id,
            "exported": exported,
            "skipped_short": skipped_short,
            "skipped_long": skipped_long,
            "skipped_other": skipped_other,
        }
        log.info(
            f"  {call_id}: exported {exported}, "
            f"skipped {skipped_other} other + {skipped_short} short + {skipped_long} long"
        )
        all_stats.append(stats)
        total_exported += exported

    summary = {"total_exported": total_exported, "output_dir": str(output_dir), "calls": all_stats}
    with open(output_dir / "export_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Dataset: {total_exported} utterances exported to {me_dir}")
    return summary


def validate_data(data_dir: Path) -> tuple[int, int]:
    me_dir = data_dir / "me"
    if not me_dir.is_dir():
        raise FileNotFoundError(f"No training data directory: {me_dir}")
    wavs = list(me_dir.glob("*.wav"))
    labs = list(me_dir.glob("*.lab"))
    if not wavs or not labs:
        raise FileNotFoundError(f"No wav+lab pairs in {me_dir}. Run the export step first.")
    return len(wavs), len(labs)


def run_finetune(
    data_dir: Path,
    max_steps: int,
    test: bool,
    output_dir: Path,
    provider: str = "gcp",
) -> dict:
    wav_count, lab_count = validate_data(data_dir)
    log.info(f"Training data: {wav_count} wav, {lab_count} lab files")

    if provider == "gcp":
        from .backends.gcp import run
    elif provider == "aws":
        from .backends.aws import run
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run(data_dir, max_steps, test, output_dir)
