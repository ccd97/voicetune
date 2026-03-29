"""CLI entry point: python -m voicetune.export"""

import argparse
import json
import logging
from pathlib import Path

from voicetune.common import setup_logging

from .pipeline import process_call

setup_logging()
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export segmented dialogues to Fish Speech fine-tuning format"
    )
    parser.add_argument(
        "--segmented-dir", type=Path, default=Path("./output/segmented"),
        help="Directory containing segmented call output (default: ./output/segmented)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./output/fish-speech/data/me"),
        help="Output directory for .wav + .lab pairs (default: ./output/fish-speech/data/me)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=1.0,
        help="Skip turns shorter than this (seconds, default: 1.0)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=60.0,
        help="Skip turns longer than this (seconds, default: 60.0)"
    )
    args = parser.parse_args()

    dialogue_files = sorted(args.segmented_dir.glob("*/dialogue.json"))
    if not dialogue_files:
        log.warning(f"No segmented dialogues found in {args.segmented_dir}")
        return

    log.info(f"Found {len(dialogue_files)} call(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_exported = 0
    all_stats = []

    for dialogue_path in dialogue_files:
        stats = process_call(
            dialogue_path, args.output_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        total_exported += stats["exported"]
        all_stats.append(stats)

    # Write summary
    summary = {
        "total_exported": total_exported,
        "output_dir": str(args.output_dir),
        "calls": all_stats,
    }
    summary_path = args.output_dir.parent / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Total: {total_exported} utterances exported to {args.output_dir}")
    log.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
