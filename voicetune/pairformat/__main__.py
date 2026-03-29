"""CLI entry point: python -m voicetune.pairformat"""

import argparse
import logging
from pathlib import Path

from voicetune.common import setup_logging

from .pipeline import process_file

setup_logging()
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Format segmented dialogues into (input, output) pairs for S2S training"
    )
    parser.add_argument(
        "--segmented-dir", type=Path, default=Path("./output/segmented"),
        help="Directory containing segmented call output (default: ./output/segmented)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./output/pairs"),
        help="Directory for training pairs (default: ./output/pairs)"
    )
    parser.add_argument(
        "--context-turns", type=int, default=4,
        help="Max number of previous turns to include as context (default: 4)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.5,
        help="Skip response turns shorter than this (seconds, default: 0.5)"
    )
    args = parser.parse_args()

    # Find all segmented calls
    dialogue_files = sorted(args.segmented_dir.glob("*/dialogue.json"))
    if not dialogue_files:
        log.warning(f"No segmented dialogues found in {args.segmented_dir}")
        return

    log.info(f"Found {len(dialogue_files)} call(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    succeeded = 0
    failed = []

    for dialogue_path in dialogue_files:
        try:
            result = process_file(
                dialogue_path, args.output_dir,
                context_turns=args.context_turns,
                min_duration=args.min_duration,
            )
            total_pairs += result["num_pairs"]
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {dialogue_path}")
            failed.append(dialogue_path.name)

    log.info(f"Summary: {succeeded} calls, {total_pairs} total pairs")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
