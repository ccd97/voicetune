"""CLI entry point: python -m voicetune.stages.filter"""

import argparse
import logging
from pathlib import Path

from .pipeline import process_call

log = logging.getLogger(__name__)


def main():
    from voicetune.common import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Filter segmented calls: drop bad-audio turns and clean per-turn WAVs"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing segmented call directories (default: <run-dir>/segmented)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for filtered call directories (default: <run-dir>/filtered)"
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = args.run_dir / "segmented"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "filtered"

    if not args.input_dir.is_dir():
        log.error(f"Input directory does not exist: {args.input_dir}")
        return

    call_dirs = sorted(
        d for d in args.input_dir.iterdir()
        if d.is_dir() and (d / "dialogue.json").exists()
    )
    if not call_dirs:
        log.error(f"No segmented call directories found in {args.input_dir}")
        return

    log.info(f"Found {len(call_dirs)} call(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0
    skipped = 0

    for call_dir in call_dirs:
        call_id = call_dir.name
        if (args.output_dir / call_id / "dialogue.json").exists():
            skipped += 1
            continue

        result = process_call(call_dir, args.output_dir)
        if result.get("rejected"):
            rejected += 1
        else:
            accepted += 1

    if skipped:
        log.info(f"Skipped {skipped} already-filtered call(s)")
    log.info(f"Summary: {accepted} accepted, {rejected} rejected")


if __name__ == "__main__":
    main()
