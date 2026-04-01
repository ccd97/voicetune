"""CLI entry point: python -m voicetune.stages.filter"""

import argparse
import logging
from pathlib import Path

from .pipeline import process_file

log = logging.getLogger(__name__)


def main():
    from voicetune.common import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Filter validated transcripts: remove flagged turns, reject files with file-level issues"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing validated JSON files (default: <run-dir>/validated)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for filtered files (default: <run-dir>/filtered)"
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = args.run_dir / "validated"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "filtered"

    validated_files = sorted(args.input_dir.glob("*_validated.json"))
    if not validated_files:
        log.error(f"No validated JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(validated_files)} validated file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0
    skipped = 0

    for path in validated_files:
        call_id = path.name.replace("_validated.json", "")
        if (args.output_dir / f"{call_id}_filtered.json").exists():
            skipped += 1
            continue

        result = process_file(path, args.output_dir)
        if result.get("rejected"):
            rejected += 1
        else:
            accepted += 1

    if skipped:
        log.info(f"Skipped {skipped} already-filtered file(s)")
    log.info(f"Summary: {accepted} accepted, {rejected} rejected")


if __name__ == "__main__":
    main()
