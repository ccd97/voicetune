"""CLI entry point: python -m voicetune.segment"""

import argparse
import logging
from pathlib import Path

from voicetune.common import setup_logging

from .pipeline import process_file

setup_logging()
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Turn segmentation: merge same-speaker turns, cut per-turn audio (Step 4)"
    )
    parser.add_argument(
        "--diarized-dir", type=Path, default=Path("./output/diarized"),
        help="Directory containing diarized JSON files (default: ./output/diarized)"
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=Path("./output/preprocessed"),
        help="Directory containing preprocessed audio (default: ./output/preprocessed)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./output/segmented"),
        help="Directory for segmented output (default: ./output/segmented)"
    )
    parser.add_argument(
        "--merge-gap", type=float, default=0.5,
        help="Max gap (seconds) between turns to merge same-speaker segments (default: 0.5)"
    )
    args = parser.parse_args()

    if not args.diarized_dir.is_dir():
        log.error(f"Diarized directory does not exist: {args.diarized_dir}")
        return

    json_files = sorted(args.diarized_dir.glob("*_diarized.json"))
    if not json_files:
        log.warning(f"No diarized JSON files found in {args.diarized_dir}")
        return

    log.info(f"Found {len(json_files)} file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = []

    for json_file in json_files:
        try:
            process_file(json_file, args.audio_dir, args.output_dir, args.merge_gap)
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {json_file.name}")
            failed.append(json_file.name)

    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
