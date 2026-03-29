"""CLI entry point: python -m voicetune.translate"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from voicetune.common import setup_logging

from .pipeline import process_file

load_dotenv()
setup_logging()
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Add English translations to diarized call transcripts"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("./output/diarized"),
        help="Directory containing diarized JSON files (default: ./output/diarized)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./output/translated"),
        help="Directory for translated output (default: ./output/translated)"
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        log.error(f"Input directory does not exist: {args.input_dir}")
        return

    json_files = sorted(args.input_dir.glob("*_diarized.json"))
    if not json_files:
        log.warning(f"No diarized JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(json_files)} file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = []

    for json_file in json_files:
        try:
            process_file(json_file, args.output_dir)
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {json_file.name}")
            failed.append(json_file.name)

    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
