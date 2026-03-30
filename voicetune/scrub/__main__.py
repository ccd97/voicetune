"""CLI entry point: python -m voicetune.scrub"""

import argparse
import logging
from pathlib import Path

from .pipeline import process_file

log = logging.getLogger(__name__)


def main():
    from dotenv import load_dotenv

    from voicetune.common import setup_logging

    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Scrub sensitive data from diarized transcripts using a local LLM"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing diarized JSON files (default: <run-dir>/diarized)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for scrubbed output (default: <run-dir>/scrubbed)"
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = args.run_dir / "diarized"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "scrubbed"

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
