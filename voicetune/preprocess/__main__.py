"""CLI entry point: python -m voicetune.preprocess"""

import argparse
import logging
from pathlib import Path

from voicetune.common import setup_logging

from .pipeline import SUPPORTED_EXTENSIONS, process_file

setup_logging()
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-process call recordings for dataset pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("./input"), help="Directory containing raw audio files")
    parser.add_argument("--output-dir", type=Path, default=Path("./output/preprocessed"), help="Directory for processed output")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        log.error(f"Input directory does not exist: {args.input_dir}")
        return

    audio_files = sorted(
        f for f in args.input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        log.warning(f"No supported audio files found in {args.input_dir}")
        return

    log.info(f"Found {len(audio_files)} audio file(s) in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = []

    for audio_file in audio_files:
        try:
            process_file(audio_file, args.output_dir)
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {audio_file.name}")
            failed.append(audio_file.name)

    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed files: {', '.join(failed)}")


if __name__ == "__main__":
    main()
