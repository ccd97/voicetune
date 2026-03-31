"""CLI entry point: python -m voicetune.stages.preprocess"""

import argparse
import logging
from pathlib import Path

from .pipeline import SUPPORTED_EXTENSIONS, process_file

log = logging.getLogger(__name__)


def main():
    from voicetune.common import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Pre-process call recordings for dataset pipeline")
    parser.add_argument("--run-dir", type=Path, default=Path("./output"), help="Base output directory (default: ./output)")
    parser.add_argument("--input-dir", type=Path, default=Path("./input"), help="Directory containing raw audio files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for processed output (default: <run-dir>/preprocessed)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.run_dir / "preprocessed"

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
    skipped = 0
    failed = []

    for audio_file in audio_files:
        if (args.output_dir / audio_file.stem / "full_normalized.wav").exists():
            skipped += 1
            continue
        try:
            process_file(audio_file, args.output_dir)
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {audio_file.name}")
            failed.append(audio_file.name)

    if skipped:
        log.info(f"Skipped {skipped} already-processed file(s)")
    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed files: {', '.join(failed)}")


if __name__ == "__main__":
    main()
