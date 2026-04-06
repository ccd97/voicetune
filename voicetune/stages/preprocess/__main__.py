"""CLI entry point: python -m voicetune.stages.preprocess"""

import argparse
import logging
from pathlib import Path

from voicetune.common import (
    SUPPORTED_EXTENSIONS,
    bootstrap,
    resolve_stage_paths,
    run_stage_loop,
)

from .pipeline import process_file

log = logging.getLogger(__name__)


def main():
    bootstrap()

    parser = argparse.ArgumentParser(description="Pre-process call recordings for dataset pipeline")
    parser.add_argument("--run-dir", type=Path, default=Path("./output"), help="Base output directory (default: ./output)")
    parser.add_argument("--input-dir", type=Path, default=Path("./input"), help="Directory containing raw audio files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for processed output (default: <run-dir>/preprocessed)")
    args = parser.parse_args()

    resolve_stage_paths(args, output_dir="preprocessed")

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

    run_stage_loop(
        audio_files,
        lambda f: process_file(f, args.output_dir),
        done_check=lambda f: (args.output_dir / f.stem / "full_normalized.wav").exists(),
        label="audio file",
        name_fn=lambda f: f.name,
    )


if __name__ == "__main__":
    main()
