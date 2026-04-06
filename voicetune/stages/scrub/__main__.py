"""CLI entry point: python -m voicetune.stages.scrub"""

import argparse
import logging
from pathlib import Path

from voicetune.common import (
    bootstrap,
    resolve_stage_paths,
    run_stage_loop,
)

from .pipeline import process_file

log = logging.getLogger(__name__)


def main():
    bootstrap(dotenv=True)

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

    resolve_stage_paths(args, input_dir="diarized", output_dir="scrubbed")

    if not args.input_dir.is_dir():
        log.error(f"Input directory does not exist: {args.input_dir}")
        return

    json_files = sorted(args.input_dir.glob("*_diarized.json"))
    if not json_files:
        log.warning(f"No diarized JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(json_files)} file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_stage_loop(
        json_files,
        lambda p: process_file(p, args.output_dir),
        done_check=lambda p: (args.output_dir / p.name).exists(),
        label="file",
        name_fn=lambda p: p.name,
    )


if __name__ == "__main__":
    main()
