"""CLI entry point: python -m voicetune.stages.segment"""

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
    bootstrap()

    parser = argparse.ArgumentParser(
        description="Turn segmentation: merge same-speaker turns, cut per-turn audio"
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
        "--audio-dir", type=Path, default=None,
        help="Directory containing preprocessed audio (default: <run-dir>/preprocessed)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for segmented output (default: <run-dir>/segmented)"
    )
    parser.add_argument(
        "--merge-gap", type=float, default=0.5,
        help="Max gap (seconds) between turns to merge same-speaker segments (default: 0.5)"
    )
    args = parser.parse_args()

    resolve_stage_paths(
        args,
        input_dir="validated",
        audio_dir="preprocessed",
        output_dir="segmented",
    )

    if not args.input_dir.is_dir():
        log.error(f"Input directory does not exist: {args.input_dir}")
        return

    json_files = sorted(args.input_dir.glob("*_validated.json"))
    if not json_files:
        log.warning(f"No validated JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(json_files)} file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def already_segmented(path: Path) -> bool:
        call_id = path.name.replace("_validated.json", "")
        return (args.output_dir / call_id / "dialogue.json").exists()

    run_stage_loop(
        json_files,
        lambda p: process_file(p, args.audio_dir, args.output_dir, args.merge_gap),
        done_check=already_segmented,
        label="file",
        name_fn=lambda p: p.name,
    )


if __name__ == "__main__":
    main()
