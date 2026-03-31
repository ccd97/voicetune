"""CLI entry point: python -m voicetune.stages.segment"""

import argparse
import json
import logging
from pathlib import Path

from .pipeline import process_file

log = logging.getLogger(__name__)


def main():
    from voicetune.common import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Turn segmentation: merge same-speaker turns, cut per-turn audio (Step 4)"
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

    if args.input_dir is None:
        args.input_dir = args.run_dir / "validated"
    if args.audio_dir is None:
        args.audio_dir = args.run_dir / "preprocessed"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "segmented"

    if not args.input_dir.is_dir():
        log.error(f"Input directory does not exist: {args.input_dir}")
        return

    json_files = sorted(args.input_dir.glob("*_validated.json"))
    if not json_files:
        log.warning(f"No validated JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(json_files)} file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    skipped_done = 0
    skipped_rejected = 0
    failed = []

    for json_file in json_files:
        call_id = json_file.name.replace("_validated.json", "")
        if (args.output_dir / call_id / "dialogue.json").exists():
            skipped_done += 1
            continue
        with open(json_file) as f:
            data = json.load(f)
        if data.get("rejected"):
            skipped_rejected += 1
            continue
        try:
            process_file(json_file, args.audio_dir, args.output_dir, args.merge_gap)
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {json_file.name}")
            failed.append(json_file.name)

    if skipped_done:
        log.info(f"Skipped {skipped_done} already-segmented file(s)")
    log.info(f"Summary: {succeeded} succeeded, {skipped_rejected} skipped (rejected), {len(failed)} failed")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
