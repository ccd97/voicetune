"""CLI entry point: python -m voicetune.stages.label"""

import argparse
import logging
from pathlib import Path

from voicetune.common import (
    bootstrap,
    read_json,
    resolve_stage_paths,
)

from .pipeline import analyze_speakers, apply_labels, enroll

log = logging.getLogger(__name__)


def main():
    bootstrap()

    parser = argparse.ArgumentParser(
        description="Speaker labeling: enroll voiceprint or label calls as 'me' vs 'other'"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing filtered call directories (default: <run-dir>/filtered)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_parser = subparsers.add_parser("enroll", help="Create voiceprint from a reference call")
    enroll_parser.add_argument(
        "--call-id", type=str, required=True,
        help="Call ID to use as reference (e.g. 'call_recording')"
    )
    enroll_parser.add_argument(
        "--speaker", type=str, required=True,
        help="Your speaker label in that call (e.g. 'spk_0' or 'spk_1')"
    )
    enroll_parser.add_argument(
        "--voiceprint", type=Path, default=None,
        help="Where to save voiceprint (default: <run-dir>/voiceprint.npz)"
    )

    label_parser = subparsers.add_parser("label", help="Label speakers in segmented calls")
    label_parser.add_argument(
        "--voiceprint", type=Path, default=None,
        help="Path to voiceprint file (default: <run-dir>/voiceprint.npz)"
    )
    label_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for labeled dialogue.json files (default: <run-dir>/labeled)"
    )
    label_parser.add_argument(
        "--call-id", type=str, default=None,
        help="Label a specific call (default: label all calls)"
    )
    args = parser.parse_args()

    resolve_stage_paths(args, input_dir="filtered", voiceprint="voiceprint.npz")

    if args.command == "enroll":
        enroll(args.input_dir, args.call_id, args.speaker, args.voiceprint)

    elif args.command == "label":
        if not args.voiceprint.exists():
            log.error(f"Voiceprint not found: {args.voiceprint}. Run 'enroll' first.")
            return

        output_dir = args.output_dir or args.run_dir / "labeled"

        if args.call_id:
            call_ids = [args.call_id]
        else:
            call_ids = sorted(p.parent.name for p in args.input_dir.glob("*/dialogue.json"))

        if not call_ids:
            log.warning(f"No filtered calls found in {args.input_dir}")
            return

        log.info(f"Labeling {len(call_ids)} call(s) -> {output_dir}")
        labeled = 0
        skipped = 0
        skipped_done = 0
        skipped_rejected = 0
        for call_id in call_ids:
            if (output_dir / call_id / "dialogue.json").exists():
                skipped_done += 1
                continue
            if read_json(args.input_dir / call_id / "dialogue.json").get("rejected"):
                skipped_rejected += 1
                continue
            log.info(f"Processing {call_id}")
            analysis = analyze_speakers(args.input_dir, call_id, args.voiceprint)

            if analysis["best_match"] is None:
                skipped += 1
                continue

            apply_labels(analysis, analysis["best_match"], output_dir)
            labeled += 1

        if skipped_done:
            log.info(f"Skipped {skipped_done} already-labeled call(s)")
        if skipped_rejected:
            log.info(f"Skipped {skipped_rejected} rejected call(s)")
        log.info(f"Labeled {labeled} call(s)" + (f", {skipped} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
