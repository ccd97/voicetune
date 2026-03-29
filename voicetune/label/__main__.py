"""CLI entry point: python -m voicetune.label"""

import argparse
import logging
from pathlib import Path

from voicetune.common import setup_logging

from .pipeline import enroll, label_call

setup_logging()
log = logging.getLogger(__name__)

VOICEPRINT_PATH = Path("./output/voiceprint.npy")


def main():
    parser = argparse.ArgumentParser(
        description="Speaker labeling: enroll voiceprint or label calls as 'me' vs 'other'"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Enroll subcommand
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
        "--segmented-dir", type=Path, default=Path("./output/segmented"),
        help="Directory containing segmented output (default: ./output/segmented)"
    )
    enroll_parser.add_argument(
        "--voiceprint", type=Path, default=VOICEPRINT_PATH,
        help=f"Where to save voiceprint (default: {VOICEPRINT_PATH})"
    )

    # Label subcommand
    label_parser = subparsers.add_parser("label", help="Label speakers in segmented calls")
    label_parser.add_argument(
        "--segmented-dir", type=Path, default=Path("./output/segmented"),
        help="Directory containing segmented output (default: ./output/segmented)"
    )
    label_parser.add_argument(
        "--voiceprint", type=Path, default=VOICEPRINT_PATH,
        help=f"Path to voiceprint file (default: {VOICEPRINT_PATH})"
    )
    label_parser.add_argument(
        "--call-id", type=str, default=None,
        help="Label a specific call (default: label all calls)"
    )

    args = parser.parse_args()

    if args.command == "enroll":
        enroll(args.segmented_dir, args.call_id, args.speaker, args.voiceprint)

    elif args.command == "label":
        if not args.voiceprint.exists():
            log.error(f"Voiceprint not found: {args.voiceprint}. Run 'enroll' first.")
            return

        if args.call_id:
            call_ids = [args.call_id]
        else:
            call_ids = sorted(
                d.name for d in args.segmented_dir.iterdir()
                if d.is_dir() and (d / "dialogue.json").exists()
            )

        if not call_ids:
            log.warning(f"No segmented calls found in {args.segmented_dir}")
            return

        log.info(f"Labeling {len(call_ids)} call(s)")
        for call_id in call_ids:
            log.info(f"Processing {call_id}")
            label_call(args.segmented_dir, call_id, args.voiceprint)

        log.info("Done")


if __name__ == "__main__":
    main()
