"""CLI entry point: python -m voicetune.label"""

import argparse
import logging
from pathlib import Path

from .pipeline import analyze_speakers, apply_labels, enroll

log = logging.getLogger(__name__)


def prompt_speaker_selection(call_id: str, analysis: dict, auto_skip: bool) -> str | None:
    """Prompt user to pick 'me' speaker, or skip. Returns speaker label or None."""
    flags = ", ".join(analysis["quality_flags"])
    log.warning(f"  Low-confidence match for {call_id} [{flags}]")

    if auto_skip:
        log.info(f"  Auto-skipping {call_id}")
        return None

    print(f"\n--- {call_id}: manual review needed [{flags}] ---")
    speakers = sorted(analysis["similarities"], key=analysis["similarities"].get, reverse=True)
    for i, spk in enumerate(speakers):
        sim = analysis["similarities"][spk]
        samples = analysis["speaker_samples"].get(spk, [])
        print(f"  [{i + 1}] {spk} (similarity: {sim:.3f})")
        for line in samples:
            print(f"      \"{line}\"")

    print(f"  [s] Skip this call")

    while True:
        choice = input("Select speaker for 'me': ").strip().lower()
        if choice == "s":
            log.info(f"  Skipping {call_id}")
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(speakers):
            selected = speakers[int(choice) - 1]
            log.info(f"  User selected {selected} as 'me'")
            return selected
        print(f"  Invalid choice. Enter 1-{len(speakers)} or 's' to skip.")


def main():
    from voicetune.common import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Speaker labeling: enroll voiceprint or label calls as 'me' vs 'other'"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
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
        "--segmented-dir", type=Path, default=None,
        help="Directory containing segmented output (default: <run-dir>/segmented)"
    )
    enroll_parser.add_argument(
        "--voiceprint", type=Path, default=None,
        help="Where to save voiceprint (default: <run-dir>/voiceprint.npy)"
    )

    # Label subcommand
    label_parser = subparsers.add_parser("label", help="Label speakers in segmented calls")
    label_parser.add_argument(
        "--segmented-dir", type=Path, default=None,
        help="Directory containing segmented output (default: <run-dir>/segmented)"
    )
    label_parser.add_argument(
        "--voiceprint", type=Path, default=None,
        help="Path to voiceprint file (default: <run-dir>/voiceprint.npy)"
    )
    label_parser.add_argument(
        "--call-id", type=str, default=None,
        help="Label a specific call (default: label all calls)"
    )
    label_parser.add_argument(
        "--auto-skip", action="store_true",
        help="Automatically skip low-confidence matches instead of prompting"
    )

    args = parser.parse_args()

    if args.segmented_dir is None:
        args.segmented_dir = args.run_dir / "segmented"
    if args.voiceprint is None:
        args.voiceprint = args.run_dir / "voiceprint.npy"

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
        skipped = 0
        for call_id in call_ids:
            log.info(f"Processing {call_id}")
            analysis = analyze_speakers(args.segmented_dir, call_id, args.voiceprint)

            if analysis["needs_review"]:
                me_speaker = prompt_speaker_selection(call_id, analysis, args.auto_skip)
                if me_speaker is None:
                    skipped += 1
                    continue
            else:
                me_speaker = analysis["best_match"]

            apply_labels(args.segmented_dir, analysis, me_speaker)

        log.info(f"Done ({skipped} skipped)" if skipped else "Done")


if __name__ == "__main__":
    main()
