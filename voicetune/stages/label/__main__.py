"""CLI entry point: python -m voicetune.stages.label"""

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
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing segmented output (default: <run-dir>/segmented)"
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
        help="Where to save voiceprint (default: <run-dir>/voiceprint.npy)"
    )

    label_parser = subparsers.add_parser("label", help="Label speakers in segmented calls")
    label_parser.add_argument(
        "--voiceprint", type=Path, default=None,
        help="Path to voiceprint file (default: <run-dir>/voiceprint.npy)"
    )
    label_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for labeled dialogue.json files (default: <run-dir>/labeled)"
    )
    label_parser.add_argument(
        "--call-id", type=str, default=None,
        help="Label a specific call (default: label all calls)"
    )
    label_parser.add_argument(
        "--review", action="store_true",
        help="Prompt for manual review on low-confidence matches (default: skip them)"
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = args.run_dir / "segmented"
    if args.voiceprint is None:
        args.voiceprint = args.run_dir / "voiceprint.npy"

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
            call_ids = sorted(
                d.name for d in args.input_dir.iterdir()
                if d.is_dir() and (d / "dialogue.json").exists()
            )

        if not call_ids:
            log.warning(f"No segmented calls found in {args.input_dir}")
            return

        log.info(f"Labeling {len(call_ids)} call(s) -> {output_dir}")
        labeled = 0
        skipped = 0
        skipped_done = 0
        for call_id in call_ids:
            if not args.review and (output_dir / call_id / "dialogue.json").exists():
                skipped_done += 1
                continue
            log.info(f"Processing {call_id}")
            analysis = analyze_speakers(args.input_dir, call_id, args.voiceprint)

            if analysis["best_match"] is None:
                skipped += 1
                continue

            if analysis["needs_review"]:
                me_speaker = prompt_speaker_selection(call_id, analysis, not args.review)
                if me_speaker is None:
                    skipped += 1
                    continue
            else:
                me_speaker = analysis["best_match"]

            apply_labels(analysis, me_speaker, output_dir)
            labeled += 1

        if skipped_done:
            log.info(f"Skipped {skipped_done} already-labeled call(s)")
        log.info(f"Labeled {labeled} call(s)" + (f", {skipped} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
