"""CLI entry point: python -m voicetune.stages.correction"""

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main():
    from dotenv import load_dotenv

    from voicetune.common import setup_logging

    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Claude-based speaker correction using translated transcripts"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing translated JSON files (default: <run-dir>/translated)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for corrected diarization (default: <run-dir>/diarized)"
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = args.run_dir / "translated"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "diarized"

    translated_files = sorted(args.input_dir.glob("*_translated.json"))
    if not translated_files:
        log.error(f"No translated JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(translated_files)} translated file(s)")

    from .pipeline import process_file

    rejections = []
    for path in translated_files:
        try:
            result = process_file(path, args.output_dir)
            if result and result.get("rejected"):
                rejections.append(result)
                continue
            if result:
                n_turns = len(result["turns"])
                speakers = sorted(set(t["speaker"] for t in result["turns"]))
                names = result.get("speaker_names", {})
                log.info(
                    f"  {result['call_id']}: {n_turns} turns, "
                    f"speakers: {', '.join(f'{s}({names.get(s, '?')})' for s in speakers)}"
                )
        except Exception:
            log.exception(f"Failed to process {path}")

    if rejections:
        log.info(f"Rejected {len(rejections)}/{len(translated_files)} conversation(s)")
        rejected_path = args.output_dir / "rejected.json"
        with open(rejected_path, "w") as f:
            json.dump(rejections, f, indent=2)
        log.info(f"Rejection manifest: {rejected_path}")


if __name__ == "__main__":
    main()
