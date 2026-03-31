"""CLI entry point: python -m voicetune.stages.validation"""

import argparse
import logging
from pathlib import Path

from .pipeline import process_file

log = logging.getLogger(__name__)


def main():
    from dotenv import load_dotenv

    from voicetune.common import setup_logging

    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Claude-based speaker validation using translated transcripts"
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
        help="Output directory for validated diarization (default: <run-dir>/validated)"
    )
    parser.add_argument(
        "--backend", choices=["bedrock", "llamacpp"], default="llamacpp",
        help="Validation backend (default: llamacpp)"
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = args.run_dir / "translated"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "validated"

    translated_files = sorted(args.input_dir.glob("*_translated.json"))
    if not translated_files:
        log.error(f"No translated JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(translated_files)} translated file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    skipped = 0
    failed = []
    rejections = []
    for path in translated_files:
        out_name = path.name.replace("_translated.json", "_validated.json")
        if (args.output_dir / out_name).exists():
            skipped += 1
            continue
        try:
            result = process_file(path, args.output_dir, backend=args.backend)
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
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {path}")
            failed.append(path.name)

    if skipped:
        log.info(f"Skipped {skipped} already-validated file(s)")
    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")

    if rejections:
        log.info(f"Rejected {len(rejections)}/{len(translated_files)} conversation(s)")


if __name__ == "__main__":
    main()
