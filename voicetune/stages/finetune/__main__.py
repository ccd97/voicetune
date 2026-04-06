"""CLI entry point: python -m voicetune.stages.finetune"""

import argparse
import logging
from pathlib import Path

from voicetune.common import bootstrap, resolve_stage_paths

from .pipeline import prepare_dataset, run_finetune

log = logging.getLogger(__name__)


def main():
    bootstrap(dotenv=True)

    parser = argparse.ArgumentParser(
        description="Prepare dataset and fine-tune VoxCPM2 via GCP A100 VM"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--labeled-dir", type=Path, default=None,
        help="Directory containing labeled dialogue.json files (default: <run-dir>/labeled)"
    )
    parser.add_argument(
        "--filtered-dir", type=Path, default=None,
        help="Directory containing filtered per-call audio (default: <run-dir>/filtered)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory for prepared jsonl+wav dataset (default: <run-dir>/voxcpm/data)"
    )
    parser.add_argument(
        "--no-prepare", action="store_true",
        help="Skip dataset preparation (use existing data in data-dir)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Training steps (default: 200, ~3 epochs at 1k clips)"
    )
    parser.add_argument(
        "--max-turns-per-call", type=int, default=50,
        help="Per-call turn cap (default: 50)"
    )
    parser.add_argument(
        "--keep-languages", type=str, default=None,
        help="Comma-separated language-code prefixes to keep; a call is kept if its "
             "`language` starts with any prefix (e.g. --keep-languages hi,mr,en). "
             "Default: keep all."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: spot A100, 1 training step, auto-delete"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to download finetuned model (default: <run-dir>/finetune)"
    )
    args = parser.parse_args()

    resolve_stage_paths(
        args,
        labeled_dir="labeled",
        filtered_dir="filtered",
        data_dir="voxcpm/data",
        output_dir="finetune",
    )

    keep_languages: set[str] | None = None
    if args.keep_languages:
        keep_languages = {
            s.strip().lower() for s in args.keep_languages.split(",") if s.strip()
        }

    if not args.no_prepare:
        log.info(f"Preparing dataset from {args.labeled_dir}")
        prepare_dataset(
            labeled_dir=args.labeled_dir,
            filtered_dir=args.filtered_dir,
            output_dir=args.data_dir,
            max_turns_per_call=args.max_turns_per_call,
            keep_languages=keep_languages,
        )

    run_finetune(
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        test=args.test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
