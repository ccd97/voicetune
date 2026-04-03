"""CLI entry point: python -m voicetune.stages.finetune"""

import argparse
import logging
from pathlib import Path

from .pipeline import MIN_EXPORT_DURATION, prepare_dataset, run_finetune

log = logging.getLogger(__name__)


def main():
    from voicetune.common import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Prepare dataset and fine-tune Fish Speech S2 Pro via cloud GPU"
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
        help="Directory for prepared wav+lab dataset (default: <run-dir>/fish-speech/data)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=MIN_EXPORT_DURATION,
        help=f"Skip turns shorter than this (seconds, default: {MIN_EXPORT_DURATION})"
    )
    parser.add_argument(
        "--max-duration", type=float, default=60.0,
        help="Skip turns longer than this (seconds, default: 60.0)"
    )
    parser.add_argument(
        "--no-prepare", action="store_true",
        help="Skip dataset preparation (use existing data in data-dir)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=800,
        help="Training steps (default: 800)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: spot A100, 1 training step, auto-delete"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to download finetuned model (default: <run-dir>/finetune)"
    )
    parser.add_argument(
        "--provider", choices=["gcp", "aws"], default="gcp",
        help="Cloud provider for fine-tuning (default: gcp)"
    )
    args = parser.parse_args()

    if args.labeled_dir is None:
        args.labeled_dir = args.run_dir / "labeled"
    if args.filtered_dir is None:
        args.filtered_dir = args.run_dir / "filtered"
    if args.data_dir is None:
        args.data_dir = args.run_dir / "fish-speech" / "data"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "finetune"

    if not args.no_prepare:
        log.info(f"Preparing dataset from {args.labeled_dir}")
        prepare_dataset(
            labeled_dir=args.labeled_dir,
            filtered_dir=args.filtered_dir,
            output_dir=args.data_dir,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )

    run_finetune(
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        test=args.test,
        output_dir=args.output_dir,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
