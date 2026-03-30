"""CLI entry point: python -m voicetune.stages.finetune"""

import argparse
import logging
from pathlib import Path

from .pipeline import run_finetune

log = logging.getLogger(__name__)


def main():
    from voicetune.common import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Fine-tune Fish Speech S2 Pro via GCP A100 VM"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing exported wav+lab pairs (default: <run-dir>/fish-speech/data)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=4000,
        help="Training steps (default: 4000)"
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

    if args.input_dir is None:
        args.input_dir = args.run_dir / "fish-speech" / "data"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "finetune"

    run_finetune(
        data_dir=args.input_dir,
        max_steps=args.max_steps,
        test=args.test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
