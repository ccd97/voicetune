"""CLI entry point: python -m voicetune.finetune"""

import argparse
import logging
from pathlib import Path

from voicetune.common import setup_logging

from .pipeline import run_finetune

setup_logging()
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Fish Speech S2 Pro via GCP A100 VM"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./output/fish-speech/data"),
        help="Directory containing exported wav+lab pairs (default: ./output/fish-speech/data)"
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
        "--output-dir", type=Path, default=Path("./output/finetune"),
        help="Where to download finetuned model (default: ./output/finetune)"
    )
    args = parser.parse_args()

    run_finetune(
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        test=args.test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
