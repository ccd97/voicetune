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
        description="Fish Speech S2 Pro LoRA fine-tuning on exported training data"
    )
    parser.add_argument(
        "--fish-speech-dir", type=Path, required=True,
        help="Path to cloned fish-speech repository"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./output/fish-speech/data"),
        help="Directory containing exported wav+lab pairs (default: ./output/fish-speech/data)"
    )
    parser.add_argument(
        "--project", type=str, default="my-voice",
        help="Project name for training run (default: my-voice)"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=16,
        help="LoRA alpha (default: 16)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for VQ extraction (default: 16)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of workers for VQ extraction (default: 1)"
    )
    parser.add_argument(
        "--dataset-workers", type=int, default=16,
        help="Number of workers for dataset building (default: 16)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Max training steps (default: Fish Speech default)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip model download even if not present"
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip LoRA merge step"
    )
    parser.add_argument(
        "--checkpoint-step", type=int, default=None,
        help="Specific checkpoint step for merge (default: auto-detect latest)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./output/finetune"),
        help="Directory for finetune metadata (default: ./output/finetune)"
    )
    args = parser.parse_args()

    if not args.fish_speech_dir.is_dir():
        log.error(f"Fish Speech directory does not exist: {args.fish_speech_dir}")
        return

    data_me = args.data_dir / "me"
    wav_files = sorted(data_me.glob("*.wav")) if data_me.exists() else []
    lab_files = sorted(data_me.glob("*.lab")) if data_me.exists() else []
    if not wav_files or not lab_files:
        log.error(f"No wav+lab pairs found in {data_me}")
        return

    log.info(f"Found {len(wav_files)} wav + {len(lab_files)} lab files in {data_me}")

    result = run_finetune(
        fish_speech_dir=args.fish_speech_dir,
        data_dir=args.data_dir,
        project=args.project,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_workers=args.dataset_workers,
        max_steps=args.max_steps,
        skip_download=args.skip_download,
        skip_merge=args.skip_merge,
        checkpoint_step=args.checkpoint_step,
        output_dir=args.output_dir,
    )

    log.info(f"Fine-tuning complete: {result.get('finetuned_model', 'N/A')}")


if __name__ == "__main__":
    main()
