"""CLI entry point: python -m voicetune.stages.infer

Local Gradio UI for VoxCPM2 + LoRA. Requires `pip install -e '.[infer]'`.

    python -m voicetune.stages.infer
    python -m voicetune.stages.infer --lora-dir ./output/finetune/voxcpm2-lora/step_0000500
    python -m voicetune.stages.infer --base-only
    python -m voicetune.stages.infer --share
"""

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def main():
    from dotenv import load_dotenv

    from voicetune.common import setup_logging

    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Local Gradio UI for VoxCPM2 + LoRA inference"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--lora-dir", type=Path, default=None,
        help="LoRA adapter directory (default: <run-dir>/finetune/voxcpm2-lora/latest)"
    )
    parser.add_argument(
        "--sample-dir", type=Path, default=None,
        help="Directory of reference WAVs for the dropdown (default: <run-dir>/voxcpm/data/me)"
    )
    parser.add_argument(
        "--base-model", type=str, default="openbmb/VoxCPM2",
        help="HuggingFace repo id or local path (default: openbmb/VoxCPM2)"
    )
    parser.add_argument(
        "--base-only", action="store_true",
        help="Skip LoRA; infer with the base model only"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Gradio port (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Expose via gradio.live share tunnel"
    )
    args = parser.parse_args()

    if args.lora_dir is None:
        args.lora_dir = args.run_dir / "finetune" / "voxcpm2-lora" / "latest"
    if args.sample_dir is None:
        args.sample_dir = args.run_dir / "voxcpm" / "data" / "me"

    try:
        import voxcpm  # noqa: F401
        import gradio  # noqa: F401
    except ImportError as e:
        log.error(f"Missing inference deps ({e.name}).")
        log.error("Install them with:  pip install -e '.[infer]'")
        sys.exit(1)

    if args.base_only:
        lora_dir: Path | None = None
        log.info("Base-only mode: LoRA skipped.")
    elif not args.lora_dir.is_dir():
        log.warning(f"LoRA dir not found at {args.lora_dir}; falling back to base-only.")
        lora_dir = None
    else:
        has_weights = (
            (args.lora_dir / "lora_weights.safetensors").exists()
            or (args.lora_dir / "lora_weights.ckpt").exists()
        )
        if not has_weights:
            log.error(f"No lora_weights.safetensors/.ckpt in {args.lora_dir}")
            sys.exit(1)
        lora_dir = args.lora_dir.resolve()

    samples_dir = args.sample_dir.resolve() if args.sample_dir.is_dir() else None

    # Must be set before torch loads inside .pipeline.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    log.info(f"Starting Gradio on http://127.0.0.1:{args.port}")
    from .pipeline import launch
    launch(
        base_repo=args.base_model,
        lora_dir=lora_dir,
        samples_dir=samples_dir,
        port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
