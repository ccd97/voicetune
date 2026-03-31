"""Fine-tuning pipeline: data validation and provider dispatch."""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def validate_data(data_dir: Path) -> tuple[int, int]:
    me_dir = data_dir / "me"
    if not me_dir.is_dir():
        raise FileNotFoundError(f"No training data directory: {me_dir}")
    wavs = list(me_dir.glob("*.wav"))
    labs = list(me_dir.glob("*.lab"))
    if not wavs or not labs:
        raise FileNotFoundError(f"No wav+lab pairs in {me_dir}. Run the export step first.")
    return len(wavs), len(labs)


def run_finetune(
    data_dir: Path,
    max_steps: int,
    test: bool,
    output_dir: Path,
    provider: str = "gcp",
) -> dict:
    wav_count, lab_count = validate_data(data_dir)
    log.info(f"Training data: {wav_count} wav, {lab_count} lab files")

    if provider == "gcp":
        from .backends.gcp import run
    elif provider == "aws":
        from .backends.aws import run
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run(data_dir, max_steps, test, output_dir)
