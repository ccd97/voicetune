"""CLI entry point: python -m voicetune.diarize"""

import argparse
import logging
import warnings
from pathlib import Path

from .utils import find_preprocessed_wavs

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

log = logging.getLogger(__name__)


def main():
    from dotenv import load_dotenv

    from voicetune.common import setup_logging

    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Speaker diarization + transcription (Step 2 & 3)"
    )
    parser.add_argument(
        "--mode", choices=["aws", "whisperx", "whispermlx", "mlx", "llamacpp"], required=True,
        help="Diarization backend: 'aws', 'whisperx', 'whispermlx' (WhisperX on Apple Silicon via MLX), 'mlx' (Apple Silicon), or 'llamacpp' (Gemma 4 audio)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("./output/preprocessed"),
        help="Directory containing preprocessed output (default: ./output/preprocessed)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./output/diarized"),
        help="Directory for diarization output (default: ./output/diarized)"
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Expected number of speakers (default: auto-detect)"
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Force language code, e.g. 'en-US' (default: auto-detect). Use when auto-detect gets it wrong."
    )
    args = parser.parse_args()

    wav_files = find_preprocessed_wavs(args.input_dir)
    if not wav_files:
        log.error(f"No WAV files found in {args.input_dir}")
        return

    log.info(f"Found {len(wav_files)} file(s), mode: {args.mode}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import to avoid loading heavy deps for unused backends
    if args.mode == "aws":
        from .aws import diarize
    elif args.mode == "whisperx":
        from .whisperx_backend import diarize
    elif args.mode == "whispermlx":
        from .whispermlx_backend import diarize
    elif args.mode == "llamacpp":
        from .llamacpp_backend import diarize
    else:
        from .mlx_backend import diarize

    succeeded = 0
    failed = []

    for wav in wav_files:
        try:
            result = diarize(wav, args.output_dir, args.num_speakers, args.language)
            n_turns = len(result["turns"])
            speakers = set(t["speaker"] for t in result["turns"])
            log.info(f"  {wav.parent.name}: {n_turns} turns, {len(speakers)} speakers")
            succeeded += 1
        except Exception:
            log.exception(f"Failed to process {wav}")
            failed.append(str(wav))

    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
