"""CLI entry point: python -m voicetune.stages.diarize"""

import argparse
import logging
import warnings
from pathlib import Path

from .pipeline import find_preprocessed_wavs, process_file, process_files_batch_aws

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
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--mode", choices=["aws", "whisperx", "whispermlx", "mlx", "llamacpp"], required=True,
        help="Diarization backend: 'aws', 'whisperx', 'whispermlx' (WhisperX on Apple Silicon via MLX), 'mlx' (Apple Silicon), or 'llamacpp' (Gemma 4 audio)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing preprocessed output (default: <run-dir>/preprocessed)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for diarization output (default: <run-dir>/diarized)"
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

    if args.input_dir is None:
        args.input_dir = args.run_dir / "preprocessed"
    if args.output_dir is None:
        args.output_dir = args.run_dir / "diarized"

    wav_files = find_preprocessed_wavs(args.input_dir)
    if not wav_files:
        log.error(f"No WAV files found in {args.input_dir}")
        return

    log.info(f"Found {len(wav_files)} file(s), mode: {args.mode}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    skipped = 0
    failed = []

    pending = []
    for wav in wav_files:
        if (args.output_dir / f"{wav.parent.name}_diarized.json").exists():
            skipped += 1
        else:
            pending.append(wav)

    if args.mode == "aws":
        for audio_path, result in process_files_batch_aws(
            pending, args.output_dir, args.num_speakers, args.language
        ):
            if isinstance(result, dict):
                n_turns = len(result["turns"])
                speakers = set(t["speaker"] for t in result["turns"])
                log.info(f"  {audio_path.parent.name}: {n_turns} turns, {len(speakers)} speakers")
                succeeded += 1
            else:
                log.error(f"Failed to process {audio_path}: {result}")
                failed.append(str(audio_path))
    else:
        for wav in pending:
            try:
                result = process_file(wav, args.output_dir, args.mode, args.num_speakers, args.language)
                n_turns = len(result["turns"])
                speakers = set(t["speaker"] for t in result["turns"])
                log.info(f"  {wav.parent.name}: {n_turns} turns, {len(speakers)} speakers")
                succeeded += 1
            except Exception:
                log.exception(f"Failed to process {wav}")
                failed.append(str(wav))

    if skipped:
        log.info(f"Skipped {skipped} already-diarized file(s)")
    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
