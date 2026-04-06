"""CLI entry point: python -m voicetune.stages.diarize"""

import argparse
import logging
import warnings
from pathlib import Path

from voicetune.common import (
    bootstrap,
    resolve_stage_paths,
    run_stage_loop,
    unique_speakers,
)
from voicetune.stages.preprocess.paths import find_preprocessed_wavs

from .pipeline import process_file, process_files_batch_aws

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

log = logging.getLogger(__name__)


def main():
    bootstrap(dotenv=True)

    parser = argparse.ArgumentParser(
        description="Speaker diarization + transcription (Step 2 & 3)"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--mode", choices=["aws", "whisperx", "mlx", "llamacpp"], required=True,
        help="Diarization backend: 'aws', 'whisperx', 'mlx' (Apple Silicon), or 'llamacpp' (Gemma 4 audio)"
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

    resolve_stage_paths(args, input_dir="preprocessed", output_dir="diarized")

    wav_files = find_preprocessed_wavs(args.input_dir)
    if not wav_files:
        log.error(f"No WAV files found in {args.input_dir}")
        return

    log.info(f"Found {len(wav_files)} file(s), mode: {args.mode}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def already_diarized(wav: Path) -> bool:
        return (args.output_dir / f"{wav.parent.name}_diarized.json").exists()

    if args.mode == "aws":
        pending = [w for w in wav_files if not already_diarized(w)]
        skipped = len(wav_files) - len(pending)

        succeeded = 0
        failed: list[str] = []
        for audio_path, result in process_files_batch_aws(
            pending, args.output_dir, args.num_speakers, args.language
        ):
            if isinstance(result, dict):
                n_turns = len(result["turns"])
                speakers = unique_speakers(result["turns"])
                log.info(f"  {audio_path.parent.name}: {n_turns} turns, {len(speakers)} speakers")
                succeeded += 1
            else:
                log.error(f"Failed to process {audio_path}: {result}")
                failed.append(str(audio_path))

        if skipped:
            log.info(f"Skipped {skipped} already-diarized file(s)")
        log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
        if failed:
            log.info(f"Failed: {', '.join(failed)}")
        return

    def process_one(wav: Path) -> None:
        result = process_file(wav, args.output_dir, args.mode, args.num_speakers, args.language)
        n_turns = len(result["turns"])
        speakers = unique_speakers(result["turns"])
        log.info(f"  {wav.parent.name}: {n_turns} turns, {len(speakers)} speakers")

    run_stage_loop(
        wav_files,
        process_one,
        done_check=already_diarized,
        label="file",
    )


if __name__ == "__main__":
    main()
