"""CLI entry point: python -m voicetune.stages.validation"""

import argparse
import logging
from pathlib import Path

from voicetune.common import (
    bootstrap,
    resolve_stage_paths,
    run_stage_loop,
    unique_speakers,
)

from .pipeline import process_file

log = logging.getLogger(__name__)


def main():
    bootstrap(dotenv=True)

    parser = argparse.ArgumentParser(
        description="LLM-based speaker validation using diarized transcripts"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory (default: ./output)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing diarized JSON files (default: <run-dir>/diarized)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for validated diarization (default: <run-dir>/validated)"
    )
    args = parser.parse_args()

    resolve_stage_paths(args, input_dir="diarized", output_dir="validated")

    diarized_files = sorted(args.input_dir.glob("*_diarized.json"))
    if not diarized_files:
        log.error(f"No diarized JSON files found in {args.input_dir}")
        return

    log.info(f"Found {len(diarized_files)} diarized file(s)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rejections: list[dict] = []

    def process_one(path: Path) -> None:
        result = process_file(path, args.output_dir)
        if result and result.get("rejected"):
            rejections.append(result)
            return
        if result:
            n_turns = len(result["turns"])
            speakers = unique_speakers(result["turns"])
            names = result.get("speaker_names", {})
            speakers_fmt = ", ".join(f"{s}({names.get(s, '?')})" for s in speakers)
            log.info(f"  {result['call_id']}: {n_turns} turns, speakers: {speakers_fmt}")

    def already_validated(path: Path) -> bool:
        out_name = path.name.replace("_diarized.json", "_validated.json")
        return (args.output_dir / out_name).exists()

    run_stage_loop(
        diarized_files,
        process_one,
        done_check=already_validated,
        label="file",
        name_fn=lambda p: p.name,
    )

    if rejections:
        log.info(f"Rejected {len(rejections)}/{len(diarized_files)} conversation(s)")


if __name__ == "__main__":
    main()
