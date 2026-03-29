"""One-click pipeline: preprocess → diarize → translate → correction → segment → label → pairformat → export → finetune."""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

STEPS = [
    "preprocess",     # 1
    "diarize",        # 2
    "translate",      # 3
    "correction",     # 4
    "segment",        # 5
    "label",          # 6
    "pairformat",     # 7
    "export",         # 8
    "finetune",       # 9
]


def parse_steps(spec: str) -> list[str]:
    """Parse a step spec into a list of step names.

    Accepts:
      - Single number: "5"
      - Range: "3-6"
      - Comma-separated: "1,3,5"
      - Mixed: "1,3-5,8"
      - Step name: "segment"
      - "all" (default)
    """
    if spec.lower() == "all":
        return list(STEPS)

    # If it's a known step name, return just that
    if spec in STEPS:
        return [spec]

    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if lo < 1 or hi > len(STEPS) or lo > hi:
                raise ValueError(f"Invalid range {lo}-{hi} (steps are 1-{len(STEPS)})")
            selected.update(range(lo, hi + 1))
        elif part.isdigit():
            n = int(part)
            if n < 1 or n > len(STEPS):
                raise ValueError(f"Invalid step {n} (steps are 1-{len(STEPS)})")
            selected.add(n)
        elif part in STEPS:
            selected.add(STEPS.index(part) + 1)
        else:
            raise ValueError(f"Unknown step: {part}")

    return [STEPS[i - 1] for i in sorted(selected)]


def _prompt_speaker(seg_dir: Path, call_id: str) -> str:
    """Show detected speakers with sample text and ask user to pick one."""
    dialogue_path = seg_dir / call_id / "dialogue.json"
    with open(dialogue_path) as f:
        dialogue = json.load(f)

    # Collect sample turns per speaker
    speaker_samples: dict[str, list[str]] = {}
    for turn in dialogue["turns"]:
        spk = turn["speaker"]
        if spk not in speaker_samples:
            speaker_samples[spk] = []
        if len(speaker_samples[spk]) < 3:
            text = turn["text"][:80]
            speaker_samples[spk].append(text)

    print("\n" + "=" * 60)
    print("SPEAKER SELECTION")
    print("=" * 60)
    print(f"\nDetected {len(speaker_samples)} speakers in '{call_id}':\n")

    turns_dir = seg_dir / call_id / "turns"
    speakers = sorted(speaker_samples.keys())
    for i, spk in enumerate(speakers, 1):
        # Find first audio file for this speaker
        audio_files = sorted(turns_dir.glob(f"*_{spk}.wav")) if turns_dir.exists() else []
        print(f"  [{i}] {spk}")
        if audio_files:
            print(f"      audio: {audio_files[0]}")
        for sample in speaker_samples[spk]:
            print(f"      \"{sample}\"")
        print()

    while True:
        choice = input("Which speaker is you? Enter number or label: ").strip()
        # Accept number
        if choice.isdigit() and 1 <= int(choice) <= len(speakers):
            selected = speakers[int(choice) - 1]
            break
        # Accept label directly
        if choice in speakers:
            selected = choice
            break
        print(f"Invalid choice. Enter 1-{len(speakers)} or a speaker label.")

    print(f"\n  Selected: {selected}\n")
    return selected


def run_step(name: str, args: list[str], python: str = sys.executable):
    """Run a pipeline step as a subprocess."""
    cmd = [python, "-m", f"voicetune.{name}", *args]
    log.info(f"{'=' * 60}")
    log.info(f"STEP: {name}")
    log.info(f"  cmd: {' '.join(cmd)}")
    log.info(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error(f"STEP {name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)
    log.info(f"STEP {name} completed in {elapsed:.1f}s")
    return elapsed


def main():
    from dotenv import load_dotenv

    from voicetune.common import setup_logging

    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run the full audio processing pipeline"
    )
    parser.add_argument(
        "--mode", choices=["aws", "whisperx", "mlx"], default="mlx",
        help="Diarization backend (default: mlx)"
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Expected number of speakers (default: auto-detect)"
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Force language code (default: auto-detect)"
    )
    parser.add_argument(
        "--steps", type=str, default="all",
        help="Steps to run: 'all', a number (5), range (3-6), list (1,3,5), "
             "or mixed (1,3-5,8). Step names also accepted."
    )
    parser.add_argument(
        "--python", type=str, default=None,
        help="Python interpreter to use (default: auto-detect based on mode)"
    )
    parser.add_argument(
        "--fish-speech-dir", type=Path, default=None,
        help="Path to cloned fish-speech repo (required for finetune step)"
    )
    args = parser.parse_args()

    # Pick the right python for the mode
    if args.python:
        python = args.python
    elif args.mode in ("whisperx", "mlx"):
        venv = Path(".venv-whisperx/bin/python")
        if venv.exists():
            python = str(venv)
        else:
            log.error(f".venv-whisperx not found. Create it for {args.mode} mode.")
            sys.exit(1)
    else:
        python = sys.executable

    # For steps that use Claude API (translate, correction), use .venv
    api_python = str(Path(".venv/bin/python")) if Path(".venv/bin/python").exists() else sys.executable

    timings = {}

    try:
        steps_to_run = parse_steps(args.steps)
    except ValueError as e:
        log.error(str(e))
        log.info("Available steps:")
        for i, name in enumerate(STEPS, 1):
            log.info(f"  {i}. {name}")
        sys.exit(1)

    log.info(f"Steps to run: {', '.join(f'{STEPS.index(s)+1}.{s}' for s in steps_to_run)}")

    # 1. Preprocess
    if "preprocess" in steps_to_run:
        timings["preprocess"] = run_step("preprocess", [], python=api_python)

    # 2. Diarize
    if "diarize" in steps_to_run:
        diarize_args = ["--mode", args.mode]
        if args.num_speakers:
            diarize_args += ["--num-speakers", str(args.num_speakers)]
        if args.language:
            diarize_args += ["--language", args.language]
        timings["diarize"] = run_step("diarize", diarize_args, python=python)

    # 3. Translate
    if "translate" in steps_to_run:
        timings["translate"] = run_step("translate", [], python=api_python)

    # 4. Correction
    if "correction" in steps_to_run:
        timings["correction"] = run_step("correction", [], python=api_python)
        # Copy corrected output over diarized so segment picks it up
        diarized_dir = Path("output/diarized")
        for f in diarized_dir.glob("*_corrected.json"):
            target = diarized_dir / f.name.replace("_corrected.json", "_diarized.json")
            shutil.copy2(f, target)
            log.info(f"  Copied {f.name} -> {target.name}")

    # 5. Segment
    if "segment" in steps_to_run:
        timings["segment"] = run_step("segment", [], python=api_python)

    # 6-8. Label, Pairformat, Export — prompt user for speaker selection
    if "label" in steps_to_run:
        seg_dir = Path("output/segmented")
        call_ids = sorted(
            d.name for d in seg_dir.iterdir()
            if d.is_dir() and (d / "dialogue.json").exists()
        ) if seg_dir.exists() else []

        if not call_ids:
            log.error("No segmented calls found for labeling")
            sys.exit(1)

        voiceprint = Path("output/voiceprint.npy")
        if voiceprint.exists():
            log.info(f"Using existing voiceprint: {voiceprint}")
        else:
            # First time — prompt user to pick their speaker
            speaker = _prompt_speaker(seg_dir, call_ids[0])
            timings["label-enroll"] = run_step(
                "label",
                ["enroll", "--call-id", call_ids[0], "--speaker", speaker],
                python=api_python,
            )

        timings["label"] = run_step("label", ["label"], python=api_python)

    if "pairformat" in steps_to_run:
        timings["pairformat"] = run_step("pairformat", [], python=api_python)

    if "export" in steps_to_run:
        timings["export"] = run_step("export", [], python=api_python)

    # 9. Finetune
    if "finetune" in steps_to_run:
        if not args.fish_speech_dir:
            log.error("--fish-speech-dir is required for the finetune step")
            sys.exit(1)
        finetune_args = ["--fish-speech-dir", str(args.fish_speech_dir)]
        timings["finetune"] = run_step("finetune", finetune_args, python=api_python)

    # Summary
    log.info(f"{'=' * 60}")
    log.info("PIPELINE COMPLETE")
    log.info(f"{'=' * 60}")
    total = 0
    for step, elapsed in timings.items():
        log.info(f"  {step:20s} {elapsed:6.1f}s")
        total += elapsed
    log.info(f"  {'TOTAL':20s} {total:6.1f}s")


if __name__ == "__main__":
    main()
