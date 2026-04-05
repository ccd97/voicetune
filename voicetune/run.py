"""Pipeline runner: preprocess → diarize → scrub → validation → segment → filter → label → finetune."""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STEPS = [
    "preprocess",     # 1
    "diarize",        # 2
    "scrub",          # 3
    "validation",     # 4
    "segment",        # 5
    "filter",         # 6
    "label",          # 7
    "finetune",       # 8
]

OPTIONAL_STEPS = {"scrub", "validation"}


class Manifest:

    def __init__(self, path: Path, config: dict | None = None):
        self.path = path
        self.data = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": config or {},
            "steps": {},
        }

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        with open(path) as f:
            raw = json.load(f)
        m = cls.__new__(cls)
        m.path = path
        m.data = raw
        return m

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def record(self, step: str, status: str, elapsed: float, error: str | None = None):
        entry = {"status": status, "elapsed": round(elapsed, 1)}
        if error:
            entry["error"] = error
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.data["steps"][step] = entry
        self.save()

    def completed_steps(self) -> set[str]:
        return {name for name, info in self.data["steps"].items() if info["status"] == "complete"}

    def first_incomplete(self, planned: list[str]) -> list[str]:
        done = self.completed_steps()
        for i, step in enumerate(planned):
            if step not in done:
                return planned[i:]
        return []


def get_skip_steps() -> set[str]:
    skipped = set()
    for step in OPTIONAL_STEPS:
        if os.environ.get(f"SKIP_{step.upper()}", "").lower() in ("1", "true", "yes"):
            skipped.add(step)
    return skipped


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

    speaker_samples: dict[str, list[str]] = {}
    for turn in dialogue["turns"]:
        samples = speaker_samples.setdefault(turn["speaker"], [])
        if len(samples) < 3:
            samples.append(turn["text"][:80])

    print("\n" + "=" * 60)
    print("SPEAKER SELECTION")
    print("=" * 60)
    print(f"\nDetected {len(speaker_samples)} speakers in '{call_id}':\n")

    turns_dir = seg_dir / call_id / "turns"
    speakers = sorted(speaker_samples.keys())
    for i, spk in enumerate(speakers, 1):
        audio_files = sorted(turns_dir.glob(f"*_{spk}.wav")) if turns_dir.exists() else []
        print(f"  [{i}] {spk}")
        if audio_files:
            print(f"      audio: {audio_files[0]}")
        for sample in speaker_samples[spk]:
            print(f"      \"{sample}\"")
        print()

    while True:
        choice = input("Which speaker is you? Enter number or label: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(speakers):
            selected = speakers[int(choice) - 1]
            break
        if choice in speakers:
            selected = choice
            break
        print(f"Invalid choice. Enter 1-{len(speakers)} or a speaker label.")

    print(f"\n  Selected: {selected}\n")
    return selected


def run_step(name: str, args: list[str], python: str = sys.executable,
             manifest: Manifest | None = None, run_dir: Path | None = None):
    run_dir_args = ["--run-dir", str(run_dir)] if run_dir else []
    cmd = [python, "-m", f"voicetune.stages.{name}", *run_dir_args, *args]
    log.info(f"{'=' * 60}")
    log.info(f"STEP: {name}")
    log.info(f"  cmd: {' '.join(cmd)}")
    log.info(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        if manifest:
            manifest.record(name, "failed", elapsed, error=f"exit code {result.returncode}")
        log.error(f"STEP {name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)
    if manifest:
        manifest.record(name, "complete", elapsed)
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
        "--mode", choices=["aws", "whisperx", "whispermlx", "mlx", "llamacpp"], default="mlx",
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
        "--run-dir", type=Path, default=Path("./output"),
        help="Base output directory for all pipeline artifacts (default: ./output)"
    )
    parser.add_argument(
        "--python", type=str, default=None,
        help="Python interpreter to use (default: current interpreter)"
    )
    parser.add_argument(
        "--finetune-test", action="store_true",
        help="Finetune in test mode (spot A100, 1 step)"
    )
    parser.add_argument(
        "--validation-backend", choices=["bedrock", "llamacpp"], default="llamacpp",
        help="Validation backend (default: llamacpp)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last failed/incomplete step using manifest.json"
    )
    args = parser.parse_args()

    python = args.python or sys.executable
    run_dir = args.run_dir
    manifest_path = run_dir / "manifest.json"

    timings = {}

    try:
        steps_to_run = parse_steps(args.steps)
        skip = get_skip_steps()
    except ValueError as e:
        log.error(str(e))
        log.info("Available steps:")
        for i, name in enumerate(STEPS, 1):
            opt = " (optional)" if name in OPTIONAL_STEPS else ""
            log.info(f"  {i}. {name}{opt}")
        sys.exit(1)

    if skip:
        steps_to_run = [s for s in steps_to_run if s not in skip]
        log.info(f"SKIP_STEPS: {', '.join(sorted(skip))}")

    if args.resume:
        if not manifest_path.exists():
            log.error(f"No manifest found at {manifest_path} — nothing to resume")
            sys.exit(1)
        prev = Manifest.load(manifest_path)
        remaining = prev.first_incomplete(steps_to_run)
        if not remaining:
            log.info("All steps already complete — nothing to resume")
            sys.exit(0)
        done = prev.completed_steps()
        skipped = [s for s in steps_to_run if s in done]
        if skipped:
            log.info(f"Skipping completed: {', '.join(skipped)}")
        steps_to_run = remaining
        manifest = prev
        manifest.data["resumed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        manifest = Manifest(manifest_path, config={
            "mode": args.mode,
            "num_speakers": args.num_speakers,
            "language": args.language,
            "steps_requested": args.steps,
            "run_dir": str(run_dir),
        })

    manifest.save()
    log.info(f"Steps to run: {', '.join(f'{STEPS.index(s)+1}.{s}' for s in steps_to_run)}")

    step_kw = dict(python=python, manifest=manifest, run_dir=run_dir)

    if "preprocess" in steps_to_run:
        timings["preprocess"] = run_step("preprocess", [], **step_kw)

    if "diarize" in steps_to_run:
        diarize_args = ["--mode", args.mode]
        if args.num_speakers:
            diarize_args += ["--num-speakers", str(args.num_speakers)]
        if args.language:
            diarize_args += ["--language", args.language]
        timings["diarize"] = run_step("diarize", diarize_args, **step_kw)

    if "scrub" in steps_to_run:
        timings["scrub"] = run_step("scrub", [], **step_kw)

    if "validation" in steps_to_run:
        validation_args = ["--backend", args.validation_backend]
        scrubbed_dir = run_dir / "scrubbed"
        if "scrub" in steps_to_run and scrubbed_dir.exists() and any(scrubbed_dir.glob("*_diarized.json")):
            validation_args += ["--input-dir", str(scrubbed_dir)]
        timings["validation"] = run_step("validation", validation_args, **step_kw)

    if "segment" in steps_to_run:
        timings["segment"] = run_step("segment", [], **step_kw)

    if "filter" in steps_to_run:
        timings["filter"] = run_step("filter", [], **step_kw)

    if "label" in steps_to_run:
        filtered_dir = run_dir / "filtered"
        call_ids = sorted(
            d.name for d in filtered_dir.iterdir()
            if d.is_dir() and (d / "dialogue.json").exists()
        ) if filtered_dir.exists() else []

        if not call_ids:
            log.error("No filtered calls found for labeling")
            sys.exit(1)

        voiceprint = run_dir / "voiceprint.npy"
        if voiceprint.exists():
            log.info(f"Using existing voiceprint: {voiceprint}")
        else:
            speaker = _prompt_speaker(filtered_dir, call_ids[0])
            timings["label-enroll"] = run_step(
                "label",
                ["enroll", "--call-id", call_ids[0], "--speaker", speaker],
                **step_kw,
            )

        timings["label"] = run_step("label", ["label"], **step_kw)

    if "finetune" in steps_to_run:
        finetune_args = []
        if args.finetune_test:
            finetune_args.append("--test")
        timings["finetune"] = run_step("finetune", finetune_args, **step_kw)

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
