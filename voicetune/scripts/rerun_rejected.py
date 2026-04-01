"""Re-run diarization for calls rejected by the validation step.

Scans output/validated/ for _validated.json files with "rejected": true
and re-runs diarization with the chosen backend. Optionally force a language.

Usage:
  python scripts/rerun_rejected.py --mode mlx
  python scripts/rerun_rejected.py --mode mlx --reasons language_mismatch --language mr
  python scripts/rerun_rejected.py --mode aws --reasons garbled_transcript
"""

import argparse
import json
import logging
from pathlib import Path

from voicetune.stages.validation.pipeline import RejectReason

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

INPUT_DIR = Path("./output/preprocessed")
OUTPUT_DIR = Path("./output/diarized")
SCRUB_DIR = Path("./output/scrubbed")
VALIDATED_DIR = Path("./output/validated")

BACKENDS = ["aws", "whisperx", "mlx"]


def main():
    valid_reasons = [r.value for r in RejectReason]

    parser = argparse.ArgumentParser(description="Re-run diarization for validation-rejected calls")
    parser.add_argument("--mode", choices=BACKENDS, required=True, help="Diarization backend")
    parser.add_argument("--language", type=str, default=None, help="Force language (e.g. 'mr', 'hi')")
    parser.add_argument("--num-speakers", type=int, default=None, help="Expected number of speakers")
    parser.add_argument("--reasons", nargs="+", choices=valid_reasons, default=None,
                        help="Only re-run calls matching these reasons")
    parser.add_argument("--validated-dir", type=Path, default=VALIDATED_DIR,
                        help="Directory with validated JSON files (default: ./output/validated)")
    args = parser.parse_args()

    rejections = []
    for path in sorted(args.validated_dir.glob("*_validated.json")):
        with open(path) as f:
            data = json.load(f)
        if not data.get("rejected"):
            continue
        rejections.append({
            "call_id": data["call_id"],
            "reasons": data.get("reject_reasons", []),
            "confidence": data.get("validation_confidence", -1.0),
        })

    if args.reasons:
        filter_set = set(args.reasons)
        rejections = [r for r in rejections if filter_set & set(r["reasons"])]

    if not rejections:
        log.info("No rejected calls to re-run")
        return

    log.info(f"Found {len(rejections)} rejected call(s), backend: {args.mode}")
    for r in rejections:
        log.info(f"  {r['call_id']}: {', '.join(r['reasons'])} (confidence: {r['confidence']:.2f})")

    from voicetune.stages.diarize.pipeline import process_file as diarize_file
    from voicetune.stages.scrub.pipeline import process_file as scrub_file
    SCRUB_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for r in rejections:
        call_id = r["call_id"]
        wav = INPUT_DIR / call_id / "full_normalized.wav"
        if not wav.exists():
            log.error(f"Not found: {wav}")
            failed.append(call_id)
            continue

        stale = args.validated_dir / f"{call_id}_validated.json"
        if stale.exists():
            stale.unlink()
            log.info(f"  Deleted stale {stale}")

        lang_str = f", language={args.language}" if args.language else ""
        log.info(f"Re-running {call_id}{lang_str}")
        try:
            result = diarize_file(wav, OUTPUT_DIR, args.mode, args.num_speakers, args.language)
            log.info(f"  {len(result['turns'])} turns, {len(set(t['speaker'] for t in result['turns']))} speakers")
            diarized_path = OUTPUT_DIR / f"{result['call_id']}_diarized.json"
            scrub_file(diarized_path, SCRUB_DIR)
        except Exception:
            log.exception(f"Failed: {call_id}")
            failed.append(call_id)

    log.info(f"Done: {len(rejections) - len(failed)}/{len(rejections)} succeeded")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
