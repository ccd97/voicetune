"""Re-run MLX diarization for specific calls with a forced language.

Usage: python scripts/rerun_mlx.py
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# ---- Configure here ----
LANGUAGE = "mr"
MATCH = [
    "xxxx",
    "yyyy",
]
# -------------------------

INPUT_DIR = Path("./output/preprocessed")
OUTPUT_DIR = Path("./output/diarized")
SCRUB_DIR = Path("./output/scrubbed")


def find_matching_calls(substrings: list[str]) -> list[Path]:
    matches = []
    for d in sorted(INPUT_DIR.iterdir()):
        if not d.is_dir():
            continue
        if any(s.lower() in d.name.lower() for s in substrings):
            wav = d / "full_normalized.wav"
            if wav.exists():
                matches.append(wav)
            else:
                log.warning(f"Matched {d.name} but no full_normalized.wav found")
    return matches


def main():
    from voicetune.diarize.mlx_backend import diarize
    from voicetune.scrub.pipeline import process_file as scrub_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRUB_DIR.mkdir(parents=True, exist_ok=True)

    wavs = find_matching_calls(MATCH)
    if not wavs:
        log.error(f"No calls matched patterns: {PATTERNS}")
        return

    log.info(f"Matched {len(wavs)} call(s)")

    failed = []
    for wav in wavs:
        call_id = wav.parent.name
        log.info(f"Processing {call_id} (language={LANGUAGE})")
        try:
            result = diarize(wav, OUTPUT_DIR, language=LANGUAGE)
            log.info(f"  {len(result['turns'])} turns, {len(set(t['speaker'] for t in result['turns']))} speakers")
            diarized_path = OUTPUT_DIR / f"{result['call_id']}_diarized.json"
            scrub_file(diarized_path, SCRUB_DIR)
        except Exception:
            log.exception(f"Failed: {call_id}")
            failed.append(call_id)

    log.info(f"Done: {len(wavs) - len(failed)}/{len(wavs)} succeeded")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
