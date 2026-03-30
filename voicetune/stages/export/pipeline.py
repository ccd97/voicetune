"""Export pipeline for Fish Speech fine-tuning format.

Converts segmented + labeled dialogues into the directory structure
expected by Fish Speech:

    data/
      me/
        turn_001.wav
        turn_001.lab
        ...
"""

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

MIN_DURATION = 1.0   # skip turns shorter than 1s
MAX_DURATION = 60.0  # skip turns longer than 60s


def process_call(
    dialogue_path: Path,
    output_dir: Path,
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> dict:
    """Export 'me' turns from a segmented call as .wav + .lab pairs."""
    call_dir = dialogue_path.parent

    with open(dialogue_path) as f:
        dialogue = json.load(f)

    call_id = dialogue["call_id"]
    turns = dialogue["turns"]

    exported = 0
    skipped_short = 0
    skipped_long = 0
    skipped_other = 0
    skipped_quality = 0

    for turn in turns:
        # Only export "me" turns
        if turn.get("speaker_label") != "me":
            skipped_other += 1
            continue

        # Skip turns with quality issues
        flags = turn.get("quality_flags", [])
        if flags:
            skipped_quality += 1
            continue

        duration = turn["duration"]
        if duration < min_duration:
            skipped_short += 1
            continue
        if duration > max_duration:
            skipped_long += 1
            continue

        # Source audio
        src_audio = call_dir / turn["audio_path"]
        if not src_audio.exists():
            log.warning(f"  Missing audio: {src_audio}")
            continue

        # Output filenames: call_id + turn number for uniqueness across calls
        base_name = f"{call_id}_turn_{turn['turn']:03d}"

        dst_wav = output_dir / f"{base_name}.wav"
        shutil.copy2(str(src_audio), str(dst_wav))

        dst_lab = output_dir / f"{base_name}.lab"
        dst_lab.write_text(turn["text"].strip())

        exported += 1

    stats = {
        "call_id": call_id,
        "exported": exported,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "skipped_other": skipped_other,
        "skipped_quality": skipped_quality,
    }

    log.info(
        f"  {call_id}: exported {exported}, "
        f"skipped {skipped_other} other + {skipped_short} short + {skipped_long} long + {skipped_quality} quality"
    )
    return stats
