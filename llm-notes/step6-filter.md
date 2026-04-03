# Step 6: Filter (Audio Quality + Clip Cleaning)

## Purpose

Consume segmented call directories, drop bad-audio turns, and emit cleaned per-turn WAVs ready for downstream speaker labeling and fine-tuning.

Two stages run per call:

1. **Audio-quality drops** — `mostly_silence` (RMS < −40 dB) and `low_energy` (RMS < −30 dB).
2. **Clip cleaning** — loudness-normalize each surviving clip to −20 dBFS RMS, linearly boost the tail on clips with 1–5 dB amplitude decay, drop clips with unrepairable decay (> 5 dB first-third vs last-third drop).

Design bias: we would rather drop a clip than train on one that will teach the LoRA a bad habit (fading voice, whispery tail-offs).

## Module

`voicetune/stages/filter/` — run via `python -m voicetune.stages.filter`

## CLI Args

| Flag           | Default                | Description                                                                               |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| `--input-dir`  | `./output/segmented`   | Directory containing segmented call directories (`{call_id}/dialogue.json` + `turns/*.wav`) |
| `--output-dir` | `./output/filtered`    | Output directory for filtered call directories (same layout as input, cleaned)            |

## Clip cleaning

Phone-call turns often trail off in volume at the end (natural conversational hand-off). If left in, the LoRA learns the fade pattern and produces audio that gradually quiets out. Constants in `pipeline.py`:

| Constant             | Value  | Meaning                                                     |
| -------------------- | ------ | ----------------------------------------------------------- |
| `TARGET_RMS_DBFS`    | `-20`  | Every kept clip is normalized to this RMS                   |
| `MAX_TAIL_DECAY_DB`  | `5`    | Clips with last-third > 5 dB quieter than first-third are dropped |
| `FADE_FRAME_MS`      | `30`   | Frame size for per-clip RMS envelope                        |
| `SPEECH_FLOOR_DB`    | `-30`  | Frames below this are treated as silence, not speech        |

Clips with 1–5 dB decay get a linear gain ramp applied from the one-third mark to the end that compensates for the measured drop, so the amplitude envelope is roughly flat by the time the clip is normalized.

## Input/Output

**Input:** `output/segmented/{call_id}/dialogue.json` + `output/segmented/{call_id}/turns/*.wav`

**Output:** `output/filtered/{call_id}/dialogue.json` + `output/filtered/{call_id}/turns/*.wav`

The output dialogue.json mirrors the input but retains only kept turns, and adds:

```json
{
  "original_turn_count": 25,
  "filtered_turn_count": 22,
  "filter_removed": {"tail_decay": 2, "mostly_silence": 1}
}
```

Turn indices and `audio_path` values are preserved from the input (kept turns keep their original `turn` number, so you can cross-reference with the segmented output).

Calls where every turn is dropped are skipped (no output directory written) and counted as `rejected` in the summary.

## Removal codes

| Code             | Source                                      |
|------------------|---------------------------------------------|
| `mostly_silence` | Clip RMS < −40 dB                           |
| `low_energy`     | Clip RMS in [−40 dB, −30 dB)                |
| `tail_decay`     | Last-third > 5 dB quieter than first-third  |

## Key Implementation Details

- Filter does NOT re-apply validation-level codes; those are handled upstream in the segment step.
- Already-filtered calls in the output directory are skipped (resume-safe, presence of `{call_id}/dialogue.json`).
- Final peak-limiting clips to 0.99 to avoid overflow when tail-boost + RMS normalization would otherwise clip.

## Dependencies

`soundfile`, `numpy`
