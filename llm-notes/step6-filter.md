# Step 6: Filter (All Turn/File Dropping)

## Purpose

The only stage that drops turns or rejects files. Validation codes, duration bounds, and audio-quality checks all run here. Downstream stages (label, finetune) assume their input is clean and only do selection.

Bias toward dropping clips rather than keeping marginal ones — training on fading voice, whispery endings, or cross-talk teaches the LoRA bad habits.

## Module

`voicetune/stages/filter/` — run via `python -m voicetune.stages.filter`

## CLI Args

| Flag             | Default                | Description                                                                                 |
| ---------------- | ---------------------- | ------------------------------------------------------------------------------------------- |
| `--input-dir`    | `./output/segmented`   | Directory containing segmented call directories (`{call_id}/dialogue.json` + `turns/*.wav`) |
| `--output-dir`   | `./output/filtered`    | Output directory for filtered call directories (same layout as input, cleaned)              |
| `--min-duration` | `3.0`                  | Drop turns shorter than this (seconds)                                                      |
| `--max-duration` | `30.0`                 | Drop turns longer than this (seconds)                                                       |

## What It Does

Applied per call, in this order:

1. **File-level rejection** — if the segmented `dialogue.json` has `rejected: true` and the `reject_reasons` include any of the file-level codes (`incorrect_speaker_count`, `nonsensical_conversation`, `language_mismatch`, `language_not_allowed`, `mono_speaker`, `low_confidence`), write a `dialogue.json` with `rejected: true` and no turns, and stop.
2. **Validation per-turn codes** — drop any turn whose `issues` array contains a validation turn code (`improper_diarization`, `incorrect_speaker_assignment`, `garbled_transcript`, `too_short`, `too_long`).
3. **Duration bounds** — drop turns with `duration < --min-duration` or `duration > --max-duration`. Tighten these here; downstream finetune consumes `output/filtered/` as-is.
4. **Audio quality** — drop `mostly_silence` (RMS < −40 dB), `low_energy` (RMS in [−40 dB, −30 dB)), and `tail_decay` (last-third > 5 dB quieter than first-third).
5. **Write** — surviving WAVs are copied as-is (no amplitude changes; preprocess already normalized loudness) with a new `dialogue.json`. Calls where every turn was dropped are written as `rejected: true` with `all_turns_filtered`.

## Audio-quality constants

Phone-call turns often trail off in volume. Training on fading clips teaches the LoRA to produce audio that quiets out, so decayed clips are dropped outright.

| Constant             | Value  | Meaning                                                     |
| -------------------- | ------ | ----------------------------------------------------------- |
| `SILENCE_RMS_DB`     | `-40`  | Clip-level RMS below this → `mostly_silence`                |
| `LOW_ENERGY_RMS_DB`  | `-30`  | Clip-level RMS in [−40, −30) → `low_energy`                 |
| `SPEECH_FLOOR_DB`    | `-30`  | Per-frame RMS below this counts as silence                  |
| `MAX_TAIL_DECAY_DB`  | `5`    | Last-third mean > 5 dB below first-third mean → `tail_decay`|
| `DECAY_FRAME_MS`     | `30`   | Frame size for the per-clip RMS envelope                    |

## Input/Output

**Input:** `output/segmented/{call_id}/dialogue.json` + `output/segmented/{call_id}/turns/*.wav` (possibly with top-level `rejected: true` and per-turn `issues`).

**Output:** `output/filtered/{call_id}/dialogue.json` + `output/filtered/{call_id}/turns/*.wav`

Accepted dialogue.json:

```json
{
  "original_turn_count": 25,
  "filtered_turn_count": 22,
  "filter_removed": {"tail_decay": 2, "mostly_silence": 1},
  "turns": [ ... ]
}
```

Rejected dialogue.json (file-level or all-turns-filtered):

```json
{
  "call_id": "call_recording",
  "rejected": true,
  "reject_reasons": ["mono_speaker"],
  "original_turn_count": 12,
  "filtered_turn_count": 0,
  "turns": []
}
```

## Removal codes

| Code                             | Source                                              |
|----------------------------------|-----------------------------------------------------|
| `improper_diarization`           | Validation LLM per-turn issue                       |
| `incorrect_speaker_assignment`   | Validation LLM per-turn issue                       |
| `garbled_transcript`             | Validation LLM per-turn issue                       |
| `too_short`                      | Validation tagged or `duration < --min-duration`    |
| `too_long`                       | Validation tagged or `duration > --max-duration`    |
| `mostly_silence`                 | Clip RMS < −40 dB                                   |
| `low_energy`                     | Clip RMS in [−40 dB, −30 dB)                        |
| `tail_decay`                     | Last-third > 5 dB quieter than first-third          |

## Key Implementation Details

- `FILE_REJECT_CODES` and `VALIDATION_TURN_CODES` are sourced from `voicetune.stages.validation.pipeline`.
- Already-filtered calls are skipped (resume-safe: presence of `{call_id}/dialogue.json` in the output).
- Kept turns have their `issues` key stripped on output.
- Calls with fewer than 2 surviving turns are rejected with `single_turn`.
- Duration defaults: `MIN_TURN_DURATION = 3.0 s`, `MAX_TURN_DURATION = 30.0 s`. Finetune reads `output/filtered/` as-is.

## Dependencies

`soundfile`, `numpy`
