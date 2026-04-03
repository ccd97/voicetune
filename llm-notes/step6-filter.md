# Step 6: Filter (All Turn/File Dropping)

## Purpose

The one and only stage in the pipeline that drops turns or rejects files. Every filtering decision — validation issue codes, duration bounds, audio quality, clip cleaning — lives here. Downstream stages (label, finetune) assume their input is already clean and only do selection, not quality filtering.

Design bias: we would rather drop a clip than train on one that will teach the LoRA a bad habit (fading voice, whispery tail-offs, wrong speaker, cross-talk).

## Module

`voicetune/stages/filter/` — run via `python -m voicetune.stages.filter`

## CLI Args

| Flag             | Default                | Description                                                                                 |
| ---------------- | ---------------------- | ------------------------------------------------------------------------------------------- |
| `--input-dir`    | `./output/segmented`   | Directory containing segmented call directories (`{call_id}/dialogue.json` + `turns/*.wav`) |
| `--output-dir`   | `./output/filtered`    | Output directory for filtered call directories (same layout as input, cleaned)              |
| `--min-duration` | `2.5`                  | Drop turns shorter than this (seconds)                                                      |
| `--max-duration` | `60.0`                 | Drop turns longer than this (seconds)                                                       |

## What It Does

Applied per call, in this order:

1. **File-level rejection** — if the segmented `dialogue.json` has `rejected: true` and the `reject_reasons` include any of the file-level codes (`incorrect_speaker_count`, `nonsensical_conversation`, `language_mismatch`, `language_not_allowed`, `mono_speaker`, `low_confidence`), write a `dialogue.json` with `rejected: true` and no turns, and stop.
2. **Validation per-turn codes** — drop any turn whose `issues` array contains a validation turn code (`improper_diarization`, `incorrect_speaker_assignment`, `garbled_transcript`, `too_short`, `too_long`).
3. **Duration bounds** — drop turns with `duration < --min-duration` or `duration > --max-duration`. This catches turns that validation didn't flag at 0.5 s but that still aren't long enough to train on at 2.5 s, and turns that became too long after segment-time merge.
4. **Audio quality** — drop `mostly_silence` (RMS < −40 dB) and `low_energy` (RMS in [−40 dB, −30 dB)).
5. **Clip cleaning** — loudness-normalize each surviving clip to −20 dBFS RMS, linearly boost the tail on clips with 1–5 dB amplitude decay, drop clips with unrepairable decay (> 5 dB first-third vs last-third drop, code `tail_decay`).
6. **Write** the surviving cleaned WAVs and a new `dialogue.json`. Calls where every turn was dropped are written as `rejected: true` with `all_turns_filtered`.

## Clip cleaning constants

Phone-call turns often trail off in volume at the end (natural conversational hand-off). If left in, the LoRA learns the fade pattern and produces audio that gradually quiets out.

| Constant             | Value  | Meaning                                                     |
| -------------------- | ------ | ----------------------------------------------------------- |
| `TARGET_RMS_DBFS`    | `-20`  | Every kept clip is normalized to this RMS                   |
| `MAX_TAIL_DECAY_DB`  | `5`    | Clips with last-third > 5 dB quieter than first-third are dropped |
| `FADE_FRAME_MS`      | `30`   | Frame size for per-clip RMS envelope                        |
| `SPEECH_FLOOR_DB`    | `-30`  | Frames below this are treated as silence, not speech        |

Clips with 1–5 dB decay get a linear gain ramp applied from the one-third mark to the end that compensates for the measured drop, so the amplitude envelope is roughly flat by the time the clip is normalized.

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

- `FILE_REJECT_CODES` and `VALIDATION_TURN_CODES` are sourced from `voicetune.stages.validation.pipeline` to stay in sync with validation's code definitions.
- Already-filtered calls in the output directory are skipped (resume-safe, presence of `{call_id}/dialogue.json`).
- Kept turns have their `issues` key stripped on output (no outstanding issues remain).
- Final peak-limiting clips to 0.99 to avoid overflow when tail-boost + RMS normalization would otherwise clip.
- Duration bounds default to `MIN_TURN_DURATION = 2.5 s` and `MAX_TURN_DURATION = 60 s`, matching the old finetune `MIN_EXPORT_DURATION` / `MAX_EXPORT_DURATION` that used to live in the finetune stage.

## Dependencies

`soundfile`, `numpy`
