# Step 5: Segment (Turn Segmentation)

## Purpose

Consume validated transcripts, merge consecutive same-speaker turns, cut per-turn WAV files from the preprocessed audio, and emit a structured dialogue JSON. Segment is a pure audio-cutter: it does **not** drop turns or reject files. All filtering decisions happen in the filter step (step 6).

## Module

`voicetune/stages/segment/` — run via `python -m voicetune.stages.segment`

## CLI Args

| Flag             | Default                 | Description                                      |
| ---------------- | ----------------------- | ------------------------------------------------ |
| `--run-dir`      | `./output`              | Base output directory                            |
| `--input-dir`    | `<run-dir>/validated`    | Directory with validated JSON files              |
| `--audio-dir`    | `<run-dir>/preprocessed` | Directory with preprocessed WAV files            |
| `--output-dir`   | `<run-dir>/segmented`    | Where to write segmented output                  |
| `--merge-gap`    | `0.5`                   | Max gap (seconds) to merge same-speaker segments |

## What It Does

For each `*_validated.json`: merge consecutive same-speaker turns when their gap is ≤ `--merge-gap` (default 0.5 s; merged turns OR their `issues` together), load the preprocessed `full_normalized.wav`, cut per-turn segments by sample index, and write `dialogue.json` plus per-turn WAVs. `rejected`, `reject_reasons`, `validation_confidence`, and per-turn `issues` are copied through untouched for the filter step.

Segment runs on every file, including ones flagged `rejected` — keeps the invariant "all filtering happens in the filter stage". A few wasted cuts are cheaper than scattering policy across stages.

## Input/Output

**Input:**

- `output/validated/{call_id}_validated.json` (turn boundaries + text + issue codes + possibly `rejected: true`)
- `output/preprocessed/{call_id}/full_normalized.wav` (source audio)

**Output:**

```
output/segmented/{call_id}/
  dialogue.json          # every turn from the validated JSON, audio cut
  turns/
    turn_001_spk_0.wav   # per-turn 16-bit PCM WAV
    turn_002_spk_1.wav
    ...
```

**dialogue.json structure (accepted file):**

```json
{
  "call_id": "call_recording",
  "language": "en-US",
  "speakers": ["spk_0", "spk_1"],
  "num_turns": 25,
  "total_duration": 650.5,
  "validation_confidence": 0.85,
  "turns": [
    {
      "turn": 1,
      "speaker": "spk_0",
      "start": 0.0, "end": 3.2,
      "duration": 3.2,
      "audio_path": "turns/turn_001_spk_0.wav",
      "text": "Hi, how can I help you today?",
      "issues": ["too_short"]
    }
  ]
}
```

**dialogue.json structure (validation-rejected file):** `"rejected": true` and `"reject_reasons": [...]` are also set at the top level. The turns and audio are still written so downstream filter can inspect them.

## Key Implementation Details

- `find_audio_for_call()` tries `{audio_dir}/{call_id}/full_normalized.wav` first, then `{call_id}.wav`.
- Turn audio is extracted by sample index: `int(start * sr)` to `int(end * sr)`.
- Already-segmented calls are skipped (presence of `{call_id}/dialogue.json` in the output).
- Per-turn WAVs are 16-bit PCM at 16 kHz mono (same as preprocessed).
- Segment has no imports from validation — it passes codes through without knowing what they mean.

## Dependencies

`soundfile`, `numpy`
