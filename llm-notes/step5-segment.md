# Step 5: Segment (Turn Segmentation)

## Purpose

Consume validated transcripts, merge consecutive same-speaker turns, cut per-turn WAV files from the preprocessed audio, and emit a structured dialogue JSON. Segment is a pure audio-cutter: it does **not** drop turns or reject files. All filtering decisions happen in the filter step (step 6).

## Module

`voicetune/stages/segment/` — run via `python -m voicetune.stages.segment`

## CLI Args

| Flag             | Default                 | Description                                      |
| ---------------- | ----------------------- | ------------------------------------------------ |
| `--input-dir`    | `./output/validated`    | Directory with validated JSON files              |
| `--audio-dir`    | `./output/preprocessed` | Directory with preprocessed WAV files            |
| `--output-dir`   | `./output/segmented`    | Where to write segmented output                  |
| `--merge-gap`    | `0.5`                   | Max gap (seconds) to merge same-speaker segments |

## What It Does

1. **Read** each `*_validated.json` from the validation output.
2. **Merge** consecutive turns from the same speaker when gap <= `--merge-gap` (default 0.5 s). Issue codes on merged turns are combined and carried forward.
3. **Load** the preprocessed `full_normalized.wav` for the call.
4. **Cut** per-turn audio segments by sample index.
5. **Write** per-turn WAV files and a `dialogue.json` manifest.
6. **Propagate** `rejected`, `reject_reasons`, `validation_confidence`, and per-turn `issues` from the validated JSON untouched so the filter step can apply them.

Segment processes every file, including ones validation flagged as `rejected`. This keeps the architectural invariant "all filtering happens in the filter stage"; the few wasted cuts on already-rejected files are cheaper than scattering policy logic across stages.

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
- Already-segmented calls in the output directory are skipped (resume-safe).
- Turn audio is extracted by sample index: `int(start * sr)` to `int(end * sr)`.
- All per-turn WAVs are 16-bit PCM at 16kHz mono (same as preprocessed).
- The dialogue.json is the central artifact that the filter step consumes next.
- Segment has **no imports from validation** — it doesn't need to know what validation's codes mean, only that they should be copied through.

## Dependencies

`soundfile`, `numpy`
