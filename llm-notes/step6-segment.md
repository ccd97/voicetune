# Step 6: Segment (Turn Segmentation)

## Purpose

Merge consecutive same-speaker turns, cut per-turn WAV files from the preprocessed audio, drop turns with audio quality issues, and produce a structured dialogue JSON.

## Module

`voicetune/stages/segment/` — run via `python -m voicetune.stages.segment`

## CLI Args

| Flag             | Default                 | Description                                      |
| ---------------- | ----------------------- | ------------------------------------------------ |
| `--input-dir`    | `./output/filtered`     | Directory with filtered JSON files               |
| `--audio-dir`    | `./output/preprocessed` | Directory with preprocessed WAV files            |
| `--output-dir`   | `./output/segmented`    | Where to write segmented output                  |
| `--merge-gap`    | `0.5`                   | Max gap (seconds) to merge same-speaker segments |

## What It Does

1. **Read** each `*_filtered.json` from the filter step output
2. **Merge** consecutive turns from the same speaker when gap <= merge-gap (default 0.5s)
3. **Load** the preprocessed `full_normalized.wav` for the call
4. **Cut** per-turn audio segments based on start/end timestamps
5. **Drop** turns that fail audio quality checks:
  - `mostly_silence` — RMS < -40 dB
  - `low_energy` — RMS < -30 dB (but not silence)
6. **Write** per-turn WAV files and a dialogue.json manifest

Timing-based quality checks (`too_short`, `too_long`) are handled earlier by the validation + filter steps.

## Input/Output

**Input:**

- `output/filtered/{call_id}_filtered.json` (turn boundaries + text, already cleaned by filter)
- `output/preprocessed/{call_id}/full_normalized.wav` (source audio)

**Output:**

```
output/segmented/{call_id}/
  dialogue.json          # structured dialogue with passing turns only
  turns/
    turn_001_spk_0.wav   # per-turn 16-bit PCM WAV
    turn_002_spk_1.wav
    ...
```

**dialogue.json structure:**

```json
{
  "call_id": "call_recording",
  "language": "en-US",
  "speakers": ["spk_0", "spk_1"],
  "num_turns": 25,
  "total_duration": 650.5,
  "turns": [
    {
      "turn": 1,
      "speaker": "spk_0",
      "start": 0.0, "end": 3.2,
      "duration": 3.2,
      "audio_path": "turns/turn_001_spk_0.wav",
      "text": "Hi, how can I help you today?"
    }
  ]
}
```

## Key Implementation Details

- `find_audio_for_call()` tries `{audio_dir}/{call_id}/full_normalized.wav` first, then `{call_id}.wav`
- Turn audio is extracted by sample index: `int(start * sr)` to `int(end * sr)`
- All per-turn WAVs are 16-bit PCM at 16kHz mono (same as preprocessed)
- Turns that fail audio quality checks are silently dropped (logged at DEBUG level)
- The dialogue.json is the central artifact that the label step consumes

## Dependencies

`soundfile`, `numpy`
