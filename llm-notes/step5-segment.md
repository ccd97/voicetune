# Step 5: Segment (Turn Segmentation)

## Purpose

Consume validated transcripts, merge consecutive same-speaker turns, cut per-turn WAV files from the preprocessed audio, and emit a structured dialogue JSON. Audio-quality filtering and multi-speaker checks happen in the next step (filter).

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

1. **Read** each `*_validated.json` from the validation output
2. **Skip** files with file-level rejection codes (`incorrect_speaker_count`, `nonsensical_conversation`, `language_mismatch`, `language_not_allowed`, `mono_speaker`, `low_confidence`)
3. **Drop turns** whose `issues` array contains per-turn removal codes (`garbled_transcript`, `improper_diarization`, `incorrect_speaker_assignment`, `too_short`, `too_long`)
4. **Merge** consecutive turns from the same speaker when gap <= `--merge-gap` (default 0.5s)
5. **Load** the preprocessed `full_normalized.wav` for the call
6. **Cut** per-turn audio segments by sample index
7. **Write** per-turn WAV files and a dialogue.json manifest

The segment step does NOT drop turns based on audio content. All audio-quality drops (silence, low energy, multi-speaker, amplitude decay) happen in the filter step.

## Input/Output

**Input:**

- `output/validated/{call_id}_validated.json` (turn boundaries + text + issue codes)
- `output/preprocessed/{call_id}/full_normalized.wav` (source audio)

**Output:**

```
output/segmented/{call_id}/
  dialogue.json          # all turns that survived validation codes
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
  ],
  "validation_removed": {"too_short": 3}
}
```

## Key Implementation Details

- `find_audio_for_call()` tries `{audio_dir}/{call_id}/full_normalized.wav` first, then `{call_id}.wav`
- `FILE_REJECT_CODES` and `TURN_REMOVE_CODES` are sourced from `voicetune.stages.validation.pipeline` to stay in sync with validation's code definitions
- Already-segmented calls in the output directory are skipped (resume-safe)
- Turn audio is extracted by sample index: `int(start * sr)` to `int(end * sr)`
- All per-turn WAVs are 16-bit PCM at 16kHz mono (same as preprocessed)
- The dialogue.json is the central artifact that the filter step consumes next

## Dependencies

`soundfile`, `numpy`
