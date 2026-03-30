# Step 9: Export (Fish Speech Format)

## Purpose
Export "me" turns as `.wav` + `.lab` pairs in the directory structure expected by Fish Speech for fine-tuning.

## Module
`voicetune/export/` — run via `python -m voicetune.export`

## CLI Args
| Flag | Default | Description |
|------|---------|-------------|
| `--segmented-dir` | `./output/segmented` | Directory with segmented call output |
| `--output-dir` | `./output/fish-speech/data/me` | Output directory for .wav + .lab pairs |
| `--min-duration` | `1.0` | Skip turns shorter than this (seconds) |
| `--max-duration` | `60.0` | Skip turns longer than this (seconds) |

## What It Does
1. **Scan** each `dialogue.json` for turns where `speaker_label == "me"`
2. **Filter** by:
   - Skip if `speaker_label != "me"` (other speakers)
   - Skip if turn has any `quality_flags` (from segment step)
   - Skip if duration < 1.0s or > 60.0s
3. **Copy** the turn's WAV file to the output directory
4. **Write** a `.lab` file containing just the plain text transcription (no timestamps, no markup)
5. **Write** an `export_summary.json` with per-call stats

## Input/Output

**Input:** `output/segmented/{call_id}/dialogue.json` + `turns/*.wav`

**Output:**
```
output/fish-speech/
  data/
    me/
      call_recording_turn_003.wav    # audio clip
      call_recording_turn_003.lab    # plain text transcription
      call_recording_turn_005.wav
      call_recording_turn_005.lab
      ...
  export_summary.json               # stats per call
```

**.lab file contents** (example):
```
Sure, I can help you with that. Let me pull up your account.
```

**export_summary.json:**
```json
{
  "total_exported": 18,
  "output_dir": "output/fish-speech/data/me",
  "calls": [
    {
      "call_id": "call_recording",
      "exported": 18,
      "skipped_short": 2,
      "skipped_long": 0,
      "skipped_other": 25,
      "skipped_quality": 3
    }
  ]
}
```

## Fish Speech Training Format
Fish Speech expects:
```
data/
  SPEAKER_ID/
    utterance.wav    # audio clip
    utterance.lab    # plain text transcription
```

After export, the next steps (from finetune-plan.md) are:
1. Loudness normalize: `fap loudness-norm data-raw data --clean`
2. Extract semantic tokens with S2 Pro codec
3. Pack into protobuf dataset
4. LoRA fine-tune on S2 Pro

## Key Implementation Details
- Uses `shutil.copy2` for WAV files (preserves metadata timestamps)
- Lab files use `.write_text()` with stripped text
- Filenames include call_id + turn number for uniqueness across calls
- Quality flags from the segment step act as a filter gate — any flag = skip
- The summary JSON goes one level up from `data/me/` (in `fish-speech/`)

## Dependencies
`shutil`, `soundfile`, `numpy` (inherited, not directly used)
