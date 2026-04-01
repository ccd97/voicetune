# Step 7: Label (Speaker Identification + Dataset Preparation)

## Purpose
Label speakers as "me" vs "other" using a voiceprint embedding, then prepare the fine-tuning dataset by exporting "me" turns as `.wav` + `.lab` pairs. Two-phase process: first enroll (create voiceprint from a reference call), then label all calls and optionally export.

## Module
`voicetune/stages/label/` — run via `python -m voicetune.stages.label {enroll|label}`

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `./output/segmented` | Segmented output directory |

### Enroll subcommand
| Flag | Default | Description |
|------|---------|-------------|
| `--call-id` | (required) | Reference call to use for enrollment |
| `--speaker` | (required) | Your speaker label in that call (e.g. `spk_0`) |
| `--voiceprint` | `./output/voiceprint.npy` | Where to save voiceprint |

### Label subcommand
| Flag | Default | Description |
|------|---------|-------------|
| `--voiceprint` | `./output/voiceprint.npy` | Path to voiceprint file |
| `--call-id` | all calls | Label a specific call only |
| `--auto-skip` | `False` | Skip low-confidence matches without prompting |
| `--prepare` | `False` | Also prepare dataset after labeling |
| `--dataset-dir` | `./output/fish-speech/data/me` | Output directory for .wav + .lab pairs |
| `--min-duration` | `1.0` | Skip turns shorter than this (seconds) |
| `--max-duration` | `60.0` | Skip turns longer than this (seconds) |

## What It Does

### Enrollment Phase
1. Load the dialogue.json for the specified call
2. Find all turns matching the specified speaker label
3. Take up to 10 longest turns for robustness
4. Extract speaker embeddings using resemblyzer's `VoiceEncoder`
5. Average embeddings into a single voiceprint vector
6. Save as `.npy` file

### Labeling Phase
1. For each call's dialogue.json, group turns by speaker
2. Extract averaged embeddings per speaker (up to 10 longest turns each)
3. Compute cosine similarity between each speaker embedding and the reference voiceprint
4. Assign "me" to the most similar speaker, "other" to the rest
5. Update dialogue.json in-place with:
   - `speaker_label` field on each turn ("me" or "other")
   - `speaker_labels` map (e.g. `{"spk_0": "me", "spk_1": "other"}`)
   - `speaker_similarities` scores

### Dataset Preparation (with `--prepare`)
After labeling, exports "me" turns as `.wav` + `.lab` pairs in Fish Speech format:
1. Filter for turns where `speaker_label == "me"`
2. Skip turns outside duration bounds (default 1.0s–60.0s)
3. Copy turn WAV files to the dataset directory
4. Write `.lab` files with plain text transcriptions
5. Write `export_summary.json` with per-call stats

## Input/Output

**Input:**
- `output/segmented/{call_id}/dialogue.json`
- `output/segmented/{call_id}/turns/*.wav`
- `output/voiceprint.npy` (for labeling)

**Output:** Updates `dialogue.json` in-place, adding:
```json
{
  "speaker_labels": {"spk_0": "me", "spk_1": "other"},
  "speaker_similarities": {"spk_0": 0.87, "spk_1": 0.42},
  "turns": [
    {"speaker": "spk_0", "speaker_label": "me", ...}
  ]
}
```

With `--prepare`, also outputs:
```
output/fish-speech/
  data/
    me/
      call_recording_turn_003.wav
      call_recording_turn_003.lab
      ...
  export_summary.json
```

## Quality Checks & Interactive Review

After computing similarities, the labeling step flags potential issues:

| Flag | Condition | Meaning |
|------|-----------|---------|
| `low_similarity` | Best match < 0.60 | Voiceprint may not match any speaker |
| `ambiguous_match` | Top-two margin < 0.10 | Assignment isn't confident |
| `insufficient_audio:{spk}` | Speaker has 1-2 usable turns | Embedding may be unreliable |

When `low_similarity` or `ambiguous_match` is flagged, the CLI pauses and prompts the user to manually select which speaker is "me" — showing each speaker's similarity score and sample text. The user can also skip the call entirely (no dialogue.json update). Use `--auto-skip` to skip all low-confidence calls without prompting.

Pipeline is split into three functions: `analyze_speakers()` (compute similarities + flags, no side effects), `apply_labels()` (write dialogue.json with the chosen speaker), and `prepare_dataset()` (export .wav + .lab pairs). The CLI orchestrates the interactive logic between them.

Thresholds are module-level constants (`MIN_SIMILARITY`, `MIN_MARGIN`, `MIN_USABLE_TURNS`, `MIN_EXPORT_DURATION`, `MAX_EXPORT_DURATION`).

## Key Implementation Details
- Uses resemblyzer `VoiceEncoder` (singleton, loaded once)
- Audio is preprocessed via `preprocess_wav()` before embedding extraction
- Minimum 1600 samples required for a meaningful embedding (skips shorter clips)
- Cosine similarity: `dot(ref, emb) / (norm(ref) * norm(emb))`
- The `run.py` orchestrator prompts the user interactively to select their speaker during first enrollment (shows sample text and audio paths)
- When run via `run.py`, `--prepare` is passed automatically

## Dependencies
`resemblyzer`, `soundfile`, `numpy`, `shutil`
