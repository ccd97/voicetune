# Step 6: Label (Speaker Identification)

## Purpose
Label speakers as "me" vs "other" using a voiceprint embedding. Two-phase process: first enroll (create voiceprint from a reference call), then label all calls.

## Module
`voicetune/label/` — run via `python -m voicetune.label {enroll|label}`

## CLI Args

### Enroll subcommand
| Flag | Default | Description |
|------|---------|-------------|
| `--call-id` | (required) | Reference call to use for enrollment |
| `--speaker` | (required) | Your speaker label in that call (e.g. `spk_0`) |
| `--segmented-dir` | `./output/segmented` | Segmented output directory |
| `--voiceprint` | `./output/voiceprint.npy` | Where to save voiceprint |

### Label subcommand
| Flag | Default | Description |
|------|---------|-------------|
| `--segmented-dir` | `./output/segmented` | Segmented output directory |
| `--voiceprint` | `./output/voiceprint.npy` | Path to voiceprint file |
| `--call-id` | all calls | Label a specific call only |

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

## Key Implementation Details
- Uses resemblyzer `VoiceEncoder` (singleton, loaded once)
- Audio is preprocessed via `preprocess_wav()` before embedding extraction
- Minimum 1600 samples required for a meaningful embedding (skips shorter clips)
- Cosine similarity: `dot(ref, emb) / (norm(ref) * norm(emb))`
- The `run.py` orchestrator prompts the user interactively to select their speaker during first enrollment (shows sample text and audio paths)

## Dependencies
`resemblyzer`, `soundfile`, `numpy`
