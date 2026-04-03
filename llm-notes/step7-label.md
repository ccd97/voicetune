# Step 7: Label (Speaker Identification)

## Purpose
Label speakers as "me" vs "other" using a voiceprint embedding. Two-phase process: first enroll (create voiceprint from a reference call), then label all calls.

## Module
`voicetune/stages/label/` — run via `python -m voicetune.stages.label {enroll|label}`

## CLI Args

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `./output/filtered` | Filtered call directories (dialogue.json + turns/*.wav) |

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
| `--output-dir` | `./output/labeled` | Output directory for labeled dialogue.json files |
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
5. Write labeled dialogue.json to `output/labeled/{call_id}/dialogue.json` (does **not** modify segmented files)

## Input/Output

**Input:**
- `output/filtered/{call_id}/dialogue.json`
- `output/filtered/{call_id}/turns/*.wav`
- `output/voiceprint.npy` (for labeling)

**Output:** `output/labeled/{call_id}/dialogue.json` — copy of dialogue with added fields:
```json
{
  "speaker_labels": {"spk_0": "me", "spk_1": "other"},
  "speaker_similarities": {"spk_0": 0.87, "spk_1": 0.42},
  "label_quality_flags": [],
  "turns": [
    {"speaker": "spk_0", "speaker_label": "me", ...}
  ]
}
```

## Quality Checks

After computing similarities, the labeling step records diagnostic flags on the output:

| Flag | Condition | Meaning |
|------|-----------|---------|
| `low_similarity` | Best match < 0.60 | Voiceprint may not match any speaker |
| `ambiguous_match` | Top-two margin < 0.10 | Assignment isn't confident |
| `insufficient_audio:{spk}` | Speaker has 1-2 usable turns | Embedding may be unreliable |

Flags are informational only. The speaker with the highest similarity is always selected as "me"; calls are only skipped when no embedding can be extracted at all (`no_speakers`).

Pipeline is split into two functions: `analyze_speakers()` (compute similarities + flags, no side effects) and `apply_labels()` (writes labeled dialogue.json to output dir).

Thresholds are module-level constants (`MIN_SIMILARITY`, `MIN_MARGIN`, `MIN_USABLE_TURNS`).

## Key Implementation Details
- Uses resemblyzer `VoiceEncoder` (singleton, loaded once)
- Audio is preprocessed via `preprocess_wav()` before embedding extraction
- Minimum 1600 samples required for a meaningful embedding (skips shorter clips)
- Cosine similarity: `dot(ref, emb) / (norm(ref) * norm(emb))`
- The `run.py` orchestrator prompts the user interactively to select their speaker during first enrollment (shows sample text and audio paths)

## Dependencies
`resemblyzer`, `soundfile`, `numpy`
