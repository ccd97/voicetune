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
| `--voiceprint` | `./output/voiceprint.npz` | Where to save voiceprint |

### Label subcommand
| Flag | Default | Description |
|------|---------|-------------|
| `--voiceprint` | `./output/voiceprint.npz` | Path to voiceprint file |
| `--output-dir` | `./output/labeled` | Output directory for labeled dialogue.json files |
| `--call-id` | all calls | Label a specific call only |

## What It Does

### Enrollment Phase
1. Load the dialogue.json for the specified call
2. Find all turns matching the specified speaker label
3. Take up to 10 longest turns for robustness
4. Extract 256-d speaker embeddings using the WeSpeakerResNet34 model bundled inside `pyannote/speaker-diarization-community-1` (subfolder `embedding`) — the same model step 2 diarization used for clustering
5. Average embeddings into a single voiceprint vector
6. Save as `.npz` with `{embedding, model, subfolder}` so stale voiceprints are detected on load

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
- `output/voiceprint.npz` (for labeling)

**Output:** `output/labeled/{call_id}/dialogue.json` — copy of dialogue with added fields:
```json
{
  "speaker_labels": {"spk_0": "me", "spk_1": "other"},
  "speaker_similarities": {"spk_0": 0.87, "spk_1": 0.42},
  "label_quality_flags": [],
  "speaker_embedding_model": "pyannote/speaker-diarization-community-1#embedding",
  "turns": [
    {"speaker": "spk_0", "speaker_label": "me", ...}
  ]
}
```

## Quality Checks

After computing similarities, the labeling step records diagnostic flags on the output:

| Flag | Condition | Meaning |
|------|-----------|---------|
| `low_similarity` | Best match < 0.70 | Voiceprint may not match any speaker |
| `ambiguous_match` | Top-two margin < 0.15 | Assignment isn't confident |
| `insufficient_audio:{spk}` | Speaker has 1-2 usable turns | Embedding may be unreliable |

Flags are informational only. The speaker with the highest similarity is always selected as "me"; calls are only skipped when no embedding can be extracted at all (`no_speakers`).

Pipeline is split into two functions: `analyze_speakers()` (compute similarities + flags, no side effects) and `apply_labels()` (writes labeled dialogue.json to output dir).

Thresholds are module-level constants (`MIN_SIMILARITY`, `MIN_MARGIN`, `MIN_USABLE_TURNS`). These defaults were raised from 0.60/0.10 when we migrated from resemblyzer to pyannote — WeSpeaker cosines concentrate higher and separate better.

## Key Implementation Details
- Uses `pyannote.audio.Model.from_pretrained(EMBEDDING_MODEL, subfolder="embedding")` (singleton, loaded once)
- Model is `WeSpeakerResNet34`, 256-d, 16 kHz input, already cached by step 2's diarization
- Device selection mirrors the diarize backend: CUDA → MPS → CPU
- Audio is fed raw as a `(1, 1, T)` float32 torch tensor (no external VAD pre-trim — WeSpeaker was trained on untrimmed utterances)
- Skips clips shorter than 0.5 s
- Cosine similarity: `dot(ref, emb) / (norm(ref) * norm(emb))`
- `_load_voiceprint()` refuses `.npy` files and mismatched model names, so re-enrollment is forced when the encoder changes
- `apply_labels()` stamps `speaker_embedding_model` into each labeled `dialogue.json`
- The `run.py` orchestrator prompts the user interactively to select their speaker during first enrollment (shows sample text and audio paths)

## Migration Notes
- Re-enroll once after upgrading: `python -m voicetune.stages.label enroll --call-id <ref> --speaker <spk>`. The new voiceprint lands at `output/voiceprint.npz`; old `.npy` files are ignored.
- `output/labeled/` from the previous resemblyzer run is **not** auto-invalidated — delete it (or pass `--output-dir` elsewhere) if you want every call re-labeled with the new encoder. New per-call outputs carry `speaker_embedding_model` so future code can detect staleness.
- The evaluate-finetune-checkpoints skill still scores via resemblyzer off `output/voiceprint.npy`; that path is independent of this stage.

## Dependencies
`pyannote.audio>=4.0` (local extra), `soundfile`, `numpy`, `torch`. Requires `HF_TOKEN` with pyannote model access.
