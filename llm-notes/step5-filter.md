# Step 5: Filter (Turn/File Filtering)

## Purpose

Remove individual turns flagged with per-turn issues by validation, and reject entire files that have file-level rejection codes. This step sits between validation and segment, ensuring only clean turns reach the audio segmentation stage.

## Module

`voicetune/stages/filter/` — run via `python -m voicetune.stages.filter`

## CLI Args

| Flag           | Default              | Description                              |
| -------------- | -------------------- | ---------------------------------------- |
| `--input-dir`  | `./output/validated` | Directory with validated JSON files      |
| `--output-dir` | `./output/filtered`  | Output directory for filtered files      |

## What It Does

1. **Read** each `*_validated.json` from validation output
2. **Reject entire file** if it has file-level rejection codes (`rejected: true` with file-level reasons)
3. **Remove individual turns** that have per-turn issues in their `issues` array
4. **Write** cleaned files to `output/filtered/` with filtered turns removed

### File-level rejection codes (entire file rejected)

| Code | Origin |
|------|--------|
| `incorrect_speaker_count` | LLM file-level issue |
| `nonsensical_conversation` | LLM file-level issue |
| `language_mismatch` | LLM file-level issue |
| `language_not_allowed` | Pre-LLM check |
| `mono_speaker` | Pre-LLM check |
| `low_confidence` | Confidence gate |

### Per-turn removal codes (turn dropped, rest of conversation kept)

| Code | Origin |
|------|--------|
| `garbled_transcript` | LLM per-turn issue |
| `improper_diarization` | LLM per-turn issue |
| `incorrect_speaker_assignment` | LLM per-turn issue |
| `too_short` | Timing check (duration < 0.5s) |
| `too_long` | Timing check (duration > 60s) |

## Input/Output

**Input:** `output/validated/{call_id}_validated.json`

**Output:** `output/filtered/{call_id}_filtered.json`

```json
{
  "call_id": "call_recording",
  "mode": "aws",
  "language": "en-US",
  "validation_confidence": 0.85,
  "original_turn_count": 25,
  "filtered_turn_count": 22,
  "turns": [
    {"speaker": "spk_0", "start": 0.5, "end": 3.2, "text": "Hello, how can I help?"}
  ]
}
```

Files where all turns are removed are treated as rejected (not output). Files already rejected by validation (file-level codes) are skipped.

## Key Implementation Details

- Imports `FILE_ISSUES`, `TURN_ISSUES`, and `RejectReason` from `voicetune.stages.validation.pipeline` to stay in sync with validation's code definitions
- Already-filtered files in the output directory are skipped (resume-safe)
- Adds `original_turn_count` and `filtered_turn_count` metadata to the output
- Strips `rejected` and `reject_reasons` keys from output files (only clean files are written)

## Dependencies

None beyond stdlib (reads/writes JSON).
