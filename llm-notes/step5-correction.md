# Step 5: Correction (Claude-based Speaker Correction)

## Purpose

Correct speaker diarization errors using Claude's understanding of conversational context. AWS Transcribe often misattributes turns — Claude uses dialogue flow, names, and context to fix these errors.

## Module

`voicetune/correction/` — run via `python -m voicetune.correction`

## CLI Args


| Flag           | Default               | Description                             |
| -------------- | --------------------- | --------------------------------------- |
| `--input-dir`  | `./output/translated` | Directory with translated JSON files    |
| `--output-dir` | `./output/diarized`   | Output directory (overwrites diarized!) |


## What It Does

1. **Read** each `*_translated.json` from the translated output
2. **Build a prompt** showing Claude the full transcript with turn indices, speaker labels, and timestamps
3. **Ask Claude** to review and correct speaker assignments based on:
  - Conversational flow (who responds to whom)
  - Names mentioned (people referring to each other)
  - Consistency of speaking style
  - Turn-taking patterns
4. **Parse** Claude's JSON array response mapping each turn index to corrected speaker + optional name
5. **Apply corrections** and write output in the same format as diarized JSON (compatible with segment step)
6. **Log changes** with reasoning for each reassignment

## Input/Output

**Input:** `output/translated/{call_id}_translated.json`

**Output:**

```json
// output/diarized/{call_id}_corrected.json
{
  "call_id": "call_recording",
  "mode": "aws",
  "language": "hi-IN",
  "speaker_names": {"spk_0": "Amit", "spk_1": "Customer"},
  "turns": [
    {"speaker": "spk_0", "start": 0.0, "end": 3.2, "text": "...", "text_en": "..."}
  ]
}
```

## Pipeline Integration

- Output goes to `output/diarized/` as `*_corrected.json`
- The `run.py` orchestrator copies `*_corrected.json` over `*_diarized.json` so the segment step picks up corrected labels
- Uses `text_en` (English translation) in the prompt so Claude can reason about non-English calls

## Claude Prompt Strategy

- Presents transcript as numbered lines: `[index] speaker (start - end): text`
- Asks for JSON array with `{index, speaker, name, reasoning}` per turn
- Rules: keep original labels where correct, identify speakers by name when possible, reflect actual number of participants
- Model: `CORRECTION_MODEL` env var, defaults to Claude Haiku 4.5
- `max_tokens=8192`, timeout 120s

## Environment Variables

Same as translate step: `ANTHROPIC_BEDROCK_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `CORRECTION_MODEL`, `NODE_EXTRA_CA_CERTS`

## Dependencies

`httpx`, `python-dotenv`

## Quality Gate

Claude returns a confidence score (0.0–1.0) and an `issues` array alongside each batch. The response format is: `{"confidence": 0.85, "issues": [], "assignments": [...]}`.

### LLM-reported issues (`RejectReason` enum)

| Value | Meaning |
|-------|---------|
| `improper_diarization` | Speaker boundaries clearly wrong (mid-sentence splits, misattributed overlap) |
| `incorrect_speaker_assignment` | Speakers systematically swapped or confused throughout |
| `incorrect_speaker_count` | Speaker count doesn't match reality (one person split, or two merged) |
| `garbled_transcript` | Mostly unintelligible ASR output |
| `nonsensical_conversation` | No coherent conversation flow even after correction |
| `language_mismatch` | Transcript language doesn't match actual spoken language |

### Deterministic checks (not in the enum)

- Confidence < 0.60 (min across batches)
- Fewer than 4 turns
- Fewer than 2 speakers (mono-speaker)
- More than 4 speakers

A conversation is rejected if it has any LLM-reported issues, fails confidence, or fails any deterministic check. Rejected calls produce no `_corrected.json`; the CLI writes a `rejected.json` manifest with call IDs, reasons, confidence, and counts. The re-run script (`scripts/rerun_mlx_rejected.py`) can filter by `--reasons` using the enum values.

## Key Implementation Details

- Response parsing finds JSON array boundaries (`[` to `]`) in Claude's response
- Warns if number of assignments doesn't match number of turns (uses original labels for missing)
- Per-turn changes logged at DEBUG level; summary at INFO
- Output format is compatible with both the diarized and translated formats (has turns[] with speaker, start, end, text fields)
- **Batching:** Transcripts are processed in batches of 80 turns (BATCH_SIZE) to stay within the `max_tokens=8192` response limit. Each batch after the first includes the last 10 corrected turns (CONTEXT_OVERLAP) as read-only context so Claude maintains speaker consistency across batch boundaries. Short calls (<= 80 turns) go through in a single request.

