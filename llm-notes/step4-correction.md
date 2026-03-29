# Step 4: Correction (Claude-based Speaker Correction)

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

## Key Implementation Details

- Response parsing finds JSON array boundaries (`[` to `]`) in Claude's response
- Warns if number of assignments doesn't match number of turns (uses original labels for missing)
- Tracks and logs every speaker change with reasoning
- Output format is compatible with both the diarized and translated formats (has turns[] with speaker, start, end, text fields)

