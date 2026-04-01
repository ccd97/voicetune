# Step 3: Scrub (PII Redaction via Local LLM)

## Purpose

Detect and discard conversation turns containing sensitive data before transcripts are sent to an LLM for validation. Runs entirely on a local LLM via llama-cpp-python so no PII leaves the machine.

## Module

`voicetune/scrub/` — run via `python -m voicetune.scrub`

## CLI Args

| Flag           | Default             | Description                        |
| -------------- | ------------------- | ---------------------------------- |
| `--input-dir`  | `./output/diarized` | Directory with diarized JSON files |
| `--output-dir` | `./output/scrubbed` | Where to write scrubbed JSON       |

## What It Does

1. **Read** each `*_diarized.json` from the diarized output
2. **Batch classify** turns (batches of 20) by sending them to the local LLM
3. **Discard** any turn the LLM flags as containing sensitive data
4. **Write** output with only clean turns (same JSON structure, same filename)

Entire turns are discarded rather than redacted — this avoids partial/broken text flowing into downstream steps.

## Configurable Sensitive Data Types

The `SensitiveDataType` enum at the top of `pipeline.py` defines what gets flagged:

- Person name
- Phone number
- Account or ID number
- Physical address
- Email address
- Date of birth
- Financial info (card number, bank account, balance)

Edit the enum to add or remove categories.

## Input/Output

**Input:** `output/diarized/{call_id}_diarized.json`

**Output:** `output/scrubbed/{call_id}_diarized.json` (same filename so downstream steps' globs work unchanged)

```json
{
  "call_id": "call_recording",
  "mode": "mlx",
  "language": "hi-IN",
  "turns": [
    {"speaker": "spk_0", "start": 3.5, "end": 8.1, "text": "..."}
  ]
}
```

Turns containing sensitive data are absent from the output. Original diarized files are untouched.

## llama.cpp Integration

Uses llama-cpp-python to load the GGUF model directly (text-only, no mmproj needed). Model is cached across files within a single run. Sends turns with `temperature: 0` for deterministic classification. Response format is one line per turn: `1. CLEAN` or `2. SENSITIVE`.

## Environment Variables

| Var                  | Default | Description                  |
| -------------------- | ------- | ---------------------------- |
| `LLAMACPP_MODEL_PATH` | (required) | Path to GGUF model file |

## Dependencies

`llama-cpp-python`, `python-dotenv` (both already in the project)

## Key Implementation Details

- Batch size: 20 turns per LLM call
- Model loaded once and cached for the entire run
- Prompt asks for binary CLEAN/SENSITIVE verdict per turn — no explanation, easy to parse
- Falls back to CLEAN if a turn's response line can't be parsed (conservative: keeps the turn)
- Output preserves the `_diarized.json` suffix so downstream steps' globs work without modification — only the input directory changes (`output/scrubbed/` instead of `output/diarized/`)
