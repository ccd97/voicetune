# Step 3: Scrub (PII Redaction via Local LLM)

## Purpose

Detect and discard conversation turns containing sensitive data before transcripts are sent to an LLM for validation. Runs entirely on a local LLM via llama-cpp-python so no PII leaves the machine.

## Module

`voicetune/stages/scrub/` — run via `python -m voicetune.stages.scrub`

## CLI Args

| Flag           | Default             | Description                        |
| -------------- | ------------------- | ---------------------------------- |
| `--run-dir`    | `./output`          | Base output directory              |
| `--input-dir`  | `<run-dir>/diarized` | Directory with diarized JSON files |
| `--output-dir` | `<run-dir>/scrubbed` | Where to write scrubbed JSON       |

## What It Does

Reads each `*_diarized.json`, batch-classifies turns through the local LLM (20 per call, `temperature=0`, binary `CLEAN`/`SENSITIVE` verdict per line), and writes an output with only clean turns. Entire turns are dropped rather than redacted — partial/broken text would corrupt downstream stages.

## Configurable Sensitive Data Types

The `SensitiveDataType` enum at the top of `pipeline.py` defines what gets flagged:

- Person name
- Phone number
- Account or ID number
- Physical address
- Email address
- Date of birth
- Financial info (card number, bank account, balance)
- Personal info (marital status, health conditions, family details, political opinions, religious beliefs, sexual orientation)
- Confidential business info (internal policies, trade secrets, proprietary data)
- Government ID (SSN, Aadhaar, passport number, driver's license)
- Credentials (passwords, PINs, security questions and answers)
- Real-time location or specific whereabouts

Edit the enum to add or remove categories.

## Input/Output

**Input:** `output/diarized/{call_id}_diarized.json`

**Output:** `output/scrubbed/{call_id}_diarized.json` — the `_diarized.json` suffix is preserved so validation's glob works unchanged; only the input directory differs.

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

## Environment & Dependencies

`LLAMACPP_MODEL_PATH` (required) — path to the GGUF model (text-only, no mmproj). Deps: `llama-cpp-python`, `python-dotenv`.

## Key Implementation Details

- Model loaded once and cached across files in the run.
- Unparseable response lines fall back to `CLEAN` (conservative — keeps the turn).
