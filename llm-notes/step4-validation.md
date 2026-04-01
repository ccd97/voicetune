# Step 4: Validation (LLM-based Speaker Validation)

## Purpose

Validate and fix speaker diarization errors using an LLM's understanding of conversational context. Diarization often misattributes turns — the LLM uses dialogue flow, names, and context to fix these errors. Supports Bedrock (Claude) and llama.cpp backends.

## Module

`voicetune/stages/validation/` — run via `python -m voicetune.stages.validation`

## CLI Args


| Flag           | Default               | Description                             |
| -------------- | --------------------- | --------------------------------------- |
| `--input-dir`  | `./output/diarized`   | Directory with diarized JSON files      |
| `--output-dir` | `./output/validated`  | Output directory for validated files    |
| `--backend`    | `llamacpp`            | Backend: `bedrock` or `llamacpp`        |


## What It Does

1. **Read** each `*_diarized.json` from the diarized (or scrubbed) output
2. **Build a prompt** showing the LLM the full transcript with turn indices, speaker labels, and timestamps
3. **Ask the LLM** to review and correct speaker assignments based on:
  - Conversational flow (who responds to whom)
  - Names mentioned (people referring to each other)
  - Consistency of speaking style
  - Turn-taking patterns
4. **Parse** the JSON array response mapping each turn index to validated speaker + optional name
5. **Apply fixes** and write output in the same format as diarized JSON (compatible with segment step)
6. **Log changes** with reasoning for each reassignment

## Input/Output

**Input:** `output/diarized/{call_id}_diarized.json` (or `output/scrubbed/` if the scrub step ran)

**Output:** `output/validated/{call_id}_validated.json` — one file per recording, always.

Accepted:
```json
{
  "call_id": "call_recording",
  "mode": "aws",
  "language": "hi-IN",
  "speaker_names": {"spk_0": "Amit", "spk_1": "Customer"},
  "validation_confidence": 0.85,
  "turns": [
    {"speaker": "spk_0", "start": 0.0, "end": 3.2, "text": "..."}
  ]
}
```

Rejected (file-level issue — per-turn issues preserved but don't cause rejection):
```json
{
  "call_id": "call_recording",
  "mode": "aws",
  "language": "hi-IN",
  "rejected": true,
  "reject_reasons": ["incorrect_speaker_count"],
  "validation_confidence": 0.40,
  "turns": [
    {"speaker": "spk_0", "start": 0.0, "end": 3.2, "text": "...", "issues": ["garbled_transcript"]},
    {"speaker": "spk_1", "start": 3.5, "end": 6.0, "text": "..."}
  ]
}
```

Non-rejected with per-turn issues (filter step will remove flagged turns):
```json
{
  "call_id": "call_recording_2",
  "mode": "aws",
  "language": "en-US",
  "validation_confidence": 0.85,
  "turns": [
    {"speaker": "spk_0", "start": 0.0, "end": 0.3, "text": "Hi", "issues": ["too_short"]},
    {"speaker": "spk_1", "start": 0.5, "end": 3.2, "text": "Hello, how can I help?"}
  ]
}
```

## Pipeline Integration

- Output goes to `output/validated/` as `*_validated.json`
- Step 5 (filter) reads from `output/validated/`, removing flagged turns and rejecting files with file-level issues

## Prompt Strategy

- Presents transcript as numbered lines: `[index] speaker (start - end): text`
- Asks for JSON array with `{index, speaker, name, reasoning}` per turn
- Rules: keep original labels where correct, identify speakers by name when possible, reflect actual number of participants
- Same prompt template (`validation.j2`) used by both backends
- `max_tokens=8192`

### Bedrock backend

- Model: `CORRECTION_MODEL` env var, defaults to Claude Haiku 4.5
- Uses httpx with 120s timeout

### llama.cpp backend

- Uses llama-cpp-python with `create_chat_completion` (text-only)
- Model cached across files via module-level `_get_llm()`
- `n_ctx=8192` to match the response token budget
- `temperature=0` for deterministic output

## Environment Variables


| Var                          | Backend   | Description                     |
| ---------------------------- | --------- | ------------------------------- |
| `ANTHROPIC_BEDROCK_BASE_URL` | bedrock   | Bedrock gateway URL             |
| `ANTHROPIC_AUTH_TOKEN`       | bedrock   | Auth bearer token               |
| `VALIDATION_MODEL`           | bedrock   | Claude model ID (optional)      |
| `NODE_EXTRA_CA_CERTS`        | bedrock   | Custom CA certs path (optional) |
| `LLAMACPP_MODEL_PATH`        | llamacpp  | Path to GGUF model file         |

## Dependencies

`httpx`, `python-dotenv`, `llama-cpp-python` (for llamacpp backend)

## Quality Gate

The LLM returns a confidence score (0.0–1.0) and an `issues` array alongside each batch. The response format is: `{"confidence": 0.85, "issues": [], "assignments": [...]}`.

### Per-turn issues (on each assignment)

| Value | Meaning |
|-------|---------|
| `improper_diarization` | This turn's speaker boundary is wrong (split mid-sentence, overlap misattributed) |
| `incorrect_speaker_assignment` | This turn is attributed to the wrong speaker |
| `garbled_transcript` | This turn's text is unintelligible or full of ASR errors |
| `too_short` | Turn duration < 0.5s (timing check, not from LLM) |
| `too_long` | Turn duration > 60s (timing check, not from LLM) |

These appear in each turn's `"issues"` array in the output. Per-turn issues do **not** cause file rejection — the filter step (step 5) handles turn removal based on these codes.

### File-level issues (top-level "issues")

| Value | Meaning |
|-------|---------|
| `incorrect_speaker_count` | Speaker count doesn't match reality (one person split, or two merged) |
| `nonsensical_conversation` | No coherent conversation flow even after correction |
| `language_mismatch` | Transcript language doesn't match actual spoken language |

### Pre-LLM checks

- Language not in `ALLOWED_LANGS` env var
- Mono speaker (`num_speakers < 2`)

### Confidence check

- Confidence < 0.60 (min across batches)

A conversation is rejected only for file-level issues, pre-LLM check failures, or low confidence. Per-turn issues are preserved on individual turns but do **not** trigger file rejection — the filter step removes those turns instead. Every call produces a `_validated.json` — rejected ones have `"rejected": true`, `"reject_reasons"`, and validated turns with per-turn `"issues"`. Files that already exist in the output directory are skipped, so interrupted runs can be resumed. The re-run script (`scripts/rerun_rejected.py`) globs `output/validated/` and filters by `"rejected": true`; use `--reasons` to filter by specific values.

## Key Implementation Details

- Response parsing finds JSON object/array boundaries in the LLM response
- Warns if number of assignments doesn't match number of turns (uses original labels for missing)
- Per-turn changes logged at DEBUG level; summary at INFO
- Output format is compatible with the diarized format (has turns[] with speaker, start, end, text fields)
- **Batching:** Transcripts are processed in batches of 80 turns (BATCH_SIZE) to stay within the `max_tokens=8192` response limit. Each batch after the first includes the last 10 validated turns (CONTEXT_OVERLAP) as read-only context so the LLM maintains speaker consistency across batch boundaries. Short calls (<= 80 turns) go through in a single request.
