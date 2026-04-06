# Step 4: Validation (LLM-based Speaker Validation)

## Purpose

Validate and fix speaker diarization errors using an LLM's understanding of conversational context. Diarization often misattributes turns — the LLM uses dialogue flow, names, and context to fix these errors. Runs on a local llama.cpp model.

## Module

`voicetune/stages/validation/` — run via `python -m voicetune.stages.validation`

## CLI Args

| Flag           | Default               | Description                             |
| -------------- | --------------------- | --------------------------------------- |
| `--run-dir`    | `./output`            | Base output directory                   |
| `--input-dir`  | `<run-dir>/diarized`  | Directory with diarized JSON. When invoked by `voicetune.run`, the orchestrator overrides this to `<run-dir>/scrubbed` if scrub produced output. |
| `--output-dir` | `<run-dir>/validated` | Output directory for validated files    |

## What It Does

Reads each `*_diarized.json`, sends the transcript to the LLM as numbered lines (`[index] speaker (start - end): text`), and asks it to re-assign each turn's speaker based on dialogue flow, names used, and turn-taking. The LLM responds with `{"confidence": 0.85, "issues": [], "assignments": [...]}`; the pipeline parses it, applies the fixes, tags per-turn/file-level issues, and writes a `_validated.json`. Prompt template: `validation.j2`. Rules given to the LLM: keep original labels where correct, use names when possible, reflect the actual speaker count.

## Input/Output

**Input:** `output/diarized/{call_id}_diarized.json` (or `output/scrubbed/` if the scrub step ran).

**Output:** `output/validated/{call_id}_validated.json` — one file per recording, always. Already-present files are skipped so interrupted runs resume.

```json
// Accepted. If rejected, adds: "rejected": true, "reject_reasons": [...].
// Per-turn issues attach directly to the offending turn and are passed through
// unchanged — the filter step (step 6) is what actually drops those turns.
{
  "call_id": "call_recording",
  "mode": "aws",
  "language": "hi-IN",
  "speaker_names": {"spk_0": "Amit", "spk_1": "Customer"},
  "validation_confidence": 0.85,
  "turns": [
    {"speaker": "spk_0", "start": 0.0, "end": 3.2, "text": "Hi"},
    {"speaker": "spk_1", "start": 3.5, "end": 6.0, "text": "...", "issues": ["too_short"]}
  ]
}
```

## Issue Codes

Per-turn (attached to the turn; consumed by filter):

| Code | Meaning |
|------|---------|
| `improper_diarization` | Speaker boundary wrong (split mid-sentence, overlap misattributed) |
| `incorrect_speaker_assignment` | Turn attributed to the wrong speaker |
| `garbled_transcript` | Text unintelligible / ASR garbage |
| `too_short` | Duration < 0.5 s (timing check, not from LLM) |
| `too_long` | Duration > 60 s (timing check, not from LLM) |

File-level (trigger rejection):

| Code | Meaning |
|------|---------|
| `incorrect_speaker_count` | Speaker count doesn't match reality (split or merged) |
| `nonsensical_conversation` | No coherent flow even after correction |
| `language_mismatch` | Transcript language doesn't match spoken language |
| `language_not_allowed` | Pre-LLM: language not in `ALLOWED_LANGS` env var |
| `mono_speaker` | Pre-LLM: `num_speakers < 2` |
| `low_confidence` | Min batch confidence < 0.60 |

To re-run a subset (e.g. only rejected files, or those with a specific reason) use `voicetune/scripts/rerun.py` — loads validated JSON into an in-memory SQLite DB and filters via `--where` / `--sql` / `--ids`. See `scripts.md`.

## Backend

llama-cpp-python `create_chat_completion` (text-only), `n_ctx=8192`, `temperature=0`, model cached across files via `load_llamacpp()`. Requires `LLAMACPP_MODEL_PATH`.

## Key Implementation Details

- Response parser locates JSON object/array boundaries in free-form LLM output.
- Warns and falls back to original labels if assignment count doesn't match turn count.
- Per-turn changes logged at DEBUG, summary at INFO.
- Batching: 80 turns per request (`BATCH_SIZE`) to fit under `max_tokens=8192`. Each batch after the first prefixes the last 10 validated turns (`CONTEXT_OVERLAP`) as read-only context so speaker labels stay consistent across batch boundaries. Calls ≤80 turns go in one shot.
