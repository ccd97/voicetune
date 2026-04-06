# Scripts (Out-of-Pipeline Utilities)

Standalone helpers that sit alongside the numbered pipeline stages. None of them produce artifacts consumed by a later step — they exist for stats and selective re-runs.

| Script | Purpose |
|--------|---------|
| `voicetune/scripts/filter_stats.py` | Compact per-stage dashboard across `output/` |
| `voicetune/scripts/rerun.py` | SQL-driven selective re-run / turn repair |

## filter_stats.py — Pipeline dashboard

**Run:** `python voicetune/scripts/filter_stats.py [--run-dir ./output] [--stages preprocess,diarize,validation,segment,filter,label,finetune]`

Walks each stage's output directory under `--run-dir` and prints a boxed summary per stage: file count, accept/reject split, total turns, total duration, total audio bytes, text chars, per-language duration breakdown, and stage-specific extras (e.g. `label` surfaces `me turns`, `other turns`, and `label_quality_flags` counters; `finetune` reads `voxcpm/data/{train,val}.jsonl` and falls back to counting WAVs under `me/` if the manifests are missing).

Purely read-only; no side effects. Use after any stage to sanity-check volumes and language mix before moving on.

## rerun.py — SQL-driven re-run and turn repair

**Run:**
```bash
python voicetune/scripts/rerun.py --where "language LIKE 'hi%'" --steps 2-4 --execute
python voicetune/scripts/rerun.py --where "call_id IN (SELECT call_id FROM turn_issues WHERE issue='garbled_transcript')" --fix-turns --execute
python voicetune/scripts/rerun.py --schema
```

Builds an in-memory SQLite database by scanning `output/validated/*_validated.json` + `output/preprocessed/*/metadata.json`, then selects `call_id`s via a `--where` clause (against the `calls` table) or a full `--sql` query. Schema: `calls`, `reject_reasons`, `turns`, `turn_issues` (print via `--schema`).

Two modes:

1. **Delete outputs for selective re-run** (default) — for the matched calls, deletes intermediate artifacts for the steps specified in `--steps` (2=diarize, 3=scrub, 4=validation; range `2-4` or list `2,4`). Next time you run the pipeline those stages regenerate only those calls.
2. **Fix turns in place** (`--fix-turns`) — for turns with `issues` in matched calls, clips the turn's audio out of `output/preprocessed/{call}/full_normalized.wav`, sends it to Claude (Vertex AI) with ±3 turns of context, and rewrites `output/validated/{call}_validated.json` with the corrected transcript. Skips a turn if its neighbors are also flagged (context would be garbage). Model defaults to `claude-haiku-4-5`; override with `FIX_TURNS_MODEL`. Requires `GCP_PROJECT_ID` in `.env` (and optionally `VERTEX_LOCATION`, default `us-east5`). Runs up to 10 calls concurrently.

Always a dry run unless `--execute` is passed. Use `--list` to print matched call IDs (with per-call turn-issue counts) before committing.
