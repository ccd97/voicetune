# Scripts (Out-of-Pipeline Utilities)

Standalone helpers that sit alongside the numbered pipeline stages. None of them produce artifacts consumed by a later step — they exist for stats and selective re-runs.

| Script | Purpose |
|--------|---------|
| `voicetune/scripts/filter_stats.py` | Compact per-stage dashboard across `output/` |
| `voicetune/scripts/invalidate.py` | SQL-driven selective re-run |

## filter_stats.py — Pipeline dashboard

**Run:** `python voicetune/scripts/filter_stats.py [--run-dir ./output] [--stages preprocess,diarize,validation,segment,filter,label,finetune]`

Walks each stage's output directory under `--run-dir` and prints a boxed summary per stage: file count, accept/reject split, total turns, total duration, total audio bytes, text chars, per-language duration breakdown, and stage-specific extras (e.g. `label` surfaces `me turns`, `other turns`, and `label_quality_flags` counters; `finetune` reads `voxcpm/data/{train,val}.jsonl` and falls back to counting WAVs under `me/` if the manifests are missing).

Purely read-only; no side effects. Use after any stage to sanity-check volumes and language mix before moving on.

## invalidate.py — SQL-driven re-run

**Run:**
```bash
python voicetune/scripts/invalidate.py --where "language LIKE 'hi%'" --steps 2-4 --execute
python voicetune/scripts/invalidate.py --where "call_id IN (SELECT call_id FROM turn_issues WHERE issue='garbled_transcript')" --execute
python voicetune/scripts/invalidate.py --schema
```

Builds an in-memory SQLite database by scanning `output/validated/*_validated.json` + `output/preprocessed/*/metadata.json`, then selects `call_id`s via a `--where` clause (against the `calls` table) or a full `--sql` query. Schema: `calls`, `reject_reasons`, `turns`, `turn_issues` (print via `--schema`).

Deletes matched calls' intermediate artifacts for the steps given in `--steps` (2=diarize, 3=scrub, 4=validation; range `2-4` or list `2,4`).

Always a dry run unless `--execute` is passed. `--list` prints matched call IDs (with per-call turn-issue counts) before committing.

After invalidating, just re-run the pipeline as usual (`python -m voicetune.run ...`) — each stage skips calls whose output files still exist, so it will only regenerate the artifacts that were deleted and leave everything else untouched.
