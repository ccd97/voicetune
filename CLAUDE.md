# voicetune

Audio processing pipeline for voice fine-tuning. Run the full pipeline via `python -m voicetune` or individual steps via `python -m voicetune.<step>`.

## First steps

**Before writing any code, read the `llm-notes/` directory.** These files document every pipeline step — purpose, CLI args, I/O format, dependencies, and implementation details. Read the relevant `llm-notes/step*.md` file(s) for any step you're about to touch. This is mandatory context, not optional background.

## Code style rules

- **No AI slop.** Do not add unnecessary comments, docstrings, or type annotations to code you didn't change. Do not add defensive try/catch blocks or null checks on trusted internal code paths. Do not cast to `any` to bypass type issues. Do not wrap simple logic in helper functions "for readability."
- **No redundant comments.** If the code says `write_wav(path, audio)`, do not add `# Write the WAV file`. Only comment when the *why* is non-obvious.
- **No numbered step comments** like `# 1. Do X`, `# 2. Do Y` unless the ordering is surprising or non-trivial.
- **Prefer early returns** over deep nesting.
- **Shared utilities live in `voicetune/common/`.** Before adding a helper to a module, check if it already exists in the appropriate submodule (`audio`, `cli`, `dialogue`, `gemma_llm`, `json_io`) or belongs there. Stage-specific path helpers live in `voicetune/stages/<stage>/paths.py`.
- **No module-level side effects** in library code. `load_dotenv()`, `setup_logging()`, etc. belong inside `main()` or CLI entry points only.
- **Remove unused imports.** Do not leave imports "for later."
- **Keep behavior unchanged** unless fixing a clear bug. Prefer minimal, focused edits over broad rewrites.

## End of task checklist

At the end of any task:

1. Verify the changes follow the code style rules above (no AI slop).
2. Verify no bugs are introduced in the changed files/logic. Do a dry run if needed.
3. Update `llm-notes/` if necessary. Keep notes concise and factual — no AI slop, only necessary info.
4. If the changes are large, suggest the user commit the code.
