# voicetune

Audio processing pipeline for voice fine-tuning. Run the full pipeline via `python -m voicetune` or individual steps via `python -m voicetune.<step>`.

## First steps

**Before writing any code, read the `llm-notes/` directory.** These files document every pipeline step — purpose, CLI args, I/O format, dependencies, and implementation details. Read the relevant `llm-notes/step*.md` file(s) for any step you're about to touch. This is mandatory context, not optional background.

## Code style rules

- **No AI slop.** Do not add unnecessary comments, docstrings, or type annotations to code you didn't change. Do not add defensive try/catch blocks or null checks on trusted internal code paths. Do not cast to `any` to bypass type issues. Do not wrap simple logic in helper functions "for readability."
- **No redundant comments.** If the code says `write_wav(path, audio)`, do not add `# Write the WAV file`. Only comment when the *why* is non-obvious.
- **No numbered step comments** like `# 1. Do X`, `# 2. Do Y` unless the ordering is surprising or non-trivial.
- **Prefer early returns** over deep nesting.
- **Shared utilities live in `voicetune/common.py`.** Before adding a helper to a module, check if it already exists there or belongs there.
- **No module-level side effects** in library code. `load_dotenv()`, `setup_logging()`, etc. belong inside `main()` or CLI entry points only.
- **Remove unused imports.** Do not leave imports "for later."
- **Keep behavior unchanged** unless fixing a clear bug. Prefer minimal, focused edits over broad rewrites.