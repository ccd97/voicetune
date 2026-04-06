"""JSON and JSONL read/write helpers."""

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")

_MISSING = object()


def read_json(path: Path, default: Any = _MISSING) -> Any:
    """Read JSON from path. If `default` is provided and path doesn't exist, return it."""
    if default is not _MISSING and not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def write_json(
    path: Path,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    atomic: bool = False,
) -> None:
    """Write `data` as JSON, creating parent dirs; `atomic=True` writes to a `.tmp` then `os.replace`s."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(tmp, path)
        return
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL from path; blank lines are skipped."""
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: Iterable[dict]) -> None:
    """Write entries as JSONL. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_json(
    text: str,
    *,
    prefer: str = "object",
    strip_fences: bool = True,
) -> Any:
    """Parse JSON from noisy text (e.g. fenced LLM output); `prefer` picks `{..}` vs `[..]` first."""
    text = text.strip()
    if strip_fences and text.startswith("```"):
        text = _FENCE_OPEN_RE.sub("", text)
        text = _FENCE_CLOSE_RE.sub("", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    primary = ("{", "}") if prefer == "object" else ("[", "]")
    secondary = ("[", "]") if prefer == "object" else ("{", "}")

    for open_c, close_c in (primary, secondary):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

    raise ValueError("No JSON found in text")
