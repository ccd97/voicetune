"""Generic CLI scaffolding: bootstrap, argparse helpers, stage-loop runner, range parsing."""

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)


def bootstrap(*, dotenv: bool = False) -> None:
    """Standard CLI startup: configure logging, optionally load `.env`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if dotenv:
        from dotenv import load_dotenv
        load_dotenv()


def resolve_stage_paths(args: argparse.Namespace, **subpaths: str) -> None:
    """Fill `None`-valued path args on `args` with `args.run_dir / subpath`."""
    run_dir: Path = args.run_dir
    for attr, subpath in subpaths.items():
        if getattr(args, attr, None) is None:
            setattr(args, attr, run_dir / subpath)


def run_stage_loop(
    items: Iterable,
    process_fn: Callable[[Any], Any],
    *,
    done_check: Callable[[Any], bool] | None = None,
    label: str = "file",
    name_fn: Callable[[Any], str] = str,
) -> dict:
    """Apply `process_fn` to each item (skipping when `done_check` is true); returns {succeeded, skipped, failed}."""
    succeeded = 0
    skipped = 0
    failed: list[str] = []
    for item in items:
        if done_check is not None and done_check(item):
            skipped += 1
            continue
        try:
            process_fn(item)
            succeeded += 1
        except Exception:
            name = name_fn(item)
            log.exception(f"Failed to process {name}")
            failed.append(name)
    if skipped:
        log.info(f"Skipped {skipped} already-processed {label}(s)")
    log.info(f"Summary: {succeeded} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed: {', '.join(failed)}")
    return {"succeeded": succeeded, "skipped": skipped, "failed": failed}


def parse_int_ranges(spec: str, lo: int, hi: int) -> set[int]:
    """Parse a comma/range spec like `"1,3-5,8"` into a set of ints within [lo, hi]."""
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
            if a < lo or b > hi or a > b:
                raise ValueError(f"Invalid range {a}-{b} (must be within {lo}-{hi})")
            selected.update(range(a, b + 1))
        elif part.isdigit():
            n = int(part)
            if n < lo or n > hi:
                raise ValueError(f"Invalid value {n} (must be within {lo}-{hi})")
            selected.add(n)
        else:
            raise ValueError(f"Unknown token: {part!r}")
    return selected
