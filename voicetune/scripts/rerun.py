"""Select recordings via SQL and either delete outputs for re-run or fix flagged turns.

Usage:
  python voicetune/scripts/rerun.py --where "language LIKE 'hi%'" --steps 2-4 --execute
  python voicetune/scripts/rerun.py --where "call_id LIKE '%Blanchi%' AND num_speakers = 1"
  python voicetune/scripts/rerun.py --where "call_id IN (SELECT call_id FROM turn_issues WHERE issue='garbled_transcript')" --fix-turns --execute
  python voicetune/scripts/rerun.py --schema
"""

import argparse
import logging
import os
import sqlite3
from pathlib import Path

from voicetune.common import (
    bootstrap,
    clip_turn_audio,
    extract_json,
    parse_int_ranges,
    read_json,
    turn_duration,
    unique_speakers,
    write_json,
)

log = logging.getLogger(__name__)

PREPROCESSED_DIR = Path("./output/preprocessed")
DIARIZED_DIR = Path("./output/diarized")
SCRUBBED_DIR = Path("./output/scrubbed")
VALIDATED_DIR = Path("./output/validated")

STALE_MAP = {
    2: lambda cid: DIARIZED_DIR / f"{cid}_diarized.json",
    3: lambda cid: SCRUBBED_DIR / f"{cid}_diarized.json",
    4: lambda cid: VALIDATED_DIR / f"{cid}_validated.json",
}

SCHEMA_SQL = """\
CREATE TABLE calls (
    call_id               TEXT PRIMARY KEY,
    mode                  TEXT,
    language              TEXT,
    num_speakers          INTEGER,
    validation_confidence REAL,
    rejected              BOOLEAN,
    duration              REAL,
    snr_db                REAL
);
CREATE TABLE reject_reasons (
    call_id TEXT,
    reason  TEXT
);
CREATE TABLE turns (
    call_id    TEXT,
    idx        INTEGER,
    speaker    TEXT,
    start_time REAL,
    end_time   REAL,
    text       TEXT
);
CREATE TABLE turn_issues (
    call_id  TEXT,
    turn_idx INTEGER,
    issue    TEXT
);"""


def build_db(validated_dir: Path, preprocessed_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)

    for path in sorted(validated_dir.glob("*_validated.json")):
        data = read_json(path)

        cid = data["call_id"]

        meta_path = preprocessed_dir / cid / "metadata.json"
        duration = None
        snr_db = None
        if meta_path.exists():
            meta = read_json(meta_path)
            duration = meta.get("output_duration_seconds")
            snr_db = meta.get("snr_db")

        conn.execute(
            "INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?,?,?)",
            (cid, data.get("mode"), data.get("language"),
             data.get("num_speakers"), data.get("validation_confidence"),
             data.get("rejected", False), duration, snr_db),
        )

        for reason in data.get("reject_reasons", []):
            conn.execute("INSERT INTO reject_reasons VALUES (?,?)", (cid, reason))

        for i, turn in enumerate(data.get("turns", [])):
            conn.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,?)",
                (cid, i, turn.get("speaker"), turn.get("start"),
                 turn.get("end"), turn.get("text")),
            )
            for issue in turn.get("issues", []):
                conn.execute("INSERT INTO turn_issues VALUES (?,?,?)", (cid, i, issue))

    conn.commit()
    return conn


def run_query(conn: sqlite3.Connection, sql: str) -> list[str]:
    rows = conn.execute(sql).fetchall()
    return sorted({row[0] for row in rows})


def delete_outputs(call_id: str, steps: set[int]) -> int:
    deleted = 0
    for step in sorted(steps):
        if step not in STALE_MAP:
            continue
        path = STALE_MAP[step](call_id)
        if path.exists():
            path.unlink()
            log.info(f"  Deleted {path.name}")
            deleted += 1
    return deleted


def print_schema():
    print(SCHEMA_SQL)


CONTEXT_TURNS = 3

TRANSCRIBE_PROMPT = """\
You are correcting a bad transcript turn. The audio clip corresponds to the \
CURRENT turn below. The surrounding turns are provided for speaker context only.

Language: {language}
Speakers in this conversation: {speakers}

{context}

Listen to the audio and respond with a JSON object. Rules:
- If the audio is clear, transcribe it. The turn may contain multiple speakers — \
if so, split into separate turns with the correct speaker labels.
- If the audio is unintelligible or silence, set "turns" to null.
- Use ONLY the speaker labels listed above.

Response format (no other text):
{{"turns": [{{"speaker": "spk_0", "text": "..."}}]}}
or
{{"turns": null}}"""


def _format_context(turns: list[dict], target_idx: int) -> str:
    lines = []
    lo = max(0, target_idx - CONTEXT_TURNS)
    hi = min(len(turns), target_idx + CONTEXT_TURNS + 1)
    for i in range(lo, hi):
        t = turns[i]
        marker = ">>> CURRENT (re-transcribe this) <<<" if i == target_idx else ""
        lines.append(f"[{t['speaker']} {t['start']:.1f}-{t['end']:.1f}]: {t['text']}"
                     + (f"  {marker}" if marker else ""))
    return "\n".join(lines)


FIX_TURNS_MODEL = "claude-haiku-4-5"


def transcribe_turn(client, model: str, b64_audio: str,
                    language: str, context_text: str,
                    speakers: list[str]) -> list[dict] | None:
    prompt = TRANSCRIBE_PROMPT.format(
        language=language,
        speakers=", ".join(speakers),
        context=context_text,
    )
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "source": {
                        "type": "base64",
                        "media_type": "audio/wav",
                        "data": b64_audio,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw = response.content[0].text

    try:
        parsed = extract_json(raw, prefer="object", strip_fences=False)
    except ValueError:
        log.warning(f"Could not parse response: {raw[:120]}")
        return None
    if not isinstance(parsed, dict):
        log.warning(f"Expected object response, got {type(parsed).__name__}: {raw[:120]}")
        return None
    return parsed.get("turns")


CONCURRENCY = 10


def _fix_one_call(cid: str, validated_dir: Path, preprocessed_dir: Path,
                  project_id: str, location: str, model: str) -> tuple[int, int]:
    """Returns (fixed_count, skipped_count)."""
    from anthropic import AnthropicVertex

    val_path = validated_dir / f"{cid}_validated.json"
    wav_path = preprocessed_dir / cid / "full_normalized.wav"
    if not val_path.exists() or not wav_path.exists():
        return 0, 0

    data = read_json(val_path)

    turns = data.get("turns", [])
    language = data.get("language", "unknown")
    speakers = unique_speakers(turns)
    flagged = [(i, turn) for i, turn in enumerate(turns) if turn.get("issues")]

    if not flagged:
        return 0, 0

    log.info(f"{cid}: fixing {len(flagged)} turn(s)")
    client = AnthropicVertex(project_id=project_id, region=location)

    fixed = 0
    skipped = 0
    flagged_indices = {i for i, _ in flagged}
    for i, turn in reversed(flagged):
        lo = max(0, i - CONTEXT_TURNS)
        hi = min(len(turns), i + CONTEXT_TURNS + 1)
        neighbors_bad = any(j in flagged_indices for j in range(lo, hi) if j != i)
        if neighbors_bad:
            log.info(f"  {cid} turn {i}: skipped, neighboring turns also have issues")
            skipped += 1
            continue

        b64_audio = clip_turn_audio(wav_path, turn["start"], turn["end"])
        context_text = _format_context(turns, i)
        result = transcribe_turn(
            client, model, b64_audio, language, context_text, speakers)

        if result is None:
            log.info(f"  {cid} turn {i}: unfixable, keeping original")
            skipped += 1
            continue

        duration = turn_duration(turn)
        n = len(result)
        for j, new_turn in enumerate(result):
            new_turn["start"] = round(turn["start"] + duration * j / n, 3)
            new_turn["end"] = round(turn["start"] + duration * (j + 1) / n, 3)

        if n == 1:
            log.info(f"  {cid} turn {i}: {turn['text'][:50]!r} -> {result[0]['text'][:50]!r}")
        else:
            log.info(f"  {cid} turn {i}: split into {n} turns")

        turns[i:i+1] = result
        fixed += 1

    data["turns"] = turns
    write_json(val_path, data)

    return fixed, skipped


def fix_turns(call_ids: list[str], validated_dir: Path, preprocessed_dir: Path,
              execute: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not execute:
        total = 0
        for cid in call_ids:
            val_path = validated_dir / f"{cid}_validated.json"
            if not val_path.exists():
                continue
            data = read_json(val_path)
            n = sum(1 for t in data.get("turns", []) if t.get("issues"))
            if n:
                log.info(f"{cid}: {n} turn(s) with issues")
                total += n
        log.info(f"Would fix {total} turn(s) (pass --execute to apply)")
        return

    model = os.environ.get("FIX_TURNS_MODEL", FIX_TURNS_MODEL)
    project_id = os.environ["GCP_PROJECT_ID"]
    location = os.environ.get("VERTEX_LOCATION", "us-east5")

    total_fixed = 0
    total_skipped = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(_fix_one_call, cid, validated_dir, preprocessed_dir,
                        project_id, location, model): cid
            for cid in call_ids
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                fixed, skipped = future.result()
                total_fixed += fixed
                total_skipped += skipped
            except Exception:
                log.exception(f"Failed to fix turns for {cid}")

    log.info(f"Fixed {total_fixed}, skipped {total_skipped} unfixable turn(s) "
             f"across {len(call_ids)} recording(s)")


def main():
    bootstrap(dotenv=True)

    parser = argparse.ArgumentParser(
        description="Delete pipeline outputs for selected recordings so they can be re-run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --where "language LIKE 'hi%%'" --steps 2-4
  %(prog)s --where "call_id LIKE '%%Blanchi%%' AND num_speakers = 1"
  %(prog)s --where "call_id IN (SELECT call_id FROM reject_reasons WHERE reason='mono_speaker')"
  %(prog)s --where "duration BETWEEN 30 AND 120"
  %(prog)s --where "language LIKE 'hi%%'" --steps 2-4 --execute
  %(prog)s --sql "SELECT DISTINCT c.call_id FROM calls c JOIN turn_issues ti ON c.call_id = ti.call_id WHERE ti.issue = 'garbled_transcript'"
  %(prog)s --where "call_id IN (SELECT call_id FROM turn_issues WHERE issue='garbled_transcript')" --fix-turns
  %(prog)s --schema""",
    )

    sel = parser.add_argument_group("selection")
    sel.add_argument("--where", default=None, metavar="CLAUSE",
                     help="SQL WHERE clause against the calls table")
    sel.add_argument("--sql", default=None, metavar="QUERY",
                     help="Full SQL query (must return a call_id column)")
    sel.add_argument("--schema", action="store_true",
                     help="Print the database schema and exit")

    exe = parser.add_argument_group("execution")
    exe.add_argument("--steps", default="2,3,4",
                     help="Steps whose outputs to delete: 2, 2-4, 2,4 (default: 2,3,4)")
    exe.add_argument("--list", action="store_true",
                     help="Print matched call IDs")
    exe.add_argument("--fix-turns", action="store_true",
                     help="Re-transcribe flagged turns via Vertex AI instead of deleting outputs")
    exe.add_argument("--execute", action="store_true",
                     help="Actually apply changes (default is dry-run)")

    args = parser.parse_args()

    if args.schema:
        print_schema()
        return

    if not args.where and not args.sql:
        parser.error("--where or --sql is required")
    if args.where and args.sql:
        parser.error("--where and --sql are mutually exclusive")

    conn = build_db(VALIDATED_DIR, PREPROCESSED_DIR)

    if args.where:
        sql = f"SELECT DISTINCT call_id FROM calls WHERE {args.where}"
    else:
        sql = args.sql

    call_ids = run_query(conn, sql)

    if not call_ids:
        conn.close()
        log.info("No recordings matched the query")
        return

    if args.list:
        issue_counts = dict(conn.execute(
            "SELECT call_id, COUNT(*) FROM turn_issues GROUP BY call_id").fetchall())
        total_issues = 0
        for cid in call_ids:
            n = issue_counts.get(cid, 0)
            total_issues += n
            suffix = f"  ({n} turn issue{'s' if n != 1 else ''})" if n else ""
            print(f"{cid}{suffix}")
        print(f"\n{len(call_ids)} recording(s), {total_issues} turn issue(s)")

    conn.close()

    if args.fix_turns:
        log.info(f"Selected {len(call_ids)} recording(s) for turn fixing")
        fix_turns(call_ids, VALIDATED_DIR, PREPROCESSED_DIR, args.execute)
        return

    steps = parse_int_ranges(args.steps, 2, 4)
    log.info(f"Selected {len(call_ids)} recording(s), will delete outputs for steps {sorted(steps)}")

    if not args.execute:
        return

    total_deleted = 0
    for call_id in call_ids:
        deleted = delete_outputs(call_id, steps)
        total_deleted += deleted

    log.info(f"Deleted {total_deleted} file(s) across {len(call_ids)} recording(s)")


if __name__ == "__main__":
    main()
