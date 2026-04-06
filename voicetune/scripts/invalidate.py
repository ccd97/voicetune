"""Select recordings via SQL and delete stage outputs so they get regenerated.

Usage:
  python voicetune/scripts/invalidate.py --where "language LIKE 'hi%'" --steps 2-4 --execute
  python voicetune/scripts/invalidate.py --where "call_id LIKE '%Blanchi%' AND num_speakers = 1"
  python voicetune/scripts/invalidate.py --schema
"""

import argparse
import logging
import sqlite3
from pathlib import Path

from voicetune.common import (
    bootstrap,
    parse_int_ranges,
    read_json,
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
