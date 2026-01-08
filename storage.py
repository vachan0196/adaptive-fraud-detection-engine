# storage.py
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------- Database path ----------
DB_PATH = Path("artifacts/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------- Connection helper ----------
def get_conn() -> sqlite3.Connection:
    """
    Get a SQLite connection configured for:
    - WAL: concurrent readers + single writer
    - foreign_keys: enforce cases -> live_events integrity
    """
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Pragmas: do these on EVERY connection
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------- DB initialisation ----------
def init_db() -> None:
    """
    Create tables and indexes if they do not exist.
    Safe to call on every app startup.
    """
    with get_conn() as conn:
        # ---- runs table (for demo resets + scoping) ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
              started_at TEXT NOT NULL,
              label      TEXT
            );
            """
        )

        # ---- app_state table (stores current_run_id) ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
              key   TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )

        # ---- live events (single source of truth) ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_events (
              event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id       INTEGER NOT NULL,
              ts           TEXT NOT NULL,
              decision     TEXT NOT NULL CHECK (decision IN ('APPROVE','REVIEW')),
              proba        REAL NOT NULL,
              payload_json TEXT NOT NULL,
              FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )

        # ---- cases (workflow layer on top of live_events) ----
        # event_id is PRIMARY KEY to enforce 1:1 between REVIEW event and case
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
              event_id           INTEGER PRIMARY KEY,
              run_id             INTEGER NOT NULL,
              status             TEXT NOT NULL CHECK (status IN ('PENDING','CONFIRMED_FRAUD','CONFIRMED_LEGIT','DISMISSED')),
              customer_response  TEXT,
              resolution_source  TEXT,
              created_at         TEXT NOT NULL,
              updated_at         TEXT,
              FOREIGN KEY (event_id) REFERENCES live_events(event_id) ON DELETE CASCADE,
              FOREIGN KEY (run_id)   REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )

        # ---- audit log (optional but useful) ----
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS case_audit_log (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id     INTEGER NOT NULL,
              event_id   INTEGER NOT NULL,
              old_status TEXT,
              new_status TEXT,
              source     TEXT NOT NULL,
              proba      REAL,
              ts         TEXT,
              logged_at  TEXT NOT NULL,
              FOREIGN KEY (event_id) REFERENCES live_events(event_id) ON DELETE CASCADE,
              FOREIGN KEY (run_id)   REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )

        # ---- Indexes for performance ----
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_events_run_event ON live_events(run_id, event_id DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_events_run_ts ON live_events(run_id, ts);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_events_run_dec ON live_events(run_id, decision);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cases_run_status ON cases(run_id, status);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_run_id ON case_audit_log(run_id, id DESC);"
        )

        # Ensure we have an active run_id
        ensure_current_run()


# ---------- Run helpers ----------
def ensure_current_run(label: Optional[str] = "default") -> int:
    """
    Ensure app_state.current_run_id exists. If not, create a new run.
    Returns the current run_id.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key='current_run_id' LIMIT 1;"
        ).fetchone()
        if row and str(row["value"]).strip():
            return int(row["value"])

    # If missing, create one
    return start_new_run(label=label)


def get_current_run_id() -> int:
    """
    Get the current run_id. Will create one if missing.
    """
    return ensure_current_run(label="default")


def start_new_run(label: Optional[str] = None, started_at: Optional[str] = None) -> int:
    """
    Start a new run and set it as current_run_id.

    started_at should be an ISO timestamp string (e.g., datetime.utcnow().isoformat()).
    If not provided, caller should supply it (service.py will).
    """
    if not started_at:
        # Avoid importing datetime here if you want strict control upstream.
        # But to make this drop-in, we'll generate a basic ISO timestamp.
        from datetime import datetime, timezone

        started_at = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (started_at, label) VALUES (?, ?);",
            (started_at, label),
        )
        run_id = int(cur.lastrowid)
        conn.execute(
            """
            INSERT INTO app_state (key, value) VALUES ('current_run_id', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value;
            """,
            (str(run_id),),
        )
        return run_id


# ---------- Insert helpers ----------
def insert_live_event(
    run_id: int,
    ts: str,
    decision: str,
    proba: float,
    payload: Dict[str, Any],
) -> int:
    """
    Insert a live event into SQLite atomically.
    Returns generated event_id.
    """
    payload_json = json.dumps(payload, ensure_ascii=False)
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO live_events (run_id, ts, decision, proba, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(run_id), ts, decision, float(proba), payload_json),
        )
        return int(cur.lastrowid)


def ensure_case_for_event(run_id: int, event_id: int, created_at: str) -> None:
    """
    Create a case for a live_event (only when decision == REVIEW).
    Idempotent: will not duplicate if called multiple times.
    """
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO cases (
              event_id, run_id, status, customer_response, resolution_source, created_at, updated_at
            )
            VALUES (?, ?, 'PENDING', NULL, NULL, ?, NULL);
            """,
            (int(event_id), int(run_id), created_at),
        )


def resolve_case(
    run_id: int,
    event_id: int,
    new_status: str,
    customer_response: Optional[str],
    source: str,
    proba: Optional[float],
    ts: Optional[str],
    logged_at: str,
) -> None:
    """
    Resolve/update a case status and write an audit row in ONE transaction.
    """
    with get_conn() as conn:
        # Ensure case exists (safety)
        existing = conn.execute(
            "SELECT status FROM cases WHERE event_id=? AND run_id=? LIMIT 1;",
            (int(event_id), int(run_id)),
        ).fetchone()

        if not existing:
            # If a resolve comes in without a case (shouldn't happen),
            # create a minimal pending case then resolve it.
            conn.execute(
                """
                INSERT OR IGNORE INTO cases (
                  event_id, run_id, status, customer_response, resolution_source, created_at, updated_at
                )
                VALUES (?, ?, 'PENDING', NULL, NULL, ?, NULL);
                """,
                (int(event_id), int(run_id), logged_at),
            )
            old_status = "PENDING"
        else:
            old_status = str(existing["status"])

        conn.execute(
            """
            UPDATE cases
            SET status=?,
                customer_response=?,
                resolution_source=?,
                updated_at=?
            WHERE event_id=? AND run_id=?;
            """,
            (
                new_status,
                customer_response,
                source,
                logged_at,
                int(event_id),
                int(run_id),
            ),
        )

        conn.execute(
            """
            INSERT INTO case_audit_log (
              run_id, event_id, old_status, new_status, source, proba, ts, logged_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                int(run_id),
                int(event_id),
                old_status,
                new_status,
                source,
                float(proba) if proba is not None else None,
                ts,
                logged_at,
            ),
        )


def insert_audit_row(row: Dict[str, Any]) -> None:
    """
    Backwards-compatible helper (but prefer resolve_case()).
    """
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO case_audit_log (
                run_id,
                event_id,
                old_status,
                new_status,
                source,
                proba,
                ts,
                logged_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("run_id"),
                row.get("event_id"),
                row.get("old_status"),
                row.get("new_status"),
                row.get("source"),
                row.get("proba"),
                row.get("ts"),
                row.get("logged_at"),
            ),
        )


# ---------- Pruning / retention ----------
def prune_run(run_id: int, max_events: int = 100_000) -> int:
    """
    Enforce retention cap for a run by deleting older live_events.
    ON DELETE CASCADE removes dependent cases + audit rows automatically.
    Returns number of deleted live_events.
    """
    with get_conn() as conn:
        count_row = conn.execute(
            "SELECT COUNT(*) AS n FROM live_events WHERE run_id=?;",
            (int(run_id),),
        ).fetchone()
        n = int(count_row["n"]) if count_row else 0
        if n <= max_events:
            return 0

        to_delete = n - max_events

        # Delete the oldest 'to_delete' events in this run.
        # Using event_id ordering as insertion order.
        old_ids = conn.execute(
            """
            SELECT event_id
            FROM live_events
            WHERE run_id=?
            ORDER BY event_id ASC
            LIMIT ?;
            """,
            (int(run_id), int(to_delete)),
        ).fetchall()

        if not old_ids:
            return 0

        ids = [int(r["event_id"]) for r in old_ids]
        placeholders = ",".join(["?"] * len(ids))

        cur = conn.execute(
            f"DELETE FROM live_events WHERE run_id=? AND event_id IN ({placeholders});",
            (int(run_id), *ids),
        )
        return int(cur.rowcount or 0)


# ---------- Demo reset ----------
def reset_demo(label: Optional[str] = "demo reset") -> int:
    """
    Soft reset: start a new run_id and make it current.
    This avoids wiping the DB and eliminates historical confusion in the UI.
    Returns the new run_id.
    """
    return start_new_run(label=label)


def hard_reset_all() -> None:
    """
    Hard reset (optional): deletes ALL data.
    Use only if you truly want a clean DB file state.
    """
    with get_conn() as conn:
        conn.execute("DELETE FROM case_audit_log;")
        conn.execute("DELETE FROM cases;")
        conn.execute("DELETE FROM live_events;")
        conn.execute("DELETE FROM runs;")
        conn.execute("DELETE FROM app_state;")


# ---------- Minimal read helpers (optional convenience) ----------
def get_run_stats(run_id: int) -> Dict[str, Any]:
    """
    Convenience read helper; Streamlit can also run its own SELECTs.
    """
    with get_conn() as conn:
        ev = conn.execute(
            "SELECT COUNT(*) AS n FROM live_events WHERE run_id=?;", (int(run_id),)
        ).fetchone()
        cs = conn.execute(
            "SELECT COUNT(*) AS n FROM cases WHERE run_id=?;", (int(run_id),)
        ).fetchone()
        return {"run_id": int(run_id), "live_events": int(ev["n"]), "cases": int(cs["n"])}
