# storage.py
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

# ---------- Database path ----------
DB_PATH = Path("artifacts/app.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------- Connection helper ----------
def get_conn() -> sqlite3.Connection:
    """
    Get a SQLite connection with WAL mode enabled.
    WAL allows concurrent readers + writers and is ideal for
    FastAPI (writes) + Streamlit (reads).
    """
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# ---------- DB initialisation ----------
def init_db() -> None:
    """
    Create tables and indexes if they do not exist.
    Safe to call on every app startup.
    """
    with get_conn() as conn:
        # ---- live events table ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS live_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            decision TEXT NOT NULL,
            proba REAL NOT NULL,
            payload_json TEXT NOT NULL
        );
        """)

        # ---- audit log table ----
        conn.execute("""
        CREATE TABLE IF NOT EXISTS case_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            old_status TEXT,
            new_status TEXT,
            source TEXT,
            proba REAL,
            ts TEXT,
            logged_at TEXT
        );
        """)

        # ---------- Indexes for performance ----------
        # Speeds up time-based queries (last 60 min, resampling, etc.)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_events_ts
        ON live_events(ts);
        """)

        # Speeds up filtering by decision (REVIEW / APPROVE)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_events_decision
        ON live_events(decision);
        """)


# ---------- Insert helpers ----------
def insert_live_event(
    ts: str,
    decision: str,
    proba: float,
    payload: Dict[str, Any]
) -> int:
    """
    Insert a live event into SQLite atomically.
    Returns the generated event_id.
    """
    payload_json = json.dumps(payload, ensure_ascii=False)

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO live_events (ts, decision, proba, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (ts, decision, float(proba), payload_json),
        )
        return int(cur.lastrowid)


def insert_audit_row(row: Dict[str, Any]) -> None:
    """
    Insert a case audit row into SQLite atomically.
    """
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO case_audit_log (
                event_id,
                old_status,
                new_status,
                source,
                proba,
                ts,
                logged_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("event_id"),
                row.get("old_status"),
                row.get("new_status"),
                row.get("source"),
                row.get("proba"),
                row.get("ts"),
                row.get("logged_at"),
            ),
        )
