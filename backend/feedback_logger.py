"""
ASTRA-Screen: QA Inspector Feedback Logger & Continuous Learning Loop
Logs operator overrides, TEM/DPA laboratory findings, and ground-truth validations
to a local SQLite database for model auditing and future retraining.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "inspector_feedback.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initializes and returns a connection to the feedback SQLite database."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspector_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component_id TEXT NOT NULL,
                lot_id TEXT NOT NULL,
                ai_verdict TEXT NOT NULL,
                ai_diagnosis TEXT,
                ai_confidence REAL,
                inspector_verdict TEXT NOT NULL,
                tem_dpa_findings TEXT,
                inspector_id TEXT NOT NULL,
                notes TEXT
            )
        """)
    return conn


def log_inspector_feedback(
    component_id: str,
    lot_id: str,
    ai_verdict: str,
    ai_diagnosis: str,
    ai_confidence: float,
    inspector_verdict: str,
    tem_dpa_findings: str,
    inspector_id: str = "QA-INSP-402",
    notes: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Logs an inspection feedback entry into the SQLite database."""
    conn = get_connection(db_path)
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO inspector_feedback (
                timestamp, component_id, lot_id, ai_verdict, ai_diagnosis,
                ai_confidence, inspector_verdict, tem_dpa_findings, inspector_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                component_id,
                lot_id,
                ai_verdict,
                ai_diagnosis,
                float(ai_confidence) if ai_confidence is not None else 0.0,
                inspector_verdict,
                tem_dpa_findings,
                inspector_id,
                notes,
            ),
        )
        return cursor.lastrowid


def get_feedback_history(limit: int = 10, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves the most recent feedback entries as a pandas DataFrame."""
    conn = get_connection(db_path)
    query = """
        SELECT id, timestamp, component_id, lot_id, ai_diagnosis,
               inspector_verdict, tem_dpa_findings, inspector_id, notes
        FROM inspector_feedback
        ORDER BY id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    return df


def get_feedback_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, int]:
    """Returns total logged feedback counts and agreement rate."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM inspector_feedback")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM inspector_feedback WHERE inspector_verdict = 'CONFIRMED'")
    confirmed = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM inspector_feedback WHERE inspector_verdict = 'OVERRIDDEN'")
    overridden = cursor.fetchone()[0]

    return {
        "total_logged": total,
        "confirmed": confirmed,
        "overridden": overridden,
    }


def clear_feedback_history(db_path: str = DEFAULT_DB_PATH) -> bool:
    """Clears all logged feedback entries from the SQLite database."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM inspector_feedback")
    return True

