"""
SQLite cache layer for fetched financial data.

Used in Phase 2 (live API) — MVP uses sample_data only.
Provides a clean interface for storing/retrieving time-series data.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "cache.db"


class DataCache:
    """Simple SQLite cache for financial time-series data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS series_data (
                    series_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (series_id, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT
                )
            """)

    def store_series(self, series_id: str, df: pd.DataFrame, value_col: str = "value"):
        """Store a time series. df must have a date index and a value column."""
        now = datetime.utcnow().isoformat()
        rows = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            rows.append((series_id, date_str, float(row[value_col]), now))

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO series_data (series_id, date, value, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.execute(
                "INSERT INTO fetch_log (series_id, fetched_at, status, message) "
                "VALUES (?, ?, 'success', ?)",
                (series_id, now, f"Stored {len(rows)} rows"),
            )

    def load_series(self, series_id: str) -> pd.DataFrame:
        """Load a cached series as a DataFrame."""
        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT date, value FROM series_data WHERE series_id = ? ORDER BY date",
                conn,
                params=(series_id,),
                parse_dates=["date"],
            )
        if df.empty:
            return df
        return df.set_index("date")

    def last_fetch(self, series_id: str) -> Optional[str]:
        """Return ISO timestamp of last successful fetch for a series."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute(
                "SELECT fetched_at FROM fetch_log "
                "WHERE series_id = ? AND status = 'success' "
                "ORDER BY fetched_at DESC LIMIT 1",
                (series_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def clear_series(self, series_id: str):
        """Remove all cached data for a series."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM series_data WHERE series_id = ?", (series_id,))
            conn.execute("DELETE FROM fetch_log WHERE series_id = ?", (series_id,))
