"""
storage.py

Persists generated IncidentReport objects (and the underlying detection
events that produced them) to a local SQLite database, so reports survive
past the lifetime of the running process and can be queried later.

SQLite is used instead of a plain text/JSON log because it gives you real
querying (by date range, by object class, by whether an LLM error occurred)
without adding any third-party dependency - sqlite3 ships with Python.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from schemas import IncidentReport, WindowSummary


class ReportStorage:
    """Handles all reads/writes to the incident_reports SQLite table."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # Columns added after the original schema shipped. Kept as (name, DDL)
    # pairs so an existing database (like one from an earlier version of this
    # app) gets migrated in place via ALTER TABLE instead of silently losing
    # the new fields.
    _MIGRATIONS = [
        ("person_summary_text", "ALTER TABLE incident_reports ADD COLUMN person_summary_text TEXT"),
        ("other_summary_text", "ALTER TABLE incident_reports ADD COLUMN other_summary_text TEXT"),
        (
            "time_source",
            "ALTER TABLE incident_reports ADD COLUMN time_source TEXT NOT NULL DEFAULT 'realtime'",
        ),
    ]

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS incident_reports (
                        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                        window_start         TEXT NOT NULL,
                        window_end           TEXT NOT NULL,
                        generated_at         TEXT NOT NULL,
                        model_used           TEXT NOT NULL,
                        raw_event_count      INTEGER NOT NULL,
                        summary_text         TEXT NOT NULL,
                        person_summary_text  TEXT,
                        other_summary_text   TEXT,
                        time_source          TEXT NOT NULL DEFAULT 'realtime',
                        error                TEXT,
                        events_json          TEXT,
                        created_at           TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )

                # Migrate any pre-existing database (created before these
                # columns existed) so old data keeps working with the new
                # split-report / time-source features.
                existing_cols = {
                    row["name"] for row in conn.execute("PRAGMA table_info(incident_reports)")
                }
                for column_name, ddl in self._MIGRATIONS:
                    if column_name not in existing_cols:
                        conn.execute(ddl)

                # Index to make "reports in a time range" queries fast as the table grows.
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reports_window_start "
                    "ON incident_reports (window_start)"
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to initialize report storage at '{self.db_path}': {exc}") from exc

    def save(self, report: IncidentReport, summary: Optional[WindowSummary] = None) -> Optional[int]:
        """
        Persists one report. Returns the new row id, or None if the write
        failed - a storage failure should never crash the video/report loop,
        so errors are caught and logged here rather than raised.
        """
        events_json: Optional[str] = None
        if summary is not None:
            try:
                events_json = json.dumps(
                    [json.loads(event.model_dump_json()) for event in summary.events]
                )
            except (TypeError, ValueError) as exc:
                print(f"[ReportStorage] Could not serialize event details (report will still be saved): {exc}")

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO incident_reports
                        (window_start, window_end, generated_at, model_used,
                         raw_event_count, summary_text, person_summary_text,
                         other_summary_text, time_source, error, events_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report.window_start.isoformat(),
                        report.window_end.isoformat(),
                        report.generated_at.isoformat(),
                        report.model_used,
                        report.raw_event_count,
                        report.summary_text,
                        report.person_summary_text,
                        report.other_summary_text,
                        report.time_source,
                        report.error,
                        events_json,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to save report to database: {exc}")
            return None

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent `limit` reports, newest first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM incident_reports ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to read recent reports: {exc}")
            return []

    def get_between(self, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        """Returns all reports whose window overlaps [start, end], oldest first."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM incident_reports
                    WHERE window_start <= ? AND window_end >= ?
                    ORDER BY id ASC
                    """,
                    (end.isoformat(), start.isoformat()),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to query reports by time range: {exc}")
            return []

    def search_by_class(self, class_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns reports whose stored events include a given object class name."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM incident_reports
                    WHERE events_json LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (f'%"class_name": "{class_name}"%', limit),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to search reports by class: {exc}")
            return []

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns every stored report, oldest first. Used by PDF export so the
        generated PDFs include the full history unless a time range is given."""
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM incident_reports ORDER BY id ASC").fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to read all reports: {exc}")
            return []

    def clear_all(self) -> int:
        """Deletes every stored report and resets the id counter. Returns the
        number of rows deleted. This is what backs 'view_reports.py clear' -
        it clears the SOURCE data, so any PDF/JSONL you export afterwards
        will be empty/fresh; it does not touch any files you've already
        exported (those are separate static files, see view_reports.py)."""
        try:
            with self._connect() as conn:
                deleted = conn.execute("SELECT COUNT(*) AS c FROM incident_reports").fetchone()["c"]
                conn.execute("DELETE FROM incident_reports")
                # Reset AUTOINCREMENT counter so new reports start at id 1 again.
                conn.execute("DELETE FROM sqlite_sequence WHERE name = 'incident_reports'")
                conn.commit()
                return int(deleted)
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to clear reports: {exc}")
            return 0

    def count(self) -> int:
        try:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) AS c FROM incident_reports").fetchone()
                return int(row["c"]) if row else 0
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to count reports: {exc}")
            return 0

    def export_jsonl(self, output_path: str) -> int:
        """Exports every stored report to a JSONL file. Returns the number of rows written."""
        try:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM incident_reports ORDER BY id ASC").fetchall()
        except sqlite3.Error as exc:
            print(f"[ReportStorage] Failed to read reports for export: {exc}")
            return 0

        written = 0
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(dict(row)) + "\n")
                    written += 1
        except OSError as exc:
            print(f"[ReportStorage] Failed to write export file '{output_path}': {exc}")
            return written

        return written