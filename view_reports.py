"""
view_reports.py

Small command-line utility for browsing incident reports stored by main.py
in the SQLite database. This is a separate, optional tool - it does not
run the video pipeline, it only reads the existing database.

Usage:
    python view_reports.py recent [--limit 20]
    python view_reports.py search --class person [--limit 20]
    python view_reports.py between --start "2026-07-25 09:00:00" --end "2026-07-25 10:00:00"
    python view_reports.py export --out reports_export.jsonl
    python view_reports.py pdf [--start ...] [--end ...] [--person-out person_report.pdf] [--objects-out other_objects_report.pdf]
    python view_reports.py clear [--yes]
    python view_reports.py count
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from config import Config
from pdf_report import generate_objects_pdf, generate_person_pdf
from storage import ReportStorage


def _print_reports(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No reports found.")
        return

    for row in rows:
        print("-" * 72)
        print(f"ID: {row['id']}")
        print(f"Window: {row['window_start']}  ->  {row['window_end']}")
        print(f"Generated at: {row['generated_at']}   Model: {row['model_used']}")
        print(f"Event count: {row['raw_event_count']}")
        if row.get("error"):
            print(f"LLM error recorded: {row['error']}")
        print(f"Report:\n{row['summary_text']}")
    print("-" * 72)
    print(f"\n{len(rows)} report(s) shown.")


def _parse_datetime(value: str) -> datetime:
    formats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Could not parse datetime '{value}'. Use format 'YYYY-MM-DD HH:MM:SS'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse stored YOLO + Mistral incident reports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recent_parser = subparsers.add_parser("recent", help="Show the most recent reports.")
    recent_parser.add_argument("--limit", type=int, default=20)

    search_parser = subparsers.add_parser("search", help="Search reports by detected object class.")
    search_parser.add_argument("--class", dest="class_name", required=True)
    search_parser.add_argument("--limit", type=int, default=50)

    between_parser = subparsers.add_parser("between", help="Show reports overlapping a time range.")
    between_parser.add_argument("--start", type=_parse_datetime, required=True)
    between_parser.add_argument("--end", type=_parse_datetime, required=True)

    export_parser = subparsers.add_parser("export", help="Export all reports to a JSONL file.")
    export_parser.add_argument("--out", default="reports_export.jsonl")

    pdf_parser = subparsers.add_parser(
        "pdf",
        help="Export two PDF reports: one person-only, one for every other detected object class.",
    )
    pdf_parser.add_argument("--start", type=_parse_datetime, default=None, help="Optional range start.")
    pdf_parser.add_argument("--end", type=_parse_datetime, default=None, help="Optional range end.")
    pdf_parser.add_argument(
        "--person-out",
        default=None,
        help=f"Output path for the person PDF (default: <{Config.REPORTS_PDF_DIR}>/person_report.pdf)",
    )
    pdf_parser.add_argument(
        "--objects-out",
        default=None,
        help=f"Output path for the other-objects PDF (default: <{Config.REPORTS_PDF_DIR}>/other_objects_report.pdf)",
    )

    subparsers.add_parser("count", help="Print the total number of stored reports.")

    clear_parser = subparsers.add_parser(
        "clear",
        help="Delete ALL stored reports from the database (source data for future PDF/JSONL exports).",
    )
    clear_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (use in scripts/automation).",
    )

    args = parser.parse_args()

    storage = ReportStorage(db_path=Config.REPORTS_DB_PATH)

    if args.command == "recent":
        _print_reports(storage.get_recent(limit=args.limit))
    elif args.command == "search":
        _print_reports(storage.search_by_class(args.class_name, limit=args.limit))
    elif args.command == "between":
        _print_reports(storage.get_between(args.start, args.end))
    elif args.command == "export":
        written = storage.export_jsonl(args.out)
        print(f"Exported {written} report(s) to '{args.out}'.")
    elif args.command == "pdf":
        if args.start and args.end:
            rows = storage.get_between(args.start, args.end)
        else:
            rows = storage.get_all()

        os.makedirs(Config.REPORTS_PDF_DIR, exist_ok=True)
        person_out = args.person_out or os.path.join(Config.REPORTS_PDF_DIR, "person_report.pdf")
        objects_out = args.objects_out or os.path.join(Config.REPORTS_PDF_DIR, "other_objects_report.pdf")

        _, person_count = generate_person_pdf(storage, person_out, rows=rows)
        _, objects_count = generate_objects_pdf(storage, objects_out, rows=rows)

        print(f"Wrote person report ({person_count} window(s) included) -> '{person_out}'")
        print(f"Wrote other-objects report ({objects_count} window(s) included) -> '{objects_out}'")
    elif args.command == "count":
        print(f"Total stored reports: {storage.count()}")
    elif args.command == "clear":
        current = storage.count()
        if current == 0:
            print("Nothing to clear - the database is already empty.")
        else:
            if not args.yes:
                answer = input(
                    f"This will permanently delete all {current} stored report(s) from "
                    f"'{Config.REPORTS_DB_PATH}'. Type 'yes' to confirm: "
                )
                if answer.strip().lower() != "yes":
                    print("Cancelled - no reports were deleted.")
                    sys.exit(0)
            deleted = storage.clear_all()
            print(f"Deleted {deleted} report(s). The database is now empty.")
            print(
                "Note: this only clears the source data. Any .pdf/.jsonl files you already "
                "exported still exist on disk - delete those files directly if you want them gone too."
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()