"""
pdf_report.py

Generates the two PDF deliverables companies asked for, built directly from
data already stored in SQLite by main.py:

  1. Person Detection Report  - only "person" class activity.
  2. Other Objects Report     - every other detected COCO class (up to the
                                 remaining 79 classes: vehicles, animals,
                                 bags, furniture, electronics, etc.).

Design notes
------------
- The per-window text (`person_summary_text` / `other_summary_text`) already
  comes pre-split from llm_reporter.py, so this module does not call an LLM
  itself - it only formats what's already been generated and stored. That
  keeps PDF export fast, offline-capable, and free of any risk of the two
  PDFs disagreeing with what was already reported live.
- Each window also gets a small table built straight from `events_json`
  (the raw per-class stats saved alongside every report), so the PDF shows
  the authoritative peak/unique/occurrence counts, not just prose - this is
  what makes the PDF "accurate without unnecessary detail": one paragraph of
  narrative plus one compact numeric table per window, nothing else.
- Windows with no relevant data for a given PDF (e.g. a window with only
  cars in it contributes nothing to the person PDF) are skipped entirely
  rather than padded with empty sections.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from storage import ReportStorage


def _time_label(time_source: str) -> str:
    return "video time (elapsed)" if time_source == "video" else "real-world clock time"


def _fmt_window(row: Dict[str, Any]) -> str:
    """Renders 'HH:MM:SS - HH:MM:SS', tolerating either ISO datetimes or
    plain time strings already stored in the row."""

    def _hms(value: str) -> str:
        try:
            return datetime.fromisoformat(value).strftime("%H:%M:%S")
        except (TypeError, ValueError):
            return str(value)

    return f"{_hms(row.get('window_start', ''))} - {_hms(row.get('window_end', ''))}"


def _events_for(row: Dict[str, Any], person_only: bool) -> List[Dict[str, Any]]:
    raw = row.get("events_json")
    if not raw:
        return []
    try:
        events = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if person_only:
        return [e for e in events if e.get("class_name") == "person"]
    return [e for e in events if e.get("class_name") != "person"]


def _event_table(events: List[Dict[str, Any]]) -> Table:
    header = ["Class", "Peak simultaneous", "Distinct tracked", "Total detections", "Max confidence"]
    data = [header]
    for event in sorted(events, key=lambda e: e.get("peak_concurrent_count", 0), reverse=True):
        data.append(
            [
                str(event.get("class_name", "")),
                str(event.get("peak_concurrent_count", "")),
                str(event.get("unique_count", "")),
                str(event.get("occurrences", "")),
                f"{float(event.get('max_confidence', 0) or 0):.2f}",
            ]
        )

    table = Table(data, hAlign="LEFT", colWidths=[1.7 * inch, 1.15 * inch, 1.1 * inch, 1.15 * inch, 1.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#233240")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f6")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _build_pdf(
    rows: List[Dict[str, Any]],
    title: str,
    subtitle: str,
    person_only: bool,
    output_path: str,
) -> Tuple[str, int]:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title=title,
    )
    styles = getSampleStyleSheet()
    window_style = ParagraphStyle("WindowHeading", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], textColor=colors.HexColor("#555555"), fontSize=9)
    error_style = ParagraphStyle("ErrorNote", parent=styles["Italic"], textColor=colors.HexColor("#8a5a00"), fontSize=9)

    story: List[Any] = [
        Paragraph(title, styles["Title"]),
        Paragraph(subtitle, styles["Normal"]),
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"from {len(rows)} stored monitoring window(s).",
            meta_style,
        ),
        Spacer(1, 14),
    ]

    included = 0
    for row in rows:
        events = _events_for(row, person_only)
        text = (row.get("person_summary_text") if person_only else row.get("other_summary_text")) or ""
        text = text.strip()

        if not events and not text:
            continue
        included += 1

        story.append(
            Paragraph(
                f"Window: {_fmt_window(row)} &nbsp;&mdash;&nbsp; {_time_label(row.get('time_source') or 'realtime')}",
                window_style,
            )
        )
        if row.get("error"):
            story.append(Paragraph("LLM call failed for this window; fallback summary shown.", error_style))

        story.append(Paragraph(text if text else "No detections recorded for this window.", styles["Normal"]))

        if events:
            story.append(Spacer(1, 6))
            story.append(_event_table(events))

        story.append(Spacer(1, 12))

    if included == 0:
        story.append(Paragraph("No matching detections were recorded in the selected range.", styles["Normal"]))

    doc.build(story)
    return output_path, included


def generate_person_pdf(
    storage: ReportStorage,
    output_path: str,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, int]:
    """Writes the person-only PDF. Returns (output_path, windows_included)."""
    rows = rows if rows is not None else storage.get_all()
    return _build_pdf(
        rows,
        title="Person Detection Incident Report",
        subtitle="Covers only the 'person' class across all included monitoring windows.",
        person_only=True,
        output_path=output_path,
    )


def generate_objects_pdf(
    storage: ReportStorage,
    output_path: str,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, int]:
    """Writes the other-objects PDF (every non-person COCO class). Returns (output_path, windows_included)."""
    rows = rows if rows is not None else storage.get_all()
    return _build_pdf(
        rows,
        title="Other Object Detections Report",
        subtitle="Covers every detected object class EXCEPT 'person' across all included monitoring windows.",
        person_only=False,
        output_path=output_path,
    )
