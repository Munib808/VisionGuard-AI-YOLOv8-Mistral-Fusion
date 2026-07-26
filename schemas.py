"""
schemas.py

Pydantic data models used across the YOLO + Mistral Fusion application.
These provide validation and a single source of truth for the shape of
detection, aggregation, and report data as it flows through the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

PERSON_CLASS_NAME = "person"


class Detection(BaseModel):
    """A single object detection produced by YOLO for one frame."""

    class_name: str = Field(..., description="COCO class name, e.g. 'person'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    bbox: List[float] = Field(
        ..., min_length=4, max_length=4, description="[x1, y1, x2, y2] pixel coordinates."
    )
    frame_index: int = Field(..., ge=0, description="Index of the frame this detection came from.")
    timestamp: datetime = Field(..., description="Wall-clock timestamp when the frame was captured.")
    track_id: Optional[int] = Field(
        default=None,
        description=(
            "Stable identity assigned by the tracker (e.g. ByteTrack) so the same "
            "physical person keeps the same ID across consecutive frames. None if "
            "tracking was unavailable for this detection."
        ),
    )

    @field_validator("class_name")
    @classmethod
    def _strip_class_name(cls, v: str) -> str:
        return v.strip()


class AggregatedEvent(BaseModel):
    """
    A summarized record of a class of object seen during a sliding window.

    Two distinct counts are kept because they answer different questions:
      - unique_count: how many DIFFERENT individuals were seen at any point
        during the window (based on tracker IDs), e.g. two different people
        who each showed up once.
      - peak_concurrent_count: the highest number of that class present
        in a SINGLE frame at once, e.g. two people visible together at the
        same time. This is the number a report should use for phrases like
        "N people were present".
    """

    class_name: str
    occurrences: int = Field(..., ge=1, description="Total number of frame-level detections.")
    unique_count: int = Field(..., ge=1, description="Distinct tracked individuals seen in the window.")
    peak_concurrent_count: int = Field(..., ge=1, description="Max number seen at the same time in one frame.")
    first_seen: datetime
    last_seen: datetime
    max_confidence: float = Field(..., ge=0.0, le=1.0)

    @property
    def duration_seconds(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds()

    def to_log_line(self) -> str:
        """Render this event as a single human-readable log line for the LLM prompt."""
        first_str = self.first_seen.strftime("%H:%M:%S")
        last_str = self.last_seen.strftime("%H:%M:%S")
        return (
            f"- Object: '{self.class_name}' | Peak simultaneous count: {self.peak_concurrent_count} | "
            f"Distinct individuals tracked: {self.unique_count} | "
            f"Total frame detections: {self.occurrences} | "
            f"First seen: {first_str} | Last seen: {last_str} "
            f"(present for {self.duration_seconds:.1f}s) | "
            f"Max confidence: {self.max_confidence:.2f}"
        )


class WindowSummary(BaseModel):
    """All aggregated events collected during a single reporting window."""

    window_start: datetime
    window_end: datetime
    events: List[AggregatedEvent]
    total_frames_processed: int = Field(..., ge=0)
    time_source: str = Field(
        default="realtime",
        description=(
            "'video' if window_start/window_end are elapsed time within an uploaded "
            "video file (counted from 00:00:00 at the start of the video), or "
            "'realtime' if they are actual wall-clock time (e.g. from a live webcam)."
        ),
    )

    def is_empty(self) -> bool:
        return len(self.events) == 0

    def person_events(self) -> List[AggregatedEvent]:
        """Events for the 'person' class only."""
        return [e for e in self.events if e.class_name == PERSON_CLASS_NAME]

    def other_events(self) -> List[AggregatedEvent]:
        """Events for every detected class EXCEPT 'person' (the other ~79 COCO classes)."""
        return [e for e in self.events if e.class_name != PERSON_CLASS_NAME]

    def peak_person_count(self) -> int:
        """Deterministic ground truth for 'how many people at once', independent of the LLM."""
        for event in self.events:
            if event.class_name == PERSON_CLASS_NAME:
                return event.peak_concurrent_count
        return 0

    def peak_counts(self) -> Dict[str, int]:
        """Deterministic ground truth peak-simultaneous-count for EVERY detected
        class in this window (not just person), keyed by class name."""
        return {event.class_name: event.peak_concurrent_count for event in self.events}

    def _time_label(self) -> str:
        return "elapsed video time (counted from the start of the video)" if self.time_source == "video" else "real-world wall-clock time"

    def to_prompt_text(self) -> str:
        """Render the whole window as text suitable for inclusion in an LLM prompt."""
        if self.is_empty():
            return "No objects were detected during this monitoring window."

        lines = [event.to_log_line() for event in self.events]
        header = (
            f"Monitoring window ({self._time_label()}): "
            f"{self.window_start.strftime('%H:%M:%S')} to {self.window_end.strftime('%H:%M:%S')} "
            f"({self.total_frames_processed} frames analyzed)"
        )

        counts = self.peak_counts()
        count_parts = [f"{name}: {count}" for name, count in counts.items()]
        footer = (
            "\nAuthoritative peak-simultaneous counts for this window (use these exact "
            "numbers, per class, and never estimate or round them): " + ", ".join(count_parts) + "."
        )
        return header + "\n" + "\n".join(lines) + footer


class IncidentReport(BaseModel):
    """The final natural-language report generated by the Mistral LLM.

    `summary_text` holds a combined narrative (person + other objects) for
    console/log display. `person_summary_text` / `other_summary_text` hold
    the same content already split apart, which is what feeds the two
    separate PDF reports (person-only vs. all-other-objects)."""

    window_start: datetime
    window_end: datetime
    summary_text: str
    person_summary_text: str = ""
    other_summary_text: str = ""
    raw_event_count: int = Field(..., ge=0)
    peak_person_count: int = Field(default=0, ge=0)
    time_source: str = Field(
        default="realtime",
        description="'video' (elapsed video time) or 'realtime' (wall-clock time). See WindowSummary.",
    )
    model_used: str
    generated_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = Field(
        default=None, description="Populated if report generation failed; summary_text will be a fallback."
    )
