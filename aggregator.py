"""
aggregator.py

Accumulates raw per-frame Detection objects into a WindowSummary over a
sliding time/frame window, so the LLM is called once per window instead of
once per frame (which would be slow, expensive, and redundant).

Counting logic
---------------
This module deliberately tracks THREE different numbers per class, because
collapsing them into one is exactly what caused reports like "A person was
detected" when two people were actually on screen at once:

  1. occurrences            -> total frame-level detections (noisy, not a
                                headcount; the same person contributes one
                                occurrence per frame they're visible in).
  2. unique_count            -> count of DISTINCT tracker IDs seen at any
                                point in the window (two different people
                                who each appeared once = 2).
  3. peak_concurrent_count   -> the max number of that class seen together
                                in a SINGLE frame (two people standing next
                                to each other at the same moment = 2). This
                                is the number that should drive phrases like
                                "N people were present" in a report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Set

from schemas import AggregatedEvent, Detection, WindowSummary


class EventAggregator:
    """
    Collects detections across frames and produces a WindowSummary once the
    configured time or frame-count threshold is reached.
    """

    def __init__(self, window_seconds: float, window_max_frames: int, time_source: str = "realtime") -> None:
        self.window_seconds = window_seconds
        self.window_max_frames = window_max_frames
        # 'video' (elapsed video time, for uploaded files) or 'realtime'
        # (wall-clock time, for a live webcam) -- see VideoSource.time_source.
        # Carried straight through onto every WindowSummary this aggregator
        # produces so reports/PDFs know which kind of timestamp they have.
        self.time_source = time_source

        self._window_start: Optional[datetime] = None
        self._frames_in_window: int = 0
        # class_name -> running stats
        self._class_stats: Dict[str, Dict] = {}

    def add_detections(self, detections: List[Detection], timestamp: datetime) -> None:
        """
        Registers all detections found in a single frame. `detections` is
        expected to be every detection produced for ONE frame, so counting
        how many belong to a given class here gives us the concurrent count
        for that instant.
        """
        if self._window_start is None:
            self._window_start = timestamp

        self._frames_in_window += 1

        # Count how many of each class appear together in THIS frame, so we
        # can track the peak simultaneous count across the whole window.
        per_frame_class_counts: Dict[str, int] = {}
        for det in detections:
            per_frame_class_counts[det.class_name] = per_frame_class_counts.get(det.class_name, 0) + 1

        for det in detections:
            stats = self._class_stats.get(det.class_name)
            if stats is None:
                stats = {
                    "occurrences": 0,
                    "first_seen": det.timestamp,
                    "last_seen": det.timestamp,
                    "max_confidence": det.confidence,
                    "track_ids": set(),  # type: Set[int]
                    "untracked_occurrences": 0,
                    "peak_concurrent_count": 0,
                }
                self._class_stats[det.class_name] = stats

            stats["occurrences"] += 1
            stats["last_seen"] = det.timestamp
            stats["max_confidence"] = max(stats["max_confidence"], det.confidence)

            if det.track_id is not None:
                stats["track_ids"].add(det.track_id)
            else:
                # No tracker available (e.g. tracking disabled/failed) - fall
                # back to counting every detection as a potentially distinct
                # individual so unique_count never under-reports.
                stats["untracked_occurrences"] += 1

            stats["peak_concurrent_count"] = max(
                stats["peak_concurrent_count"], per_frame_class_counts[det.class_name]
            )

    def is_window_complete(self, current_timestamp: datetime) -> bool:
        """Checks whether the current window has hit either the time or frame limit."""
        if self._window_start is None:
            return False

        elapsed_seconds = (current_timestamp - self._window_start).total_seconds()
        if elapsed_seconds >= self.window_seconds:
            return True
        if self._frames_in_window >= self.window_max_frames:
            return True
        return False

    def flush(self, current_timestamp: datetime) -> WindowSummary:
        """
        Produces a WindowSummary from everything accumulated so far and resets
        internal state so the next window starts clean.
        """
        window_start = self._window_start or current_timestamp

        events: List[AggregatedEvent] = []
        for class_name, stats in self._class_stats.items():
            tracked_unique = len(stats["track_ids"])
            # unique_count is at least the number of distinct tracked IDs, and
            # at least the peak concurrent count (you can't have more people
            # on screen at once than total distinct people seen).
            unique_count = max(
                tracked_unique + stats["untracked_occurrences"],
                stats["peak_concurrent_count"],
                1,
            )
            events.append(
                AggregatedEvent(
                    class_name=class_name,
                    occurrences=stats["occurrences"],
                    unique_count=unique_count,
                    peak_concurrent_count=max(stats["peak_concurrent_count"], 1),
                    first_seen=stats["first_seen"],
                    last_seen=stats["last_seen"],
                    max_confidence=stats["max_confidence"],
                )
            )

        # Present the most frequently observed objects first; this makes the
        # eventual LLM prompt easier to reason over for the most salient items.
        events.sort(key=lambda e: e.occurrences, reverse=True)

        summary = WindowSummary(
            window_start=window_start,
            window_end=current_timestamp,
            events=events,
            total_frames_processed=self._frames_in_window,
            time_source=self.time_source,
        )

        # Reset state for the next window.
        self._window_start = None
        self._frames_in_window = 0
        self._class_stats = {}

        return summary
