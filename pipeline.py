"""
pipeline.py

Reusable, framework-agnostic pipeline that both the Flask app (flask_app.py)
and the Streamlit app (streamlit_app.py) call into. This is the ONE place
that:

  1. Reads an uploaded video file frame by frame.
  2. Runs YOLO detection + tracking on every frame (vision.Detector).
  3. Draws bounding boxes and writes an annotated output video so the UI can
     show detections on-screen.
  4. Aggregates detections into rolling windows (aggregator.EventAggregator).
  5. Turns each window into a split person/other-objects incident report,
     via Mistral if an API key is configured, otherwise via a deterministic
     offline fallback -- so the app is always usable end-to-end even without
     an API key.
  6. Persists every report to a per-job SQLite database (storage.ReportStorage).
  7. Generates the two PDF deliverables (pdf_report.py) on demand.

Both UIs call `process_video()` and `build_pdfs()` below, so behavior can
never drift between the two interfaces.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import cv2

from aggregator import EventAggregator
from config import Config
from llm_reporter import LLMReporterError, MistralReporter
from pdf_report import generate_objects_pdf, generate_person_pdf
from schemas import IncidentReport, WindowSummary
from storage import ReportStorage

JOBS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_data")
os.makedirs(JOBS_ROOT, exist_ok=True)

ProgressCallback = Optional[Callable[[float, str], None]]


@dataclass
class JobResult:
    job_id: str
    job_dir: str
    db_path: str
    annotated_video_path: str
    total_frames: int = 0
    fps: float = 30.0
    duration_seconds: float = 0.0
    peak_person_count: int = 0
    peak_object_classes: Dict[str, int] = field(default_factory=dict)
    total_windows: int = 0
    reports: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def job_dir_for(job_id: str) -> str:
    d = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _open_writer(path: str, fps: float, size) -> cv2.VideoWriter:
    """Tries a browser-friendly H.264 fourcc first, falls back to mp4v if the
    local OpenCV/ffmpeg build doesn't support it."""
    for fourcc_name in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not initialize a video writer with any known codec.")


def _make_reporter() -> Optional[MistralReporter]:
    """Returns a MistralReporter if an API key is configured, else None
    (the caller then uses the deterministic offline fallback text)."""
    if not Config.MISTRAL_API_KEY:
        return None
    try:
        return MistralReporter(api_key=Config.MISTRAL_API_KEY, model=Config.MISTRAL_MODEL)
    except LLMReporterError:
        return None


def _generate_report(reporter: Optional[MistralReporter], summary: WindowSummary) -> IncidentReport:
    """Generates one window's report, via Mistral if available, else via the
    same deterministic fallback text the live pipeline uses on API failure."""
    if reporter is not None:
        return reporter.generate_report(summary)

    person_text, other_text = MistralReporter._build_fallback_text(summary)  # noqa: SLF001
    return IncidentReport(
        window_start=summary.window_start,
        window_end=summary.window_end,
        summary_text=f"{person_text}\n\n{other_text}",
        person_summary_text=person_text,
        other_summary_text=other_text,
        raw_event_count=len(summary.events),
        peak_person_count=summary.peak_person_count(),
        time_source=summary.time_source,
        model_used="offline-fallback (no MISTRAL_API_KEY configured)",
        error="No Mistral API key configured; using deterministic offline summary.",
    )


def process_video(
    video_path: str,
    job_id: Optional[str] = None,
    on_progress: ProgressCallback = None,
) -> JobResult:
    """
    Runs the full detection -> aggregation -> reporting pipeline over an
    uploaded video file and returns a JobResult. `on_progress(fraction, message)`
    is called periodically (fraction in [0, 1]) so a caller can drive a
    progress bar in either UI.
    """
    from vision import Detector  # local import: keeps module import light for callers that only need job helpers

    job_id = job_id or new_job_id()
    jdir = job_dir_for(job_id)
    db_path = os.path.join(jdir, "reports.db")
    annotated_path = os.path.join(jdir, "annotated.mp4")

    def report_progress(frac: float, msg: str) -> None:
        if on_progress:
            on_progress(min(max(frac, 0.0), 1.0), msg)

    report_progress(0.02, "Opening video and loading YOLO model...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open uploaded video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    detector = Detector(
        model_path=Config.YOLO_MODEL_PATH,
        confidence_threshold=Config.YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold=Config.YOLO_IOU_THRESHOLD,
        img_size=Config.YOLO_IMG_SIZE,
        target_classes=Config.TARGET_CLASSES,
        enable_tracking=Config.ENABLE_TRACKING,
        tracker_config=Config.TRACKER_CONFIG,
    )

    aggregator = EventAggregator(
        window_seconds=Config.WINDOW_SECONDS,
        window_max_frames=Config.WINDOW_MAX_FRAMES,
        time_source="video",
    )

    storage = ReportStorage(db_path=db_path)
    reporter = _make_reporter()

    writer = _open_writer(annotated_path, fps, (width, height))

    result = JobResult(
        job_id=job_id, job_dir=jdir, db_path=db_path,
        annotated_video_path=annotated_path, fps=fps,
    )
    if reporter is None:
        result.warnings.append(
            "No MISTRAL_API_KEY configured -- using deterministic offline report text "
            "instead of LLM-generated narrative."
        )

    from datetime import timedelta

    epoch = datetime(2000, 1, 1)
    frame_index = 0
    report_progress(0.05, "Running detection on frames...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Elapsed video time, counted from a fixed epoch (matches vision.VideoSource).
            timestamp = epoch + timedelta(seconds=frame_index / fps)

            detections = detector.detect(frame, frame_index, timestamp)
            annotated = detector.draw_detections(frame, detections)
            writer.write(annotated)

            aggregator.add_detections(detections, timestamp)

            if aggregator.is_window_complete(timestamp):
                summary = aggregator.flush(timestamp)
                if not summary.is_empty():
                    report = _generate_report(reporter, summary)
                    storage.save(report, summary)
                    result.total_windows += 1
                    result.peak_person_count = max(result.peak_person_count, report.peak_person_count)
                    for evt in summary.events:
                        result.peak_object_classes[evt.class_name] = max(
                            result.peak_object_classes.get(evt.class_name, 0),
                            evt.peak_concurrent_count,
                        )

            frame_index += 1
            if total_frames_hint > 0:
                frac = 0.05 + 0.85 * (frame_index / total_frames_hint)
                if frame_index % 10 == 0:
                    report_progress(frac, f"Analyzing frame {frame_index}/{total_frames_hint}...")

        # Flush any partial trailing window.
        final_ts = epoch + timedelta(seconds=frame_index / fps)
        if aggregator._window_start is not None:  # noqa: SLF001
            summary = aggregator.flush(final_ts)
            if not summary.is_empty():
                report = _generate_report(reporter, summary)
                storage.save(report, summary)
                result.total_windows += 1
                result.peak_person_count = max(result.peak_person_count, report.peak_person_count)
                for evt in summary.events:
                    result.peak_object_classes[evt.class_name] = max(
                        result.peak_object_classes.get(evt.class_name, 0), evt.peak_concurrent_count
                    )
    finally:
        cap.release()
        writer.release()

    result.total_frames = frame_index
    result.duration_seconds = frame_index / fps
    result.reports = storage.get_all()

    report_progress(0.95, "Finalizing report database...")
    report_progress(1.0, "Done.")
    return result


def build_pdfs(db_path: str, out_dir: str) -> Dict[str, str]:
    """Generates the person-only and other-objects PDFs from a job's stored
    reports. Returns {'person': path, 'objects': path}."""
    os.makedirs(out_dir, exist_ok=True)
    storage = ReportStorage(db_path=db_path)
    person_path = os.path.join(out_dir, "person_report.pdf")
    objects_path = os.path.join(out_dir, "other_objects_report.pdf")
    generate_person_pdf(storage, person_path)
    generate_objects_pdf(storage, objects_path)
    return {"person": person_path, "objects": objects_path}


def load_reports(db_path: str) -> List[Dict]:
    return ReportStorage(db_path=db_path).get_all()
