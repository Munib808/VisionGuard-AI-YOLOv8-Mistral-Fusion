
"""
main.py

Entrypoint for the "YOLO + Mistral Fusion: Explain What You See" application.

Pipeline:
  Video frame -> YOLO detection (person + all other COCO classes, tracked)
              -> Event Aggregator -> Mistral LLM incident report (split into
              a person-only section and an "other objects" section, on a
              background thread) -> console output (+ optional live
              annotated video window) -> optional PDF export
              (see view_reports.py pdf)

Why a background thread for reporting?
---------------------------------------
The original version called the Mistral API and wrote to SQLite directly
inside the video capture loop, once per window. That call is a network
round-trip (typically 0.5-3+ seconds, more if the API is slow), and while
it's in flight, cv2.imshow()/cv2.waitKey() never run -- so the preview
window visibly freezes every WINDOW_SECONDS. Moving report generation onto
a dedicated worker thread (communicating via a thread-safe Queue) means the
capture/display loop is never blocked by network I/O, so video stays live
continuously while reports are generated and stored in the background.

Run with:
    python main.py

Stop with Ctrl+C at any time; the application shuts down cleanly, releasing
the camera/video resources, draining any in-flight report, and printing a
final summary.
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import types
from typing import Optional

import cv2

from aggregator import EventAggregator
from config import Config
from llm_reporter import LLMReporterError, MistralReporter
from schemas import IncidentReport, WindowSummary
from storage import ReportStorage
from vision import Detector, VideoSource, VideoSourceError

# Sentinel used to tell the report worker thread to stop.
_SHUTDOWN_SENTINEL = object()


class ReportWorker:
    """
    Runs on a background thread, consuming WindowSummary objects from a
    queue, calling the (potentially slow) Mistral API, and persisting the
    result -- entirely off the video capture/render thread.
    """

    def __init__(self, reporter: MistralReporter, storage: ReportStorage, max_queue_size: int) -> None:
        self.reporter = reporter
        self.storage = storage
        self.queue: "queue.Queue" = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._run, name="ReportWorker", daemon=True)
        self._reports_generated = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread.start()

    def submit(self, summary: WindowSummary) -> None:
        """Non-blocking enqueue. If the queue is full (LLM is falling behind
        the video), drop the oldest pending window with a warning instead of
        blocking the video loop -- staying live is more important than
        reporting on every single window when the backlog builds up."""
        try:
            self.queue.put_nowait(summary)
        except queue.Full:
            try:
                dropped = self.queue.get_nowait()
                print(
                    f"[ReportWorker] Report queue full; dropping oldest pending window "
                    f"({dropped.window_start.strftime('%H:%M:%S')}-{dropped.window_end.strftime('%H:%M:%S')}) "
                    f"so the video feed keeps running smoothly."
                )
                self.queue.put_nowait(summary)
            except queue.Empty:
                pass

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is _SHUTDOWN_SENTINEL:
                self.queue.task_done()
                break
            self._process(item)
            self.queue.task_done()

    def _process(self, summary: WindowSummary) -> None:
        report = self.reporter.generate_report(summary)
        with self._lock:
            self._reports_generated += 1
            count = self._reports_generated

        row_id = self.storage.save(report, summary)

        time_label = "video time" if report.time_source == "video" else "real time"
        print("=" * 72)
        print(
            f"INCIDENT REPORT #{count} "
            f"(stored as row id {row_id}) "
            f"[{report.window_start.strftime('%H:%M:%S')} - {report.window_end.strftime('%H:%M:%S')} ({time_label})]"
        )
        print(f"Peak simultaneous person count: {report.peak_person_count}")
        if report.error:
            print(f"(Note: LLM call failed, showing fallback summary. Error: {report.error})")
        print("-- Person report " + "-" * 54)
        print(report.person_summary_text or "(no person data)")
        print("-- Other objects report " + "-" * 47)
        print(report.other_summary_text or "(no other-object data)")
        print("=" * 72 + "\n")

    def reports_generated(self) -> int:
        with self._lock:
            return self._reports_generated

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Signals the worker to stop and optionally waits for the queue to drain."""
        if wait:
            print(f"[ReportWorker] Waiting for {self.queue.qsize()} pending report(s) to finish...")
            self.queue.join()
        self.queue.put(_SHUTDOWN_SENTINEL)
        self._thread.join(timeout=timeout)


class Application:
    """Owns the full pipeline lifecycle: setup, main loop, and clean shutdown."""

    def __init__(self) -> None:
        self._running = True
        self._total_frames_processed = 0

        self.video_source: Optional[VideoSource] = None
        self.detector: Optional[Detector] = None
        self.aggregator: Optional[EventAggregator] = None
        self.reporter: Optional[MistralReporter] = None
        self.storage: Optional[ReportStorage] = None
        self.report_worker: Optional[ReportWorker] = None

    def setup(self) -> None:
        """Initializes every pipeline component, failing fast with clear errors."""
        Config.validate()

        try:
            self.video_source = VideoSource()
        except VideoSourceError as exc:
            print(f"[Application] Fatal: could not initialize any video source: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            self.detector = Detector(
                model_path=Config.YOLO_MODEL_PATH,
                confidence_threshold=Config.YOLO_CONFIDENCE_THRESHOLD,
                iou_threshold=Config.YOLO_IOU_THRESHOLD,
                img_size=Config.YOLO_IMG_SIZE,
                target_classes=Config.TARGET_CLASSES,
                enable_tracking=Config.ENABLE_TRACKING,
                tracker_config=Config.TRACKER_CONFIG,
            )
            print(
                f"[Application] Detector ready. Model: '{Config.YOLO_MODEL_PATH}', "
                f"classes: {Config.TARGET_CLASSES}, tracking: {Config.ENABLE_TRACKING}."
            )
        except RuntimeError as exc:
            print(f"[Application] Fatal: {exc}", file=sys.stderr)
            sys.exit(1)

        self.aggregator = EventAggregator(
            window_seconds=Config.WINDOW_SECONDS,
            window_max_frames=Config.WINDOW_MAX_FRAMES,
            time_source=self.video_source.time_source,
        )
        print(
            f"[Application] Time source for reports: '{self.video_source.time_source}' "
            f"({'elapsed video time' if self.video_source.time_source == 'video' else 'real-world wall-clock time'})."
        )

        try:
            self.reporter = MistralReporter(
                api_key=Config.MISTRAL_API_KEY,
                model=Config.MISTRAL_MODEL,
            )
        except LLMReporterError as exc:
            print(f"[Application] Fatal: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            self.storage = ReportStorage(db_path=Config.REPORTS_DB_PATH)
            existing_count = self.storage.count()
            print(
                f"[Application] Report storage ready at '{Config.REPORTS_DB_PATH}' "
                f"({existing_count} report(s) already stored)."
            )
        except RuntimeError as exc:
            print(f"[Application] Fatal: {exc}", file=sys.stderr)
            sys.exit(1)

        # Reports are generated on a background thread so a slow/blocking
        # Mistral API call can never freeze the live video window.
        self.report_worker = ReportWorker(
            reporter=self.reporter,
            storage=self.storage,
            max_queue_size=Config.REPORT_QUEUE_MAX_SIZE,
        )
        self.report_worker.start()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        print("[Application] Setup complete. Starting main loop. Press Ctrl+C (or 'q' in the video window) to stop.\n")

    def _handle_interrupt(self, signum: int, frame: Optional[types.FrameType]) -> None:
        print(f"\n[Application] Received signal {signum}. Shutting down gracefully...")
        self._running = False

    def run(self) -> None:
        """Main video processing loop. Never blocks on network I/O."""
        assert self.video_source is not None
        assert self.detector is not None
        assert self.aggregator is not None
        assert self.report_worker is not None

        try:
            for frame, frame_index, timestamp in self.video_source.frames():
                if not self._running:
                    break

                detections = self.detector.detect(frame, frame_index, timestamp)
                self.aggregator.add_detections(detections, timestamp)
                self._total_frames_processed += 1

                if Config.SHOW_VIDEO_WINDOW:
                    self._render_frame(frame, detections)

                if self.aggregator.is_window_complete(timestamp):
                    self._flush_and_submit(timestamp)

                # Allow the OpenCV window (if shown) to process UI events and
                # let the user quit early by pressing 'q'. This runs every
                # frame regardless of report activity, so the window stays
                # responsive at all times.
                if Config.SHOW_VIDEO_WINDOW:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[Application] 'q' pressed. Shutting down...")
                        break

        except Exception as exc:  # noqa: BLE001
            print(f"[Application] Unexpected error in main loop: {exc}", file=sys.stderr)
        finally:
            self._shutdown()

    def _render_frame(self, frame, detections) -> None:
        try:
            annotated = self.detector.draw_detections(frame, detections)
            cv2.imshow("YOLO + Mistral Fusion - Explain What You See", annotated)
        except Exception as exc:  # noqa: BLE001
            # Rendering failures (e.g., headless environment with no display)
            # should never crash the detection/reporting pipeline.
            print(f"[Application] Could not render video window (continuing headless): {exc}")
            Config.SHOW_VIDEO_WINDOW = False

    def _flush_and_submit(self, current_timestamp) -> None:
        """Flushes the current window and hands it to the background report
        worker. This call itself is essentially instant (no network I/O), so
        the video loop keeps running at full frame rate."""
        summary = self.aggregator.flush(current_timestamp)

        if summary.is_empty():
            print(
                f"[{summary.window_end.strftime('%H:%M:%S')}] "
                f"No objects detected in this {Config.WINDOW_SECONDS:.0f}s window "
                f"({summary.total_frames_processed} frames)."
            )
            if Config.STORE_EMPTY_WINDOWS:
                empty_report = IncidentReport(
                    window_start=summary.window_start,
                    window_end=summary.window_end,
                    summary_text="No activity detected during this monitoring window.",
                    person_summary_text="No people were detected during this monitoring window.",
                    other_summary_text="No other objects were detected during this monitoring window.",
                    raw_event_count=0,
                    peak_person_count=0,
                    time_source=summary.time_source,
                    model_used="none",
                    error=None,
                )
                self.storage.save(empty_report, summary)
            return

        self.report_worker.submit(summary)

    def _shutdown(self) -> None:
        if self.video_source is not None:
            self.video_source.release()
        if Config.SHOW_VIDEO_WINDOW:
            cv2.destroyAllWindows()

        reports_generated = 0
        if self.report_worker is not None:
            self.report_worker.shutdown(wait=True, timeout=15.0)
            reports_generated = self.report_worker.reports_generated()

        print("\n[Application] Shutdown complete.")
        print(f"[Application] Total frames processed: {self._total_frames_processed}")
        print(f"[Application] Total incident reports generated: {reports_generated}")
        if self.storage is not None:
            print(f"[Application] Total reports now stored in '{Config.REPORTS_DB_PATH}': {self.storage.count()}")

            if Config.EXPORT_PDF_ON_EXIT:
                self._export_pdfs()

    def _export_pdfs(self) -> None:
        """Writes the two PDF deliverables (person-only, other-objects) on shutdown."""
        try:
            import os

            from pdf_report import generate_objects_pdf, generate_person_pdf

            os.makedirs(Config.REPORTS_PDF_DIR, exist_ok=True)
            person_path = os.path.join(Config.REPORTS_PDF_DIR, "person_report.pdf")
            objects_path = os.path.join(Config.REPORTS_PDF_DIR, "other_objects_report.pdf")

            _, person_count = generate_person_pdf(self.storage, person_path)
            _, objects_count = generate_objects_pdf(self.storage, objects_path)

            print(
                f"[Application] Wrote person PDF ({person_count} window(s)) to '{person_path}' "
                f"and other-objects PDF ({objects_count} window(s)) to '{objects_path}'."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Application] Could not export PDF reports: {exc}")


def main() -> None:
    app = Application()
    app.setup()
    app.run()


if __name__ == "__main__":
    main()
