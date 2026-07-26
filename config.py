"""
config.py

Centralized configuration for the YOLO + Mistral Fusion application.
Loads secrets from a local .env file via python-dotenv and exposes
typed, validated settings for the rest of the application to consume.
"""

from __future__ import annotations

import os
import sys
from typing import List

from dotenv import load_dotenv

# Load variables from a .env file in the current working directory, if present.
load_dotenv()


class Config:
    """Static configuration container. Values are read once at import time."""

    # --- Mistral API settings ---
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "").strip()
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()

    # --- YOLO settings ---
    # yolov8n.pt is fastest but least accurate. yolov8s.pt (small) is a good
    # accuracy/speed tradeoff for person detection on CPU; use yolov8m.pt if
    # you have a GPU and want even better accuracy.
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt").strip()
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
    # IoU threshold used by non-max suppression; lower = fewer duplicate/overlapping boxes.
    YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    # Inference resolution. Larger = more accurate, slower. Must be a multiple of 32.
    YOLO_IMG_SIZE: int = int(os.getenv("YOLO_IMG_SIZE", "640"))

    # --- Class filtering ---
    # Comma-separated list of COCO class names to detect, e.g. "person,car,dog".
    # Leave EMPTY (the default) to detect ALL 80 COCO classes -- this is what
    # lets the pipeline report on people AND every other object type (cars,
    # bags, animals, etc.), which is then split into two separate reports
    # (person-only vs. everything else) downstream.
    TARGET_CLASSES: List[str] = [
        c.strip() for c in os.getenv("TARGET_CLASSES", "").split(",") if c.strip()
    ]

    # The one class name that is always broken out into its own "person"
    # report/PDF; every other detected class goes into the "other objects"
    # report/PDF. Changing this is not recommended/supported.
    PERSON_CLASS_NAME: str = "person"

    # --- Tracking (gives every detected person a stable ID across frames) ---
    # This is what makes "2 people" possible to report correctly: without a
    # tracker, the pipeline only knows "some person-shaped box appeared N
    # times", not "how many distinct people were on screen".
    ENABLE_TRACKING: bool = os.getenv("ENABLE_TRACKING", "true").strip().lower() == "true"
    TRACKER_CONFIG: str = os.getenv("TRACKER_CONFIG", "bytetrack.yaml").strip()

    # --- Video source settings ---
    # Camera index used for cv2.VideoCapture. Ignored if FALLBACK_VIDEO_PATH is forced.
    WEBCAM_INDEX: int = int(os.getenv("WEBCAM_INDEX", "0"))
    # Optional path to a sample video file used if the webcam cannot be opened.
    FALLBACK_VIDEO_PATH: str = os.getenv("FALLBACK_VIDEO_PATH", "sample_video.mp4").strip()
    # Controls which source is used:
    #   "auto"      -> try webcam, then FALLBACK_VIDEO_PATH, then synthetic (default)
    #   "file"      -> skip the webcam entirely and use FALLBACK_VIDEO_PATH directly
    #   "webcam"    -> only try the webcam (no file/synthetic fallback)
    #   "synthetic" -> skip webcam and file, go straight to the synthetic generator
    VIDEO_SOURCE_MODE: str = os.getenv("VIDEO_SOURCE_MODE", "auto").strip().lower()

    # --- Aggregation window settings ---
    # A report is generated whichever of these two limits is hit first.
    WINDOW_SECONDS: float = float(os.getenv("WINDOW_SECONDS", "10.0"))
    WINDOW_MAX_FRAMES: int = int(os.getenv("WINDOW_MAX_FRAMES", "300"))

    # --- Display settings ---
    SHOW_VIDEO_WINDOW: bool = os.getenv("SHOW_VIDEO_WINDOW", "true").strip().lower() == "true"

    # --- Background reporting settings ---
    # The Mistral API call + DB write happen on a background thread so the
    # video window never freezes waiting on the network. This caps how many
    # pending windows can queue up if the LLM is slow, so memory can't grow
    # unbounded; if the queue is full, the oldest window is dropped with a
    # warning rather than blocking the video loop.
    REPORT_QUEUE_MAX_SIZE: int = int(os.getenv("REPORT_QUEUE_MAX_SIZE", "5"))

    # --- Report storage settings ---
    # SQLite database file where every generated IncidentReport is persisted.
    REPORTS_DB_PATH: str = os.getenv("REPORTS_DB_PATH", "incident_reports.db").strip()
    # Whether to also persist "no activity detected" windows (they have no
    # LLM-generated text, just a placeholder). Off by default to avoid
    # bloating the database with empty rows.
    STORE_EMPTY_WINDOWS: bool = os.getenv("STORE_EMPTY_WINDOWS", "false").strip().lower() == "true"

    # --- PDF export settings ---
    # Directory where "python view_reports.py pdf" writes the two PDF
    # reports (person-only, and all-other-objects). Companies asked for PDF
    # deliverables specifically, so this is separate from the JSONL export.
    REPORTS_PDF_DIR: str = os.getenv("REPORTS_PDF_DIR", "reports_pdf").strip()
    # If true, main.py automatically writes both PDFs on clean shutdown, in
    # addition to being generatable on demand via view_reports.py.
    EXPORT_PDF_ON_EXIT: bool = os.getenv("EXPORT_PDF_ON_EXIT", "false").strip().lower() == "true"

    @classmethod
    def validate(cls) -> None:
        """
        Validate required configuration and fail fast with a clear message
        rather than letting the program crash deep inside the Mistral SDK.
        """
        problems = []

        if not cls.MISTRAL_API_KEY:
            problems.append(
                "MISTRAL_API_KEY is not set. Create a .env file with "
                "MISTRAL_API_KEY=your_key_here, or export it as an environment variable."
            )

        if cls.WINDOW_SECONDS <= 0:
            problems.append("WINDOW_SECONDS must be a positive number.")

        if cls.WINDOW_MAX_FRAMES <= 0:
            problems.append("WINDOW_MAX_FRAMES must be a positive integer.")

        if cls.YOLO_CONFIDENCE_THRESHOLD < 0 or cls.YOLO_CONFIDENCE_THRESHOLD > 1:
            problems.append("YOLO_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0.")

        # Note: TARGET_CLASSES is allowed to be empty on purpose -- it means
        # "detect all 80 COCO classes" (see vision.Detector), not an error.

        valid_modes = {"auto", "file", "webcam", "synthetic"}
        if cls.VIDEO_SOURCE_MODE not in valid_modes:
            problems.append(
                f"VIDEO_SOURCE_MODE must be one of {sorted(valid_modes)}, got '{cls.VIDEO_SOURCE_MODE}'."
            )

        if cls.VIDEO_SOURCE_MODE == "file" and not os.path.isfile(cls.FALLBACK_VIDEO_PATH):
            problems.append(
                f"VIDEO_SOURCE_MODE is 'file' but FALLBACK_VIDEO_PATH "
                f"('{cls.FALLBACK_VIDEO_PATH}') does not exist. Check the path in your .env file."
            )

        if problems:
            print("Configuration error(s) detected:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            sys.exit(1)