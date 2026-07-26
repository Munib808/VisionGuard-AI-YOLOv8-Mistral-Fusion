"""
vision.py

Handles all computer-vision concerns:
  - VideoSource: opens a webcam, falls back to a sample video file, and
    falls back again to a synthetic frame generator so the application
    can always run end-to-end even with no camera or sample file present.
  - Detector: wraps an Ultralytics YOLO model and converts raw model
    output into validated Detection objects.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Generator, List, Optional, Tuple  # noqa: F401

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config
from schemas import Detection


class VideoSourceError(Exception):
    """Raised when no usable video source could be initialized at all."""


class VideoSource:
    """
    Provides a unified frame generator regardless of where frames come from.

    Resolution order:
      1. Webcam at Config.WEBCAM_INDEX
      2. Video file at Config.FALLBACK_VIDEO_PATH
      3. Synthetic procedurally generated frames (so the app never hard-fails
         just because no camera or sample video is available in this
         environment).

    Timestamps and `time_source`
    -----------------------------
    Reports need to know whether a timestamp means "this many seconds into
    the uploaded video" or "the real-world wall-clock time it was seen":
      - mode == "file"  -> time_source = "video".  Timestamps are computed
        purely from frame_index / fps, counting up from 00:00:00 at the
        start of the video, regardless of how fast/slow the machine actually
        processes frames. This is what makes an uploaded video's report
        times match the video's own timeline instead of however long the
        analysis happened to take on this machine.
      - mode == "webcam" -> time_source = "realtime". Timestamps are the
        actual wall-clock time (datetime.now()) each frame was captured,
        since a live camera has no other meaningful timeline.
      - mode == "synthetic" -> time_source = "realtime" (there's no source
        video to have its own timeline, so wall-clock is the only sensible
        choice here too).
    """

    # Fixed epoch used purely as an anchor for "video time" -- only the
    # elapsed duration since this point matters, never the absolute date.
    _VIDEO_TIME_EPOCH = datetime(2000, 1, 1)

    def __init__(self) -> None:
        self.capture: Optional[cv2.VideoCapture] = None
        self.mode: str = "uninitialized"
        self.time_source: str = "realtime"
        self.fps: float = 30.0
        self._synthetic_frame_index: int = 0
        self._open_source()

    def _open_source(self) -> None:
        mode = Config.VIDEO_SOURCE_MODE

        # --- Explicit "file" mode: skip webcam entirely, use FALLBACK_VIDEO_PATH directly. ---
        if mode == "file":
            if not os.path.isfile(Config.FALLBACK_VIDEO_PATH):
                raise VideoSourceError(
                    f"VIDEO_SOURCE_MODE is 'file' but the file "
                    f"'{Config.FALLBACK_VIDEO_PATH}' was not found."
                )
            cap = cv2.VideoCapture(Config.FALLBACK_VIDEO_PATH)
            if cap is None or not cap.isOpened():
                raise VideoSourceError(
                    f"VIDEO_SOURCE_MODE is 'file' but OpenCV could not open "
                    f"'{Config.FALLBACK_VIDEO_PATH}'. The file may be corrupt or in an "
                    f"unsupported codec."
                )
            self.capture = cap
            self.mode = "file"
            self.time_source = "video"
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(
                f"[VideoSource] VIDEO_SOURCE_MODE=file: using '{Config.FALLBACK_VIDEO_PATH}'. "
                "Report timestamps will reflect elapsed VIDEO time, not wall-clock time."
            )
            return

        # --- Explicit "synthetic" mode: skip webcam and file entirely. ---
        if mode == "synthetic":
            self.capture = None
            self.mode = "synthetic"
            self.time_source = "realtime"
            self.fps = 30.0
            print("[VideoSource] VIDEO_SOURCE_MODE=synthetic: using the synthetic frame generator.")
            return

        # --- Explicit "webcam" mode: only try the webcam, no fallback. ---
        if mode == "webcam":
            cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self.capture = cap
                    self.mode = "webcam"
                    self.time_source = "realtime"
                    self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    print(
                        f"[VideoSource] VIDEO_SOURCE_MODE=webcam: using camera index {Config.WEBCAM_INDEX}. "
                        "Report timestamps will reflect real-world wall-clock time."
                    )
                    return
                cap.release()
            raise VideoSourceError(
                f"VIDEO_SOURCE_MODE is 'webcam' but camera index {Config.WEBCAM_INDEX} "
                "could not be opened or read from."
            )

        # --- "auto" mode (default): webcam -> file -> synthetic, in that order. ---
        # 1. Try webcam first.
        try:
            cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self.capture = cap
                    self.mode = "webcam"
                    self.time_source = "realtime"
                    self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    print(
                        f"[VideoSource] Using webcam index {Config.WEBCAM_INDEX}. "
                        "Report timestamps will reflect real-world wall-clock time."
                    )
                    return
                cap.release()
        except Exception as exc:  # noqa: BLE001 - we deliberately want to continue to fallback
            print(f"[VideoSource] Webcam initialization raised an exception: {exc}")

        # 2. Try fallback video file.
        try:
            if Config.FALLBACK_VIDEO_PATH and os.path.isfile(Config.FALLBACK_VIDEO_PATH):
                cap = cv2.VideoCapture(Config.FALLBACK_VIDEO_PATH)
                if cap is not None and cap.isOpened():
                    self.capture = cap
                    self.mode = "file"
                    self.time_source = "video"
                    self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    print(
                        f"[VideoSource] Using fallback video file '{Config.FALLBACK_VIDEO_PATH}'. "
                        "Report timestamps will reflect elapsed VIDEO time, not wall-clock time."
                    )
                    return
                if cap is not None:
                    cap.release()
            else:
                print(
                    f"[VideoSource] Fallback video path '{Config.FALLBACK_VIDEO_PATH}' "
                    "not found; skipping to synthetic frame generator."
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[VideoSource] Fallback video file initialization raised an exception: {exc}")

        # 3. Synthetic frame generator: guarantees the pipeline can always run.
        self.capture = None
        self.mode = "synthetic"
        self.time_source = "realtime"
        self.fps = 30.0
        print(
            "[VideoSource] No webcam or sample video available. "
            "Falling back to a synthetic frame generator so the pipeline can still run."
        )

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Produces a simple procedurally animated frame (moving rectangle on a
        gray background) so the rest of the pipeline has real image data to
        run YOLO inference on, even with zero real video sources available.
        """
        height, width = 480, 640
        frame = np.full((height, width, 3), 60, dtype=np.uint8)

        # Animate a rectangle moving across the frame to keep frames non-identical.
        offset = (self._synthetic_frame_index * 4) % (width - 100)
        cv2.rectangle(frame, (offset, 150), (offset + 80, 330), (90, 160, 90), thickness=-1)
        cv2.putText(
            frame,
            "SYNTHETIC FEED (no camera/video found)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._synthetic_frame_index += 1
        return frame

    def _frame_timestamp(self, frame_index: int) -> datetime:
        """
        Returns the timestamp to attach to a given frame, computed according
        to `self.time_source`:
          - "video": elapsed video time (frame_index / fps) counted up from a
            fixed epoch, so it always matches the video's own timeline no
            matter how fast/slow this machine processes frames.
          - "realtime": actual wall-clock time (datetime.now()).
        """
        if self.time_source == "video":
            fps = self.fps if self.fps and self.fps > 0 else 30.0
            elapsed_seconds = frame_index / fps
            return self._VIDEO_TIME_EPOCH + timedelta(seconds=elapsed_seconds)
        return datetime.now()

    def frames(self) -> Generator[Tuple[np.ndarray, int, datetime], None, None]:
        """
        Yields (frame, frame_index, timestamp) tuples indefinitely for webcam
        and synthetic sources, or until exhausted for a file source. See
        `_frame_timestamp` / the class docstring for what `timestamp` means
        depending on `self.time_source`.
        """
        frame_index = 0
        while True:
            if self.mode == "synthetic":
                frame = self._generate_synthetic_frame()
                yield frame, frame_index, self._frame_timestamp(frame_index)
                frame_index += 1
                time.sleep(1.0 / self.fps)
                continue

            if self.capture is None:
                raise VideoSourceError("Video capture is not initialized.")

            ok, frame = self.capture.read()
            if not ok:
                if self.mode == "file":
                    print("[VideoSource] End of video file reached.")
                    return
                print("[VideoSource] Failed to read frame from webcam; stopping capture.")
                return

            yield frame, frame_index, self._frame_timestamp(frame_index)
            frame_index += 1

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


class Detector:
    """
    Wraps an Ultralytics YOLO model and returns validated Detection objects,
    restricted to a configured set of classes (by default: only 'person').

    Two things matter for accuracy here beyond just "run the model":
      1. Restricting `classes=[...]` at inference time, not just filtering
         the output afterwards. This avoids wasting NMS/confidence budget on
         classes we don't care about and prevents any chance of a mislabeled
         non-person box leaking into the results.
      2. Using the model's built-in tracker (`model.track(..., persist=True)`)
         instead of `model.predict(...)`. Tracking assigns a stable ID to
         each person across frames, which is what lets the aggregator later
         tell the difference between "the same person seen 40 times" and
         "2 different people seen at once".
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        target_classes: Optional[List[str]] = None,
        enable_tracking: bool = True,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.enable_tracking = enable_tracking
        self.tracker_config = tracker_config

        try:
            self.model = YOLO(model_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load YOLO model from '{model_path}'. "
                f"Ensure the weights file is available or downloadable. Original error: {exc}"
            ) from exc

        # Resolve requested class names (e.g. "person", "car") to the model's
        # internal class IDs. Fail fast with a clear message if a name
        # doesn't exist in this model's label set (e.g. a typo).
        #
        # If no target_classes are given, self.class_ids is left as None,
        # which means "don't restrict inference at all" -- YOLO will detect
        # every one of its ~80 COCO classes (person + all 79 other object
        # types), not just person. This is what lets the pipeline generate
        # both a person report and a separate "everything else" report.
        self.class_ids: Optional[List[int]] = None
        if target_classes:
            name_to_id = {name: cid for cid, name in self.model.names.items()}
            self.class_ids = []
            for name in target_classes:
                if name not in name_to_id:
                    available = ", ".join(sorted(name_to_id.keys()))
                    raise RuntimeError(
                        f"TARGET_CLASSES contains '{name}', which is not a class this model "
                        f"knows about. Available classes: {available}"
                    )
                self.class_ids.append(name_to_id[name])

        self._tracking_available = enable_tracking

    def detect(self, frame: np.ndarray, frame_index: int, timestamp: datetime) -> List[Detection]:
        """Runs inference (with tracking, if enabled) on a single frame."""
        results = self._run_inference(frame, frame_index)
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections: List[Detection] = []
        for box in result.boxes:
            try:
                class_id = int(box.cls[0])
                class_name = self.model.names.get(class_id, f"class_{class_id}")
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                track_id: Optional[int] = None
                if getattr(box, "id", None) is not None:
                    track_id = int(box.id[0])

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        track_id=track_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[Detector] Skipping a malformed detection box: {exc}")
                continue

        return detections

    def _run_inference(self, frame: np.ndarray, frame_index: int):
        common_kwargs = dict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
        )
        # Only pass `classes=` when we actually want to restrict inference;
        # passing `classes=None` explicitly behaves correctly in Ultralytics
        # too, but omitting the key entirely is the most defensive choice
        # across model/SDK versions.
        if self.class_ids is not None:
            common_kwargs["classes"] = self.class_ids

        if self._tracking_available:
            try:
                return self.model.track(
                    persist=True,
                    tracker=self.tracker_config,
                    **common_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                # Tracking can fail to initialize in some environments (e.g.
                # missing lap/cython-bbox). Fall back to plain detection
                # rather than crashing the whole pipeline; unique-person
                # counting degrades gracefully (see aggregator.py).
                print(
                    f"[Detector] Tracking failed on frame {frame_index}, "
                    f"falling back to detection-only mode: {exc}"
                )
                self._tracking_available = False

        try:
            return self.model.predict(**common_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[Detector] YOLO inference failed on frame {frame_index}: {exc}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draws bounding boxes, labels, and track IDs on a copy of the frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            id_part = f"#{det.track_id} " if det.track_id is not None else ""
            label = f"{id_part}{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

        person_count = sum(1 for det in detections if det.class_name == "person")
        count_label = f"Objects: {len(detections)}  (person: {person_count})"
        cv2.putText(
            annotated,
            count_label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated