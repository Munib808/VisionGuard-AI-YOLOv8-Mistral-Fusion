"""
flask_app.py

Professional Flask web interface for the YOLO + Mistral Fusion pipeline.

Flow:
  1. POST /api/upload           -> saves the uploaded video, starts a
                                    background thread running pipeline.process_video(),
                                    returns {job_id}.
  2. GET  /api/status/<job_id>  -> polling endpoint: {state, progress, message, result?}
  3. GET  /media/<job_id>/video -> streams the annotated (bounding-boxes-drawn) video
  4. GET  /api/pdf/<job_id>/<kind> -> downloads the person or objects PDF
                                       (generated on first request, cached after)

Run with:
    python flask_app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import threading
import traceback
from typing import Dict

from flask import Flask, Response, jsonify, render_template, request, send_file

import pipeline
from shared_ui import CSS, header_html

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB upload cap

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory job registry. Fine for a single-process demo/dev server; for
# production use, back this with Redis or a DB instead.
JOBS: Dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _set_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kwargs)


def _get_job(job_id: str) -> dict:
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


def _run_pipeline_in_background(job_id: str, video_path: str) -> None:
    def on_progress(frac: float, msg: str) -> None:
        _set_job(job_id, state="processing", progress=frac, message=msg)

    try:
        _set_job(job_id, state="processing", progress=0.0, message="Starting...")
        result = pipeline.process_video(video_path, job_id=job_id, on_progress=on_progress)
        _set_job(
            job_id,
            state="done",
            progress=1.0,
            message="Analysis complete.",
            result={
                "job_id": result.job_id,
                "total_frames": result.total_frames,
                "fps": round(result.fps, 2),
                "duration_seconds": round(result.duration_seconds, 1),
                "peak_person_count": result.peak_person_count,
                "peak_object_classes": result.peak_object_classes,
                "total_windows": result.total_windows,
                "reports": result.reports,
                "warnings": result.warnings,
            },
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _set_job(job_id, state="error", progress=0.0, message=str(exc))


@app.route("/")
def index():
    return render_template("index.html", header_html=header_html("Video Intelligence"))


@app.route("/static/style.css")
def style_css():
    return Response(CSS, mimetype="text/css")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided under field name 'video'."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    job_id = pipeline.new_job_id()
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    file.save(saved_path)

    _set_job(job_id, state="queued", progress=0.0, message="Queued for processing.")
    thread = threading.Thread(
        target=_run_pipeline_in_background, args=(job_id, saved_path), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id."}), 404
    return jsonify(job)


@app.route("/media/<job_id>/video")
def media_video(job_id: str):
    path = os.path.join(pipeline.job_dir_for(job_id), "annotated.mp4")
    if not os.path.isfile(path):
        return jsonify({"error": "Annotated video not ready yet."}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/api/pdf/<job_id>/<kind>")
def pdf(job_id: str, kind: str):
    if kind not in ("person", "objects"):
        return jsonify({"error": "kind must be 'person' or 'objects'."}), 400

    jdir = pipeline.job_dir_for(job_id)
    db_path = os.path.join(jdir, "reports.db")
    if not os.path.isfile(db_path):
        return jsonify({"error": "No reports found for this job."}), 404

    pdf_paths = pipeline.build_pdfs(db_path, os.path.join(jdir, "pdf"))
    filename = "person_report.pdf" if kind == "person" else "other_objects_report.pdf"
    return send_file(pdf_paths[kind], mimetype="application/pdf", as_attachment=True, download_name=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
