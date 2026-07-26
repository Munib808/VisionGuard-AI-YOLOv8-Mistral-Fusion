# VisionGuard AI — YOLOv8 × Mistral Fusion

**Real-time object detection and AI-generated incident reports.**
YOLOv8 detection + ByteTrack tracking feed a Mistral LLM that writes
natural-language incident reports every few seconds, exportable as PDF —
available as a CLI pipeline, a Flask web app, and a Streamlit app.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![Mistral](https://img.shields.io/badge/LLM-Mistral-red)

---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Pipeline](#1-cli-pipeline)
  - [Flask Web App](#2-flask-web-app)
  - [Streamlit App](#3-streamlit-app)
  - [Exporting PDF Reports](#4-exporting-pdf-reports)
- [Configuration](#configuration)
- [Design Notes](#design-notes)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [License](#license)

---

## Overview

VisionGuard AI watches a video feed (webcam, uploaded file, or a synthetic
fallback), detects and tracks every one of the 80 COCO object classes with
YOLOv8 + ByteTrack, and batches detections into rolling time windows. Each
window is summarized by Mistral into two scoped reports — **person activity**
and **other objects** — which are stored in SQLite and can be exported to PDF
at any time. Two ready-made front-ends (Flask and Streamlit) let you upload a
video, watch detections drawn live, and download the reports as PDF, all in
a shared, professional UI.

## Features

- 🎯 **80-class detection + tracking** via YOLOv8 and ByteTrack, with correct
  peak-concurrent and unique-individual counts per class
- 🧠 **LLM incident reports**, split per-window into person / other-objects
  sections, with a deterministic offline fallback if no API key is set
- 🕒 **Accurate timestamps** — video-elapsed time for uploaded files, real
  wall-clock time for a live feed
- 🧵 **Non-blocking pipeline** — LLM calls and DB writes run on a background
  worker so the video never freezes
- 📄 **PDF export** — separate Person Detection and Other Objects reports
- 🌐 **Two matching web UIs** (Flask, Streamlit) sharing one design system,
  each with drag-and-drop upload, live progress, an annotated video player,
  a metrics dashboard, and one-click PDF downloads
- 🛡️ **Resilient by design** — graceful fallbacks for missing webcam,
  missing tracker deps, and missing/failed LLM calls

## Architecture

```
Video source (webcam / file / synthetic)
        │
        ▼
  Detector (YOLOv8 + ByteTrack)  ──► annotated frames (drawn boxes)
        │
        ▼
  EventAggregator (rolling window: N seconds / M frames)
        │
        ▼
  MistralReporter  ──► IncidentReport (person + other-objects text)
        │
        ▼
  ReportStorage (SQLite)  ──► view_reports.py / pdf_report.py (PDF export)
```

`pipeline.py` wires this whole chain into a single reusable function
(`process_video`) called identically by `main.py` (CLI), `flask_app.py`, and
`streamlit_app.py` — so all three interfaces behave exactly the same way.

## Project Structure

```
├── main.py             # CLI entrypoint (live webcam / file loop)
├── pipeline.py          # Shared upload → detect → aggregate → report pipeline
├── config.py             # Environment configuration loader
├── schemas.py            # Pydantic models (Detection, WindowSummary, IncidentReport...)
├── vision.py              # VideoSource + Detector (YOLO + ByteTrack wrapper)
├── aggregator.py          # Rolling-window detection aggregation
├── llm_reporter.py        # Mistral prompt + response parsing
├── storage.py             # SQLite persistence
├── pdf_report.py          # PDF report generation
├── view_reports.py        # CLI: browse / search / export / PDF stored reports
├── shared_ui.py           # Shared design system (CSS/branding) for both web apps
├── flask_app.py           # Flask web app (REST API + HTML/JS UI)
├── streamlit_app.py       # Streamlit web app
├── templates/index.html   # Flask front-end page
├── requirements.txt
└── .env.example
```

## Installation

```bash
git clone <your-repo-url>
cd visionguard-ai
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then add your MISTRAL_API_KEY
```

Get a Mistral API key at [console.mistral.ai](https://console.mistral.ai/).
The app works without one too — LLM reports fall back to deterministic
offline text.

## Usage

### 1. CLI Pipeline

```bash
python main.py
```
Opens a live annotated video window (webcam → fallback video → synthetic
feed, in that order) and prints an incident report to the console every
`WINDOW_SECONDS`. Press `q` in the video window or `Ctrl+C` to stop.

### 2. Flask Web App

```bash
python flask_app.py
# open http://127.0.0.1:5000
```
Upload a video, watch progress live, view the annotated result inline, and
download both PDF reports.

### 3. Streamlit App

```bash
streamlit run streamlit_app.py
```
Same workflow and design as the Flask app, as a single Python script.

### 4. Exporting PDF Reports

```bash
python view_reports.py pdf
# -> reports_pdf/person_report.pdf
# -> reports_pdf/other_objects_report.pdf
```
Set `EXPORT_PDF_ON_EXIT=true` in `.env` to auto-export on every clean CLI
shutdown, or use the **Generate PDF** buttons in either web app.

## Configuration

All settings live in `.env` (see `.env.example`). Key variables:

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | *(optional)* | Enables LLM reports; falls back to offline text if unset |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral model used for reports |
| `YOLO_MODEL_PATH` | `yolov8s.pt` | YOLO weights — use `yolov8n.pt` for speed, `yolov8m.pt`+ for accuracy |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `TARGET_CLASSES` | *(all 80)* | Comma-separated COCO classes to restrict to, e.g. `person,car` |
| `ENABLE_TRACKING` | `true` | Enables ByteTrack for stable object IDs |
| `VIDEO_SOURCE_MODE` | `auto` | `auto` / `file` / `webcam` / `synthetic` |
| `WINDOW_SECONDS` | `3.0` | Max seconds of detections per report window |
| `SHOW_VIDEO_WINDOW` | `true` | Show the live OpenCV window (CLI mode) |
| `REPORTS_PDF_DIR` | `reports_pdf` | Output folder for exported PDFs |

Full reference is documented inline in `.env.example`.

## Design Notes

- **Aggregation over per-frame LLM calls** — one Mistral call per time
  window, not per frame, keeps latency and cost sane.
- **Background worker thread** — the LLM call and SQLite write happen off
  the video loop, so frame rate never drops while waiting on the network.
- **Tracking, not just detection** — `model.track(persist=True)` gives every
  object a stable ID across frames, enabling correct peak-concurrent and
  unique-individual counts instead of naive per-frame box counts.
- **One pipeline, three interfaces** — `pipeline.py` and `shared_ui.py` are
  the single source of truth so the CLI, Flask, and Streamlit experiences
  never drift apart.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'Mistral' from 'mistralai'` | `pip install "mistralai<2.0.0" --force-reinstall` |
| YOLO weights fail to download | Check internet access, or manually place `yolov8n.pt`/`yolov8s.pt` in the project root |
| No webcam / headless environment | Expected — app auto-falls back to a synthetic feed |
| Tracking disabled warning | `pip install lapx --break-system-packages` (ByteTrack dependency) |
| Annotated video won't play in browser | Re-encode: `ffmpeg -i job_data/<job_id>/annotated.mp4 -vcodec libx264 out.mp4` |
| Laggy video on CPU | Lower `YOLO_IMG_SIZE` or switch to `YOLO_MODEL_PATH=yolov8n.pt` |

## Security

Never commit a real `.env` file. If an API key was ever exposed in this
repo's history, rotate it in the [Mistral console](https://console.mistral.ai/)
immediately and issue a new one.

## License

MIT — see `LICENSE`.
