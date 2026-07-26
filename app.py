"""
streamlit_app.py

Streamlit interface for the YOLO + Mistral Fusion pipeline. Deliberately
mirrors flask_app.py / templates/index.html section-for-section (same
shared_ui.CSS, same header, same metric cards, same "Person report / Other
objects report" window layout, same two PDF download buttons) so the two
apps present as one consistent product regardless of which framework
the person happens to open.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile
import time

import streamlit as st

import pipeline
from shared_ui import APP_NAME, CSS, header_html, window_html

st.set_page_config(
    page_title=f"{APP_NAME} \u2014 YOLO + Mistral Fusion",
    page_icon="\U0001F6E1\uFE0F",
    layout="centered",
)

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown(header_html("Video Intelligence"), unsafe_allow_html=True)

# ------------------------------------------------------------------ state --
if "job_result" not in st.session_state:
    st.session_state.job_result = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def reset_state():
    st.session_state.job_result = None
    st.session_state.job_id = None
    st.session_state.processing = False


# --------------------------------------------------------------- upload UI --
st.markdown('<div class="vg-card">', unsafe_allow_html=True)
st.markdown(
    "<h3>\U0001F4FA Upload Surveillance Video</h3>"
    "<p class='vg-sub'>Detects people and 79 other COCO object classes, tracks them across "
    "frames, and generates AI incident reports per time window.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drag & drop a video, or click to browse",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    label_visibility="collapsed",
)

col_a, col_b = st.columns([1, 1])
with col_a:
    run_clicked = st.button(
        "\u26A1 Run AI Analysis", disabled=(uploaded_file is None or st.session_state.processing)
    )
with col_b:
    if st.session_state.job_result is not None:
        if st.button("Start Over"):
            reset_state()
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------- processing --
if run_clicked and uploaded_file is not None:
    st.session_state.processing = True

    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    st.markdown('<div class="vg-card">', unsafe_allow_html=True)
    st.markdown("<h3>\u23F3 Processing Video</h3>", unsafe_allow_html=True)
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    st.markdown("</div>", unsafe_allow_html=True)

    def on_progress(frac: float, msg: str) -> None:
        progress_bar.progress(min(max(frac, 0.0), 1.0))
        status_placeholder.markdown(f"<p class='vg-sub'>{msg}</p>", unsafe_allow_html=True)

    try:
        result = pipeline.process_video(tmp_path, on_progress=on_progress)
        st.session_state.job_result = {
            "job_id": result.job_id,
            "db_path": result.db_path,
            "annotated_video_path": result.annotated_video_path,
            "total_frames": result.total_frames,
            "duration_seconds": round(result.duration_seconds, 1),
            "peak_person_count": result.peak_person_count,
            "peak_object_classes": result.peak_object_classes,
            "total_windows": result.total_windows,
            "reports": result.reports,
            "warnings": result.warnings,
        }
        st.session_state.job_id = result.job_id
    except Exception as exc:  # noqa: BLE001
        st.error(f"Processing failed: {exc}")
    finally:
        st.session_state.processing = False
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        st.rerun()

# ---------------------------------------------------------------- results --
result = st.session_state.job_result
if result:
    st.markdown('<div class="vg-card">', unsafe_allow_html=True)
    st.markdown("<h3>\U0001F4CA Detection Summary</h3>", unsafe_allow_html=True)
    if result["warnings"]:
        st.markdown(
            f"<p class='vg-sub'>{' '.join(result['warnings'])}</p>", unsafe_allow_html=True
        )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames Analyzed", result["total_frames"])
    m2.metric("Video Duration", f"{result['duration_seconds']}s")
    m3.metric("Peak Persons", result["peak_person_count"])
    m4.metric("Report Windows", result["total_windows"])
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="vg-card">', unsafe_allow_html=True)
        st.markdown(
            "<h3>\U0001F3AC Annotated Video</h3>"
            "<p class='vg-sub'>Bounding boxes, class labels, and track IDs drawn on every frame.</p>",
            unsafe_allow_html=True,
        )
        video_path = result["annotated_video_path"]
        if os.path.isfile(video_path):
            st.video(video_path)
        else:
            st.info("Annotated video not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="vg-card">', unsafe_allow_html=True)
        st.markdown(
            "<h3>\U0001F4BE Export Reports</h3>"
            "<p class='vg-sub'>Two separate PDF deliverables, generated from the stored incident reports.</p>",
            unsafe_allow_html=True,
        )

        chips = "".join(
            f'<span class="vg-pill {"ok" if cls == "person" else "warn"}" style="margin:3px 4px 3px 0;">{cls}: {cnt}</span>'
            for cls, cnt in (result["peak_object_classes"] or {}).items()
        )
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

        pdf_out_dir = os.path.join(os.path.dirname(result["db_path"]), "pdf")
        if st.button("\U0001F4C4 Generate PDF Reports", key="gen_pdf"):
            with st.spinner("Building PDF reports..."):
                pdf_paths = pipeline.build_pdfs(result["db_path"], pdf_out_dir)
                st.session_state["pdf_paths"] = pdf_paths

        pdf_paths = st.session_state.get("pdf_paths")
        if pdf_paths:
            pc1, pc2 = st.columns(2)
            with open(pdf_paths["person"], "rb") as f:
                pc1.download_button("\U0001F464 Person Report PDF", f.read(), file_name="person_report.pdf", mime="application/pdf")
            with open(pdf_paths["objects"], "rb") as f:
                pc2.download_button("\U0001F4E6 Other Objects PDF", f.read(), file_name="other_objects_report.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="vg-card">', unsafe_allow_html=True)
    st.markdown(
        "<h3>\U0001F4CB AI Incident Reports by Window</h3>"
        "<p class='vg-sub'>Generated by Mistral LLM (or offline fallback text if no API key is configured).</p>",
        unsafe_allow_html=True,
    )
    if not result["reports"]:
        st.markdown("<div class='vg-empty'>No activity windows were recorded for this video.</div>", unsafe_allow_html=True)
    else:
        for row in result["reports"]:
            st.markdown(window_html(row), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='vg-footer'>VisionGuard AI &middot; Streamlit interface &middot; "
    "powered by YOLOv8 + Mistral</div>",
    unsafe_allow_html=True,
)
