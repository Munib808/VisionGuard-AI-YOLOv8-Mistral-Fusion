"""
shared_ui.py

Single source of truth for the visual design system used by BOTH the Flask
web app (flask_app.py) and the Streamlit app (streamlit_app.py), so the two
interfaces look and feel identical -- same colors, typography, spacing,
cards, buttons, and copy -- regardless of which framework renders them.

Keeping this in one file means a design tweak only has to happen once.
"""

from __future__ import annotations

APP_NAME = "VisionGuard AI"
APP_TAGLINE = "YOLOv8 Detection \u00d7 Mistral LLM Incident Reporting"

# --- Design tokens -----------------------------------------------------
COLOR_BG = "#0b1120"
COLOR_BG_ALT = "#0f172a"
COLOR_SURFACE = "#111c33"
COLOR_SURFACE_ALT = "#16213f"
COLOR_BORDER = "#1f2c4d"
COLOR_TEXT = "#e6ecf7"
COLOR_TEXT_DIM = "#8b9bc2"
COLOR_ACCENT = "#22d3ee"      # cyan
COLOR_ACCENT_2 = "#10b981"    # emerald (person / success)
COLOR_ACCENT_3 = "#f59e0b"    # amber (other objects / warning)
COLOR_DANGER = "#ef4444"

FONT_STACK = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)

# --- Shared CSS ----------------------------------------------------------
# Written once, injected verbatim into both the Flask Jinja template
# (as static/style.css) and the Streamlit app (as an <style> block).
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
    --bg: {COLOR_BG};
    --bg-alt: {COLOR_BG_ALT};
    --surface: {COLOR_SURFACE};
    --surface-alt: {COLOR_SURFACE_ALT};
    --border: {COLOR_BORDER};
    --text: {COLOR_TEXT};
    --text-dim: {COLOR_TEXT_DIM};
    --accent: {COLOR_ACCENT};
    --accent-2: {COLOR_ACCENT_2};
    --accent-3: {COLOR_ACCENT_3};
    --danger: {COLOR_DANGER};
}}

html, body, [class*="css"] {{
    font-family: {FONT_STACK};
}}

body, .stApp {{
    background: radial-gradient(circle at 15% 0%, #12213f 0%, {COLOR_BG} 45%) fixed !important;
    color: var(--text);
}}

/* ---------- Top brand header ---------- */
.vg-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 26px;
    background: linear-gradient(120deg, rgba(34,211,238,0.10), rgba(16,185,129,0.06));
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 22px;
}}
.vg-header .vg-logo {{
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    font-size: 22px; font-weight: 800; color: #041018;
    flex-shrink: 0;
}}
.vg-header h1 {{
    font-size: 21px; font-weight: 800; margin: 0; letter-spacing: -0.02em;
    color: var(--text);
}}
.vg-header p {{
    margin: 2px 0 0 0; font-size: 13px; color: var(--text-dim); font-weight: 500;
}}
.vg-badge {{
    margin-left: auto;
    font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
    padding: 6px 12px; border-radius: 999px;
    background: rgba(16,185,129,0.12); color: var(--accent-2);
    border: 1px solid rgba(16,185,129,0.35);
    text-transform: uppercase;
}}

/* ---------- Cards ---------- */
.vg-card {{
    background: linear-gradient(180deg, var(--surface), var(--surface-alt));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}}
.vg-card h3 {{
    margin: 0 0 4px 0; font-size: 15px; font-weight: 700; color: var(--text);
    display: flex; align-items: center; gap: 8px;
}}
.vg-card .vg-sub {{
    margin: 0 0 16px 0; font-size: 12.5px; color: var(--text-dim);
}}

/* ---------- Metrics ---------- */
.vg-metric-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
}}
.vg-metric {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px 18px;
}}
.vg-metric .val {{
    font-size: 26px; font-weight: 800; color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
}}
.vg-metric .lbl {{
    font-size: 11.5px; color: var(--text-dim); text-transform: uppercase;
    letter-spacing: 0.05em; margin-top: 4px; font-weight: 600;
}}

/* ---------- Status pill ---------- */
.vg-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 700; padding: 5px 12px; border-radius: 999px;
}}
.vg-pill.ok {{ background: rgba(16,185,129,0.15); color: var(--accent-2); }}
.vg-pill.warn {{ background: rgba(245,158,11,0.15); color: var(--accent-3); }}
.vg-pill.err {{ background: rgba(239,68,68,0.15); color: var(--danger); }}
.vg-pill.busy {{ background: rgba(34,211,238,0.15); color: var(--accent); }}

/* ---------- Buttons ---------- */
.vg-btn, .stButton>button, .stDownloadButton>button {{
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #041018 !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; padding: 0.6em 1.3em !important;
    letter-spacing: 0.01em; box-shadow: 0 4px 14px rgba(34,211,238,0.20);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.vg-btn:hover, .stButton>button:hover, .stDownloadButton>button:hover {{
    transform: translateY(-1px); box-shadow: 0 6px 18px rgba(34,211,238,0.32);
}}
.vg-btn.secondary {{
    background: var(--surface-alt) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; box-shadow: none;
}}

/* ---------- Report window entries ---------- */
.vg-window {{
    border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
    margin-bottom: 10px; background: var(--surface);
}}
.vg-window .win-time {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--accent);
    font-weight: 700; margin-bottom: 6px;
}}
.vg-window .win-section-label {{
    font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em;
    font-weight: 700; color: var(--text-dim); margin: 8px 0 2px 0;
}}
.vg-window .win-person {{ border-left: 3px solid var(--accent-2); padding-left: 10px; }}
.vg-window .win-other {{ border-left: 3px solid var(--accent-3); padding-left: 10px; margin-top: 6px; }}
.vg-window p {{ font-size: 13px; line-height: 1.5; color: var(--text); margin: 2px 0 0 0; }}

/* ---------- Upload zone (Flask only, native) ---------- */
.vg-drop {{
    border: 2px dashed var(--border); border-radius: 14px; padding: 34px 20px;
    text-align: center; color: var(--text-dim); cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease;
}}
.vg-drop:hover, .vg-drop.dragover {{
    border-color: var(--accent); background: rgba(34,211,238,0.05);
}}

/* ---------- Progress bar ---------- */
.vg-progress-track {{
    width: 100%; height: 10px; border-radius: 999px; background: var(--surface-alt);
    overflow: hidden; border: 1px solid var(--border);
}}
.vg-progress-fill {{
    height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2));
    transition: width 0.3s ease;
}}

/* Streamlit-specific cleanup */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem !important; max-width: 1080px; }}
[data-testid="stFileUploader"] section {{
    background: var(--surface) !important; border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
}}
[data-testid="stMetric"] {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 10px 14px;
}}
video {{ border-radius: 14px; border: 1px solid var(--border); }}
"""


def _flatten_html(html: str) -> str:
    """Strips leading whitespace from every line of an HTML snippet.

    Streamlit's (and standard Markdown's) parser treats any line indented by
    4+ spaces as a code block, which silently HTML-escapes and literally
    prints tags like `<div class="...">` instead of rendering them. Every
    HTML string handed to st.markdown(..., unsafe_allow_html=True) must be
    flattened to one tag per line with no leading indentation to avoid this.
    """
    return "\n".join(line.strip() for line in html.strip().splitlines())


def header_html(badge_text: str = "Video Intelligence") -> str:
    """Returns the shared brand header as raw HTML (used verbatim by Flask
    and injected via st.markdown(unsafe_allow_html=True) by Streamlit)."""
    return _flatten_html(f"""
    <div class="vg-header">
        <div class="vg-logo">VG</div>
        <div>
            <h1>{APP_NAME}</h1>
            <p>{APP_TAGLINE}</p>
        </div>
        <div class="vg-badge">{badge_text}</div>
    </div>
    """)


def window_html(row: dict) -> str:
    """Renders one stored report row as the shared 'report window' card markup."""
    from datetime import datetime

    def _hms(v):
        try:
            return datetime.fromisoformat(v).strftime("%H:%M:%S")
        except Exception:
            return str(v)

    time_label = "video time" if (row.get("time_source") or "video") == "video" else "real time"
    person_text = (row.get("person_summary_text") or "No person data.").strip()
    other_text = (row.get("other_summary_text") or "No other-object data.").strip()
    err_html = (
        f'<div class="vg-pill err" style="margin-top:6px;">LLM fallback used</div>'
        if row.get("error")
        else ""
    )
    return _flatten_html(f"""
    <div class="vg-window">
        <div class="win-time">WINDOW {_hms(row.get('window_start',''))} \u2192 {_hms(row.get('window_end',''))} &middot; {time_label}</div>
        {err_html}
        <div class="win-section-label">Person report</div>
        <div class="win-person"><p>{person_text}</p></div>
        <div class="win-section-label">Other objects report</div>
        <div class="win-other"><p>{other_text}</p></div>
    </div>
    """)
