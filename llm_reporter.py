"""
llm_reporter.py

Wraps the official `mistralai` Python SDK and turns a WindowSummary of
aggregated detection events into a natural-language incident report,
using Mistral's chat completion endpoint.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Tuple

from mistralai import Mistral

from schemas import IncidentReport, WindowSummary

# Marker strings the model is instructed to emit verbatim, so the response
# can be split into two independent reports afterwards (see _split_sections).
_PERSON_MARKER = "PERSON_REPORT:"
_OTHER_MARKER = "OTHER_OBJECTS_REPORT:"

SYSTEM_PROMPT = f"""You are an automated surveillance analyst reviewing object-detection \
logs produced by a computer vision system (YOLO + object tracking) monitoring a video feed. \
The system detects PEOPLE as well as other objects, drawn from the standard 80 COCO classes \
(vehicles, animals, bags, furniture, electronics, etc.). Every entry in the log was tracked \
with a stable ID across frames where possible.

For each class in the log you will be given:
  - "Peak simultaneous count": the maximum number of that class visible together in a single \
    frame at any point during the window. This is the authoritative headcount for that class \
    in this window.
  - "Distinct individuals tracked": the total number of different tracked instances of that \
    class seen at any point in the window, which can be higher than the peak if instances \
    entered and left one at a time.
  - "Total frame detections": a noisy raw count of frame-level boxes, NOT a headcount (the \
    same object contributes many detections while visible). Never use this number as a count.
The log also states "Authoritative peak-simultaneous counts for this window" per class — \
always use exactly those numbers; never estimate or round them.

The monitoring window header tells you whether times are elapsed VIDEO time (counted from \
00:00:00 at the start of an uploaded video) or real-world WALL-CLOCK time (from a live \
camera). Use whichever format the header gives you, exactly as given, and never convert \
between the two or imply the other kind of time is being used.

Your job is to write TWO separate, concise, professional incident reports from the SAME log: \
one covering ONLY the "person" class, and one covering EVERY OTHER detected class (never \
mention "person" in the second one). Output them in EXACTLY this format, with no other text \
before, between, or after:

{_PERSON_MARKER}
<2-4 sentences about person detections only>

{_OTHER_MARKER}
<2-4 sentences about every other detected object class, or "No other objects were detected \
during this monitoring window." if none are present>

Follow these rules strictly for both sections:

1. Only report facts that are present in the log. Do not invent objects, people, actions, or \
   events that are not in the data provided. Do not add speculative detail, background, or \
   commentary that isn't directly supported by the log.
2. If a section's class has no data in the log, state clearly that nothing of that kind was \
   observed and keep that section to one short sentence.
3. Grammar and counting are critical: if a class's authoritative peak count is 1, use the \
   singular (e.g. "a person", "a car"). If it is 2 or more, you MUST use the plural form and \
   the exact number, e.g. "2 people were present" or "three cars were observed" — never say \
   "a car" or "the car" when the count is greater than 1. If distinct individuals tracked is \
   higher than the peak count for a class, you may note that instances came and went rather \
   than being present simultaneously.
4. Describe activity level factually (how persistent the presence was, based on first/last \
   seen times and duration) without speculating about intent, identity, or motive.
5. Do not use the words 'suspicious' or 'threat' unless the log explicitly indicates a \
   higher-risk situation; otherwise use neutral, factual language.
6. Keep each section to 2-4 sentences, plain prose, no headers/bullets/markdown beyond the \
   two marker lines above, and no unnecessary filler or repeated information.
"""


def _split_sections(text: str) -> Tuple[str, str]:
    """Splits a model response formatted with PERSON_REPORT:/OTHER_OBJECTS_REPORT:
    markers into (person_text, other_text). Tolerates minor formatting drift
    (extra whitespace, markdown emphasis around the markers)."""
    pattern = re.compile(
        rf"{re.escape(_PERSON_MARKER)}\s*(.*?)\s*{re.escape(_OTHER_MARKER)}\s*(.*)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        raise LLMReporterError(f"Could not find both report markers in model output: {text[:200]!r}")

    person_text = match.group(1).strip()
    other_text = match.group(2).strip()
    if not person_text or not other_text:
        raise LLMReporterError("One or both report sections were empty after parsing.")
    return person_text, other_text


class LLMReporterError(Exception):
    """Raised when the Mistral API call fails and no usable report can be produced."""


class MistralReporter:
    """Generates natural-language incident reports from aggregated detection windows."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMReporterError("A Mistral API key is required to initialize MistralReporter.")
        self.model = model
        try:
            self.client = Mistral(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            raise LLMReporterError(f"Failed to initialize Mistral client: {exc}") from exc

    def generate_report(self, summary: WindowSummary) -> IncidentReport:
        """
        Calls the Mistral chat completion endpoint with the window summary and
        returns a validated IncidentReport. On any failure, returns a report
        with a safe fallback summary_text and the error populated, rather than
        raising, so the main video loop is never interrupted by an LLM outage.
        """
        prompt_text = summary.to_prompt_text()

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Here is the detection log for the current monitoring window:\n\n"
                            f"{prompt_text}\n\n"
                            "Write the incident report now."
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=500,
            )

            choice = response.choices[0]
            report_text = choice.message.content

            if isinstance(report_text, list):
                # Some SDK versions may return content as a list of content blocks;
                # normalize defensively to a single string.
                report_text = " ".join(
                    part.text if hasattr(part, "text") else str(part) for part in report_text
                )

            report_text = (report_text or "").strip()
            if not report_text:
                raise LLMReporterError("Mistral returned an empty report.")

            person_text, other_text = _split_sections(report_text)
            combined_text = f"{person_text}\n\n{other_text}"

            return IncidentReport(
                window_start=summary.window_start,
                window_end=summary.window_end,
                summary_text=combined_text,
                person_summary_text=person_text,
                other_summary_text=other_text,
                raw_event_count=len(summary.events),
                peak_person_count=summary.peak_person_count(),
                time_source=summary.time_source,
                model_used=self.model,
                error=None,
            )

        except Exception as exc:  # noqa: BLE001
            person_text, other_text = self._build_fallback_text(summary)
            print(f"[MistralReporter] Report generation failed, using fallback summary: {exc}")
            return IncidentReport(
                window_start=summary.window_start,
                window_end=summary.window_end,
                summary_text=f"{person_text}\n\n{other_text}",
                person_summary_text=person_text,
                other_summary_text=other_text,
                raw_event_count=len(summary.events),
                peak_person_count=summary.peak_person_count(),
                time_source=summary.time_source,
                model_used=self.model,
                error=str(exc),
            )

    @staticmethod
    def _build_fallback_text(summary: WindowSummary) -> Tuple[str, str]:
        """Deterministic, non-LLM fallback used if the API call (or response
        parsing) fails. Returns (person_text, other_text) using the same
        authoritative counts as the LLM path, so grammar/counts stay correct
        even when the Mistral API is unavailable."""
        # -- Person section --
        if not summary.person_events():
            person_text = "No people were detected during this monitoring window (LLM report unavailable)."
        else:
            count = summary.peak_person_count()
            noun = "person was" if count == 1 else "people were"
            person_text = (
                f"LLM report unavailable. {count} {noun} detected at peak during this window "
                f"({summary.total_frames_processed} frames analyzed)."
            )

        # -- Other-objects section --
        other_events = summary.other_events()
        if not other_events:
            other_text = "No other objects were detected during this monitoring window (LLM report unavailable)."
        else:
            parts = []
            for event in sorted(other_events, key=lambda e: e.peak_concurrent_count, reverse=True):
                noun = event.class_name if event.peak_concurrent_count == 1 else f"{event.class_name}s"
                parts.append(f"{event.peak_concurrent_count} {noun}")
            other_text = (
                f"LLM report unavailable. Detected at peak during this window: "
                f"{', '.join(parts)} ({summary.total_frames_processed} frames analyzed)."
            )

        return person_text, other_text
