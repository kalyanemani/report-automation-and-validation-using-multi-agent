from __future__ import annotations

"""
QDR·AI — Live Pipeline Progress Tracker
──────────────────────────────────────────
Process-wide, thread-safe progress state. The inspection pipeline runs on
a background thread (see app.py) so the Streamlit UI thread stays free to
poll "what's happening right now" — current stage, and for the page-
parallel OCR stage, which page numbers are actively being processed.
"""

import threading

import time

_lock = threading.Lock()
_state = {
    "stage": "",
    "total_pages": 0,
    "pages_in_flight": set(),
    "pages_done": 0,
    "stage_current": 0,
    "stage_total": 0,
    "percentage": 0,
    "stage_detail": "",
    "stage_start_time": 0.0,
}


def reset() -> None:
    with _lock:
        _state["stage"] = ""
        _state["total_pages"] = 0
        _state["pages_in_flight"] = set()
        _state["pages_done"] = 0
        _state["stage_current"] = 0
        _state["stage_total"] = 0
        _state["percentage"] = 0
        _state["stage_detail"] = ""
        _state["stage_start_time"] = 0.0


def set_stage(stage: str, percentage: int = 0) -> None:
    with _lock:
        _state["stage"] = stage
        _state["stage_current"] = 0
        _state["stage_total"] = 0
        _state["percentage"] = percentage
        _state["stage_detail"] = ""
        _state["stage_start_time"] = time.time()


def set_total_pages(total: int) -> None:
    with _lock:
        _state["total_pages"] = total


def start_page(page_idx: int) -> None:
    with _lock:
        _state["pages_in_flight"].add(page_idx)


def finish_page(page_idx: int) -> None:
    with _lock:
        _state["pages_in_flight"].discard(page_idx)
        _state["pages_done"] += 1
        if _state["total_pages"] > 0:
            _state["percentage"] = min(99, int(round((_state["pages_done"] / _state["total_pages"]) * 100)))


def set_stage_progress(current: int, total: int, detail: str = "") -> None:
    """Set current progress count out of total items for the active stage."""
    with _lock:
        _state["stage_current"] = current
        _state["stage_total"] = total
        _state["stage_detail"] = detail
        if total > 0:
            _state["percentage"] = min(100, int(round((current / total) * 100)))
        else:
            _state["percentage"] = 0


def set_percentage(pct: int, detail: str = "") -> None:
    """Explicitly set a completion percentage (0-100)."""
    with _lock:
        _state["percentage"] = min(100, max(0, pct))
        _state["stage_detail"] = detail


def snapshot() -> dict:
    """Point-in-time read for the polling UI thread."""
    with _lock:
        pct = _state["percentage"]
        stage = _state["stage"]
        start_t = _state["stage_start_time"]
        total = _state["total_pages"]
        done = _state["pages_done"]

        # 1. Page-based progress (OCR extraction / Visual inspection)
        if stage in ("ocr_extract", "scan_all_pages_ocr", "inspect_page_visual") and total > 0:
            calc_pct = int(round((done / total) * 100))
            pct = max(pct, min(99, calc_pct))
        elif start_t > 0 and pct < 95:
            # 2. Smooth dynamic progression for long-running stages
            elapsed = time.time() - start_t
            rate = 5.0 if stage in ("search_specs", "classify_document", "extract_requirements", "build_annotations") else 1.8
            estimated_pct = min(92, max(pct, int(min(92, 5 + elapsed * rate))))
            pct = estimated_pct

        return {
            "stage": stage,
            "total_pages": total,
            "pages_done": done,
            "pages_in_flight": sorted(_state["pages_in_flight"]),
            "stage_current": _state["stage_current"],
            "stage_total": _state["stage_total"],
            "percentage": min(100, max(0, pct)),
            "stage_detail": _state["stage_detail"],
        }

