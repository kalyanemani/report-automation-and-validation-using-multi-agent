from __future__ import annotations

"""
QDR·AI — Usage & Adoption Log (append-only)
────────────────────────────────────────────
Records when the tool is used, on what, and how it went — so the plant
engineering team's real usage can be analysed and fed back into the product.

Persists to:
  - data/logs/usage/usage_log.jsonl   append-only, one JSON object per event,
                                      never rewritten or truncated
  - data/logs/usage/usage_YYYY-MM.csv flat monthly mirror of the same events,
                                      for opening directly in Excel

WHY APPEND-ONLY, AND WHY TWO FILES
──────────────────────────────────
The existing run artefacts in data/parsed (`llm_traces_latest.json`,
`rag_results_latest.json`) are deliberately *last-run-wins* snapshots: each run
overwrites them, because they exist for debugging the run you just did. That
makes them useless for the question this log answers — "how has the team been
using this over the last three months?" — since every previous run's evidence
is gone the moment someone clicks Inspect again.

So this file only ever *appends*. Every write opens with mode "a"; there is no
rotation, no truncation and no size cap, because history is the entire point.
Records are well under a kilobyte, so a thousand inspections costs under a
megabyte. The .csv mirror carries the same events in flat columns, because the
people who need to read this (QA leads, not developers) will open it in Excel
rather than parse JSON.

WHAT AN "EVENT" IS
──────────────────
One run produces at least two lines, correlated by `usage_run_id`:

    run_started    written the moment the pipeline is dispatched
    run_completed  written when it returns a report   (or)
    run_failed     written when it raises

Two lines rather than one mutated record is what append-only buys: if the host
is killed or the browser tab closed mid-run, the `run_started` line survives
with no terminal partner, so an abandoned or crashed run stays visible in the
data instead of vanishing. `session_started` marks each new browser session —
that is the "when did the team start using the tool" signal — and `run_blocked`
/ `input_missing` record the attempts that never became runs, which is where
friction in the UI shows up.

IDENTITY
────────
No person's name is stored. Each browser session gets a random `session_id`,
and the server `host` is recorded; that is enough to count sessions, runs per
session and repeat usage without turning this into a record of named
individuals' activity. `session_id` is not stable across browser reloads — a
reload is a new session by design, since that is also how Streamlit sees it.

FAILURE POLICY
──────────────
Telemetry must never take down an inspection. Every public function swallows
its own exceptions and degrades to a pipeline_logger warning, so a read-only
disk or a locked file loses log lines, not the user's review.
"""

import csv
import json
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from src.config import LOGS_DIR
from src.utils.logger import pipeline_logger

USAGE_DIR = LOGS_DIR / "usage"
USAGE_DIR.mkdir(parents=True, exist_ok=True)

USAGE_LOG_FILE = USAGE_DIR / "usage_log.jsonl"

_HOST = socket.gethostname()

# One lock for both files. Streamlit serves each browser session on its own
# script thread and the pipeline runs on a worker thread, so concurrent writes
# are normal here; without this, two appends can interleave inside one line.
_lock = threading.Lock()

# Column order for the CSV mirror. Append new columns at the END only — a
# monthly file already on disk keeps the header it was created with, so
# reordering would silently misalign values against that existing header.
CSV_FIELDS: list[str] = [
    "timestamp",
    "event",
    "usage_run_id",
    "session_id",
    "host",
    "pid",
    "document_filename",
    "document_chars",
    "discipline",
    "guide_type",
    "review_mode",
    "supplier",
    "doc_no",
    "revision",
    "criteria_count",
    "status",
    "duration_sec",
    "disposition",
    "score",
    "findings_total",
    "findings_hold",
    "findings_major",
    "pages_processed",
    "llm_calls",
    "llm_run_id",
    "stage_errors",
    "error_type",
    "error_message",
    "detail",
    "timestamp_utc",
]


def _now() -> tuple[str, str]:
    """(local ISO timestamp with UTC offset, UTC ISO timestamp)."""
    utc = datetime.now(timezone.utc)
    return utc.astimezone().isoformat(timespec="seconds"), utc.isoformat(timespec="seconds")


def _append(record: dict[str, Any]) -> None:
    """Append one event to the JSONL log and to the current month's CSV mirror."""
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        month = str(record.get("timestamp", ""))[:7] or datetime.now().strftime("%Y-%m")
        csv_path = USAGE_DIR / f"usage_{month}.csv"

        with _lock:
            with open(USAGE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            new_file = not csv_path.exists()
            # newline="" is required on Windows, else csv writes \r\r\n rows.
            with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow({k: record.get(k, "") for k in CSV_FIELDS})
    except Exception as e:  # never let telemetry break a review
        pipeline_logger.warning(f"[usage-log] could not write {record.get('event')} event: {e}")


def _base(event: str, session_id: str) -> dict[str, Any]:
    local, utc = _now()
    return {
        "timestamp": local,
        "timestamp_utc": utc,
        "event": event,
        "session_id": session_id,
        "host": _HOST,
        "pid": os.getpid(),
    }


def new_session_id() -> str:
    """Random per-browser-session identifier. Carries no personal information."""
    return uuid.uuid4().hex[:12]


def log_session_start(session_id: str, detail: str = "") -> None:
    """Records that someone opened the app. Call once per browser session."""
    record = _base("session_started", session_id)
    if detail:
        record["detail"] = detail
    _append(record)


def log_event(event: str, session_id: str, detail: str = "", **fields: Any) -> None:
    """
    Records a one-off event that is not a full run — `run_blocked` (a second
    inspection attempted while one was already running), `input_missing`
    (Inspect clicked with no document loaded), or anything else worth counting.
    """
    record = _base(event, session_id)
    if detail:
        record["detail"] = detail
    record.update(fields)
    _append(record)


def _count_severities(findings: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings or []:
        if isinstance(finding, dict):
            sev = str(finding.get("severity") or "UNSPECIFIED").upper()
            counts[sev] = counts.get(sev, 0) + 1
    return counts


class UsageRun:
    """
    One inspection run's entry in the usage log.

    Created by `start_run()`, which immediately writes the `run_started` line;
    the caller then closes it with exactly one of `completed()` / `failed()`.
    Both are idempotent, so a `failed()` in an `except` followed by a catch-all
    in `finally` cannot double-count the same run.
    """

    def __init__(self, session_id: str, inputs: dict[str, Any]):
        self.session_id = session_id
        self.usage_run_id = uuid.uuid4().hex[:12]
        self.inputs = inputs
        self._t0 = time.perf_counter()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def duration_sec(self) -> float:
        return round(time.perf_counter() - self._t0, 2)

    def _record(self, event: str) -> dict[str, Any]:
        record = _base(event, self.session_id)
        record["usage_run_id"] = self.usage_run_id
        record.update(self.inputs)
        return record

    def _started(self) -> None:
        _append(self._record("run_started"))

    def completed(self, result: dict[str, Any] | None, llm_run_id: str = "") -> None:
        """Terminal event for a run that produced a report."""
        if self._closed:
            return
        self._closed = True
        result = result or {}
        try:
            findings = result.get("findings") or result.get("annotated_findings") or []
            severities = _count_severities(findings)
            record = self._record("run_completed")
            record.update(
                {
                    "status": "COMPLETED",
                    "duration_sec": self.duration_sec,
                    "disposition": result.get("disposition", ""),
                    "score": result.get("score", ""),
                    "findings_total": len(findings),
                    "findings_hold": severities.get("HOLD", 0),
                    "findings_major": severities.get("MAJOR", 0),
                    "findings_by_severity": severities,
                    "pages_processed": len(result.get("page_ocr_results") or []),
                    "llm_calls": len(result.get("llm_traces") or []),
                    # Correlates this run with its reasoning traces in
                    # data/logs/llm_traces.jsonl, which are keyed by run_id.
                    "llm_run_id": llm_run_id,
                    "stage_errors": len(result.get("errors") or []),
                    "stage_timings": [
                        {"stage": m.get("stage"), "sec": m.get("duration_sec"), "status": m.get("status")}
                        for m in (result.get("pipeline_metrics") or [])
                        if isinstance(m, dict)
                    ],
                }
            )
            _append(record)
        except Exception as e:
            pipeline_logger.warning(f"[usage-log] could not summarise completed run: {e}")

    def failed(self, error: BaseException | str, error_kind: str = "", llm_run_id: str = "") -> None:
        """Terminal event for a run that raised. `error_kind` groups the cause."""
        if self._closed:
            return
        self._closed = True
        record = self._record("run_failed")
        # An agent error message can carry a whole LLM response; keep the line
        # readable and keep one CSV cell from swallowing a page of text.
        message = str(error)[:2000].replace("\n", " ").strip()
        record.update(
            {
                "status": "FAILED",
                "duration_sec": self.duration_sec,
                "error_type": error_kind or type(error).__name__,
                "error_message": message,
                "llm_run_id": llm_run_id,
            }
        )
        _append(record)


    def interrupted(self, reason: str = "", llm_run_id: str = "") -> None:
        """
        Terminal event for a run the UI stopped waiting on — a browser reload
        or rerun tears down the Streamlit script thread while the pipeline
        worker carries on (see src/utils/run_lock.py).

        Kept separate from `failed()` so UI reloads don't inflate the failure
        rate: nothing is known to be broken here, the user just walked away
        from the result. A cluster of these is its own signal — it usually
        means runs are taking longer than people are willing to wait.
        """
        if self._closed:
            return
        self._closed = True
        record = self._record("run_interrupted")
        record.update(
            {
                "status": "INTERRUPTED",
                "duration_sec": self.duration_sec,
                "error_type": reason or "interrupted",
                "llm_run_id": llm_run_id,
            }
        )
        _append(record)


def start_run(
    session_id: str,
    document_filename: str = "",
    document_chars: int = 0,
    discipline: str = "",
    guide_type: str = "",
    review_mode: str = "",
    supplier: str = "",
    doc_no: str = "",
    revision: str = "",
    criteria_count: int = 0,
) -> UsageRun:
    """
    Opens a usage-log entry for an inspection about to start, writing its
    `run_started` line immediately. The returned handle must be closed with
    `completed()` or `failed()`.

    The inputs are repeated on both the started and the terminal line on
    purpose: each CSV row then stands on its own, so a pivot over the monthly
    file can group runs by supplier or document without first joining lines
    together by run id.
    """
    inputs = {
        "document_filename": document_filename,
        "document_chars": document_chars,
        "discipline": discipline,
        "guide_type": guide_type,
        "review_mode": review_mode,
        "supplier": supplier,
        "doc_no": doc_no,
        "revision": revision,
        "criteria_count": criteria_count,
    }
    run = UsageRun(session_id, inputs)
    run._started()
    return run
