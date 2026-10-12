from __future__ import annotations

"""
QDR·AI — Per-Stage Pipeline Log (append-only, live-tailable)
─────────────────────────────────────────────────────────────
One record per pipeline stage, every run, kept forever — what ran, in what
order, how long it took, and whether it fell back or failed.

Persists to:
  - data/logs/stages/stages_live.log    flat aligned text, built to be tailed
                                        while a run is in progress
  - data/logs/stages/stage_log.jsonl    append-only structured record per event
  - data/logs/stages/stages_YYYY-MM.csv flat monthly mirror, for Excel pivots
                                        over stage durations

WHY, GIVEN pipeline.log ALREADY EXISTS
──────────────────────────────────────
`data/logs/pipeline.log` interleaves stage boundaries with every other log line
the pipeline emits — OCR diagnostics, retrieval hits, LLM prompt sizes. Tailing
it to answer "which stage is running right now, and how long has it been
there?" means reading past hundreds of unrelated lines per stage. These files
carry *only* stage transitions, so:

    Get-Content -Wait -Tail 20 data\\logs\\stages\\stages_live.log

is a live progress view of the pipeline, and the .csv answers "what is our
slowest stage across the last 200 runs" without any parsing.

APPEND-ONLY
───────────
Same contract as src/utils/usage_log.py: every write opens with mode "a", and
nothing here rotates or truncates. A stage-timing history is only useful if it
spans releases — that is how you see that OCR got slower after a change.

CORRELATION
───────────
Every record carries `run_id`, the same id `llm_tracer` stamps on its traces,
which `usage_log` records as `llm_run_id`. So one id joins all three views of a
run: what the user did (usage), which stages ran (here), and what the LLM was
asked (traces).

EVENTS
──────
    run_begin        pipeline entry, before the first stage
    stage_started    a StageTracker context was entered
    stage_fallback   the stage degraded but carried on
    stage_completed  exited cleanly
    stage_failed     exited with an exception
    run_end          pipeline exit, with total wall time and stage count

FAILURE POLICY
──────────────
Instrumentation must never break a review: every public function swallows its
own exceptions down to a pipeline_logger warning.
"""

import csv
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src.config import LOGS_DIR
from src.utils.logger import pipeline_logger

STAGES_DIR = LOGS_DIR / "stages"
STAGES_DIR.mkdir(parents=True, exist_ok=True)

LIVE_LOG_FILE = STAGES_DIR / "stages_live.log"
STAGE_LOG_FILE = STAGES_DIR / "stage_log.jsonl"

_HOST = socket.gethostname()

_lock = threading.Lock()

# thread ident -> run_id, registered by begin_run() on the pipeline thread.
#
# The run id has to be looked up rather than passed down, because the 23
# StageTracker call sites are spread across the agent modules and none of them
# hold a run handle. A stage can also be logged from a thread that never called
# begin_run() — the OCR stage fans pages out to a pool — so the lookup falls
# back to the single active run when there is exactly one. If two pipelines are
# somehow running at once (see src/utils/run_lock.py for how that used to
# happen), the fallback is ambiguous, and an ambiguous stage is recorded with an
# empty run_id rather than being attributed to the wrong run.
_RUN_BY_THREAD: dict[int, str] = {}
_SEQ: dict[str, int] = {}          # run_id -> stages logged so far
_STARTED: dict[str, float] = {}    # run_id -> perf_counter at begin_run
_DEPTH = threading.local()          # nesting level, per thread

CSV_FIELDS: list[str] = [
    "timestamp",
    "event",
    "run_id",
    "seq",
    "stage",
    "status",
    "duration_sec",
    "depth",
    "detail",
    "error",
    "pages_done",
    "total_pages",
    "host",
    "pid",
    "thread",
    "timestamp_utc",
]


def _now() -> tuple[str, str]:
    utc = datetime.now(timezone.utc)
    return utc.astimezone().isoformat(timespec="seconds"), utc.isoformat(timespec="seconds")


def _depth() -> int:
    return getattr(_DEPTH, "value", 0)


def _run_id() -> str:
    """Run id for the calling thread — see the note on _RUN_BY_THREAD."""
    own = _RUN_BY_THREAD.get(threading.get_ident())
    if own:
        return own
    active = set(_RUN_BY_THREAD.values())
    return next(iter(active)) if len(active) == 1 else ""


def _next_seq(run_id: str) -> int:
    _SEQ[run_id] = _SEQ.get(run_id, 0) + 1
    return _SEQ[run_id]


def _progress() -> dict[str, Any]:
    """
    Live page-count context for the record.

    Only the real counters are taken. `snapshot()["percentage"]` is deliberately
    NOT recorded: for any stage that isn't page-based it is a time-based
    animation the UI uses to keep a progress bar moving (see the
    `estimated_pct` branch in src/utils/progress.py), so persisting it would put
    a fabricated number in a diagnostics file that later analysis would read as
    measured.

    Imported lazily: src.utils.progress is what the UI polls, and importing it
    at module scope would make this module's import order matter to it.
    """
    try:
        from src.utils import progress as pipeline_progress

        snap = pipeline_progress.snapshot()
        return {
            "pages_done": snap.get("pages_done", 0),
            "total_pages": snap.get("total_pages", 0),
        }
    except Exception:
        return {}


def _live_line(record: dict[str, Any]) -> str:
    """
    One aligned, ASCII-only text line for the tailable log.

    ASCII on purpose: this file gets read with `Get-Content -Wait` in a
    PowerShell console, where a box-drawing or emoji marker can arrive as
    mojibake depending on the console code page. Indentation by nesting depth
    makes a stage that runs inside another stage read as a subtree.
    """
    marker = {
        "run_begin": "=== RUN  ",
        "run_end": "=== END  ",
        "stage_started": ">> START ",
        "stage_completed": "<< DONE  ",
        "stage_failed": "!! FAIL  ",
        "stage_fallback": "~~ WARN  ",
    }.get(record["event"], "   ....  ")

    stamp = str(record.get("timestamp", ""))[:19].replace("T", " ")
    run = (record.get("run_id") or "-")[:12]
    indent = "  " * int(record.get("depth", 0) or 0)
    stage = f"{indent}{record.get('stage', '')}"

    secs = record.get("duration_sec", "")
    timing = f"{float(secs):8.2f}s" if secs not in ("", None) else " " * 9

    tail_parts = []
    if record.get("status") and record["event"] != "stage_started":
        tail_parts.append(str(record["status"]))
    if record.get("total_pages"):
        tail_parts.append(f"pg {record.get('pages_done', 0)}/{record['total_pages']}")
    if record.get("detail"):
        tail_parts.append(str(record["detail"]))
    if record.get("error"):
        tail_parts.append(f"ERROR: {record['error']}")

    tail = " | ".join(tail_parts)
    return f"[{stamp}] [{run}] {marker} {stage:<34} {timing}  {tail}".rstrip()


def _append(record: dict[str, Any]) -> None:
    """Append one stage event to all three files."""
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        month = str(record.get("timestamp", ""))[:7] or datetime.now().strftime("%Y-%m")
        csv_path = STAGES_DIR / f"stages_{month}.csv"
        text = _live_line(record)

        with _lock:
            with open(STAGE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            # Flushed per line so a tail sees each transition as it happens
            # rather than when a buffer happens to fill.
            with open(LIVE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(text + "\n")
                f.flush()

            new_file = not csv_path.exists()
            with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow({k: record.get(k, "") for k in CSV_FIELDS})
    except Exception as e:
        pipeline_logger.warning(f"[stage-log] could not write {record.get('event')} event: {e}")


def _base(event: str, run_id: str) -> dict[str, Any]:
    local, utc = _now()
    return {
        "timestamp": local,
        "timestamp_utc": utc,
        "event": event,
        "run_id": run_id,
        "host": _HOST,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
    }


def begin_run(run_id: str, document: str = "", mode: str = "") -> None:
    """
    Registers the calling thread as the owner of `run_id` and writes the
    run_begin header. Called at pipeline entry, before the first stage, so
    every stage that follows can find the run id without being handed it.
    """
    try:
        with _lock:
            # Drop registrations for threads that have exited, so a long-lived
            # Streamlit process does not accumulate one entry per run it has
            # ever executed — and, more importantly, so _run_id()'s
            # single-active-run fallback is not confused by dead runs.
            live = {t.ident for t in threading.enumerate()}
            for ident in [i for i in _RUN_BY_THREAD if i not in live]:
                _RUN_BY_THREAD.pop(ident, None)
            _RUN_BY_THREAD[threading.get_ident()] = run_id
            _SEQ[run_id] = 0
            _STARTED[run_id] = time.perf_counter()
        _DEPTH.value = 0

        record = _base("run_begin", run_id)
        record.update({"seq": 0, "stage": "PIPELINE", "detail": document, "status": mode})
        _append(record)
    except Exception as e:
        pipeline_logger.warning(f"[stage-log] could not open run {run_id}: {e}")


def end_run(status: str = "OK", error: str = "") -> None:
    """
    Closes the run with a total wall time and stage count. Written so a tail of
    stages_live.log shows where one run ends and the next begins, instead of
    leaving the reader to infer it from the last stage that happened to log.
    """
    try:
        run_id = _run_id()
        with _lock:
            started = _STARTED.pop(run_id, None)
            stages = _SEQ.get(run_id, 0)
            _RUN_BY_THREAD.pop(threading.get_ident(), None)

        record = _base("run_end", run_id)
        record.update(
            {
                "seq": stages,
                "stage": "PIPELINE",
                "status": status,
                "detail": f"{stages} stage events",
                "error": str(error)[:1000].replace("\n", " ").strip() if error else "",
            }
        )
        if started is not None:
            record["duration_sec"] = round(time.perf_counter() - started, 3)
        record.update(_progress())
        _append(record)
    except Exception as e:
        pipeline_logger.warning(f"[stage-log] could not close run: {e}")


def stage_started(stage: str, detail: str = "") -> float:
    """Records stage entry. Returns the monotonic start time for the caller."""
    started = time.perf_counter()
    try:
        run_id = _run_id()
        with _lock:
            seq = _next_seq(run_id)
        record = _base("stage_started", run_id)
        record.update({"seq": seq, "stage": stage, "detail": detail, "depth": _depth()})
        record.update(_progress())
        _append(record)
        _DEPTH.value = _depth() + 1
    except Exception as e:
        pipeline_logger.warning(f"[stage-log] could not log start of {stage}: {e}")
    return started


def stage_fallback(stage: str, reason: str, started: float | None = None) -> None:
    """
    Records that a stage degraded but kept going. Logged as its own event, at
    the moment it happens, because the reason a run produced a thin report is
    usually a fallback midway through a stage that still exited "OK".
    """
    try:
        record = _base("stage_fallback", _run_id())
        record.update(
            {
                "stage": stage,
                "status": "FALLBACK",
                "detail": reason,
                # depth-1: the stage is on the stack, so its own events belong
                # at the level it was logged at, not one further in.
                "depth": max(_depth() - 1, 0),
            }
        )
        if started is not None:
            record["duration_sec"] = round(time.perf_counter() - started, 3)
        record.update(_progress())
        _append(record)
    except Exception as e:
        pipeline_logger.warning(f"[stage-log] could not log fallback of {stage}: {e}")


def stage_ended(
    stage: str,
    started: float,
    status: str = "OK",
    error: str = "",
    detail: str = "",
) -> None:
    """Records stage exit — `stage_failed` when it raised, else `stage_completed`."""
    try:
        _DEPTH.value = max(_depth() - 1, 0)
        run_id = _run_id()
        with _lock:
            seq = _next_seq(run_id)
        event = "stage_failed" if status == "ERROR" else "stage_completed"
        record = _base(event, run_id)
        record.update(
            {
                "seq": seq,
                "stage": stage,
                "status": status,
                "duration_sec": round(time.perf_counter() - started, 3),
                "depth": _depth(),
                "detail": detail,
                "error": str(error)[:1000].replace("\n", " ").strip() if error else "",
            }
        )
        record.update(_progress())
        _append(record)
    except Exception as e:
        pipeline_logger.warning(f"[stage-log] could not log end of {stage}: {e}")
