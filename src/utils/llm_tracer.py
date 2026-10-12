from __future__ import annotations

"""
QDR·AI — Unified LLM Call Tracer & Response Logger
─────────────────────────────────────────────────────
Captures and persists full traces (system prompts, user prompts,
raw text responses, structured JSON outputs, durations, and token metrics)
for every LLM invocation across the review pipeline.

Persists to:
  - data/parsed/llm_traces_latest.json (live inspectable JSON)
  - data/logs/llm_traces.jsonl (append-only historical log)

PER-RUN ISOLATION
─────────────────
Traces are keyed by the thread that produced them. The buffer used to be one
module-global list, so two pipeline runs in the same process interleaved into a
single file: on 2026-08-20 `llm_traces_latest.json` held five entries from two
concurrent runs, with duplicate call_ids at unrelated timestamps and a
"Total traces: 5" counter that belonged to neither run. Anyone debugging from
that file was reading a blend of two documents' reasoning.

The pipeline runs in a worker thread (see app.py), so the thread identity is a
faithful per-run key. `llm_traces_latest.json` now holds only the traces of the
run that wrote it, and every entry carries `run_id` so the append-only .jsonl
can be split by run after the fact.
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Optional

from src.config import LOGS_DIR, PARSED_DIR
from src.utils.logger import pipeline_logger

logger = logging.getLogger("qdr_ai.llm_tracer")

_lock = threading.Lock()
# thread ident -> {"run_id": str, "traces": list}
_RUNS: dict[int, dict[str, Any]] = {}


def _bucket() -> dict[str, Any]:
    """Trace bucket for the calling thread, created on first use."""
    ident = threading.get_ident()
    b = _RUNS.get(ident)
    if b is None:
        b = {"run_id": uuid.uuid4().hex[:12], "traces": []}
        _RUNS[ident] = b
    return b


def reset_llm_traces() -> None:
    """
    Start a fresh trace buffer for the calling thread.

    Called at pipeline entry. Also prunes buckets belonging to threads that no
    longer exist, so a long-lived Streamlit process does not accumulate the
    traces of every run it has ever executed.
    """
    with _lock:
        live = {t.ident for t in threading.enumerate()}
        for ident in [i for i in _RUNS if i not in live]:
            _RUNS.pop(ident, None)
        _RUNS[threading.get_ident()] = {"run_id": uuid.uuid4().hex[:12], "traces": []}
        try:
            latest_file = PARSED_DIR / "llm_traces_latest.json"
            latest_file.write_text("[]", encoding="utf-8")
        except Exception:
            pass


def current_run_id() -> str:
    with _lock:
        return _bucket()["run_id"]


def record_llm_trace(
    call_id: str,
    stage_name: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    parsed_output: Any,
    duration_s: float,
    status: str = "SUCCESS",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Record an LLM call trace and immediately persist to disk.
    """
    with _lock:
        bucket = _bucket()
        run_id = bucket["run_id"]
        traces = bucket["traces"]

        trace_entry: dict[str, Any] = {
            "run_id": run_id,
            "call_id": call_id,
            "stage": stage_name,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "duration_s": round(duration_s, 4),
            "status": status,
            "metadata": metadata or {},
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_output": parsed_output,
        }
        traces.append(trace_entry)
        snapshot = list(traces)
        count = len(traces)

    # Persist to data/parsed/llm_traces_latest.json — this run's traces only.
    try:
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        latest_file = PARSED_DIR / "llm_traces_latest.json"
        latest_file.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write llm_traces_latest.json: {e}")

    # Append to data/logs/llm_traces.jsonl (all runs, separable by run_id).
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "llm_traces.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to append to llm_traces.jsonl: {e}")

    pipeline_logger.info(
        f"[LLM TRACE] Recorded {stage_name} ({model_name}) in {duration_s:.2f}s "
        f"| Status: {status} | run={run_id} | traces this run: {count}"
    )

    return trace_entry


def record_rag_trace(
    query: str,
    test_type: str,
    material: str,
    retrieved_chunks: list[dict[str, Any]],
    graph_rules: dict[str, Any],
    review_guide_rules: list[str],
    duration_s: float,
    status: str = "SUCCESS",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Record a RAG & Knowledge Graph retrieval trace and persist to disk.
    """
    with _lock:
        bucket = _bucket()
        run_id = bucket["run_id"]

        rag_entry: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "test_type": test_type,
            "material": material,
            "duration_s": round(duration_s, 4),
            "status": status,
            "chunks_retrieved_count": len(retrieved_chunks),
            "retrieved_chunks": [
                {
                    "source_file": c.get("source_file", c.get("metadata", {}).get("filename", "Spec")),
                    "section": c.get("metadata", {}).get("section", "Clause"),
                    "score": c.get("score", c.get("adjusted_score", 0.0)),
                    "content_snippet": c.get("content", "")[:350],
                }
                for c in retrieved_chunks
            ],
            "knowledge_graph_rules": graph_rules,
            "review_guide_rules": review_guide_rules,
            "metadata": metadata or {},
        }

    # Persist to data/parsed/rag_traces_latest.json
    try:
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        latest_file = PARSED_DIR / "rag_traces_latest.json"
        latest_file.write_text(json.dumps(rag_entry, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write rag_traces_latest.json: {e}")

    # Append to data/logs/rag_traces.jsonl
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "rag_traces.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rag_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to append to rag_traces.jsonl: {e}")

    pipeline_logger.info(
        f"[RAG TRACE] Retrieved {len(retrieved_chunks)} chunks + Graph rules in {duration_s:.2f}s | run={run_id}"
    )

    return rag_entry


def get_current_traces() -> list[dict[str, Any]]:
    """Return all traces recorded by the calling thread's run."""
    with _lock:
        return list(_bucket()["traces"])

