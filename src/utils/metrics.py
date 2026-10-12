from __future__ import annotations

"""
QDR·AI — Latency & Pipeline Health Monitor
─────────────────────────────────────────────
Context manager and timing utilities for stage latency & health tracking.
"""

import time
from typing import Any, Optional
from src.utils.logger import pipeline_logger
from src.utils import progress as pipeline_progress
from src.utils import stage_log


class StageTracker:
    """Context manager to measure execution time & health status of each tool stage."""

    def __init__(self, stage_name: str, details: str = ""):
        self.stage_name = stage_name
        self.details = details
        self.start_time = 0.0
        self.duration_sec = 0.0
        self.status = "OK"  # OK | FALLBACK | ERROR
        self.error_msg: Optional[str] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        pipeline_progress.set_stage(self.stage_name)
        pipeline_logger.info(f"[START] Stage [{self.stage_name}] {self.details}")
        # Every stage in the pipeline is entered through this context manager,
        # so mirroring the three transitions here is what gives the per-stage
        # log (data/logs/stages/) complete coverage without touching the 23
        # call sites individually.
        stage_log.stage_started(self.stage_name, self.details)
        return self

    def set_fallback(self, reason: str):
        self.status = "FALLBACK"
        pipeline_logger.warning(f"[WARN] FALLBACK Stage [{self.stage_name}]: {reason}")
        stage_log.stage_fallback(self.stage_name, reason, started=self.start_time)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_sec = round(time.perf_counter() - self.start_time, 4)
        if exc_type is not None:
            self.status = "ERROR"
            self.error_msg = str(exc_val)
            pipeline_logger.error(f"[FAIL] Stage [{self.stage_name}] after {self.duration_sec}s: {exc_val}")
        else:
            pipeline_logger.info(
                f"[COMPLETE] Stage [{self.stage_name}] in {self.duration_sec}s | Status: {self.status}"
            )
        stage_log.stage_ended(
            self.stage_name,
            self.start_time,
            status=self.status,
            error=self.error_msg or "",
            detail=self.details,
        )

    def to_dict(self) -> dict[str, Any]:
        res = {
            "stage": self.stage_name,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "details": self.details,
        }
        if self.error_msg:
            res["error"] = self.error_msg
        return res
