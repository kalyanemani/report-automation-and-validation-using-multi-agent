from __future__ import annotations

"""
QDR·AI — Single-Flight Pipeline Lock
──────────────────────────────────────
Streamlit re-executes the whole script on every rerun, and a hard page reload
can't interrupt a thread already blocked inside a long HTTP call — so a user
reloading mid-run and clicking "Inspect Document" again ends up running the
pipeline twice, concurrently, on the same CPU.

WHY THE PREVIOUS VERSION DID NOT PREVENT THAT
─────────────────────────────────────────────
It was a plain bool released from the Streamlit *script* thread's `finally`,
while the pipeline itself runs in a separate daemon worker thread. On a browser
reload Streamlit raises inside the script thread at its next `st.*` call, the
`finally` ran, the flag cleared — and the worker carried on regardless. The next
click then passed `try_start()` and started a second concurrent pipeline. That
was observed on 2026-08-20: runs at 10:48:42 and 10:52:37 overlapped, each
LLM call taking ~2.6x its uncontended time (561s and 574s against a measured
215s) and both marginal calls timing out as a direct result.

THE FIX
───────
The lock now owns a reference to the worker thread and treats *thread liveness*
as the source of truth. Releasing is idempotent and, crucially, a release from
the script thread cannot free the lock while the worker it started is still
alive. The worker releases the lock itself when it actually finishes.
"""

import threading
from typing import Optional

_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
_running = False


def try_start() -> bool:
    """
    Attempt to claim the lock. Returns False if a run is already in flight.

    "In flight" means either the flag is set or a previously registered worker
    thread is still alive — so an abandoned script thread cannot hand the lock
    to a second run while the first is still computing.
    """
    global _running
    with _lock:
        if _running or (_worker is not None and _worker.is_alive()):
            return False
        _running = True
        return True


def register_worker(thread: threading.Thread) -> None:
    """
    Attach the thread actually executing the pipeline.

    Once registered, its liveness governs `is_running()` regardless of what
    happens to the Streamlit script thread that started it.
    """
    global _worker
    with _lock:
        _worker = thread


def finish(thread: Optional[threading.Thread] = None) -> None:
    """
    Release the lock.

    Pass the worker thread when releasing from that worker (the authoritative
    path). Called with no argument — e.g. from the script thread's `finally` —
    it refuses to release while the registered worker is still alive, which is
    the specific hole that allowed concurrent runs.
    """
    global _running, _worker
    with _lock:
        if thread is None and _worker is not None and _worker.is_alive():
            # Script thread unwound (Streamlit rerun / reload) but the pipeline
            # is still going. Keep the lock held; the worker will release it.
            return
        _running = False
        _worker = None


def is_running() -> bool:
    with _lock:
        return _running or (_worker is not None and _worker.is_alive())
