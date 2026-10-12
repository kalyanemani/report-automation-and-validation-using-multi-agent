"""
QDR·AI — Configuration Module
─────────────────────────────
Central configuration loaded from environment variables / .env file.
Supports GPU-accelerated Remote Ollama (http://172.16.116.228:11434) with local fallback.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# ── Load .env file if present ────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chromadb")))
DOCUMENTS_DIR = DATA_DIR / "documents"
PARSED_DIR = DATA_DIR / "parsed"
LOGS_DIR = DATA_DIR / "logs"
REPORTS_DIR = DATA_DIR / "reports"

# Create directories
for d in [DATA_DIR, CHROMA_PERSIST_DIR, DOCUMENTS_DIR, PARSED_DIR, LOGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama Host & Model Priority Configuration ──────────────────────
# Preference 1: Remote Primary Server (172.16.116.228:11435)
# Preference 2: Localhost (localhost:11434)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_GPU_HOST", "http://172.16.116.228:11435"))
OLLAMA_GPU_HOST = OLLAMA_BASE_URL
OLLAMA_LOCAL_HOST = os.getenv("OLLAMA_LOCAL_HOST", "http://localhost:11434")

# Primary models: qwen2.5:7b-instruct (Text, Schema & Reasoning) with fallback
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "qwen2.5:7b-instruct")
REASONING_MODEL = os.getenv("REASONING_MODEL", "qwen2.5:7b-instruct")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_LLM_MODEL)
OLLAMA_LOCAL_MODEL = os.getenv("OLLAMA_LOCAL_MODEL", "qwen2.5:3b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")

OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300")) if os.getenv("OLLAMA_TIMEOUT") and os.getenv("OLLAMA_TIMEOUT").lower() != "none" else None
OLLAMA_VISION_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT")) if os.getenv("OLLAMA_VISION_TIMEOUT") and os.getenv("OLLAMA_VISION_TIMEOUT").lower() != "none" else None

# Material/compliance review no longer calls the LLM once per page. Deterministic
# rule-based extraction runs first for every row; only ambiguous/borderline rows
# are batched into one (or a few) LLM calls per document. Real per-candidate cost
# on this CPU-only host varies with its thermal/load state (observed ~20-90s),
# and a batch that times out wastes just as much time as one that succeeds — so
# this is sized with headroom for a full candidate-budget-sized batch (see
# MAX_LLM_CANDIDATES_MATERIAL / MAX_LLM_CANDIDATES_COMPLIANCE below) to actually
# finish rather than fail right at the boundary.
OLLAMA_BATCH_TIMEOUT = int(os.getenv("OLLAMA_BATCH_TIMEOUT", "300"))

# Finding Analyzer (LLM Call #2) sends the whole-document failure summary in one
# request and is by far the largest generation in the pipeline — measured at 215s
# uncontended and 561-574s with two pipeline runs sharing this CPU. It previously
# hardcoded 600 at the call site; keep that as the default but make it tunable,
# since it is the one call whose timeout genuinely needs to exceed OLLAMA_TIMEOUT.
OLLAMA_ANALYZER_TIMEOUT = int(os.getenv("OLLAMA_ANALYZER_TIMEOUT", "600"))

# Context window for the analyzer call. Ollama's default is 4096, and with the
# retrieved clauses + reading summary the prompt reached ~3,900 tokens — leaving
# ~187 tokens of output room for a schema needing 600-1,000. The model then
# emitted a collapsed shape with fabricated numbers, which is what a truncated
# generation looks like. 8192 gives the response real space. Raising this costs
# prefill time and RAM (both cheap here); lowering it risks silent truncation.
OLLAMA_ANALYZER_NUM_CTX = int(os.getenv("OLLAMA_ANALYZER_NUM_CTX", "8192"))

# This machine's Ollama runs 100% on CPU with no GPU offload — two concurrent
# generate calls contend for the same cores instead of running side by side,
# so both end up slower than running them one at a time. Keep this at 1 unless
# Ollama is later moved to a GPU host that can actually serve parallel requests.
OLLAMA_BATCH_CONCURRENCY = int(os.getenv("OLLAMA_BATCH_CONCURRENCY", "1"))

# LLM-only compliance review sends table rows sequentially in bounded batches.
# Keep batches modest: schema-constrained responses must return one verdict per
# row, and CPU-only Ollama can otherwise spend minutes decoding one response.
LLM_AUDIT_BATCH_SIZE = int(os.getenv("LLM_AUDIT_BATCH_SIZE", "12"))

# Hard cap on how many ambiguous/borderline rows get sent to the LLM per stage,
# per document. This is what makes total runtime predictable regardless of how
# messy a given document's table turns out to be — rows beyond the cap keep
# their rule-based result instead of blowing the time budget open-ended.
MAX_LLM_CANDIDATES_MATERIAL = int(os.getenv("MAX_LLM_CANDIDATES_MATERIAL", "0"))
MAX_LLM_CANDIDATES_COMPLIANCE = int(os.getenv("MAX_LLM_CANDIDATES_COMPLIANCE", "2"))

# Set local Ollama server (localhost:11434) as default override if true
OLLAMA_FORCE_LOCAL = os.getenv("OLLAMA_FORCE_LOCAL", "false").lower() == "true"

VISION_DPI = int(os.getenv("VISION_DPI", "120"))
_cached_ollama_host: Optional[str] = None
_last_probe_time: float = 0.0
_PROBE_TTL_SECONDS: float = 30.0


def get_ollama_host() -> str:
    """
    Returns active Ollama host.
    Prefers the configured Primary server (http://172.16.116.228:11435) and falls back to local
    (http://localhost:11434) if it is unreachable. Caches the reachability result for 30s.
    """
    global _cached_ollama_host, _last_probe_time
    if OLLAMA_FORCE_LOCAL:
        return OLLAMA_LOCAL_HOST

    now = time.time()
    if _cached_ollama_host and (now - _last_probe_time) < _PROBE_TTL_SECONDS:
        return _cached_ollama_host

    import requests
    # 1. Prefer the primary server (172.16.116.228:11435)
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if r.status_code == 200:
            _cached_ollama_host = OLLAMA_BASE_URL
            _last_probe_time = now
            return OLLAMA_BASE_URL
    except Exception:
        pass

    # 2. Fall back to the local server if primary is unreachable
    try:
        r = requests.get(f"{OLLAMA_LOCAL_HOST}/api/tags", timeout=2.0)
        if r.status_code == 200:
            _cached_ollama_host = OLLAMA_LOCAL_HOST
            _last_probe_time = now
            return OLLAMA_LOCAL_HOST
    except Exception:
        pass

    return OLLAMA_LOCAL_HOST


def get_active_model_name(host: Optional[str] = None) -> str:
    """Returns the optimal model name for the given/active host with fallback."""
    active_host = host or get_ollama_host()
    if "172.16.116.228" in active_host:
        return DEFAULT_LLM_MODEL  # qwen2.5:7b-instruct
    return OLLAMA_LOCAL_MODEL  # qwen2.5:3b fallback for localhost


# ── Vision Model Host (Primary 11435 or Localhost fallback) ─────────
OLLAMA_VISION_HOST = os.getenv("OLLAMA_VISION_HOST", OLLAMA_BASE_URL)


def get_vision_ollama_host() -> str:
    """Returns Ollama host specifically for Vision LLM."""
    host = get_ollama_host()
    return host


# Backward-compatible global default
OLLAMA_HOST = get_ollama_host()

# ── Tesseract OCR ────────────────────────────────────────────────────
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

# ── Embedding Model ──────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
EMBED_TOP_K = int(os.getenv("EMBED_TOP_K", "15"))

# ── Agent Pipeline Mode ──────────────────────────────────────────────
# "langgraph" = New LLM multi-agent pipeline (Material Resolver + Compliance Auditor)
# "rule_based" = Legacy hardcoded rule engine (fast, no LLM required)
AGENT_MODE = os.getenv("AGENT_MODE", "langgraph")

# ── Strict LLM Mode ──────────────────────────────────────────────────
# When true, an LLM agent that fails (timeout, unparseable response, schema
# mismatch) raises LLMAgentError instead of silently substituting hardcoded
# constants or templated prose. The point is that a run either reflects real
# agent reasoning or it fails visibly — a report built from fallback data is
# indistinguishable from a real one, which is not acceptable when the output
# is a REJECTED disposition on a supplier submittal.
#
# Note this trades availability for honesty: on a CPU-only host where calls
# routinely approach OLLAMA_TIMEOUT, expect runs to fail outright rather than
# complete with substituted values. Raise OLLAMA_TIMEOUT (or move Ollama onto
# a GPU) before enabling this in anger.
STRICT_LLM_MODE = os.getenv("STRICT_LLM_MODE", "false").lower() == "true"


class LLMAgentError(RuntimeError):
    """
    An LLM agent could not produce a usable result.

    Raised only under STRICT_LLM_MODE. Carries the call_id so the UI can name
    which agent failed rather than reporting a generic pipeline error.
    """

    def __init__(self, call_id: str, message: str) -> None:
        self.call_id = call_id
        super().__init__(f"[{call_id}] {message}")


class EmbeddingModelError(RuntimeError):
    """
    The real embedding model could not be loaded.

    Always raised, independent of STRICT_LLM_MODE: substituting random vectors
    for BGE-M3 turns RAG retrieval into noise while leaving every downstream
    citation looking legitimate, so there is no mode in which that is a
    reasonable degradation.
    """

# ── Backend ──────────────────────────────────────────────────────────
BACKEND_HOST = os.getenv("QDR_BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("QDR_BACKEND_PORT", "8001"))

# ── Document Types ───────────────────────────────────────────────────
DOC_TYPES = {
    "review_guide":       "Review Guides (R0, R1)",
    "specification":      "Specs, Codes, Procedures",
    "supplier_submittal": "Supplier HRDR Submittals",
    "ne_comments":        "N/E Review Comments",
    "revision":           "Supplier Revised Versions",
}

# ── Chunking Parameters ─────────────────────────────────────────────
CHUNK_SIZE_CHILD = 400       # 200–500 tokens target
CHUNK_SIZE_PARENT = 1500     # 1000–2000 tokens target
CHUNK_OVERLAP = 50           # Overlap tokens target
