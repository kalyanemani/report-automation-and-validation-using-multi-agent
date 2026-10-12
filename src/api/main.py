from __future__ import annotations

"""
QDR·AI — FastAPI Backend Server
──────────────────────────────────
REST API server for Quality Document Review inspection & search endpoints.

Endpoints:
  - GET /          : Health & info
  - GET /health    : Health check & loaded model status
  - POST /inspect  : Run multi-agent quality inspection pipeline
  - GET /search    : Search ChromaDB vector store
"""

import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.orchestrator import run_inspection_pipeline
from src.config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME,
    BACKEND_PORT,
    STRICT_LLM_MODE,
    EmbeddingModelError,
    LLMAgentError,
)
from src.vectorstore.store import VectorStoreManager

logger = logging.getLogger("qdr_ai.api")

app = FastAPI(
    title="QDR·AI — Inspection & RAG API",
    description="REST API server for 3rd-party Supplier Quality Review Inspection & Vector Store Retrieval",
    version="2.0.0",
)

# Enable CORS for Streamlit / external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InspectionRequest(BaseModel):
    submittal_text: str = Field(..., description="Plain text extracted from supplier submittal PDF")
    submittal_filename: str = Field(default="submittal.pdf", description="Filename of submittal PDF")
    submittal_filepath: str = Field(default="", description="Filepath of submittal PDF if available locally")
    guide_code: str = Field(default="references_procedures", description="Review guide identifier")
    supplier: str = Field(default="Supplier", description="Supplier / Contractor name")
    doc_no: str = Field(default="—", description="Document / Report number")
    revision: str = Field(default="0", description="Revision code")
    reviewer: str = Field(default="Quality Inspector", description="Reviewer name")
    gen_params_input: Optional[Dict[str, str]] = Field(default_factory=dict, description="Document control inputs")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    guide_code: str = Field(default="references_procedures", description="Vector collection guide code")
    k: int = Field(default=5, description="Number of results to return")


@app.get("/")
def root_endpoint():
    return {
        "system": "QDR·AI Quality Document Review Engine",
        "status": "running",
        "backend_url": f"http://localhost:{BACKEND_PORT}",
        "docs": f"http://localhost:{BACKEND_PORT}/docs",
    }


@app.on_event("startup")
def _warm_embeddings() -> None:
    """
    Load BGE-M3 at boot rather than on the first retrieval.

    Cold-loading it inside a run costs ~17s on the first search_specs call
    (measured 18.3s cold vs 1.3s warm). Paying it once at startup keeps that
    out of every inspection's critical path.
    """
    try:
        import time as _t
        from src.vectorstore.embeddings import get_embedding_model
        _t0 = _t.time()
        get_embedding_model()
        logger.info(f"Embedding model warmed in {_t.time() - _t0:.1f}s")
    except Exception as exc:
        # Never block startup: a failure here surfaces on the first retrieval,
        # where EmbeddingModelError already reports it properly.
        logger.warning(f"Embedding warm-up skipped: {exc}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_store": "ChromaDB",
        "strict_llm_mode": STRICT_LLM_MODE,
    }


@app.post("/inspect")
def inspect_document_endpoint(req: InspectionRequest):
    if not req.submittal_text.strip():
        raise HTTPException(status_code=400, detail="submittal_text cannot be empty")
    
    try:
        result = run_inspection_pipeline(
            submittal_text=req.submittal_text,
            submittal_filename=req.submittal_filename,
            submittal_filepath=req.submittal_filepath,
            guide_code=req.guide_code,
            supplier=req.supplier,
            doc_no=req.doc_no,
            revision=req.revision,
            reviewer=req.reviewer,
            gen_params_input=req.gen_params_input or {},
        )
        return result
    except LLMAgentError as e:
        # Strict mode: an agent could not produce a usable result, so no report
        # exists. 422 rather than 500 — the pipeline worked correctly by
        # refusing, this is not an internal fault.
        logger.error(f"Inspection aborted — {e.call_id}: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "agent_failure",
                "call_id": e.call_id,
                "message": str(e),
                "hint": "STRICT_LLM_MODE is enabled; no fallback values were substituted.",
            },
        )
    except EmbeddingModelError as e:
        logger.error(f"Inspection aborted — embedding model: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "retrieval_unavailable", "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Inspection pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inspect/annotated-pdf")
def inspect_and_annotate_endpoint(req: InspectionRequest):
    """Run pipeline and return the generated annotated PDF directly."""
    from fastapi.responses import Response
    from src.ingestion.annotator import annotate_pdf_with_findings
    import os
    
    if not req.submittal_filepath or not os.path.exists(req.submittal_filepath):
        raise HTTPException(status_code=400, detail="Valid submittal_filepath required for PDF annotation")
        
    try:
        result = run_inspection_pipeline(
            submittal_text=req.submittal_text,
            submittal_filename=req.submittal_filename,
            submittal_filepath=req.submittal_filepath,
            guide_code=req.guide_code,
            supplier=req.supplier,
            doc_no=req.doc_no,
            revision=req.revision,
            reviewer=req.reviewer,
            gen_params_input=req.gen_params_input or {},
        )
        pdf_bytes = annotate_pdf_with_findings(
            pdf_path=req.submittal_filepath,
            findings=result.get("findings", []),
            disposition=result.get("disposition", "REJECTED"),
            score=result.get("score", 0),
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=annotated_{req.submittal_filename}"},
        )
    except Exception as e:
        logger.error(f"Annotated PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search_vector_store_endpoint(query: str, guide_code: str = "references_procedures", k: int = 5):
    try:
        store = VectorStoreManager()
        results = store.search(guide_code=guide_code, query=query, k=k, return_parents=True)
        return {"query": query, "guide_code": guide_code, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
def get_pipeline_logs(lines: int = 50):
    """Retrieve recent pipeline log entries."""
    from src.utils.logger import PIPELINE_LOG_FILE
    if not PIPELINE_LOG_FILE.exists():
        return {"log_file": str(PIPELINE_LOG_FILE), "lines": []}
    
    content = PIPELINE_LOG_FILE.read_text(encoding="utf-8").splitlines()
    recent = content[-lines:] if len(content) > lines else content
    return {"log_file": str(PIPELINE_LOG_FILE), "count": len(recent), "lines": recent}


