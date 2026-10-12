from __future__ import annotations

"""
QDR·AI — Corrective RAG (CRAG) Evaluator Node
────────────────────────────────────────────────
Evaluates the quality and relevance of retrieved BGE-M3 vector store chunks.

If retrieved context relevance is below threshold:
  1. Rewrites / expands query keywords
  2. Executes fallback retrieval attempt
"""

import logging
import re
from src.agents.state import ReviewState
from src.vectorstore.store import VectorStoreManager

logger = logging.getLogger("qdr_ai.agent.crag_evaluator")

RELEVANCE_KEYWORDS = [
    "shall", "must", "requirement", "specification", "acceptance",
    "hardness", "min", "max", "table", "limit", "procedure", "test"
]


def evaluate_retrieval(state: ReviewState) -> dict:
    """LangGraph node: Evaluate retrieval quality and apply CRAG fallback if needed."""
    retrieved_context = state.get("retrieved_context", [])
    guide_code = state.get("guide_code", "references_procedures")
    requirements = state.get("requirements", [])
    doc_type = state.get("doc_type", "unknown")

    logger.info("Executing Corrective RAG (CRAG) Evaluator...")

    # Calculate relevance score based on keyword coverage and vector distance
    good_chunks = 0
    for chunk in retrieved_context:
        content = chunk.get("content", "").lower()
        score = chunk.get("score", 1.0)  # Lower score is better in Chroma L2/Cosine

        # If score is good or keywords are present
        has_keywords = any(kw in content for kw in RELEVANCE_KEYWORDS)
        if score < 0.8 or has_keywords:
            good_chunks += 1

    total_retrieved = len(retrieved_context)
    confidence = (good_chunks / total_retrieved) if total_retrieved > 0 else 0.0

    logger.info(f"CRAG Confidence: {confidence:.2f} ({good_chunks}/{total_retrieved} relevant chunks)")

    # If confidence is low, perform query expansion and re-retrieve
    if confidence < 0.5 and total_retrieved < 30:
        logger.info("CRAG triggered: Low retrieval confidence. Rewriting queries and executing fallback retrieval...")
        
        store = VectorStoreManager()
        expanded_queries = [
            f"mandatory requirements for {doc_type}",
            "acceptance criteria maximum minimum allowable hardness",
            "test method procedure calibration requirements",
        ]
        
        fallback_context = list(retrieved_context)
        seen_contents = {c.get("content") for c in retrieved_context}

        for eq in expanded_queries:
            results = store.search(guide_code, eq, k=3, return_parents=True)
            for r in results:
                if r["content"] not in seen_contents:
                    seen_contents.add(r["content"])
                    fallback_context.append({
                        "content": r["content"][:500],
                        "source": r["metadata"].get("source_file", ""),
                        "section": r["metadata"].get("section", ""),
                        "score": r["score"],
                    })

        logger.info(f"CRAG Fallback expanded context to {len(fallback_context)} chunks.")
        
        return {
            "retrieved_context": fallback_context,
            "retrieved_criteria": [c["content"] for c in fallback_context],
            "current_agent": "crag_evaluator",
        }

    return {
        "current_agent": "crag_evaluator",
    }
