from __future__ import annotations

"""
QDR·AI — Hybrid Retriever (Vector + Graph + Review Guide)
──────────────────────────────────────────────────────────
Combines:
  1. Semantic Vector Search (ChromaDB + BGE-M3 parent-child retrieval)
  2. Deterministic Knowledge Graph Traversal (Entities, Limits & Multipliers)
  3. Dynamic Review Guide Checklist Rules (Pasted / Uploaded criteria)

Produces unified, hallucination-free context for AI Review and Validation Agents.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from src.config import EMBED_TOP_K
from src.knowledge_graph.graph_builder import KnowledgeGraph, get_knowledge_graph
from src.vectorstore.store import VectorStoreManager

logger = logging.getLogger("qdr_ai.hybrid_retriever")


class HybridRetriever:
    """
    Unified Hybrid RAG Retriever combining:
      - ChromaDB Vector Store (unstructured text / tables)
      - Knowledge Graph (structured rules, limits, formulas)
      - Active Review Guide (runtime checklist criteria)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.knowledge_graph = knowledge_graph or get_knowledge_graph()

    def retrieve(
        self,
        query: str,
        test_type: str = "hydrostatic",
        material: Optional[str] = None,
        collection_name: str = "references_procedures",
        topics: Optional[List[str]] = None,
        active_review_guide: Optional[Dict[str, Any]] = None,
        top_k: int = EMBED_TOP_K,
    ) -> Dict[str, Any]:
        """
        Execute hybrid search across Vector DB, Knowledge Graph, and dynamic Review Guide.

        Returns structured dict with:
          - 'vector_chunks': list of retrieved text / table chunks
          - 'graph_rules': structured dictionary of applicable graph constraints
          - 'review_guide_items': active items from runtime review guide
          - 'formatted_context': unified prompt-ready markdown string
        """
        # ── 1. Vector Search ─────────────────────────────────────────
        start_time = time.time()
        vector_results = []
        try:
            # Search combined references_procedures collection
            raw_results = self.vector_store.search(
                guide_code=collection_name,
                query=query,
                k=top_k,
                return_parents=True,
            )
            # Filter / boost results by test_type if available
            for res in raw_results:
                meta = res.get("metadata", {})
                chunk_tt = meta.get("test_type", "").lower()
                chunk_topics = meta.get("topics", "")

                # Score boost if topic matches
                score = res.get("score", 1.0)
                if test_type and chunk_tt in (test_type.lower(), "general_spec"):
                    score -= 0.1
                if topics:
                    for t in topics:
                        if t.lower() in chunk_topics.lower():
                            score -= 0.15

                vector_results.append({**res, "adjusted_score": round(score, 4)})

            # Sort by adjusted score (lower distance is better)
            vector_results.sort(key=lambda x: x.get("adjusted_score", 1.0))
            vector_results = vector_results[:top_k]

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        # ── 2. Knowledge Graph Traversal ─────────────────────────────
        graph_summary = self.knowledge_graph.get_summary_context(
            test_type=test_type, material=material
        )

        specific_rules = {}
        if test_type.lower() in ("hydrostatic", "hydr"):
            is_ss = bool(material and ("SS" in material.upper() or "STAINLESS" in material.upper()))
            specific_rules["water_chemistry"] = self.knowledge_graph.find_water_chemistry_limits(is_stainless_steel=is_ss)
            specific_rules["pressure_multipliers"] = {
                "Section_I": self.knowledge_graph.find_pressure_multiplier_rule("Section I"),
                "Section_VIII": self.knowledge_graph.find_pressure_multiplier_rule("Section VIII"),
            }
            specific_rules["gauge_requirements"] = self.knowledge_graph.find_gauge_requirements()

        elif test_type.lower() in ("hardness", "hrdr") and material:
            specific_rules["material_rules"] = self.knowledge_graph.find_rules_for_material(material, test_type="Hardness")

        # ── 3. Review Guide Context ──────────────────────────────────
        review_guide_items = []
        if active_review_guide and "items" in active_review_guide:
            review_guide_items = active_review_guide["items"]

        # ── 4. Build Unified Formatted Context ───────────────────────
        context_blocks = []

        # Block A: Active Review Guide
        if review_guide_items:
            mode_desc = active_review_guide.get("mode", "all_criteria")
            fn = active_review_guide.get("filename", "Active Review Criteria")
            context_blocks.append(
                f"### [RUNTIME REVIEW CRITERIA — {fn.upper()}] (Mode: {mode_desc})\n"
                + "\n".join(f"- {item}" for item in review_guide_items)
            )

        # Block B: Deterministic Knowledge Graph Rules
        if graph_summary:
            context_blocks.append(f"### [DETERMINISTIC SPECIFICATION RULES]\n{graph_summary}")

        # Block C: Semantic Vector Chunks & Tables
        if vector_results:
            context_blocks.append("### [PROCEDURAL SPECIFICATION EXCERPTS & TABLES]")
            for i, r in enumerate(vector_results, 1):
                source = r.get("source_file", r.get("metadata", {}).get("filename", "Spec"))
                sec = r.get("metadata", {}).get("section", "Clause")
                content = r.get("content", "").strip()
                context_blocks.append(f"#### Excerpt {i} ({source} — {sec}):\n{content}")

        formatted_context = "\n\n".join(context_blocks)
        elapsed = time.time() - start_time

        from src.utils.llm_tracer import record_rag_trace
        record_rag_trace(
            query=query,
            test_type=test_type,
            material=material,
            retrieved_chunks=vector_results,
            graph_rules=specific_rules,
            review_guide_rules=review_guide_items,
            duration_s=elapsed,
            status="SUCCESS",
        )

        return {
            "query": query,
            "test_type": test_type,
            "material": material,
            "vector_chunks": vector_results,
            "graph_rules": specific_rules,
            "review_guide_items": review_guide_items,
            "formatted_context": formatted_context,
            "duration_s": round(elapsed, 3),
        }


# Global singleton
_hybrid_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_instance
    if _hybrid_instance is None:
        _hybrid_instance = HybridRetriever()
    return _hybrid_instance
