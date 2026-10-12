from __future__ import annotations

"""
QDR·AI — LangGraph Multi-Agent Orchestrator (2 LLM calls)
──────────────────────────────────────────────────────────
Orchestrates the review pipeline. Retrieval is driven by metadata OCR actually
found in the submittal plus the Review Guide checklist, and compliance
arithmetic stays deterministic:

  1. OCR & Table Extraction (PyMuPDF + Tesseract + OpenCV)            [0 LLM]
  2. RAG Context Build — OCR metadata + checklist → BGE-M3 retrieval  [0 LLM]
  3. Finding Analyzer — establishes cited hardness limits from the
     retrieved clauses and groups non-conformances by root cause      [1 LLM]
  4. Python Rule Engine — applies those limits to every reading       [0 LLM]
  5. Annotation Builder (Deterministic PDF Coordinate Highlighting)   [0 LLM]
  6. Report Synthesizer — executive narrative; scoring/disposition
     remain deterministic                                             [1 LLM]

Supports AGENT_MODE toggle:
  - "langgraph" → this multi-agent pipeline
  - "rule_based" → Legacy hardcoded rule engine

Under STRICT_LLM_MODE (see src/config.py) an agent that cannot produce a usable
result aborts the run instead of substituting hardcoded limits or templated
prose, and a material present in the document with no established limits is a
hard failure rather than a skipped evaluation.
"""

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from src.agents.state import ReviewState
from src.config import AGENT_MODE, STRICT_LLM_MODE, LLMAgentError
from src.utils.logger import pipeline_logger
from src.utils import progress as pipeline_progress
from src.utils import stage_log

logger = logging.getLogger("qdr_ai.orchestrator")


# ── LangGraph Pipeline (2 LLM calls: analyzer + summary) ──────────────

def _build_langgraph_pipeline() -> Any:
    """
    Build the LangGraph StateGraph for the two-LLM-call pipeline.

    ARCHITECTURE
    ────────────
        ocr_extract        Python   OCR → rows, readings, materials, test blocks
        build_rag_context  Python   metadata + Review Guide checklist → RAG retrieval
        analyze_findings   *** LLM #1 ***  retrieved specs + reading summary
                                           → hardness limits (cited) +
                                             non-conformance grouping
        audit_rules        Python   applies those limits to every reading
        build_annotations  Python   findings → pixel boxes
        synthesize_report  *** LLM #2 ***  executive narrative; scoring and
                                           disposition remain deterministic

    Retrieval runs AFTER OCR so the query is built from what the document
    actually contains — its materials, drawing/spool numbers and stated test
    block ranges — rather than from a keyword scan of raw text. The previous
    graph searched specs before OCR had run, so retrieval could not be informed
    by the submittal's own metadata.

    The old LLM Call #1 (Requirement Parser) is gone. Establishing the limits is
    now part of the analyzer's job, which removes a 169-300s round trip whose
    output was provably identical to a hardcoded constant — run 1 and run 2 of
    2026-08-20 produced byte-identical findings, one from that call and one from
    its fallback.

    Compliance arithmetic stays in Python. A submittal carries ~9,000 readings
    against a 4096-token context, so per-reading comparison by an LLM is both
    impossible here and non-reproducible; `audit_rules` applies the analyzer's
    established limits to every reading exactly.
    """
    from src.agents.ocr_extractor import ocr_extractor_node
    from src.agents.llm_batch_reviewer import llm_batch_review_node
    from src.agents.annotation_builder import annotation_builder_node
    from src.agents.report_synthesizer_agent import critic_and_synthesize_node

    from src.agents.tools import classify_document, extract_requirements, search_specs
    from src.agents.materials import canonical_grade, retrieval_terms
    from src.utils.metrics import StageTracker
    from src.utils.llm_tracer import reset_llm_traces, current_run_id

    def ocr_entry_node(state: ReviewState) -> dict:
        """Entry: reset traces, then run OCR extraction first (before retrieval)."""
        reset_llm_traces()
        # Registers this thread's run id for the per-stage log before the first
        # StageTracker is entered, so every stage of this run is attributed to
        # the same id the LLM traces and the usage log record.
        stage_log.begin_run(
            current_run_id(),
            document=state.get("submittal_filename", ""),
            mode=AGENT_MODE,
        )
        pipeline_logger.info(
            "=================== PIPELINE EXECUTION STARTED (OCR -> RAG -> ANALYZER -> RULES -> REPORT) ==================="
        )
        return ocr_extractor_node(state)

    def build_rag_context_node(state: ReviewState) -> dict:
        """
        Deterministic. Builds the retrieval context for the analyzer call from
        (a) metadata OCR actually found in the submittal and (b) the Review
        Guide checklist the user ticked.
        """
        pipeline_progress.set_stage("build_rag_context")
        submittal_text = state.get("submittal_text", "")
        submittal_filename = state.get("submittal_filename", "submittal.pdf")
        guide_code = state.get("guide_code", "references_procedures")
        checked_criteria = state.get("requirements", [])
        pages = state.get("page_ocr_results", []) or []

        # ── 1. Classify + harvest metadata FROM THE OCR RESULT ────────────
        with StageTracker("classify_document", f"filename={submittal_filename}") as st:
            classification = classify_document.invoke({"submittal_text": submittal_text})
            doc_type = classification["doc_type"]
            test_method = classification["test_method"]
            primary_grade = classification["material_grade"]

            # Materials as OCR saw them: per-row hints and per-page specs are
            # direct evidence, unlike a keyword scan of the whole text blob.
            detected: list[str] = []
            drawings: set[str] = set()
            spools: set[str] = set()
            test_blocks: set[str] = set()
            stated_ranges: set[tuple] = set()

            for p in pages:
                if p.drawing_no:
                    drawings.add(p.drawing_no)
                if p.spool_no:
                    spools.add(p.spool_no)
                if p.test_block_info:
                    test_blocks.add(p.test_block_info)
                if p.acceptance_range:
                    stated_ranges.add(tuple(p.acceptance_range))
                if getattr(p, "material_spec", ""):
                    detected.append(p.material_spec.upper())
                for row in p.table_rows:
                    if row.material_hint:
                        detected.append(row.material_hint.upper())

            # Canonicalise via the shared mapping so a P-Number designation in
            # the source is recognised as the grade it denotes. The previous
            # inline substring table matched "P1" inside "P11" and had no
            # knowledge of P-Numbers at all.
            norm: list[str] = []
            for token in detected:
                grade = canonical_grade(token)
                if grade and grade not in norm:
                    norm.append(grade)

            # Fall back to the raw-text scan only if OCR found no grade at all.
            if not norm:
                upper_sample = submittal_text[:20000].upper()
                for canon, keys in (
                    ("P91", ("P91", "F91", "T91", "9CR")),
                    ("P22", ("P22", "F22", "T22", "2.25CR")),
                    ("P11", ("P11", "F11", "T11", "1.25CR")),
                    ("CS", ("CS", "A106", "A105", "WPB", "CARBON STEEL")),
                ):
                    if any(k in upper_sample for k in keys):
                        norm.append(canon)

            detected_materials = norm
            st.details += (
                f" | doc_type={doc_type}, primary={primary_grade}, "
                f"ocr_materials={'/'.join(detected_materials) or 'NONE'}, "
                f"drawings={len(drawings)}, spools={len(spools)}"
            )

        if STRICT_LLM_MODE and not detected_materials:
            raise LLMAgentError(
                "build_rag_context",
                "OCR found no material grade anywhere in the submittal. Retrieval cannot "
                "be targeted and limits cannot be established without knowing the grade.",
            )

        # ── 2. Review Guide checklist → requirements ──────────────────────
        with StageTracker("extract_requirements", f"guide={guide_code}") as st:
            reqs_raw = extract_requirements.invoke({
                "guide_code": guide_code,
                "checked_criteria": checked_criteria,
            })
            requirements = [r["requirement"] for r in reqs_raw]
            st.details += f" | requirements_count={len(requirements)}"

        # ── 3. RAG retrieval driven by that metadata + checklist ──────────
        with StageTracker("search_specs", f"materials={detected_materials}") as st:
            spec_results: list[dict] = []
            seen: set = set()

            def _add(results: list[dict]) -> None:
                for r in results:
                    h = hash(r.get("content", "")[:100])
                    if h not in seen:
                        seen.add(h)
                        spec_results.append(r)

            queries: list[tuple[str, int]] = []

            # (a) One query per material OCR observed, carrying BOTH
            #     vocabularies. The governing limits in this corpus live in a
            #     table keyed by ASME P-Number ("P5A", "P4"), which contains no
            #     occurrence of "P22"/"P11" at all — a grade-only query matches
            #     it by luck if it matches at all. See src/agents/materials.py.
            for mat in detected_materials:
                queries.append((
                    f"{retrieval_terms(mat)} {test_method} hardness acceptance criteria "
                    f"limits maximum minimum HBW weld base metal HAZ post weld heat treatment",
                    4,
                ))

            # (b) The limit table's own phrasing. BGE-M3 discriminates poorly
            #     between material names here — the same "Alloy 91 Acceptance
            #     Criteria" chunk was the top hit for the P22, P11 and CS
            #     queries alike — so query the structure that actually holds the
            #     numbers rather than the grade name.
            queries.append((
                "table material grade Brinell Hardness HB Max permissible hardness "
                "values base weld HAZ acceptance criteria P-Number",
                5,
            ))

            # (c) The stated in-submittal acceptance ranges — retrieving on the
            #     numbers the document itself claims surfaces the clause that
            #     governs them (and any clause that contradicts them).
            for lo, hi in list(stated_ranges)[:3]:
                queries.append((
                    f"hardness acceptance range {lo} to {hi} HBW permissible limit clause",
                    2,
                ))

            # (d) The ticked Review Guide checklist items, verbatim — these are
            #     the criteria the reviewer is actually asserting must be met.
            for req in requirements[:6]:
                q = req.strip()
                if len(q) > 20:
                    queries.append((q[:300], 2))

            # (e) Test block identifiers, when the report names them.
            for tb in list(test_blocks)[:2]:
                queries.append((f"test block {tb} hardness requirement", 2))

            for q, k in queries:
                try:
                    _add(search_specs.invoke({"query": q, "guide_code": guide_code, "k": k}))
                except Exception as e:
                    pipeline_logger.warning(f"Retrieval query failed ({q[:40]}...): {e}")

            st.details += f" | queries={len(queries)}, chunks_retrieved={len(spec_results)}"

        if STRICT_LLM_MODE and not spec_results:
            raise LLMAgentError(
                "build_rag_context",
                f"No specification clauses retrieved from collection '{guide_code}'. The "
                f"analyzer would have no evidence from which to establish hardness limits.",
            )

        pipeline_logger.info(
            f"RAG Context: {len(spec_results)} clauses retrieved from {len(queries)} metadata-driven "
            f"queries (materials={detected_materials}, spools={len(spools)}, "
            f"stated_ranges={len(stated_ranges)}, checklist_items={len(requirements)})"
        )

        return {
            "doc_type": doc_type,
            "test_method": test_method,
            "material_grade": primary_grade,
            "detected_materials": detected_materials,
            "requirements": requirements,
            "retrieved_criteria": [s["content"] for s in spec_results],
            "retrieved_context": spec_results,
            "current_agent": "build_rag_context",
        }

    # Build the StateGraph
    graph = StateGraph(ReviewState)

    graph.add_node("ocr_extract", ocr_entry_node)                       # Python
    graph.add_node("build_rag_context", build_rag_context_node)         # Python + RAG
    graph.add_node("llm_batch_review", llm_batch_review_node)           # LLM-only batches
    graph.add_node("build_annotations", annotation_builder_node)        # Python
    graph.add_node("synthesize_report", critic_and_synthesize_node)     # Python

    graph.set_entry_point("ocr_extract")
    graph.add_edge("ocr_extract", "build_rag_context")
    graph.add_edge("build_rag_context", "llm_batch_review")
    graph.add_edge("llm_batch_review", "build_annotations")
    graph.add_edge("build_annotations", "synthesize_report")
    graph.add_edge("synthesize_report", END)

    return graph.compile()


# ── Legacy Pipeline (Rule-Based Fallback) ──────────────────────────────

def _run_legacy_pipeline(state: ReviewState) -> ReviewState:
    """Run the legacy hardcoded rule-based pipeline."""
    from src.agents.qa_inspector_agent_legacy import run_qa_inspection
    return run_qa_inspection(state)


# ── Main Entry Point ──────────────────────────────────────────────────

_compiled_graph = None


def run_inspection_pipeline(
    submittal_text: str,
    submittal_filename: str = "submittal.pdf",
    submittal_filepath: str = "",
    guide_code: str = "references_procedures",
    guide_name: str = "References and Procedures",
    cat_code: str = "DOC",
    cat_label: str = "Inspection Review Guides",
    supplier: str = "Supplier",
    doc_no: str = "—",
    revision: str = "—",
    reviewer: str = "—",
    gen_params_input: dict[str, str] | None = None,
    checked_criteria: list[str] | None = None,
    **kwargs: Any,
) -> dict:
    """Execute the inspection pipeline end-to-end."""
    if gen_params_input is None:
        gen_params_input = {}
    if checked_criteria is None:
        checked_criteria = []

    initial_state: ReviewState = {
        "submittal_text": submittal_text,
        "submittal_filename": submittal_filename,
        "submittal_filepath": submittal_filepath,
        "guide_code": guide_code,
        "guide_name": guide_name,
        "cat_code": cat_code,
        "cat_label": cat_label,
        "supplier": supplier,
        "doc_no": doc_no,
        "revision": revision,
        "reviewer": reviewer,
        "gen_params_input": gen_params_input,
        "doc_type": "unknown",
        "test_method": "Not identified",
        "material_grade": "Not identified",
        "requirements": checked_criteria,
        "requirement_sources": [],
        "retrieved_criteria": [],
        "retrieved_context": [],
        "page_ocr_results": [],
        "hardness_rules_schema": {},
        "raw_rule_failures": [],
        "grouped_findings": [],
        "critic_verifications": [],
        "material_assignments": [],
        "compliance_findings": [],
        "gen_param_checks": [],
        "findings": [],
        "visual_observations": [],
        "gaps": [],
        "missing_fields": [],
        "ambiguities": [],
        "disposition": "REVISE AND RESUBMIT",
        "score": 0,
        "verdict": "",
        "summary": "",
        "pipeline_metrics": [],
        "llm_traces": [],
        "llm_reasoning_traces": [],
        "messages": [],
        "current_agent": "start",
        "errors": [],
    }

    mode = AGENT_MODE.lower()
    logger.info(f"Pipeline mode: {mode}")

    if mode == "langgraph":
        global _compiled_graph
        if _compiled_graph is None:
            pipeline_logger.info("Compiling LangGraph multi-agent pipeline (2 LLM calls: analyzer + summary)...")
            _compiled_graph = _build_langgraph_pipeline()

        try:
            final_state = _compiled_graph.invoke(initial_state)
            pipeline_logger.info("=================== PIPELINE FINISHED ===================")
            stage_log.end_run("OK")
            return final_state
        except Exception as e:
            pipeline_logger.error(f"LangGraph pipeline failed: {e}")
            logger.error(f"LangGraph pipeline error: {e}", exc_info=True)
            stage_log.end_run("ERROR", error=str(e))
            raise e
    else:
        return _run_legacy_pipeline(initial_state)
