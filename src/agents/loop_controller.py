from __future__ import annotations

"""
QDR·AI — Loop Engineering Controller
─────────────────────────────────────
Quality-control iterative evaluation loop:

  AI Decision → Evaluator Checks Sufficiency & Confidence
    ├─ If Confident & Evidence Complete → Finalize
    └─ If Incomplete / Low Confidence / Conflict:
        ├─ Identify problem type (Retrieval Gap vs Extraction Mismatch)
        ├─ Retry (Re-search / Graph Traversal / Re-evaluate)
        ├─ Enforce MAX_RETRIES = 3
        └─ If still unresolved after MAX_RETRIES → Escalate to Human Review
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.agents.two_stage_validator import (
    FailureCategory,
    GaugeValidationReport,
    TwoStageValidator,
    ValidationStatus,
)
from src.knowledge_graph.hybrid_retriever import HybridRetriever, get_hybrid_retriever

logger = logging.getLogger("qdr_ai.loop_controller")

MAX_RETRIES = 3


@dataclass
class LoopExecutionTrace:
    """Audit log of iterative loop attempts."""
    gauge_id: str
    attempts: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_disposition: str = "PENDING"
    human_escalation: bool = False
    escalation_reason: str = ""


class LoopController:
    """
    Manages iterative validation loops with a strict stopping condition (MAX_RETRIES=3).
    """

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        validator: Optional[TwoStageValidator] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.max_retries = max_retries
        self.validator = validator or TwoStageValidator()
        self.retriever = retriever or get_hybrid_retriever()

    def run_validation_loop(
        self,
        page_number: int,
        gauge_id: str,
        ai_reading: Dict[str, Any],
        qa_reading: Dict[str, Any],
        initial_requirement: Optional[Dict[str, Any]] = None,
        test_type: str = "hydrostatic",
        material: Optional[str] = None,
        tolerance: Optional[float] = None,
        re_extraction_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> Tuple[GaugeValidationReport, LoopExecutionTrace]:
        """
        Execute iterative quality validation loop for a gauge review.
        """
        trace = LoopExecutionTrace(gauge_id=gauge_id)
        current_ai = dict(ai_reading)
        current_qa = dict(qa_reading)
        current_req = dict(initial_requirement) if initial_requirement else {}

        report: Optional[GaugeValidationReport] = None

        for attempt in range(1, self.max_retries + 1):
            trace.attempts = attempt
            logger.info(f"Loop Attempt {attempt}/{self.max_retries} for Gauge [{gauge_id}]")

            # 1. If requirement is missing, try hybrid RAG retrieval
            if not current_req:
                retrieval = self.retriever.retrieve(
                    query=f"hydrostatic test pressure requirements for {material or 'pressure parts'}",
                    test_type=test_type,
                    material=material,
                )
                graph_rules = retrieval.get("graph_rules", {})
                if "pressure_multipliers" in graph_rules:
                    p_rule = graph_rules["pressure_multipliers"].get("Section_I", {})
                    current_req = {
                        "rule": p_rule.get("formula", "MAWP * 1.5"),
                        "description": p_rule.get("notes", "ASME Section I Hydrotest Pressure"),
                        "multiplier": p_rule.get("multiplier", 1.5),
                    }

            # 2. Evaluate through TwoStageValidator
            report = self.validator.evaluate_gauge(
                page_number=page_number,
                gauge_id=gauge_id,
                ai_reading=current_ai,
                qa_reading=current_qa,
                procedure_requirement=current_req,
                tolerance=tolerance,
            )

            # Record attempt snapshot
            attempt_snapshot = {
                "attempt": attempt,
                "ai_reading": current_ai,
                "qa_reading": current_qa,
                "stage_1_status": report.stage_1.get("status"),
                "stage_2_status": report.stage_2.get("status"),
                "disposition": report.final_disposition,
                "failure_category": report.failure_category,
            }
            trace.history.append(attempt_snapshot)

            # 3. Assess loop termination condition
            # Clear PASS or verified NON_CONFORMANCE is a definitive result
            if report.final_disposition in ("ACCEPTED", "NON_CONFORMANCE"):
                trace.final_disposition = report.final_disposition
                logger.info(f"Loop resolved on attempt {attempt}: {report.final_disposition}")
                return report, trace

            # Extraction Mismatch with high confidence on AI side
            if report.final_disposition == "EXTRACTION_MISMATCH":
                ai_conf = float(current_ai.get("confidence", 1.0))
                if ai_conf >= 0.85:
                    # Definite extraction mismatch, terminate
                    trace.final_disposition = "EXTRACTION_MISMATCH"
                    return report, trace

            # If uncertain / low confidence / missing requirement -> Attempt targeted retry
            if attempt < self.max_retries:
                if report.failure_category == FailureCategory.IMAGE_UNCLEAR.value and re_extraction_fn:
                    logger.info(f"Attempting re-extraction with enhanced prompt for Gauge [{gauge_id}]...")
                    re_result = re_extraction_fn(gauge_id=gauge_id, attempt=attempt)
                    if re_result:
                        current_ai = re_result
                elif report.failure_category == FailureCategory.MISSING_REQUIREMENT.value:
                    logger.info(f"Expanding Knowledge Graph traversal for requirement...")
                    # Re-query Knowledge Graph directly
                    rules = self.retriever.knowledge_graph.find_rules_for_material(material or "CS", test_type=test_type)
                    if rules:
                        current_req = rules[0]

        # Max retries exceeded -> Escalate to Human Review
        trace.human_escalation = True
        trace.escalation_reason = (
            f"Validation loop reached maximum retry limit ({self.max_retries}) without definitive resolution. "
            f"Last failure category: {report.failure_category if report else 'UNKNOWN'}."
        )
        trace.final_disposition = "HUMAN_REVIEW_REQUIRED"

        if report:
            report.final_disposition = "HUMAN_REVIEW_REQUIRED"
            report.annotation_text = f"[HUMAN REVIEW REQUIRED] {trace.escalation_reason}"

        return report, trace
