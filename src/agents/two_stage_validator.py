from __future__ import annotations

"""
QDR·AI — Two-Stage Validation Engine
─────────────────────────────────────
Establishes two strict levels of validation for Quality Document Review:

  Stage 1 – AI Reading Validation:
    Validates the gauge reading extracted by the Vision LLM (Qwen2.5-VL)
    against the manually reviewed reading provided by QA.
    Computes reading difference, unit alignment, and confidence.

  Stage 2 – Compliance Validation:
    Only after Stage 1 PASSES, validates the confirmed reading against the
    applicable Review Guide and Client Procedure rules (e.g. 600-001-R49, 241511CB-HYDP001).

Critical distinction:
  A gauge reading can be correctly extracted by AI (Stage 1 PASS), but still
  fail client procedural requirements (Stage 2 FAIL).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("qdr_ai.two_stage_validator")


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NOT_EVALUATED = "NOT_EVALUATED"


class FailureCategory(str, Enum):
    NONE = "NONE"
    AI_READING_MISMATCH = "AI_READING_MISMATCH"
    IMAGE_UNCLEAR = "IMAGE_UNCLEAR"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    PROCEDURE_NON_COMPLIANCE = "PROCEDURE_NON_COMPLIANCE"
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


@dataclass
class GaugeValidationReport:
    """Structured inspection data model for two-stage gauge review."""
    page_number: int
    gauge_id: str
    qa_reading: Dict[str, Any]       # {"value": 195.0, "unit": "PSIG", "source": "manual_qa"}
    ai_reading: Dict[str, Any]       # {"value": 195.0, "unit": "PSIG", "confidence": 0.93, "model": "qwen2.5vl:7b"}
    stage_1: Dict[str, Any]          # {"status": "PASS", "difference": 0.0, "tolerance": 2.0}
    procedure_requirement: Dict[str, Any] = field(default_factory=dict)
    stage_2: Dict[str, Any] = field(default_factory=lambda: {"status": "NOT_EVALUATED"})
    final_disposition: str = "PENDING"
    failure_category: str = "NONE"
    annotation_text: str = ""
    evidence_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "gauge_id": self.gauge_id,
            "qa_reading": self.qa_reading,
            "ai_reading": self.ai_reading,
            "stage_1_validation": self.stage_1,
            "stage_1": self.stage_1,
            "procedure_requirement": self.procedure_requirement,
            "stage_2_validation": self.stage_2,
            "stage_2": self.stage_2,
            "final_disposition": self.final_disposition,
            "failure_category": self.failure_category,
            "annotation_text": self.annotation_text,
            "evidence_notes": self.evidence_notes,
        }


class TwoStageValidator:
    """Evaluates Stage 1 extraction validity and Stage 2 procedural compliance."""

    def __init__(self, default_tolerance: float = 2.0, min_confidence: float = 0.65):
        self.default_tolerance = default_tolerance
        self.min_confidence = min_confidence

    def validate_stage_1(
        self,
        ai_reading: Dict[str, Any],
        qa_reading: Dict[str, Any],
        tolerance: Optional[float] = None,
    ) -> Tuple[ValidationStatus, FailureCategory, float, str]:
        """
        Stage 1: Validate AI gauge reading against QA manual recorded reading.

        Returns:
          (status, failure_category, diff, reason)
        """
        tol = tolerance if tolerance is not None else self.default_tolerance

        ai_val = ai_reading.get("value")
        qa_val = qa_reading.get("value")
        ai_unit = str(ai_reading.get("unit", "")).strip().upper()
        qa_unit = str(qa_reading.get("unit", "")).strip().upper()
        confidence = float(ai_reading.get("confidence", 1.0))

        # Check if values could not be parsed
        if ai_val is None or qa_val is None:
            return (
                ValidationStatus.REVIEW,
                FailureCategory.IMAGE_UNCLEAR,
                999.0,
                "Gauge reading value could not be resolved from image or QA report.",
            )

        try:
            ai_num = float(ai_val)
            qa_num = float(qa_val)
        except (ValueError, TypeError):
            return (
                ValidationStatus.REVIEW,
                FailureCategory.IMAGE_UNCLEAR,
                999.0,
                f"Non-numeric gauge values provided: AI='{ai_val}', QA='{qa_val}'",
            )

        # Check unit consistency
        if ai_unit and qa_unit and ai_unit != qa_unit:
            return (
                ValidationStatus.FAIL,
                FailureCategory.UNIT_MISMATCH,
                abs(ai_num - qa_num),
                f"Unit mismatch: AI reading has '{ai_unit}', but QA reading specifies '{qa_unit}'.",
            )

        diff = abs(ai_num - qa_num)

        # Low confidence signal
        if confidence < self.min_confidence:
            logger.warning(f"Low AI confidence ({confidence:.2f}) for gauge reading {ai_num}")
            if diff <= tol:
                return (
                    ValidationStatus.REVIEW,
                    FailureCategory.MANUAL_REVIEW_REQUIRED,
                    diff,
                    f"AI reading matches QA ({ai_num} {qa_unit}), but AI confidence is low ({confidence:.2f}). Human review recommended.",
                )

        # Compare values against tolerance
        if diff <= tol:
            return (
                ValidationStatus.PASS,
                FailureCategory.NONE,
                diff,
                f"AI reading ({ai_num}) matches QA reading ({qa_num} {qa_unit}) within tolerance (+/-{tol}).",
            )
        else:
            return (
                ValidationStatus.FAIL,
                FailureCategory.AI_READING_MISMATCH,
                diff,
                f"AI gauge reading ({ai_num}) does not match QA-reviewed reading ({qa_num} {qa_unit}). Difference = {diff:.2f} exceeds tolerance (+/-{tol}).",
            )

    def validate_stage_2(
        self,
        confirmed_reading: Dict[str, Any],
        requirement: Dict[str, Any],
    ) -> Tuple[ValidationStatus, FailureCategory, str]:
        """
        Stage 2: Validate confirmed reading against Client Procedure / Spec rules.

        Returns:
          (status, failure_category, reason)
        """
        val = confirmed_reading.get("value")
        if val is None:
            return (
                ValidationStatus.FAIL,
                FailureCategory.PROCEDURE_NON_COMPLIANCE,
                "No confirmed reading available for Stage 2 compliance check.",
            )

        try:
            reading_num = float(val)
        except (ValueError, TypeError):
            return (
                ValidationStatus.FAIL,
                FailureCategory.PROCEDURE_NON_COMPLIANCE,
                f"Invalid reading value '{val}'.",
            )

        req_val = requirement.get("required_value")
        min_val = requirement.get("min_value")
        max_val = requirement.get("max_value")
        rule_desc = requirement.get("description", requirement.get("rule", "Client Requirement"))

        # Test against Min / Max limits
        if min_val is not None and reading_num < float(min_val):
            return (
                ValidationStatus.FAIL,
                FailureCategory.PROCEDURE_NON_COMPLIANCE,
                f"Confirmed reading ({reading_num}) is below minimum required limit ({min_val}). Rule: {rule_desc}",
            )

        if max_val is not None and reading_num > float(max_val):
            return (
                ValidationStatus.FAIL,
                FailureCategory.PROCEDURE_NON_COMPLIANCE,
                f"Confirmed reading ({reading_num}) exceeds maximum allowable limit ({max_val}). Rule: {rule_desc}",
            )

        if req_val is not None:
            req_num = float(req_val)
            req_tol = float(requirement.get("tolerance", 0.0))
            if abs(reading_num - req_num) > req_tol and reading_num < req_num:
                return (
                    ValidationStatus.FAIL,
                    FailureCategory.PROCEDURE_NON_COMPLIANCE,
                    f"Confirmed reading ({reading_num}) does not satisfy required test value ({req_num}). Rule: {rule_desc}",
                )

        return (
            ValidationStatus.PASS,
            FailureCategory.NONE,
            f"Confirmed reading ({reading_num}) satisfies procedure requirement. ({rule_desc})",
        )

    def evaluate_gauge(
        self,
        page_number: int,
        gauge_id: str,
        ai_reading: Dict[str, Any],
        qa_reading: Dict[str, Any],
        procedure_requirement: Optional[Dict[str, Any]] = None,
        tolerance: Optional[float] = None,
    ) -> GaugeValidationReport:
        """Run full two-stage evaluation on a single gauge reading."""
        # 1. Execute Stage 1
        s1_status, s1_cat, diff, s1_reason = self.validate_stage_1(
            ai_reading=ai_reading,
            qa_reading=qa_reading,
            tolerance=tolerance,
        )

        stage_1_info = {
            "status": s1_status.value,
            "difference": round(diff, 2),
            "tolerance": tolerance or self.default_tolerance,
            "reason": s1_reason,
        }

        req = procedure_requirement or {}

        # 2. Decision Logic
        if s1_status == ValidationStatus.PASS:
            # Proceed to Stage 2
            if not req:
                # No requirement provided
                report = GaugeValidationReport(
                    page_number=page_number,
                    gauge_id=gauge_id,
                    qa_reading=qa_reading,
                    ai_reading=ai_reading,
                    stage_1=stage_1_info,
                    procedure_requirement={},
                    stage_2={"status": ValidationStatus.REVIEW.value, "reason": "No applicable procedure requirement found."},
                    final_disposition="MANUAL_REVIEW",
                    failure_category=FailureCategory.MISSING_REQUIREMENT.value,
                    annotation_text=f"Stage 1 PASS (AI=QA={qa_reading.get('value')}). Procedure requirement missing for Stage 2.",
                    evidence_notes=[s1_reason],
                )
                return report

            s2_status, s2_cat, s2_reason = self.validate_stage_2(
                confirmed_reading=qa_reading,
                requirement=req,
            )

            stage_2_info = {
                "status": s2_status.value,
                "reason": s2_reason,
            }

            if s2_status == ValidationStatus.PASS:
                disposition = "ACCEPTED"
                cat = FailureCategory.NONE.value
                annot = f"[PASSED] Validated Reading: {qa_reading.get('value')} {qa_reading.get('unit', '')}. Meets {req.get('rule', 'procedure requirement')}."
            else:
                disposition = "NON_CONFORMANCE"
                cat = s2_cat.value
                annot = f"[NON-CONFORMANCE] Validated gauge reading ({qa_reading.get('value')} {qa_reading.get('unit', '')}) violates client requirement: {s2_reason}"

            return GaugeValidationReport(
                page_number=page_number,
                gauge_id=gauge_id,
                qa_reading=qa_reading,
                ai_reading=ai_reading,
                stage_1=stage_1_info,
                procedure_requirement=req,
                stage_2=stage_2_info,
                final_disposition=disposition,
                failure_category=cat,
                annotation_text=annot,
                evidence_notes=[s1_reason, s2_reason],
            )

        elif s1_status == ValidationStatus.FAIL:
            # Stage 1 Failed -> DO NOT execute Stage 2
            return GaugeValidationReport(
                page_number=page_number,
                gauge_id=gauge_id,
                qa_reading=qa_reading,
                ai_reading=ai_reading,
                stage_1=stage_1_info,
                procedure_requirement=req,
                stage_2={"status": ValidationStatus.NOT_EVALUATED.value, "reason": "Stage 1 AI Reading validation failed; Stage 2 skipped."},
                final_disposition="EXTRACTION_MISMATCH",
                failure_category=s1_cat.value,
                annotation_text=f"[EXTRACTION MISMATCH] AI gauge reading ({ai_reading.get('value')}) does not match QA reading ({qa_reading.get('value')}). Stage 2 compliance skipped.",
                evidence_notes=[s1_reason],
            )

        else:
            # Stage 1 Review Needed
            return GaugeValidationReport(
                page_number=page_number,
                gauge_id=gauge_id,
                qa_reading=qa_reading,
                ai_reading=ai_reading,
                stage_1=stage_1_info,
                procedure_requirement=req,
                stage_2={"status": ValidationStatus.NOT_EVALUATED.value, "reason": "Stage 1 requires manual review."},
                final_disposition="HUMAN_REVIEW_REQUIRED",
                failure_category=s1_cat.value,
                annotation_text=f"[MANUAL REVIEW] {s1_reason}",
                evidence_notes=[s1_reason],
            )
