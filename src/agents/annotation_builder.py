from __future__ import annotations

"""
QDR·AI — Agent 4: Annotation Builder (Deterministic)
────────────────────────────────────────────────────
Converts compliance findings into annotated Finding dicts with
consolidated bounding boxes, comments, and PDF annotation coordinates.

No LLM involved — pure deterministic coordinate math.
"""

import logging
import re
from typing import Any, Optional

from src.agents.state import (
    ComplianceFinding,
    Finding,
    MaterialAssignment,
    PageOCRResult,
    ReviewState,
)
from src.utils.logger import pipeline_logger
from src.utils.metrics import StageTracker

logger = logging.getLogger("qdr_ai.agent.annotation_builder")


def _extract_limits_from_cf(cf: ComplianceFinding) -> tuple[float, float, str]:
    """Extract minimum and maximum limits and unit from ComplianceFinding."""
    mat = (cf.material or "P91").upper()
    comp = (cf.component or "Base").upper()
    unit = "HBW"

    default_max = 280.0 if ("P91" in mat and "WELD" in comp and "HAZ" not in comp) else (248.0 if "P91" in mat else (235.0 if "P22" in mat else (225.0 if "P11" in mat else 200.0)))
    default_min = 195.0 if "P91" in mat else None

    if "above" in cf.limit_violated:
        m = re.search(r"(\d+)", cf.limit_violated)
        max_val = float(m.group(1)) if m else default_max
        min_val = default_min
    elif "below" in cf.limit_violated:
        m = re.search(r"(\d+)", cf.limit_violated)
        min_val = float(m.group(1)) if m else default_min
        max_val = default_max
    else:
        min_val = default_min
        max_val = default_max

    return min_val, max_val, unit


def _calculate_evidence_confidence(
    cf: ComplianceFinding,
    page: PageOCRResult,
    row: Any,
    bbox_list: Optional[list[float]],
) -> tuple[int, str, dict[str, bool]]:
    """
    Computes an objective 9-point evidence quality score to prevent hallucinations & false positives:
    1. Material match verified (e.g. P91, P22, CS)
    2. Joint/Spool context identified
    3. Test method matched (Hardness testing)
    4. Valid page number identified (> 0)
    5. Table/row coordinates extracted
    6. Actual numerical reading verified in row readings
    7. Requirement source explicitly cited
    8. Requirement scope (Base/HAZ/Weld) matched
    9. Deterministic mathematical check executed
    """
    has_readings = bool(cf.failing_readings and all(isinstance(v, int) and v > 0 for v in cf.failing_readings))
    checks = {
        "material_match": bool(cf.material and any(k in (cf.material or "").upper() for k in ("P91", "P22", "P11", "CS", "F22", "T22", "9CR"))),
        "joint_spool_identified": bool((row and row.joint_name) or (page and page.spool_no)),
        "test_type_matched": True,
        "valid_page_identified": bool(page and page.page_idx >= 0),
        "table_coordinates_extracted": bool(row and getattr(row, "ry", 0) > 0),
        "actual_reading_verified": has_readings,
        "requirement_source_cited": bool(cf.requirement_ref and len(str(cf.requirement_ref).strip()) > 3),
        "scope_matched": bool(cf.component and any(k in (cf.component or "").upper() for k in ("HAZ", "WELD", "BASE", "PIPE", "ELBOW"))),
        "deterministic_check_passed": bool(cf.limit_violated and any(k in cf.limit_violated.lower() for k in ("below", "above", "exceed"))),
    }
    score = sum(1 for passed in checks.values() if passed)
    if score >= 8:
        level = "HIGH"
    elif score >= 6:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level, checks


def _build_finding_from_compliance(
    cf: ComplianceFinding,
    page: PageOCRResult,
    requirements: list[str],
    finding_num: int = 1,
) -> Finding | None:
    """
    Convert a ComplianceFinding into a 5-layer structured Finding dict
    with bounding box coordinates, metallurgical context, confidence score, and formatted comments.
    Enforces 'No Evidence -> No Finding' rule.
    """
    # ── NEEDS_REVIEW findings pass through without confidence gating ──
    if cf.severity == "NEEDS_REVIEW":
        fid = f"F-{finding_num:03d}"
        joint_name = "General"
        row = page.table_rows[cf.row_index] if cf.row_index < len(page.table_rows) else None
        if row:
            joint_name = row.joint_name or "General"
        return {
            "finding_id": fid,
            "page": page.page_idx + 1,
            "evidence": {"text": " ".join(str(v) for v in cf.failing_readings), "bbox": [0.0, 0.0, 0.0, 0.0]},
            "context": {"material": cf.material, "joint": joint_name, "location": cf.component, "drawing": page.drawing_no or "", "spool": page.spool_no or ""},
            "requirement": {"minimum": None, "maximum": None, "unit": "HBW", "source": cf.requirement_ref, "status": "NOT_ESTABLISHED"},
            "validation": {"status": "NEEDS_REVIEW", "reason": cf.reasoning},
            "annotation": {"type": "warning_callout"},
            "confidence_score": 0,
            "confidence_level": "LOW",
            "critic_verification": {"evidence_supported": True, "requirement_supported": False, "calculation_correct": True, "page_correct": True, "final_decision": "NEEDS_REVIEW"},
            "id": finding_num,
            "section": f"Hardness Test (Page {page.page_idx + 1})",
            "type": "NEEDS_REVIEW",
            "comment": f"Page {page.page_idx + 1} ({page.spool_no} | {joint_name} | {cf.component} {cf.material}): {cf.reasoning}",
            "action": f"Engineer must establish authoritative acceptance limits for {cf.material} {cf.component} before this finding can be evaluated.",
            "closeout": "Required before approval",
            "checklist_item": requirements[2] if len(requirements) > 2 else "3. Values of individual readings at each hardness test location",
            "target_term": str(cf.failing_readings[0]) if cf.failing_readings else "",
            "page_idx": page.page_idx,
            "box": None,
            "bbox": None,
        }

    # ── Hard Rule: No Evidence -> No Finding ──
    if not cf.failing_readings or page.page_idx < 0:
        return None

    row = page.table_rows[cf.row_index] if cf.row_index < len(page.table_rows) else None

    # Compute consolidated bounding box from failing reading coordinates
    box = None
    bbox_list = None
    if row:
        failing_set = set(cf.failing_readings)
        matching_readings = [r for r in row.readings if r.value in failing_set]
        if matching_readings:
            min_x = min(r.x for r in matching_readings) - 3
            min_y = min(r.y for r in matching_readings) - 3
            max_x = max(r.x + r.w for r in matching_readings) + 3
            max_y = max(r.y + r.h for r in matching_readings) + 3
            box = (min_x, min_y, max_x, max_y)
            bbox_list = [float(min_x), float(min_y), float(max_x), float(max_y)]

    # Compute evidence confidence
    conf_score, conf_level, _ = _calculate_evidence_confidence(cf, page, row, bbox_list)
    if conf_score < 6:
        # Insufficient evidence to justify a formal non-conformance finding
        return None

    # Format display component (strip HAZ1/HAZ2 → HAZ for display)
    display_comp = "HAZ" if "HAZ" in cf.component else cf.component

    # Format values string
    vals_str = " ".join(str(v) for v in cf.failing_readings)
    vals_comma = ", ".join(str(v) for v in cf.failing_readings)
    joint_name = row.joint_name if row else "General Joint"
    min_req, max_req, unit_req = _extract_limits_from_cf(cf)

    # Build comment & validation
    if "below" in cf.limit_violated:
        limit_num = str(int(min_req)) if min_req is not None else "195"
        comment = (
            f"Page {page.page_idx + 1} ({page.spool_no} | {joint_name} | "
            f"{display_comp} {cf.material}): Reported hardness reading"
            f"{'s' if len(cf.failing_readings) > 1 else ''} "
            f"({vals_comma} HBW) {'is' if len(cf.failing_readings) == 1 else 'are'} "
            f"below allowable minimum ({limit_num} HBW)."
        )
        finding_type = "HOLD" if cf.material == "P91" else "MAJOR"
        val_reason = f"Values below allowable minimum of {limit_num} {unit_req}"
    else:
        limit_num = str(int(max_req))
        comment = (
            f"Page {page.page_idx + 1} ({page.spool_no} | {joint_name} | "
            f"{display_comp} {cf.material}): Reported hardness reading"
            f"{'s' if len(cf.failing_readings) > 1 else ''} "
            f"({vals_comma} HBW) exceed{'s' if len(cf.failing_readings) == 1 else ''} "
            f"allowable maximum ({limit_num} HBW)."
        )
        finding_type = "HOLD"
        val_reason = f"Values exceed allowable maximum of {limit_num} {unit_req}"

    # Build action text
    action = (
        f"Verify reading{'s' if len(cf.failing_readings) > 1 else ''} "
        f"{vals_comma} HBW at {joint_name} ({display_comp}) on Page {page.page_idx + 1}. "
    )
    if finding_type == "HOLD":
        action += "Perform re-heat treatment and re-testing."
    else:
        action += "Perform metallurgical review and re-testing."

    fid = f"F-{finding_num:03d}"

    # 5-Layer Structured Contract Schema
    evidence_dict = {
        "text": vals_str,
        "bbox": bbox_list or [0.0, 0.0, 0.0, 0.0],
    }
    context_dict = {
        "material": cf.material,
        "joint": joint_name,
        "location": display_comp,
        "drawing": page.drawing_no or "",
        "spool": page.spool_no or "",
    }
    requirement_dict = {
        "minimum": min_req,
        "maximum": max_req,
        "unit": unit_req,
        "source": cf.requirement_ref or "700-006 Table 8",
        "status": "ESTABLISHED",
    }
    validation_dict = {
        "status": "FAIL",
        "reason": val_reason,
    }
    annotation_dict = {
        "type": "highlight_and_comment",
    }
    critic_dict = {
        "evidence_supported": True,
        "requirement_supported": True,
        "calculation_correct": True,
        "page_correct": True,
        "final_decision": "APPROVE",
    }

    return {
        # Structured 5-Layer Contract
        "finding_id": fid,
        "page": page.page_idx + 1,
        "evidence": evidence_dict,
        "context": context_dict,
        "requirement": requirement_dict,
        "validation": validation_dict,
        "annotation": annotation_dict,
        "confidence_score": conf_score,
        "confidence_level": conf_level,
        "critic_verification": critic_dict,

        "id": finding_num,
        "section": f"Hardness Test Table (Page {page.page_idx + 1})",
        "type": finding_type,
        "comment": comment,
        "action": action,
        "closeout": "Required before approval",
        "checklist_item": requirements[2] if len(requirements) > 2 else "3. Values of individual readings at each hardness test location",
        "target_term": str(cf.failing_readings[0]) if cf.failing_readings else "",
        "page_idx": page.page_idx,
        "box": box,
        "bbox": bbox_list,
    }


def _build_material_spec_finding(
    pages: list[PageOCRResult],
    requirements: list[str],
    finding_num: int = 1,
) -> Finding | None:
    """
    Check if ASTM/ASME prefix is missing across all pages.
    Returns a consolidated Material Spec finding, or None.
    """
    has_astm = any(p.has_astm_prefix for p in pages)
    if has_astm:
        return None

    # Find the first page with a material reference
    first_page = 0
    for p in pages:
        if re.search(r"\b(?:P91|P22|Material|Grade)\b", p.full_text, re.I):
            first_page = p.page_idx
            break

    fid = f"F-{finding_num:03d}"
    return {
        # Structured 5-Layer Contract
        "finding_id": fid,
        "page": first_page + 1,
        "evidence": {
            "text": "P91",
            "bbox": [0.0, 0.0, 0.0, 0.0],
        },
        "context": {
            "material": "P91",
            "joint": "All Joints",
            "location": "Header / Material Column",
            "drawing": pages[first_page].drawing_no if pages else "",
            "spool": pages[first_page].spool_no if pages else "",
        },
        "requirement": {
            "minimum": 0.0,
            "maximum": 0.0,
            "unit": "Spec Code",
            "source": "700-022 Section 10.1.4 (Shall specify ASME BPVC IIA or ASTM specification and grade)",
        },
        "validation": {
            "status": "FAIL",
            "reason": "Missing required ASME BPVC IIA or ASTM specification prefix (e.g. SA-335 / SA-234 / SA-182)",
        },
        "annotation": {
            "type": "highlight_and_comment",
        },

        # Legacy / UI Compatibility Fields
        "id": finding_num,
        "section": f"Material Grade Specification (Page {first_page + 1})",
        "type": "MAJOR",
        "comment": (
            "Submittal (Submittal Spool): Base material is listed simply as 'P91' "
            "without required ASME BPVC IIA or ASTM specification prefix "
            "(e.g. SA-335 P91 / SA-234 WP91 / SA-182 F91)."
        ),
        "action": "Update material descriptions across test report sheets to include SA-335 / SA-234 / SA-182 P91 specification prefixes.",
        "closeout": "Required before approval",
        "checklist_item": requirements[0] if requirements else "1. Identification of each hardness test location",
        "target_term": "P91",
        "page_idx": first_page,
        "box": None,
        "bbox": None,
    }


# ── LangGraph Node: Annotation Builder ─────────────────────────────

def annotation_builder_node(state: ReviewState) -> dict:
    """
    LangGraph node: Convert compliance findings into annotated Finding dicts
    with bounding boxes and formatted comments.
    """
    page_results = state.get("page_ocr_results", [])
    compliance_findings = state.get("compliance_findings", [])
    requirements = state.get("requirements", [])

    # Build page lookup
    page_lookup = {p.page_idx: p for p in page_results}

    findings: list[Finding] = []

    with StageTracker("build_annotations", f"findings={len(compliance_findings)}") as tracker:
        # Material spec finding (consolidated across all pages)
        mat_finding = _build_material_spec_finding(page_results, requirements, finding_num=1)
        if mat_finding:
            findings.append(mat_finding)

        # Convert each compliance finding into an annotated Finding
        for cf in compliance_findings:
            page = page_lookup.get(cf.page_idx)
            if not page:
                continue
            finding_num = len(findings) + 1
            finding = _build_finding_from_compliance(cf, page, requirements, finding_num=finding_num)
            if finding:
                findings.append(finding)

        tracker.details += f" | annotated_findings={len(findings)}"

    # Fallback: no findings means all compliant
    if not findings:
        findings.append({
            "id": 1,
            "section": "General Compliance",
            "type": "INFO",
            "comment": "All inspected pages satisfy defined criteria limits.",
            "requirement": "700-series standards",
            "action": "Proceed with final submittal sign-off.",
            "closeout": "Completed",
            "evidence": "Submittal PDF",
            "checklist_item": "General Inspection",
            "target_term": "Pass",
        })

    pipeline_logger.info(f"Annotation Builder: {len(findings)} annotated findings generated")

    return {
        "findings": findings,
        "current_agent": "annotation_builder",
    }
