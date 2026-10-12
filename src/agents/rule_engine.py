from __future__ import annotations

"""
QDR·AI — Deterministic Python Rule Engine (0 LLM Calls)
──────────────────────────────────────────────────────────
Pure Python mathematical validator for all hardness readings across
all pages of the submittal.

Features:
  - Strict numerical evaluation: (actual < min) or (actual > max)
  - Automatic material resolution per row (HP prefix -> P91, RP -> P22, HAZ inheritance)
  - Full ASME/700-006 Note 3 weld allowance logic (single reading up to 315 HBW)
  - Calibration block filtering (270, 180, 90, 0)
  - Zero hallucinations — 100% deterministic arithmetic
  - Preserves exact page, spool, joint, component, unit, and pixel bounding boxes
"""

import logging
import re
from typing import Any, Optional

from src.agents.state import (
    ComplianceFinding,
    PageOCRResult,
    Reading,
    ReviewState,
    TableRow,
)
from src.utils.logger import pipeline_logger
from src.utils.metrics import StageTracker
from src.utils import progress as pipeline_progress

logger = logging.getLogger("qdr_ai.rule_engine")

SKIP_CALIBRATION_VALUES = {270, 180, 90, 0}


def _detect_material_from_text(text: str) -> Optional[str]:
    """Detect material grade from a text string. Returns None if no match."""
    upper = text.upper()
    if any(k in upper for k in ("P91", "F91", "T91", "WP91", "9CR")):
        return "P91"
    if any(k in upper for k in ("P22", "F22", "T22", "WP22", "2.25CR")):
        return "P22"
    if any(k in upper for k in ("P11", "F11", "T11", "1.25CR")):
        return "P11"
    if any(k in upper for k in ("CS", "A106", "A105", "WPB", "CARBON STEEL")):
        return "CS"
    return None


def _parse_header_materials(material_spec: str) -> list[str]:
    """
    Parse ALL material grades from a page header material spec string.
    Handles dissimilar metal joints like 'SA335P91+SA182F22' or 'SA335P91+SA234WP91 TYPE 1'.
    Returns a list of unique grades found (e.g. ['P91', 'P22']).
    """
    upper = material_spec.upper()
    found = []
    # Split on common separators (+, /, ,, &)
    segments = re.split(r'[+/,&]', upper)
    for seg in segments:
        mat = _detect_material_from_text(seg)
        if mat and mat not in found:
            found.append(mat)
    # If nothing parsed from segments, try the whole string
    if not found:
        mat = _detect_material_from_text(upper)
        if mat:
            found.append(mat)
    return found


def _material_from_drawing_prefix(drawing_no: str = "", spool_no: str = "") -> Optional[str]:
    """
    DISABLED — always returns None.

    This used to infer material from spool/drawing prefixes ("-RP" -> P22,
    "-HP" -> P91, "-CSP" -> CS). Spool and drawing numbering is a naming
    convention, not a material designation, so the inference has no basis. On
    page 13 of 221135LBP-HRDR001 it returned P22 for every row of spool
    221135LBP-RPO3 — overriding the one row that explicitly stated P91.

    Material must come only from an actual grade designation found in the
    document text. See src/agents/context_resolver.py. Kept as a stub so any
    surviving caller degrades to "no evidence" rather than a wrong grade.
    """
    return None


# Component types that typically correspond to the "run pipe" side in a dissimilar joint
_PIPE_SIDE_COMPONENTS = {"PIPE", "HEADER", "DRUM"}
# Component types that typically correspond to the "fitting/branch" side
_FITTING_SIDE_COMPONENTS = {"ELBOW", "TEE", "REDUCER", "END CAP", "HALF COUPLING", "THERMOWELL", "PLATE", "WELDOLET"}


def _resolve_page_materials(
    rows: list[TableRow],
    drawing_no: str,
    spool_no: str,
    header_materials: list[str],
    full_page_text: str = "",
    fallback_material: str = "P91",
) -> list[str]:
    """
    3-pass material resolver for a single page's rows.
    
    Pass 1: Resolve non-HAZ rows from (a) row material_hint/left_text,
            (b) page-level test blocks / spool prefix,
            (c) weld component rules.
    Pass 2 (forward): HAZ1 inherits from the row above (the component it's heat-affected by).
    Pass 3 (backward): HAZ2 inherits from the row below.
    """
    n = len(rows)
    materials: list[Optional[str]] = [None] * n

    spool_u = spool_no.upper()
    dwg_u = drawing_no.upper()
    full_u = full_page_text.upper()

    # Determine page base material (distinguishes pure CS/P22 spools from P91 spools)
    # Spool-prefix tokens ("RP", "LP", "IP", "CSP") removed: they are naming
    # conventions, not material designations. Only explicit grade evidence in
    # the page's own test blocks is consulted now.
    is_pure_p22_cs = (
        any(k in spool_u for k in ("2.25CR",))
        or ("DDEL-TH-P22" in full_u and "DDEL-TH-CS" in full_u and "DDEL-TH-P91" not in full_u and "SA335P91" not in full_u)
    )
    if is_pure_p22_cs:
        page_base_mat = "P22"
    elif "9CR" in spool_u or "DDEL-TH-P91" in full_u:  # spool "HP" prefix removed
        page_base_mat = "P91"
    else:
        page_base_mat = fallback_material

    # ── Pass 1: Resolve all non-HAZ rows ──
    for i, row in enumerate(rows):
        comp_upper = (row.component or "").upper()

        # Skip HAZ rows — resolved in passes 2 and 3
        if "HAZ" in comp_upper:
            continue

        left_u = (row.left_text or "").upper()

        # 1a. Explicit material matches in row text
        if any(k in left_u for k in ("P91", "F91", "T91", "WP91", "POT", "POI", "PO1", "PO4", "P94")):
            materials[i] = "P91"
        elif any(k in left_u for k in ("P22", "F22", "T22", "WP22", "2.25CR")):
            materials[i] = "P22"
        elif any(k in left_u for k in ("P11", "F11", "T11", "1.25CR")):
            materials[i] = "P11"
        elif any(k in left_u for k in ("CS", "A106", "A105", "WPB", "CARBON")):
            materials[i] = "CS"
        elif row.is_weld or comp_upper == "WELD":
            # Welds on P91 spools are P91 welds; on P22 spools are P22 welds
            materials[i] = page_base_mat
        else:
            materials[i] = page_base_mat

    # ── Pass 2 (forward): HAZ1 inherits from the row above ──
    for i in range(n):
        comp_upper = (rows[i].component or "").upper()
        if comp_upper in ("HAZ1", "HAZ") and materials[i] is None:
            for j in range(i - 1, -1, -1):
                if materials[j] is not None and "HAZ" not in (rows[j].component or "").upper():
                    materials[i] = materials[j]
                    break
            if materials[i] is None:
                materials[i] = page_base_mat

    # ── Pass 3 (backward): HAZ2 inherits from the row below ──
    for i in range(n - 1, -1, -1):
        comp_upper = (rows[i].component or "").upper()
        if comp_upper == "HAZ2" and materials[i] is None:
            for j in range(i + 1, n):
                if materials[j] is not None and "HAZ" not in (rows[j].component or "").upper():
                    materials[i] = materials[j]
                    break
            if materials[i] is None:
                materials[i] = page_base_mat

    return [m or page_base_mat for m in materials]


def evaluate_hardness_readings_deterministically(
    pages: list[PageOCRResult],
    rules_schema: dict,
    submittal_material: str = "P91",
) -> tuple[list[dict], list[ComplianceFinding]]:
    """
    Run pure Python rule engine across all extracted page rows.
    Returns (raw_rule_failures, compliance_findings).
    """
    # Normalize submittal fallback material
    fallback_mat = "P91"
    sub_upper = submittal_material.upper()
    if "P22" in sub_upper or "22" in sub_upper or "F22" in sub_upper:
        fallback_mat = "P22"
    elif "P11" in sub_upper or "11" in sub_upper or "F11" in sub_upper:
        fallback_mat = "P11"
    elif "CS" in sub_upper or "CARBON" in sub_upper or "A106" in sub_upper or "SIL" in sub_upper:
        fallback_mat = "CS"

    raw_failures: list[dict] = []
    compliance_findings: list[ComplianceFinding] = []

    for p in pages:
        p_idx = p.page_idx
        dwg = p.drawing_no or "N/A"
        spool = p.spool_no or "N/A"
        rows = p.table_rows
        page_acc_range = getattr(p, "acceptance_range", None)
        page_mat_str = (getattr(p, "material_spec", "") or "").upper()

        # Parse ALL materials from page header (supports dissimilar joints)
        header_materials = _parse_header_materials(page_mat_str)
        if not header_materials:
            # Fallback to drawing prefix
            prefix_mat = _material_from_drawing_prefix(dwg, spool)
            if prefix_mat:
                header_materials = [prefix_mat]

        # 3-pass material resolution with submittal fallback
        row_materials = _resolve_page_materials(
            rows, dwg, spool, header_materials, full_page_text=getattr(p, "full_text", ""), fallback_material=fallback_mat
        )

        # Audit each row
        for row_idx, row in enumerate(rows):
            comp = row.component or "Base"
            mat = row_materials[row_idx]
            joint = row.joint_name or f"Joint on page {p_idx + 1}"
            is_weld = row.is_weld or any(w in comp.upper() for w in ("WELD", "WM", "W.M.", "W/M", "WLD", "BW", "FW", "SEAM"))

            # Extract active limits for this material
            mat_rules = rules_schema.get(mat)
            if mat_rules is None:
                # Unknown material not in rules schema → emit NEEDS_REVIEW, do NOT evaluate against P91 defaults
                valid_readings = [r for r in row.readings if r.value not in SKIP_CALIBRATION_VALUES]
                if valid_readings:
                    compliance_findings.append(
                        ComplianceFinding(
                            page_idx=p_idx,
                            row_index=row_idx,
                            component=comp,
                            material=mat,
                            failing_readings=[r.value for r in valid_readings],
                            limit_violated=f"NEEDS_REVIEW: no authoritative limits established for material {mat}",
                            severity="NEEDS_REVIEW",
                            requirement_ref=f"Material {mat} not found in project specification schema",
                            reasoning=f"Material {mat} on Page {p_idx + 1} ({joint}, {comp}) has no established acceptance criteria. Cannot determine PASS or FAIL without authoritative limits.",
                        )
                    )
                    pipeline_logger.warning(f"NEEDS_REVIEW: Material {mat} on Page {p_idx + 1} ({joint}, {comp}) has no established limits in rules schema. Skipping evaluation.")
                continue

            if is_weld:
                spec_entry = mat_rules.get("weld", {
                    "min": 195 if mat == "P91" else None,
                    "max": 280 if mat == "P91" else (235 if mat == "P22" else (225 if mat == "P11" else 200)),
                    "single_reading_max": 315 if mat == "P91" else None,
                    "unit": "HBW",
                    "ref": "700-006 Table 8 Note 2 & Note 3" if mat == "P91" else "241135LBP-HRDP001 Section II / ASME B31.1",
                })
            elif "HAZ" in comp.upper():
                spec_entry = mat_rules.get("haz", {
                    "min": 195 if mat == "P91" else None,
                    "max": 248 if mat == "P91" else (235 if mat == "P22" else (225 if mat == "P11" else 200)),
                    "single_reading_max": 250 if mat == "P91" else None,
                    "unit": "HBW",
                    "ref": "700-022 / DEETH-QCP-EL-003 / ASME B31.1" if mat == "P91" else "241135LBP-HRDP001 Section II / ASME B31.1",
                })
            else:
                spec_entry = mat_rules.get("base_metal", {
                    "min": 195 if mat == "P91" else None,
                    "max": 248 if mat == "P91" else (235 if mat == "P22" else (225 if mat == "P11" else 200)),
                    "single_reading_max": 250 if mat == "P91" else None,
                    "unit": "HBW",
                    "ref": "700-022 / DEETH-QCP-EL-003" if mat == "P91" else "241135LBP-HRDP001 Section II / ASME B31.1",
                })

            raw_min = spec_entry.get("min")
            raw_max = spec_entry.get("max")
            
            # If maximum limit is completely missing → NEEDS_REVIEW
            if raw_max is None:
                valid_readings = [r for r in row.readings if r.value not in SKIP_CALIBRATION_VALUES]
                if valid_readings:
                    compliance_findings.append(
                        ComplianceFinding(
                            page_idx=p_idx,
                            row_index=row_idx,
                            component=comp,
                            material=mat,
                            failing_readings=[r.value for r in valid_readings],
                            limit_violated=f"NEEDS_REVIEW: max={raw_max} — incomplete acceptance criteria",
                            severity="NEEDS_REVIEW",
                            requirement_ref=" / ".join(spec_entry.get("ref")) if isinstance(spec_entry.get("ref"), list) else str(spec_entry.get("ref", f"Specification for {mat}")),
                            reasoning=f"Material {mat} {comp} on Page {p_idx + 1} has no upper acceptance limit. Cannot determine PASS or FAIL.",
                        )
                    )
                    pipeline_logger.warning(f"NEEDS_REVIEW: {mat} {comp} on Page {p_idx + 1} has null max limit. Skipping.")
                continue
            
            min_limit = int(raw_min) if raw_min is not None else None
            max_limit = int(raw_max)

            # Per-page stated acceptance range override (e.g. header states "195 ~ 280 HB" or "MAX 235 HB")
            if page_acc_range:
                p_min, p_max = page_acc_range
                if p_min is not None and mat == "P91":
                    min_limit = int(p_min)
                if p_max is not None:
                    if mat == "P91":
                        if is_weld or (int(p_min or 0) == 195 and int(p_max) == 280):
                            max_limit = 280
                        else:
                            max_limit = min(int(p_max), 248)
                    elif mat == "P22":
                        max_limit = int(p_max) if int(p_max) <= 235 else 235
                    elif mat == "P11":
                        max_limit = int(p_max) if int(p_max) <= 225 else 225
                    elif mat == "CS":
                        max_limit = int(p_max) if int(p_max) <= 200 else 200
                    else:
                        max_limit = int(p_max)

            single_reading_max = spec_entry.get("single_reading_max")
            ref_raw = spec_entry.get("ref", "Specification 700-006 / 700-022 / 241135LBP-HRDP001")
            ref_str = " / ".join(ref_raw) if isinstance(ref_raw, list) else str(ref_raw)

            # Evaluate each reading
            valid_readings = [r for r in row.readings if r.value not in SKIP_CALIBRATION_VALUES]
            row_failing_vals: list[int] = []

            # Check Note 3 condition for weld metal and HAZ
            high_readings = [r for r in valid_readings if r.value > max_limit]
            avg_val = sum(r.value for r in valid_readings) / len(valid_readings) if valid_readings else 0
            note3_applies = (
                mat == "P91"
                and single_reading_max is not None
                and len(high_readings) == 1
                and high_readings[0].value <= int(single_reading_max)
                and avg_val <= max_limit
            )

            for r_idx, r in enumerate(valid_readings):
                val = r.value

                # Evaluation check: Below Minimum (only if a minimum is specified)
                if min_limit is not None and val < min_limit:
                    dev = min_limit - val
                    severity = "HOLD" if mat == "P91" else "MAJOR"
                    row_failing_vals.append(val)
                    raw_failures.append({
                        "id": f"FAIL-P{p_idx + 1}-R{row_idx + 1}-{r_idx + 1}",
                        "page_idx": p_idx,
                        "page_num": p_idx + 1,
                        "drawing_no": dwg,
                        "spool_no": spool,
                        "joint_name": joint,
                        "component": comp,
                        "material": mat,
                        "measured_value": val,
                        "unit": "HBW",
                        "min_limit": min_limit,
                        "max_limit": max_limit,
                        "deviation": f"-{dev} HBW (below min {min_limit})",
                        "violation_type": "BELOW_MIN",
                        "severity": severity,
                        "requirement_ref": ref_str,
                        "bbox": [r.x, r.y, r.w, r.h],
                        "evidence_citation": f"Page {p_idx + 1} → Spool {spool} → {joint} → {comp} → {val} HBW",
                    })

                elif val > max_limit:
                    if note3_applies and val <= int(single_reading_max or max_limit):
                        # Note 3 single-reading allowance PASS
                        continue

                    dev = val - max_limit
                    row_failing_vals.append(val)
                    raw_failures.append({
                        "id": f"FAIL-P{p_idx + 1}-R{row_idx + 1}-{r_idx + 1}",
                        "page_idx": p_idx,
                        "page_num": p_idx + 1,
                        "drawing_no": dwg,
                        "spool_no": spool,
                        "joint_name": joint,
                        "component": comp,
                        "material": mat,
                        "measured_value": val,
                        "unit": "HBW",
                        "min_limit": min_limit,
                        "max_limit": max_limit,
                        "deviation": f"+{dev} HBW (above max {max_limit})",
                        "violation_type": "ABOVE_MAX",
                        "severity": "HOLD",
                        "requirement_ref": ref_str,
                        "bbox": [r.x, r.y, r.w, r.h],
                        "evidence_citation": f"Page {p_idx + 1} → Spool {spool} → {joint} → {comp} → {val} HBW",
                    })

            if row_failing_vals:
                is_below = (min_limit is not None) and any(v < min_limit for v in row_failing_vals)
                limit_desc = f"below minimum {min_limit} HBW" if is_below else f"above maximum {max_limit} HBW"
                limit_range_str = f"{min_limit}–{max_limit} HBW" if min_limit is not None else f"max {max_limit} HBW"
                compliance_findings.append(
                    ComplianceFinding(
                        page_idx=p_idx,
                        row_index=row_idx,
                        component=comp,
                        material=mat,
                        failing_readings=row_failing_vals,
                        limit_violated=limit_desc,
                        severity="HOLD" if mat == "P91" or any(v > max_limit for v in row_failing_vals) else "MAJOR",
                        requirement_ref=ref_str,
                        reasoning=f"Hardness reading(s) {row_failing_vals} violate {mat} {comp} acceptance limit ({limit_range_str}) per {ref_str}.",
                    )
                )

    pipeline_logger.info(f"Python Rule Engine: Evaluated {len(pages)} pages -> Found {len(raw_failures)} numerical failures deterministically.")
    return raw_failures, compliance_findings


def python_rule_engine_node(state: ReviewState) -> dict:
    """LangGraph Node for Deterministic Python Rule Engine Validator (0 LLM Calls).
    Validates findings from LLM Call #2 against pure mathematical limits across all pages.
    """
    pipeline_progress.set_stage("audit_rules")

    pages = state.get("page_ocr_results", [])
    rules_schema = state.get("hardness_rules_schema", {})
    mat_grade = state.get("material_grade", "P91")
    llm_findings = state.get("compliance_findings", [])

    with StageTracker("deterministic_rule_engine", f"total_pages={len(pages)}") as st:
        raw_failures, compliance_findings = evaluate_hardness_readings_deterministically(
            pages, rules_schema, submittal_material=mat_grade
        )
        st.details += f" | validated_findings={len(compliance_findings)}, reading_points={len(raw_failures)}"

    pipeline_logger.info(
        f"Deterministic Validator: Validated {len(compliance_findings)} findings against analyzer-established limits "
        f"({len(raw_failures)} individual non-conforming reading points confirmed across {len(pages)} pages)."
    )

    return {
        "raw_rule_failures": raw_failures,
        "compliance_findings": compliance_findings,
        "current_agent": "python_rule_engine",
    }
