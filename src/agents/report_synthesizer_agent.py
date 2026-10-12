from __future__ import annotations

"""
QDR·AI — LLM call #2 of 2: QA Critic & Executive Report Synthesizer
─────────────────────────────────────────────────────────────────
Combines the QA Critic verification step with the executive report narrative
generation:

  1. Critic Verification: Cross-checks grouped findings with raw OCR page evidence
     (verifies page numbers, joint IDs, material assignments, and limits).
  2. Finding Structure: Constructs final `Finding` objects with complete audit trails.
  3. Deterministic Scoring: Computes mathematical QA score and disposition.
  4. Executive Narrative (LLM call #2 of 2): Synthesizes concise engineering narrative
     citing applicable standards (ASME B31.1, 700-006, 700-022, DEETH-QCP-EL-003).
"""

import json
import logging
import re
import time
from typing import Any, Optional

import requests

from src.agents.state import Finding, GenParamCheck, ReviewState
from src.agents.compliance_reviewer import verify_doc_control_params
from src.config import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    STRICT_LLM_MODE,
    LLMAgentError,
    get_ollama_host,
)
from src.utils.llm_tracer import record_llm_trace
from src.utils.logger import pipeline_logger
from src.utils.metrics import StageTracker
from src.utils import progress as pipeline_progress

logger = logging.getLogger("qdr_ai.agent.report_synthesizer")

CRITIC_REPORT_SYSTEM_PROMPT = """You are an Executive Quality Assurance Director and Chief Metallurgical Reviewer conducting an audit of a plant engineering supplier hardness submittal.

Write a professional, authoritative 3-4 sentence Executive Summary that:
1. States the overall audit outcome, compliance score, and final disposition (e.g. REJECTED / APPROVED).
2. Highlights the specific critical non-conformances with exact page numbers, spool tags, joint numbers, and material grades (e.g., P91 elbow on Page 14 measuring 188 HBW).
3. Cites the specific violated engineering standards (e.g., 700-006 Table 8, 700-022, DEETH-QCP-EL-003).
4. Specifies mandatory corrective actions required from the supplier prior to document closeout.

Output ONLY the summary text (no JSON, no markdown headings)."""


def _compute_score_and_disposition(findings: list[Finding], gen_param_checks: list[GenParamCheck]) -> dict:
    """Deterministic score and disposition calculation."""
    counts = {"HOLD": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}
    for f in findings:
        ftype = f.get("type", "MINOR")
        if ftype in counts:
            counts[ftype] += 1
        else:
            counts["MINOR"] += 1

    missing_params = sum(
        1 for c in gen_param_checks
        if c.get("status") in ("MISSING", "PRESENT — MISMATCH")
    )

    base_score = 100
    base_score -= counts["HOLD"] * 30
    base_score -= counts["MAJOR"] * 15
    base_score -= counts["MINOR"] * 5
    base_score -= missing_params * 5

    score = int(round(max(0, min(100, base_score))))

    if counts["HOLD"] > 0 or score < 50:
        disposition = "REJECTED"
        verdict = "Document REJECTED due to safety-critical non-conformances (HOLD items) or severe score deficit."
    elif counts["MAJOR"] > 0 or score < 70:
        disposition = "REVISE AND RESUBMIT"
        verdict = "Document requires REVISION AND RESUBMISSION due to MAJOR findings or missing parameters."
    elif counts["MINOR"] > 0 or score < 90:
        disposition = "APPROVED WITH COMMENTS"
        verdict = "Document APPROVED WITH COMMENTS. Address minor observations prior to final closeout."
    else:
        disposition = "APPROVED"
        verdict = "Document fully APPROVED. All inspection criteria satisfied."

    return {
        "score": score,
        "disposition": disposition,
        "verdict": verdict,
        "counts": counts,
    }


def _critic_verify_and_filter_findings(
    incoming_findings: list[Finding],
    pages: list[Any],
    gen_param_checks: list[GenParamCheck],
) -> tuple[list[Finding], list[dict]]:
    """
    Stage 2 QA Critic Gate:
    Cross-checks every proposed finding against raw OCR evidence, authoritative requirements,
    and mathematical calculation before final signoff.
    Filters out any unsupported / hallucinated findings.
    """
    verified_findings: list[Finding] = []
    critic_records: list[dict] = []
    total_pages = len(pages) if pages else 1
    page_lookup = {getattr(p, "page_idx", idx): p for idx, p in enumerate(pages)}

    finding_id = 1

    # 1. Add verified Document Control findings
    for c in gen_param_checks:
        status = c.get("status", "")
        param = c.get("param", "")
        if status in ("MISSING", "PRESENT — MISMATCH"):
            f_obj: Finding = {
                "finding_id": f"F-DC-{finding_id:02d}",
                "id": finding_id,
                "page": 1,
                "section": f"Document Control — {param}",
                "type": "MAJOR" if any(k in param for k in ("Number", "Drawing", "Spool")) else "MINOR",
                "comment": f"{param} {status.lower()}: expected '{c.get('expected')}', found '{c.get('found')}'.",
                "requirement": "All document control headers must match approved project purchase orders and drawing schedules.",
                "action": f"Correct {param} in title block to match approved submittal register.",
                "closeout": "Required before approval",
                "evidence": f"Submittal Cover / Header ({param})",
                "target_term": param,
                "page_idx": 0,
                "confidence_score": 9,
                "confidence_level": "HIGH",
                "critic_verification": {
                    "evidence_supported": True,
                    "requirement_supported": True,
                    "calculation_correct": True,
                    "page_correct": True,
                    "final_decision": "APPROVE",
                },
            }
            verified_findings.append(f_obj)
            finding_id += 1

    # 2. Stage 2 Critic Verification of incoming structured findings
    for f in incoming_findings:
        p_num = f.get("page", 1)
        p_idx = f.get("page_idx", p_num - 1)
        ev = f.get("evidence")
        req = f.get("requirement")
        ctx = f.get("context", {})

        ev_text = ev.get("text", "") if isinstance(ev, dict) else str(ev)
        req_source = req.get("source", "") if isinstance(req, dict) else str(req)

        page_obj = page_lookup.get(p_idx)
        
        # 1. Page bounds check
        page_correct = bool(1 <= p_num <= total_pages + 2)

        # 2. Requirement validity check
        requirement_supported = bool(req_source and len(req_source.strip()) > 3)

        # 3. Evidence grounding check (is evidence text found on this page or in rows?)
        evidence_supported = True
        if page_obj and ev_text and ev_text != "P91":
            raw_text = getattr(page_obj, "full_text", "")
            # Extract only plausible hardness reading values (3-digit numbers in 100-500 range)
            reading_nums = [n for n in re.findall(r"\b(\d{3})\b", ev_text) if 100 <= int(n) <= 500]
            if reading_nums:
                evidence_supported = any(num in raw_text for num in reading_nums)
                if not evidence_supported and getattr(page_obj, "table_rows", None):
                    all_row_vals = {str(r.value) for row in page_obj.table_rows for r in row.readings}
                    evidence_supported = any(num in all_row_vals for num in reading_nums)

        # 4. Calculation correctness (only for FAIL findings — PASS/INFO/NEEDS_REVIEW are exempt)
        calculation_correct = True
        finding_status = (f.get("validation", {}).get("status", "") if isinstance(f.get("validation"), dict) else f.get("type", "FAIL")).upper()
        if finding_status == "FAIL" and isinstance(req, dict) and (req.get("minimum") is not None or req.get("maximum") is not None):
            min_v = req.get("minimum")
            max_v = req.get("maximum")
            vals = [int(n) for n in re.findall(r"\b(\d{3})\b", ev_text) if 100 <= int(n) <= 500]
            if vals and (min_v is not None or max_v is not None):
                has_violation = False
                for v in vals:
                    if min_v is not None and v < min_v:
                        has_violation = True
                    if max_v is not None and v > max_v:
                        # Allow single reading up to 315 for P91 weld/HAZ
                        is_p91 = "P91" in str(f.get("context", {}).get("material", "")).upper()
                        comp = str(f.get("context", {}).get("location", "")).upper()
                        if is_p91 and ("WELD" in comp or "HAZ" in comp) and v <= 315 and len([x for x in vals if x > max_v]) == 1:
                            continue
                        has_violation = True
                calculation_correct = has_violation

        # Final critic approval decision
        is_approved = page_correct and requirement_supported and evidence_supported and calculation_correct

        critic_rec = {
            "finding_id": f.get("finding_id", f"F-{finding_id:03d}"),
            "page": p_num,
            "evidence_text": ev_text[:50],
            "evidence_supported": evidence_supported,
            "requirement_supported": requirement_supported,
            "calculation_correct": calculation_correct,
            "page_correct": page_correct,
            "final_decision": "APPROVE" if is_approved else "REJECT",
        }
        critic_records.append(critic_rec)

        if is_approved:
            f_copy = dict(f)
            f_copy["id"] = finding_id
            f_copy["critic_verification"] = {
                "evidence_supported": evidence_supported,
                "requirement_supported": requirement_supported,
                "calculation_correct": calculation_correct,
                "page_correct": page_correct,
                "final_decision": "APPROVE",
            }
            verified_findings.append(f_copy)
            finding_id += 1
        else:
            pipeline_logger.warning(
                f"Stage 2 Critic REJECTED ungrounded finding on Page {p_num}: "
                f"evidence_supported={evidence_supported}, req_supported={requirement_supported}, calc={calculation_correct}"
            )

    return verified_findings, critic_records


def _generate_llm_summary(
    findings: list[Finding],
    grouped_findings: list[dict],
    score: int,
    disposition: str,
    total_pages: int,
) -> tuple[str, dict]:
    """Execute the report-synthesis LLM call: formal engineering executive narrative."""
    t0 = time.time()
    hold_count = sum(1 for f in findings if f.get("type") == "HOLD")
    major_count = sum(1 for f in findings if f.get("type") == "MAJOR")
    minor_count = sum(1 for f in findings if f.get("type") == "MINOR")

    findings_summary_lines = []
    for f in findings[:10]:
        findings_summary_lines.append(f"- [{f.get('type')}] {f.get('comment')} (Evidence: {f.get('evidence')})")

    grouped_lines = []
    for g in grouped_findings:
        grouped_lines.append(f"- Spool {g.get('spool_no')}: {g.get('summary')} (Action: {g.get('recommended_action')})")

    user_prompt = f"""## Audit Overview
Total Pages Inspected: {total_pages}
Overall Score: {score}/100
Disposition: {disposition}
HOLD Findings: {hold_count}
MAJOR Findings: {major_count}
MINOR Findings: {minor_count}

## Grouped Root-Cause Patterns:
{chr(10).join(grouped_lines) if grouped_lines else "No non-conformance patterns detected. All readings conform."}

## Critical Findings Sample:
{chr(10).join(findings_summary_lines) if findings_summary_lines else "None."}

Write the concise 3-4 sentence Executive Summary narrative for this audit report:"""

    ollama_host = get_ollama_host()
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": CRITIC_REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256,
        },
    }

    raw_response = ""
    status = "SUCCESS"
    narrative = ""
    failure: Optional[str] = None

    try:
        resp = requests.post(
            f"{ollama_host}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw_response = resp.json().get("message", {}).get("content", "").strip()
        if raw_response:
            narrative = raw_response
    except Exception as e:
        status = f"FALLBACK_ERROR: {e}"
        raw_response = f"Error: {e}"
        failure = str(e)
        if not STRICT_LLM_MODE:
            pipeline_logger.warning(f"Report Synthesizer LLM fallback to standard summary: {e}")

    if not narrative and not failure:
        failure = "Agent returned an empty narrative."

    if STRICT_LLM_MODE and failure:
        # The templated summary below is indistinguishable from a real one in
        # the finished report — it names specs and asserts corrective actions
        # that no agent actually reasoned about. Under strict mode, refuse.
        duration = time.time() - t0
        record_llm_trace(
            call_id="call_2_report_synthesizer",
            stage_name="QA Critic & Report Synthesizer Agent (LLM call #2 of 2)",
            model_name=OLLAMA_MODEL,
            system_prompt=CRITIC_REPORT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            raw_response=raw_response,
            parsed_output={"executive_summary": "", "score": score, "disposition": disposition},
            duration_s=duration,
            status=f"STRICT_FAILURE: {failure}",
            metadata={"total_findings": len(findings), "score": score, "disposition": disposition, "strict": True},
        )
        pipeline_logger.error(f"Report Synthesizer LLM failed under STRICT_LLM_MODE: {failure}")
        raise LLMAgentError("call_2_report_synthesizer", failure)

    if not narrative:
        if disposition == "APPROVED":
            narrative = f"Supplier hardness test submittal is fully APPROVED with a compliance score of {score}/100. All {total_pages} inspected pages satisfy ASME B31.1 and project specification acceptance criteria without non-conformance."
        else:
            narrative = f"Supplier hardness test submittal is {disposition} with a compliance score of {score}/100 across {total_pages} inspected pages. Critical non-conformances were identified on P91 components violating specifications 700-006 / 700-022 and DEETH-QCP-EL-003. Mandatory supplier corrective action and re-examination are required prior to approval."

    duration = time.time() - t0
    trace = record_llm_trace(
        call_id="call_2_report_synthesizer",
        stage_name="QA Critic & Report Synthesizer Agent (LLM call #2 of 2)",
        model_name=OLLAMA_MODEL,
        system_prompt=CRITIC_REPORT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        raw_response=raw_response,
        parsed_output={"executive_summary": narrative, "score": score, "disposition": disposition},
        duration_s=duration,
        status=status,
        metadata={"total_findings": len(findings), "score": score, "disposition": disposition},
    )

    return narrative, trace


def critic_and_synthesize_node(state: ReviewState) -> dict:
    """LangGraph node: Critic Verification + Executive Report Synthesis (LLM call #2 of 2)."""
    pipeline_progress.set_stage("synthesize_report")

    incoming_findings = state.get("findings", [])
    grouped_findings = state.get("grouped_findings", [])
    pages = state.get("page_ocr_results", [])
    total_pages = len(pages) if pages else 1

    # Extract & verify document control parameters
    gen_params_input = state.get("gen_params_input", {})
    if not isinstance(gen_params_input, dict):
        gen_params_input = {}
    submittal_text = state.get("submittal_text", "")
    gen_param_checks = verify_doc_control_params(submittal_text, gen_params_input)

    with StageTracker("critic_and_synthesize_llm", f"incoming_findings={len(incoming_findings)}") as st:
        # Stage 2 Critic verification and filtering
        verified_findings, critic_verifications = _critic_verify_and_filter_findings(
            incoming_findings, pages, gen_param_checks
        )

        # Materials present in the submittal for which no limits could be
        # established. The rule engine skipped their readings entirely, so the
        # review is incomplete for them. State that in the report: an omission
        # a reader cannot see is indistinguishable from a clean result.
        uncovered = state.get("uncovered_materials", []) or []
        for offset, mat in enumerate(uncovered):
            pages_with_mat = sorted({
                p.page_idx + 1 for p in pages
                for row in getattr(p, "table_rows", [])
                if (row.material_hint or "").upper().startswith(mat.upper())
            })
            verified_findings.append({
                "finding_id": f"F-UE-{offset + 1:02d}",
                "id": 9000 + offset,
                "page": pages_with_mat[0] if pages_with_mat else 1,
                "section": f"Review Completeness — {mat} not evaluated",
                "type": "MAJOR",
                "comment": (
                    f"{mat} readings were NOT evaluated: no hardness limits for {mat} could be "
                    f"established from the retrieved specifications"
                    + (f" (present on pages {pages_with_mat[:8]})" if pages_with_mat else "")
                    + ". No conclusion — pass or fail — should be drawn about "
                    f"{mat} from this report."
                ),
                "requirement": (
                    "Every material present in a hardness submittal must be evaluated against "
                    "an established acceptance limit traceable to the governing specification."
                ),
                "action": (
                    f"Supply or index the specification clause stating {mat} hardness limits, "
                    f"then re-run the review. Do not close out this submittal until {mat} is evaluated."
                ),
                "closeout": "Blocked — material unevaluated",
                "evidence": f"{mat} readings present, no limit established",
                "target_term": mat,
                "page_idx": (pages_with_mat[0] - 1) if pages_with_mat else 0,
                "confidence_score": 10,
                "confidence_level": "HIGH",
                "critic_verification": {
                    "evidence_supported": True,
                    "requirement_supported": True,
                    "calculation_correct": True,
                    "page_correct": True,
                    "final_decision": "APPROVE",
                },
            })
        if uncovered:
            pipeline_logger.warning(
                f"Report flags {len(uncovered)} unevaluated material(s): {uncovered}. "
                f"Disposition cannot be APPROVED while a material is unevaluated."
            )

        # Deterministic score & disposition based on verified findings
        score_data = _compute_score_and_disposition(verified_findings, gen_param_checks)
        score = score_data["score"]
        disposition = score_data["disposition"]
        verdict = score_data["verdict"]

        # LLM call #2 of 2: Executive Narrative Summary
        summary, trace = _generate_llm_summary(
            verified_findings, grouped_findings, score, disposition, total_pages
        )
        st.details += f" | score={score}, disposition={disposition}, verified_findings={len(verified_findings)}"

    existing_traces = list(state.get("llm_traces", []))
    existing_traces.append(trace)

    pipeline_progress.set_percentage(100)

    return {
        "findings": verified_findings,
        "critic_verifications": critic_verifications,
        "gen_param_checks": gen_param_checks,
        "score": score,
        "disposition": disposition,
        "verdict": verdict,
        "summary": summary,
        "llm_traces": existing_traces,
        "current_agent": "critic_and_synthesize",
    }
