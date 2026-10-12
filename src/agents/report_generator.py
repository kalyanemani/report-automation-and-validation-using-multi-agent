from __future__ import annotations

"""
QDR·AI — Report Generator Agent
────────────────────────────────
Agent 5 in the review pipeline.

Synthesizes findings, document control parameter checks, and gap analysis into a
final structured inspection report with calculated score and disposition.
"""

import logging
from src.agents.state import ReviewState

logger = logging.getLogger("qdr_ai.agent.report_generator")


def generate_report(state: ReviewState) -> dict:
    """LangGraph node: Calculate disposition & score and synthesize final report."""
    findings = state.get("findings", [])
    gen_param_checks = state.get("gen_param_checks", [])
    gaps = state.get("gaps", [])

    logger.info("Executing Report Generator Agent...")

    # Count finding severities
    counts = {"HOLD": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}
    for f in findings:
        f_type = f.get("type", "MINOR")
        if f_type in counts:
            counts[f_type] += 1
        else:
            counts["MINOR"] += 1

    # Add penalty for missing gen_param checks
    missing_gen_params = sum(1 for c in gen_param_checks if c.get("status") in ("MISSING", "PRESENT — MISMATCH"))

    # Scoring Algorithm
    base_score = 100
    base_score -= counts["HOLD"] * 30
    base_score -= counts["MAJOR"] * 15
    base_score -= counts["MINOR"] * 5
    base_score -= missing_gen_params * 5

    score = max(0, min(100, base_score))

    # Disposition Logic
    if counts["HOLD"] > 0 or score < 50:
        disposition = "REJECTED"
        verdict = "Document REJECTED due to safety-critical non-conformances (HOLD items) or severe score deficit."
    elif counts["MAJOR"] > 0 or score < 70 or missing_gen_params > 1:
        disposition = "REVISE AND RESUBMIT"
        verdict = "Document requires REVISION AND RESUBMISSION due to MAJOR findings or missing parameters."
    elif counts["MINOR"] > 0 or score < 90:
        disposition = "APPROVED WITH COMMENTS"
        verdict = "Document APPROVED WITH COMMENTS. Address minor observations prior to final closeout."
    else:
        disposition = "APPROVED"
        verdict = "Document fully APPROVED. All inspection criteria and parameters satisfied."

    # Summary Generation
    summary_parts = [
        f"Independent quality review completed with a final score of {score}/100 ({disposition}).",
        f"A total of {len(findings)} findings were identified ({counts['HOLD']} HOLD, {counts['MAJOR']} MAJOR, {counts['MINOR']} MINOR, {counts['INFO']} INFO).",
    ]
    if missing_gen_params > 0:
        summary_parts.append(f"{missing_gen_params} document control parameter issue(s) require verification.")
    if gaps:
        summary_parts.append(f"Gap analysis highlighted {len(gaps)} potential completeness/coverage area(s).")

    summary = " ".join(summary_parts)

    logger.info(f"Report generated: Score={score}, Disposition={disposition}")

    return {
        "score": score,
        "disposition": disposition,
        "verdict": verdict,
        "summary": summary,
        "current_agent": "report_generator",
    }
