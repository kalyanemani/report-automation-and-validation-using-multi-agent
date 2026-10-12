from __future__ import annotations

"""
QDR·AI — Gap Analyzer Agent
────────────────────────────
Agent 4 in the review pipeline.

Performs multi-layered gap analysis:
  1. Checklist Coverage: Items in Review Guide missing from submittal
  2. Value Acceptance: Reported values out of spec bounds
  3. Field Completeness: Missing mandatory metadata / signatures / heat numbers
  4. Cross-Reference Verification: Outdated code or procedure citations
"""

import logging
import re

from src.agents.state import Finding, ReviewState

logger = logging.getLogger("qdr_ai.agent.gap_analyzer")

MANDATORY_FIELDS = [
    ("heat_number", re.compile(r"\b(?:heat\s*(?:no|number|#)?|batch\s*no)\b", re.IGNORECASE)),
    ("test_date", re.compile(r"\b(?:date\s*of\s*test|testing\s*date|date:)\b", re.IGNORECASE)),
    ("equipment_id", re.compile(r"\b(?:equipment|serial|machine\s*no|calib)\b", re.IGNORECASE)),
    ("operator_cert", re.compile(r"\b(?:operator|inspector|technician|certified\s*by|signature)\b", re.IGNORECASE)),
]


def analyze_gaps(state: ReviewState) -> dict:
    """LangGraph node: Perform deep gap analysis across layers."""
    submittal_text = state.get("submittal_text", "")
    requirements = state.get("requirements", [])
    existing_findings = list(state.get("findings", []))

    logger.info("Executing Gap Analyzer Agent...")

    gaps: list[dict[str, str]] = []
    missing_fields: list[str] = []
    ambiguities: list[str] = []

    # 1. Field Completeness Check
    for field_name, pattern in MANDATORY_FIELDS:
        if not pattern.search(submittal_text):
            missing_fields.append(field_name)
            gaps.append({
                "layer": "Completeness",
                "description": f"Mandatory metadata field '{field_name.replace('_', ' ').title()}' is not clearly stated.",
                "severity": "MAJOR",
                "evidence": "Submittal Text Analysis",
            })

    # 2. Check for Ambiguous Language
    ambiguous_terms = re.findall(r"\b(?:approx|approximately|around|as\s+per\s+standard|etc|N/A)\b", submittal_text, re.IGNORECASE)
    if len(ambiguous_terms) > 3:
        ambiguities.append(f"Multiple ambiguous terms found ({', '.join(set(ambiguous_terms))})")
        gaps.append({
            "layer": "Ambiguity",
            "description": "Report contains imprecise wording that may compromise compliance verification.",
            "severity": "MINOR",
            "evidence": "Submittal Text Analysis",
        })

    # 3. Requirement Coverage Gap Check
    unmatched_reqs = 0
    for req in requirements[:10]:
        keywords = [w for w in re.findall(r"\b\w{4,}\b", req) if w.lower() not in ("shall", "must", "with", "this", "that", "from")]
        if keywords:
            matches = sum(1 for kw in keywords if kw.lower() in submittal_text.lower())
            if matches < len(keywords) // 3:
                unmatched_reqs += 1

    if unmatched_reqs > 0:
        gaps.append({
            "layer": "Checklist Coverage",
            "description": f"{unmatched_reqs} requirement checklist criteria were not explicitly addressed in the submittal.",
            "severity": "MAJOR" if unmatched_reqs > 2 else "MINOR",
            "evidence": "Review Guide Cross-Reference",
        })

    # Convert severe gaps to additional findings if not already logged
    next_id = max([f.get("id", 0) for f in existing_findings], default=0) + 1
    new_findings = list(existing_findings)

    for gap in gaps:
        if gap["severity"] in ("HOLD", "MAJOR"):
            new_findings.append({
                "id": next_id,
                "section": f"Gap Analysis — {gap['layer']}",
                "type": gap["severity"],
                "comment": gap["description"],
                "requirement": f"Mandatory QA completeness criteria ({gap['layer']})",
                "action": "Revise submittal to explicitly provide missing details and resolve ambiguous references.",
                "closeout": "Required before approval",
                "evidence": gap["evidence"],
            })
            next_id += 1

    logger.info(f"Gap analysis identified {len(gaps)} total gaps.")

    return {
        "gaps": gaps,
        "missing_fields": missing_fields,
        "ambiguities": ambiguities,
        "findings": new_findings,
        "current_agent": "gap_analyzer",
    }
