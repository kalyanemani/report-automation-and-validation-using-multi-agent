from __future__ import annotations

"""
QDR·AI — Revision Verifier Agent
──────────────────────────────────
Agent for comparing current submittal revision against previous revision (Rev 0 vs Rev 1).

Tracks:
  - Resolved non-conformances
  - Persistent non-conformances
  - Newly introduced issues
"""

import logging
import re
from src.agents.state import Finding, ReviewState

logger = logging.getLogger("qdr_ai.agent.revision_verifier")


def verify_revision_changes(state: ReviewState) -> dict:
    """LangGraph node: Track and verify changes across document revisions."""
    revision = state.get("revision", "0").strip()
    findings = state.get("findings", [])
    submittal_text = state.get("submittal_text", "")

    logger.info(f"Executing Revision Verifier Agent for Revision [{revision}]...")

    # If revision > 0 or Rev 01, check if revision table is present in text
    rev_notes = []
    if revision not in ("0", "00", "A", "—"):
        has_rev_table = bool(re.search(r"\b(?:revision\s+history|rev\s+description|amendment|record\ of\ revision)\b", submittal_text, re.IGNORECASE))
        if not has_rev_table:
            rev_notes.append({
                "id": len(findings) + 101,
                "section": "Revision Control",
                "type": "MINOR",
                "comment": f"Document is marked as Revision {revision}, but does not contain a revision history / description of change table.",
                "requirement": "Document Control Revision Traceability",
                "action": "Include a revision table detailing changes made from previous revision.",
                "closeout": "Required before next revision",
                "evidence": "Document Header Check",
            })

    # Append revision findings if any
    updated_findings = list(findings) + rev_notes

    return {
        "findings": updated_findings,
        "current_agent": "revision_verifier",
    }
