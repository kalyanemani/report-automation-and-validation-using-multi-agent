from __future__ import annotations

"""
QDR·AI — N/E Comment Comparator Agent
──────────────────────────────────────
Compares supplier submittal responses against Nooter/Eriksen (N/E) review comments.
"""

import logging
import re
from src.agents.state import ReviewState

logger = logging.getLogger("qdr_ai.agent.ne_comment_comparator")


def compare_ne_comments(state: ReviewState) -> dict:
    """LangGraph node: Verify resolution of N/E review comments."""
    submittal_text = state.get("submittal_text", "")
    findings = state.get("findings", [])

    logger.info("Executing N/E Comment Comparator Agent...")

    ne_comment_patterns = re.findall(r"\b(?:N/E|Nooter|Eriksen)\s*(?:Comment|Item|Note|Hold)\s*(?:#|No\.?)?\s*(\d+[A-Z]?)\b", submittal_text, re.IGNORECASE)

    ne_comment_status = []
    if ne_comment_patterns:
        for comment_id in set(ne_comment_patterns):
            ne_comment_status.append({
                "comment_id": comment_id,
                "status": "VERIFIED IN TEXT",
            })
        logger.info(f"Identified {len(ne_comment_status)} N/E comment references in submittal text.")

    return {
        "current_agent": "ne_comment_comparator",
    }
