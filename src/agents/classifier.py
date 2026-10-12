from __future__ import annotations

"""
QDR·AI — Document Classifier Agent
────────────────────────────────────
Agent 1 in the review pipeline.

Classifies the submitted document:
  - Document type (HRDR, tensile, impact, PWHT, etc.)
  - Test method (Rockwell HRC, Brinell HBW, etc.)
  - Material grade (Alloy 91, Grade 92, etc.)
  - Extracts basic document metadata

Uses Ollama LLM for classification with fallback to rule-based detection.
"""

import logging
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import OLLAMA_MODEL, OLLAMA_TIMEOUT, get_ollama_host
from src.agents.state import ReviewState

logger = logging.getLogger("qdr_ai.agent.classifier")


# ── Rule-Based Classification (Fallback) ────────────────────────────

TEST_METHOD_PATTERNS = {
    "Rockwell HRC": re.compile(
        r"\b(?:Rockwell|HRC|Rc)\b", re.IGNORECASE
    ),
    "Brinell HBW": re.compile(
        r"\b(?:Brinell|HBW|HB|BHN)\b", re.IGNORECASE
    ),
    "Vickers HV": re.compile(
        r"\b(?:Vickers|HV)\b", re.IGNORECASE
    ),
    "UCI": re.compile(
        r"\b(?:UCI|MIC-10|MIC-20)\b", re.IGNORECASE
    ),
    "Telebrinell": re.compile(
        r"\bTelebrinell\b", re.IGNORECASE
    ),
}

DOC_TYPE_PATTERNS = {
    "hardness_report": re.compile(
        r"\b(?:hardness\s+test|HRC|HBW|Brinell|Rockwell|"
        r"hardness\s+report|HRDR)\b",
        re.IGNORECASE,
    ),
    "tensile_report": re.compile(
        r"\b(?:tensile\s+test|yield\s+strength|ultimate\s+tensile|"
        r"elongation|reduction\s+of\s+area)\b",
        re.IGNORECASE,
    ),
    "impact_report": re.compile(
        r"\b(?:Charpy|impact\s+test|absorbed\s+energy|"
        r"notch\s+toughness|CVN)\b",
        re.IGNORECASE,
    ),
    "pwht_report": re.compile(
        r"\b(?:PWHT|post\s*weld\s*heat\s*treat|soak\s+temp|"
        r"thermocouple|time-temperature)\b",
        re.IGNORECASE,
    ),
    "welding_procedure": re.compile(
        r"\b(?:WPS|PQR|welding\s+procedure|filler\s+metal|"
        r"preheat|interpass)\b",
        re.IGNORECASE,
    ),
    "specification": re.compile(
        r"\b(?:specification|spec\s+for|requirements\s+for)\b",
        re.IGNORECASE,
    ),
}

MATERIAL_PATTERNS = {
    "Alloy 91": re.compile(
        r"\b(?:Alloy\s*91|P91|T91|F91|Grade\s*91|SA-335.*P91|"
        r"C12A|B9|B91)\b",
        re.IGNORECASE,
    ),
    "Alloy 92": re.compile(
        r"\b(?:Alloy\s*92|P92|T92|Grade\s*92)\b",
        re.IGNORECASE,
    ),
    "Alloy 22": re.compile(
        r"\b(?:Alloy\s*22|P22|T22|2.*Chrome)\b",
        re.IGNORECASE,
    ),
    "Carbon Steel": re.compile(
        r"\b(?:carbon\s+steel|CS|SA-106|SA-516)\b",
        re.IGNORECASE,
    ),
}


def classify_rule_based(text: str) -> dict:
    """Rule-based document classification from text patterns."""
    result = {
        "doc_type": "unknown",
        "test_method": "Not identified",
        "material_grade": "Not identified",
    }

    # Detect document type
    for doc_type, pattern in DOC_TYPE_PATTERNS.items():
        if pattern.search(text[:5000]):
            result["doc_type"] = doc_type
            break

    # Detect test method
    methods_found = []
    for method, pattern in TEST_METHOD_PATTERNS.items():
        if pattern.search(text):
            methods_found.append(method)
    if methods_found:
        result["test_method"] = ", ".join(methods_found)

    # Detect material grade
    grades_found = []
    for grade, pattern in MATERIAL_PATTERNS.items():
        if pattern.search(text[:5000]):
            grades_found.append(grade)
    if grades_found:
        result["material_grade"] = ", ".join(grades_found)

    return result


# ── LLM-Based Classification ────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are an engineering document classifier for a quality review system.
Given the first portion of a document, classify it by:

1. doc_type: One of: hardness_report, tensile_report, impact_report, pwht_report,
   welding_procedure, specification, review_guide, supplier_submittal, ne_comments, unknown
2. test_method: The test method used (e.g., Rockwell HRC, Brinell HBW, Vickers HV, UCI, Telebrinell, N/A)
3. material_grade: The primary material (e.g., Alloy 91, Alloy 92, Carbon Steel, Unknown)

Return ONLY a JSON object with these three fields. No markdown, no explanation."""


def classify_with_llm(text: str) -> Optional[dict]:
    """Classify document using Ollama LLM."""
    try:
        llm = ChatOllama(
            base_url=get_ollama_host(),
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
            temperature=0.0,
        )

        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Classify this document:\n\n{text[:3000]}"
            ),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        # Parse JSON response
        import json
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        result = json.loads(content)
        logger.info(f"LLM classification: {result}")
        return result

    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
        return None


# ── Agent Node Function ─────────────────────────────────────────────

def classify_document(state: ReviewState) -> dict:
    """
    LangGraph node: Classify the submittal document.
    Prioritizes fast regex pattern match (0.001s) with fast LLM fallback.
    """
    text = state.get("submittal_text", "")
    logger.info("Executing Document Classifier Agent...")

    # Fast 0.001s Rule-Based Classification First
    rule_res = classify_rule_based(text)
    if rule_res and rule_res.get("doc_type") != "unknown":
        logger.info(f"Instant Rule-Based Classification (0.001s): {rule_res}")
        return {
            "doc_type": rule_res.get("doc_type", "hardness_report"),
            "test_method": rule_res.get("test_method", "Brinell HBW"),
            "material_grade": rule_res.get("material_grade", "Alloy 91"),
            "current_agent": "classifier",
        }

    # Fallback to Fast LLM Classification (timeout 10s, num_ctx 2048)
    llm_res = None
    try:
        llm = ChatOllama(
            base_url=get_ollama_host(),
            model=OLLAMA_MODEL,
            timeout=10,
            temperature=0.0,
            num_ctx=2048,
        )
        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this document:\n\n{text[:2000]}"),
        ]
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        llm_res = json.loads(content)
    except Exception as e:
        logger.warning(f"Fast LLM classification fallback warning: {e}")

    final_res = llm_res or rule_res or {}
    return {
        "doc_type": final_res.get("doc_type", "hardness_report"),
        "test_method": final_res.get("test_method", "Brinell HBW"),
        "material_grade": final_res.get("material_grade", "Alloy 91"),
        "current_agent": "classifier",
    }
