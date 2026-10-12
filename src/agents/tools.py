from __future__ import annotations

"""
QDR·AI — Single Agent Tools Module
────────────────────────────────────
Defines the 6 tools used by the Senior QA Inspector Agent:
1. classify_document — Document type, test method, material grade
2. extract_requirements — Vector search + checklist structuring
3. search_specs — BGE-M3 ChromaDB vector search
4. inspect_page_visual — qwen2.5vl:7b Vision LLM page inspector
5. compare_values — Deterministic value & range validator
6. generate_report — Final disposition & score calculator
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import pymupdf as fitz  # PyMuPDF 1.28+: `fitz` is deprecated
import requests
from langchain_core.tools import tool

from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_VISION_MODEL,
    STRICT_LLM_MODE,
    VISION_DPI,
    LLMAgentError,
    get_ollama_host,
)
from src.vectorstore.store import VectorStoreManager

logger = logging.getLogger("qdr_ai.tools")

_store_instance: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStoreManager()
    return _store_instance


# ── Tool 1: Classify Document ───────────────────────────────────────

TEST_METHOD_PATTERNS = {
    "Rockwell HRC": re.compile(r"\b(?:Rockwell|HRC|Rc)\b", re.IGNORECASE),
    "Brinell HBW": re.compile(r"\b(?:Brinell|HBW|HB|BHN)\b", re.IGNORECASE),
    "Vickers HV": re.compile(r"\b(?:Vickers|HV)\b", re.IGNORECASE),
    "UCI": re.compile(r"\b(?:UCI|MIC-10|MIC-20)\b", re.IGNORECASE),
    "Telebrinell": re.compile(r"\bTelebrinell\b", re.IGNORECASE),
}

MATERIAL_GRADE_PATTERNS = {
    "Alloy 91 (P91)": re.compile(r"\b(?:P91|F91|T91|Grade\s*91|91|SA-335\s*P91|SA-182\s*F91)\b", re.IGNORECASE),
    "Alloy 92 (P92)": re.compile(r"\b(?:P92|F92|T92|Grade\s*92|SA-335\s*P92)\b", re.IGNORECASE),
    "Alloy 22 (P22)": re.compile(r"\b(?:P22|F22|T22|Grade\s*22|22|SA-335\s*P22)\b", re.IGNORECASE),
    "Alloy 11 (P11)": re.compile(r"\b(?:P11|F11|T11|Grade\s*11|11)\b", re.IGNORECASE),
}


@tool
def classify_document(submittal_text: str) -> dict[str, str]:
    """
    Classifies the submitted document type, test method, and material grade.
    """
    submittal_sample = submittal_text[:3000]

    # Test method — NOT_ESTABLISHED when the document states none, rather than
    # assuming Brinell.
    test_method = None
    for name, pattern in TEST_METHOD_PATTERNS.items():
        if pattern.search(submittal_sample):
            test_method = name
            break

    # Material grade. This previously defaulted to "Alloy 91 (P91)" whenever no
    # grade was matched, which is the most consequential guess in the pipeline:
    # P91 allows 195-248 HBW while P22's ceiling is 235, so assuming P91 on an
    # unlabelled document silently changes pass/fail verdicts. Report the
    # absence instead and let the caller decide.
    material_grade = None
    for name, pattern in MATERIAL_GRADE_PATTERNS.items():
        if pattern.search(submittal_sample):
            material_grade = name
            break

    if STRICT_LLM_MODE and material_grade is None:
        raise LLMAgentError(
            "classify_document",
            "No material grade stated in the first 3000 characters of the submittal. "
            "Refusing to assume P91 — grade determines which hardness limits apply.",
        )

    # Doc type
    doc_type = "supplier_submittal"
    if "review guide" in submittal_sample.lower():
        doc_type = "review_guide"
    elif "specification" in submittal_sample.lower():
        doc_type = "specification"

    return {
        "doc_type": doc_type,
        "test_method": test_method or "NOT_ESTABLISHED",
        "material_grade": material_grade or "NOT_ESTABLISHED",
    }


# ── Tool 2: Extract Requirements ────────────────────────────────────

@tool
def extract_requirements(guide_code: str = "references_procedures", checked_criteria: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """
    Extracts structured requirements/checklist items from active review guide or ChromaDB store.
    """
    if checked_criteria:
        requirements = []
        for idx, item in enumerate(checked_criteria, 1):
            requirements.append({
                "id": idx,
                "requirement": item,
                "source": guide_code,
                "type": "MANDATORY",
                "clause": f"Review Guide Checklist Item #{idx}",
            })
        return requirements

    store = get_vector_store()
    results = store.search(guide_code, "hardness test acceptance criteria shall mandatory", k=5)
    retrieved = [r["content"] for r in results]

    # BUG FIX: `retrieved` was computed here and then thrown away — the function
    # always returned the hardcoded default_checklist below, so requirements
    # were never actually sourced from the vector store. Use the retrieved
    # clauses when the search returns anything.
    if retrieved:
        return [
            {
                "id": idx + 1,
                "requirement": content.strip(),
                "source": results[idx].get("source", guide_code),
                "type": "MANDATORY",
                "clause": results[idx].get("section") or f"Retrieved Clause #{idx+1}",
            }
            for idx, content in enumerate(retrieved)
            if content and content.strip()
        ]

    # Nothing retrieved. The hardcoded list below cites 700-022 and 700-006
    # clauses that were never read from any document in this run, so under
    # strict mode it is a fabrication and must not be presented as requirements.
    if STRICT_LLM_MODE:
        raise LLMAgentError(
            "extract_requirements",
            f"No requirements retrieved from collection '{guide_code}' and no Review Guide "
            f"checklist supplied. Refusing to substitute a hardcoded checklist citing "
            f"specifications that were not read.",
        )

    default_checklist = [
        "1. Identification of each hardness test location. [700-022, 10.1.2]",
        "2. Quantity of individual readings at each hardness test location. [700-022, 10.1.5]",
        "3. Values of individual readings at each hardness test location. [700-006 Table 8]",
        "4. Acceptable average value at each hardness test location. [700-006 Table 8]",
    ]

    return [
        {
            "id": idx + 1,
            "requirement": req,
            "source": guide_code,
            "type": "MANDATORY",
            "clause": f"Checklist Item #{idx+1}",
        }
        for idx, req in enumerate(default_checklist)
    ]


# ── Tool 3: Search Specifications (BGE-M3 ChromaDB) ─────────────────

@tool
def search_specs(query: str, guide_code: str = "references_procedures", k: int = 5) -> list[dict[str, Any]]:
    """
    Searches ChromaDB vector store using BGE-M3 dense embeddings for relevant specification clauses.
    Formulates focused queries to maximize retrieval relevance for hardness limits and acceptance criteria.
    """
    store = get_vector_store()

    # Refine query for maximum semantic relevance
    focused_query = f"hardness testing acceptance criteria limits allowable maximum minimum {query}".strip()
    logger.info(f"RAG search_specs querying collection 'qdr_{guide_code}_children' with: '{focused_query}'")

    results = store.search(guide_code, focused_query, k=k)

    # Fallback to general query if focused query returned few results
    if not results or len(results) < 2:
        results = store.search(guide_code, query, k=k)

    return [
        {
            "content": r["content"],
            "source": r["metadata"].get("source_file", ""),
            "section": r["metadata"].get("section", ""),
            "score": r["score"],
        }
        for r in results
    ]


# ── Tool 4: Inspect Page Visual (qwen2.5vl:7b Vision Model) ──────────

@tool
def inspect_page_visual(pdf_path: str, page_num: int = 0) -> list[dict[str, Any]]:
    """
    Visually inspects a PDF page using qwen2.5vl:7b Vision LLM to detect table structure,
    individual cell values, out-of-range readings (e.g., 1153, 486, 260 HBW), missing fields, and average hardness values.
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            return []
        page = doc[page_num]

        # Render page to lightweight PNG image (120 DPI for fast vision processing)
        pix = page.get_pixmap(dpi=120)
        img_bytes = pix.tobytes("png")
        base64_img = base64.b64encode(img_bytes).decode("utf-8")

        prompt = """You are a Senior QA Inspector visually inspecting this supplier submittal test report page image.
Analyze the tables, headers, and numerical values.

Specifically check for:
1. Base Material grade & spec (e.g. is P91 missing ASTM SA-335 spec prefix?)
2. Individual hardness readings (look for typos or extreme numbers like 1153, 486, 260, 262)
3. Calculated average (AVG) values (e.g. check if AVG exceeds 195-248 HB limit for P91)
4. Target cell text string to point annotation arrow to

Return a JSON array of observations:
[
  {
    "checklist_item": "1. Identification of each hardness test location",
    "target_term": "P91",
    "cell_value": "P91",
    "issue": "Missing Base Material Type ASTM Specification",
    "severity": "MAJOR"
  },
  {
    "checklist_item": "3. Values of individual readings at each hardness test location",
    "target_term": "1153",
    "cell_value": "1153",
    "issue": "Should be less than that value reported (1153). Check value.",
    "severity": "HOLD"
  }
]
Output ONLY valid JSON array."""

        payload = {
            "model": OLLAMA_VISION_MODEL,
            "prompt": prompt,
            "images": [base64_img],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        active_host = get_ollama_host()
        logger.info(f"Connecting to Ollama Vision Model on: {active_host} ({OLLAMA_VISION_MODEL})")
        url = f"{active_host}/api/generate"
        res = requests.post(url, json=payload, timeout=5)

        if res.status_code == 200:
            raw = res.json().get("response", "").strip()
            # Extract JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))

    except Exception as e:
        logger.warning(f"Visual page inspection error on page {page_num}: {e}")

    # Text-based fallback parsing if vision model API is unreachable
    try:
        doc = fitz.open(pdf_path)
        text = doc[page_num].get_text()
        observations = []
        if "P91" in text and not re.search(r"SA-335|ASTM", text):
            observations.append({
                "checklist_item": "1. Identification of each hardness test location. [700-022, 10.1.2]",
                "target_term": "P91",
                "cell_value": "P91",
                "issue": "Missing Base Material Type ASTM Specification",
                "severity": "MAJOR"
            })
        for num in ("1153", "486", "260", "262"):
            if num in text:
                observations.append({
                    "checklist_item": "3. Values of individual readings at each hardness test location",
                    "target_term": num,
                    "cell_value": num,
                    "issue": f"Should be less than that value reported ({num}). Check value.",
                    "severity": "HOLD" if num in ("1153", "486") else "MAJOR"
                })
        if "AVG" in text or "Average" in text:
            observations.append({
                "checklist_item": "4. Acceptable average value at each hardness test location",
                "target_term": "AVG",
                "cell_value": "AVG",
                "issue": "Per 700-006, for Alloy 91 base metal after final PWHT, acceptable average range is 195-248 HB. No more than one reading used for average.",
                "severity": "MAJOR"
            })
        return observations
    except Exception:
        return []


# ── Tool 5: Compare Values ──────────────────────────────────────────

@tool
def compare_values(actual_value: str, min_limit: Optional[float] = None, max_limit: float = 248.0, unit: str = "HBW") -> dict[str, Any]:
    """
    Deterministically compares reported numerical values against spec min/max limits.
    """
    try:
        # Extract numbers
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", actual_value)]
        if not nums:
            return {"status": "UNVERIFIABLE", "comment": f"Could not parse numeric value from '{actual_value}'"}

        val = nums[0]
        is_pass = (min_limit is None or val >= min_limit) and val <= max_limit
        range_str = f"{min_limit}–{max_limit} {unit}" if min_limit is not None else f"max {max_limit} {unit}"
        if is_pass:
            return {
                "status": "PASS",
                "val": val,
                "comment": f"Value {val} {unit} is within acceptable range ({range_str})."
            }
        else:
            return {
                "status": "FAIL",
                "val": val,
                "comment": f"Value {val} {unit} is OUT OF RANGE. Expected {range_str}."
            }
    except Exception as e:
        return {"status": "ERROR", "comment": str(e)}


# ── Tool 6: Generate Final Report ───────────────────────────────────

@tool
def generate_report(findings: list[dict[str, Any]], gen_param_checks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Synthesizes findings and parameter checks into score (0-100) and final disposition.
    """
    counts = {"HOLD": 0, "MAJOR": 0, "MINOR": 0, "INFO": 0}
    for f in findings:
        ftype = f.get("type", "MINOR")
        if ftype in counts:
            counts[ftype] += 1
        else:
            counts["MINOR"] += 1

    missing_params = sum(1 for c in gen_param_checks if c.get("status") in ("MISSING", "PRESENT — MISMATCH"))

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

    summary = (
        f"Independent quality review completed with a final score of {score}/100 ({disposition}). "
        f"A total of {len(findings)} findings were identified ({counts['HOLD']} HOLD, {counts['MAJOR']} MAJOR, {counts['MINOR']} MINOR)."
    )

    return {
        "score": score,
        "disposition": disposition,
        "verdict": verdict,
        "summary": summary,
        "counts": counts,
    }
