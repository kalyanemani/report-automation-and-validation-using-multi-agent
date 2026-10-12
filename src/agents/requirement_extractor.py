from __future__ import annotations

"""
QDR·AI — Requirement Extractor Agent
──────────────────────────────────────
Agent 2 in the review pipeline.

Extracts structured requirements from:
  - Review guides (checklist items)
  - Specifications (acceptance criteria)
  - Procedures (mandatory steps)

Uses vector search to retrieve relevant requirements,
then structures them into a compliance checklist.
"""

import logging
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import OLLAMA_MODEL, OLLAMA_TIMEOUT, get_ollama_host
from src.agents.state import ReviewState
from src.vectorstore.store import VectorStoreManager

logger = logging.getLogger("qdr_ai.agent.extractor")


# ── Store singleton ──────────────────────────────────────────────────
_store: Optional[VectorStoreManager] = None


def get_store() -> VectorStoreManager:
    global _store
    if _store is None:
        _store = VectorStoreManager()
    return _store


# ── Requirement Extraction Prompts ──────────────────────────────────

EXTRACTOR_SYSTEM_PROMPT = """You are a quality engineering expert extracting compliance requirements
from engineering specifications and review guides.

Given the retrieved specification/procedure text, extract a structured checklist of
requirements that a supplier document must meet.

For each requirement, output:
- requirement: The specific requirement text
- source: Which document/section it comes from
- type: MANDATORY | RECOMMENDED | INFORMATIONAL
- acceptance_criteria: Specific values/ranges if applicable (e.g., "187-229 HBW")

Return ONLY a JSON array of requirement objects. No markdown, no explanation.
Example:
[
  {
    "requirement": "All production hardness testing shall be performed per ASTM A833",
    "source": "700-022 Section 2.1",
    "type": "MANDATORY",
    "acceptance_criteria": "Per ASTM A833 latest edition"
  }
]"""


# ── Rule-Based Requirement Extraction ────────────────────────────────

REQUIREMENT_INDICATORS = re.compile(
    r"\b(?:shall|must|required|mandatory|minimum|maximum|"
    r"not\s+(?:less|more|exceed)|at\s+least|"
    r"acceptance\s+criteria|specification\s+requires)\b",
    re.IGNORECASE,
)

ACCEPTANCE_VALUE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:to|[-–—])\s*(\d+(?:\.\d+)?)\s*"
    r"(HBW|HRC|HV|HB|ksi|MPa|°F|°C|%)",
    re.IGNORECASE,
)


def extract_requirements_rule_based(text: str) -> list[dict]:
    """
    Rule-based extraction of requirement statements from text.
    Looks for 'shall', 'must', 'required' patterns and acceptance criteria.
    """
    requirements = []
    seen = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 20:
            continue

        if REQUIREMENT_INDICATORS.search(stripped):
            # Normalize and deduplicate
            normalized = " ".join(stripped.split())
            if normalized not in seen and len(normalized) < 500:
                seen.add(normalized)

                # Try to extract acceptance values
                criteria_match = ACCEPTANCE_VALUE_RE.search(stripped)
                acceptance = ""
                if criteria_match:
                    acceptance = (
                        f"{criteria_match.group(1)}-"
                        f"{criteria_match.group(2)} "
                        f"{criteria_match.group(3)}"
                    )

                requirements.append({
                    "requirement": normalized,
                    "source": "extracted",
                    "type": "MANDATORY",
                    "acceptance_criteria": acceptance,
                })

    return requirements


# ── Agent Node Function ─────────────────────────────────────────────

def extract_requirements(state: ReviewState) -> dict:
    """
    LangGraph node: Extract requirements from vectorized reference docs.

    1. Searches the vector store for requirements relevant to the submittal
    2. Extracts structured requirements using LLM (or rule-based fallback)
    3. Returns checklist for the Compliance Reviewer agent
    """
    guide_code = state.get("guide_code", "references_procedures")
    submittal_text = state.get("submittal_text", "")
    doc_type = state.get("doc_type", "unknown")
    material = state.get("material_grade", "")
    test_method = state.get("test_method", "")

    logger.info(
        f"Extracting requirements for {guide_code} "
        f"(doc_type={doc_type}, material={material})"
    )

    store = get_store()

    existing_reqs = state.get("requirements", [])

    # Build targeted search queries based on classification & active Review Guide checked items
    queries = []

    # Priority 1: Use active checked Review Guide criteria as search queries
    for req in existing_reqs:
        clean_q = re.sub(r"[^\w\s\.-]", "", req).strip()
        if clean_q:
            queries.append(clean_q)

    # Build single consolidated search query for ultra-fast vector retrieval (0.3s)
    top_req = existing_reqs[0] if existing_reqs else ""
    single_query = f"{material} {test_method} hardness limits acceptance range {doc_type} {top_req}".strip()

    all_retrieved = []
    retrieved_texts = []
    seen_content = set()

    results = store.search(guide_code, single_query, k=5, return_parents=True)
    for r in results:
        content = r["content"]
        if content not in seen_content:
            seen_content.add(content)
            all_retrieved.append(r)
            retrieved_texts.append(content)

    logger.info(f"Retrieved {len(all_retrieved)} unique chunks in 1 fast query pass.")

    # Extract requirements from retrieved text
    combined_text = "\n\n".join(retrieved_texts)

    # If checked criteria are already provided in state (from Review Guide), use them directly
    requirements = []
    if existing_reqs:
        for idx, item in enumerate(existing_reqs, 1):
            requirements.append({
                "id": idx,
                "requirement": item,
                "source": guide_code,
                "type": "MANDATORY",
                "clause": "Review Guide Criteria",
            })
        logger.info(f"Using {len(requirements)} active checked Review Guide criteria directly (bypassing redundant LLM extraction)")
    else:
        # Try LLM extraction for unguided submittals
        try:
            llm = ChatOllama(
                base_url=get_ollama_host(),
                model=OLLAMA_MODEL,
                timeout=OLLAMA_TIMEOUT,
                temperature=0.0,
                num_ctx=4096,
            )

            messages = [
                SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Document type: {doc_type}\n"
                        f"Material: {material}\n"
                        f"Test method: {test_method}\n\n"
                        f"Reference text:\n{combined_text[:4000]}"
                    )
                ),
            ]

            response = llm.invoke(messages)
            content = response.content.strip()

            # Parse JSON
            import json
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            requirements = json.loads(content)
            logger.info(
                f"LLM extracted {len(requirements)} requirements"
            )

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            requirements = extract_requirements_rule_based(combined_text)
            logger.info(
                f"Rule-based extracted {len(requirements)} requirements"
            )

    # Build requirement source mapping
    requirement_sources = []
    for req in requirements:
        requirement_sources.append({
            "requirement": req.get("requirement", ""),
            "source_doc": req.get("source", ""),
            "type": req.get("type", "MANDATORY"),
            "acceptance_criteria": req.get(
                "acceptance_criteria", ""
            ),
        })

    return {
        "requirements": [
            r.get("requirement", "") for r in requirements
        ],
        "requirement_sources": requirement_sources,
        "retrieved_criteria": retrieved_texts[:20],
        "retrieved_context": [
            {
                "content": r["content"][:500],
                "source": r["metadata"].get("source_file", ""),
                "section": r["metadata"].get("section", ""),
                "score": r["score"],
            }
            for r in all_retrieved[:20]
        ],
        "current_agent": "requirement_extractor",
    }
