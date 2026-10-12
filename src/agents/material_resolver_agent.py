from __future__ import annotations

"""
QDR·AI — Agent 2: Material Resolver (Deterministic + LLM-on-Exception)
────────────────────────────────────────────────────────────────────────
Every row gets a deterministic rule-based material grade first (fast,
no LLM). Rows whose grade came from an explicit text match are treated
as resolved. Rows whose grade came from inheritance (HAZ1/HAZ2), joint
weld context, or a bare drawing-based default are genuinely ambiguous —
those are batched into a single LLM call per document so the model
reasons only about the exceptions, not every row on every page.
"""

import json
import logging
import re
from typing import Any, Optional

import requests

from src.agents.state import (
    MaterialAssignment,
    PageOCRResult,
    ReviewState,
)
from src.config import (
    MAX_LLM_CANDIDATES_MATERIAL,
    OLLAMA_BATCH_CONCURRENCY,
    OLLAMA_BATCH_TIMEOUT,
    OLLAMA_MODEL,
    get_ollama_host,
)
from src.utils.logger import pipeline_logger
from src.utils.metrics import StageTracker
from src.utils import progress as pipeline_progress

logger = logging.getLogger("qdr_ai.agent.material_resolver")

MATERIAL_RESOLVER_SYSTEM_PROMPT = """You are a metallurgical QA reviewer classifying material grades on rows from a hardness test report that a rule-based pass could not confidently resolve.

Rules, in priority order:
1. Explicit text in the row always wins over any other rule.
2. HP drawing = P91. RP drawing = P22.
3. HAZ1 inherits its grade from the row directly above it.
4. HAZ2 inherits its grade from the row directly below it.

For every candidate you are given, confirm or correct its material grade (must be one of P91, P22, P11, CS) and give a one-sentence reasoning citing which rule you applied. Return exactly one object per candidate "id" — never skip one, never merge two into one object."""

MATERIAL_CANDIDATE_USER_PROMPT = """Review these {n} candidates and return exactly {n} objects, one per "id":

{rows_json}"""

# Forcing a JSON-array schema (not just format="json") is what actually makes the
# model emit one object per candidate — without it, it tends to answer only the
# first candidate and stop, since a single object already satisfies "valid JSON".
MATERIAL_BATCH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "material": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["id", "material", "reasoning"],
    },
}


def _call_ollama_batch(user_prompt: str, host: str, num_predict: int) -> str:
    """Call Ollama chat API for a batched candidate review (schema-constrained output)."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": MATERIAL_RESOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": MATERIAL_BATCH_SCHEMA,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    url = f"{host}/api/chat"
    res = requests.post(url, json=payload, timeout=OLLAMA_BATCH_TIMEOUT)
    if res.status_code == 200:
        return res.json().get("message", {}).get("content", "").strip()
    raise RuntimeError(f"Ollama returned status {res.status_code}: {res.text[:200]}")


def _clean_dict_keys(data: Any) -> Any:
    if isinstance(data, list):
        return [_clean_dict_keys(item) for item in data]
    elif isinstance(data, dict):
        new_d = {}
        for k, v in data.items():
            clean_k = str(k).strip("\"' ")
            new_d[clean_k] = _clean_dict_keys(v)
        return new_d
    return data


def _extract_row_objects(cleaned: str) -> list[dict]:
    """Salvage individual `{...}` row objects even if the outer array got truncated."""
    objects: list[dict] = []
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(cleaned[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def _parse_batch_response(raw: str) -> list[dict]:
    """Extract a list of row objects from a JSON-mode LLM response, tolerating truncation."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        val = json.loads(cleaned)
        if isinstance(val, list):
            return _clean_dict_keys(val)
        if isinstance(val, dict):
            for v in val.values():
                if isinstance(v, list):
                    return _clean_dict_keys(v)
    except json.JSONDecodeError:
        pass
    return _clean_dict_keys(_extract_row_objects(cleaned))


def _fallback_material(page: PageOCRResult, row, row_idx: int) -> tuple[str, bool, str]:
    """
    Rule-based grade for a single row. Returns (material, is_grounded, basis).
    Checks row text first, then checks page-level grounding (test block info,
    material specification, drawing code, and page text exclusivity).
    Only falls back to P91 default if no other material evidence exists.
    """
    t = row.left_text.lower()
    if re.search(r"\bp22\b", t):
        return "P22", True, "explicit text"
    if re.search(r"\bp11\b", t):
        return "P11", True, "explicit text"
    if re.search(r"\b(?:p91|f91|wp91|pot|por|pol|pal|p9i|po1|p9l|grade\s*91)\b", t):
        return "P91", True, "explicit text"
    if re.search(r"\b(?:gr\.?c|cs)\b", t):
        return "CS", True, "explicit text"

    # Page-level evidence
    tb = (page.test_block_info or "").upper()
    full_upper = (page.full_text or "").upper()
    mat_spec = (getattr(page, "material_spec", "") or "").upper()
    dwg = (page.drawing_no or "").upper()

    # 1. Test Block indicates material (e.g. DDEL-TH-P11-C2 -> P11, DDEL-TH-P22-D3 -> P22)
    if "P11" in tb:
        return "P11", True, f"page test block ({page.test_block_info})"
    if "P22" in tb and not re.search(r"\b(?:P91|F91)\b", full_upper):
        return "P22", True, f"page test block ({page.test_block_info})"
    if "P91" in tb:
        return "P91", True, f"page test block ({page.test_block_info})"

    # 2. Material spec explicitly on page
    if "P11" in mat_spec or "1.25CR" in mat_spec:
        return "P11", True, f"page material spec ({getattr(page, 'material_spec', '')})"
    if ("P22" in mat_spec or "2.25CR" in mat_spec) and "P91" not in mat_spec:
        return "P22", True, f"page material spec ({getattr(page, 'material_spec', '')})"
    if "P91" in mat_spec or "9CR" in mat_spec:
        return "P91", True, f"page material spec ({getattr(page, 'material_spec', '')})"

    # 3. Full text alloy exclusivity (page mentions P11 or P22 without mentioning P91)
    has_p11 = bool(re.search(r"\b(?:P11|F11|WP11|ALLOY\s*11)\b", full_upper))
    has_p22 = bool(re.search(r"\b(?:P22|F22|WP22|ALLOY\s*22)\b", full_upper))
    has_p91 = bool(re.search(r"\b(?:P91|F91|WP91|ALLOY\s*91)\b", full_upper))

    if has_p11 and not has_p91:
        return "P11", True, "page mentions P11 exclusively"
    if has_p22 and not has_p91:
        return "P22", True, "page mentions P22 exclusively"

    # 4. Drawing code heuristic
    if "SIL" in dwg and not has_p91:
        return "P11", True, f"drawing code {dwg} (SIL=P11)"
    # DISABLED: `if "RP" in dwg ...` inferred P22 from the spool/drawing
    # prefix. Spool numbering is not a material designation — see
    # src/agents/context_resolver.py. Condition forced False.
    if False and "RP" in dwg and not has_p91 and not has_p11:
        return "P22", False, f"drawing code {dwg} (RP=P22)"

    # 5. Default fallback
    return "P91", True, "default (P91 submittal)"


def _resolve_page_deterministic(
    page: PageOCRResult,
) -> tuple[list[MaterialAssignment], list[dict]]:
    """
    Rule-based material assignment for every row on a page.
    A row is "grounded" if its grade traces back to explicit text or measured
    evidence — including through an inheritance/context chain. Only rows whose
    entire chain bottoms out in a bare drawing-code guess, with no grounding
    anywhere, become LLM candidates; inheriting from an already-grounded
    neighbor is not actually ambiguous and doesn't need review.
    """
    n = len(page.table_rows)
    materials: list[Optional[str]] = [None] * n
    bases: list[Optional[str]] = [None] * n
    grounded: list[bool] = [False] * n

    page_tb = (page.test_block_info or "").upper()
    page_dwg = (page.drawing_no or "").upper()
    is_p11_page = "P11" in page_tb or "SIL" in page_dwg
    is_p22_page = "P22" in page_tb and "P91" not in page_tb

    # Pass 1: explicit text / measured-range rows (grounded).
    for idx, row in enumerate(page.table_rows):
        t = row.left_text.lower()
        vals = [r.value for r in row.readings if r.value > 0]
        if re.search(r"\b(?:p22|f22|wp22|grade\s*22)\b", t):
            materials[idx], bases[idx], grounded[idx] = "P22", "explicit text", True
        elif re.search(r"\b(?:p11|f11|wp11|grade\s*11)\b", t):
            materials[idx], bases[idx], grounded[idx] = "P11", "explicit text", True
        elif re.search(r"\b(?:p91|f91|wp91|pot|por|pol|pal|p9i|po1|p9l|grade\s*91)\b", t):
            materials[idx], bases[idx], grounded[idx] = "P91", "explicit text", True
        elif re.search(r"\b(?:gr\.?c|cs)\b", t):
            materials[idx], bases[idx], grounded[idx] = "CS", "explicit text", True
        elif row.component == "Plate" and vals and all(v <= 235 for v in vals):
            materials[idx], bases[idx], grounded[idx] = "P22", "readings in P22 plate range (max 235 HBW)", True
        elif not is_p11_page and not is_p22_page and vals and any(v > 235 for v in vals) and all(180 <= v <= 315 for v in vals):
            materials[idx], bases[idx], grounded[idx] = (
                "P91", "readings exceed P22/P11/CS max (235 HBW) — must be P91", True,
            )

    # Pass 2: inherit from the nearest explicitly-graded row anywhere on the page
    MAX_INHERIT_DISTANCE = 4
    explicit_idxs = [i for i in range(n) if grounded[i]]
    for idx, row in enumerate(page.table_rows):
        if materials[idx] is not None:
            continue
        vals = [r.value for r in row.readings if r.value > 0]

        nearest = None
        if explicit_idxs:
            min_dist = min(abs(j - idx) for j in explicit_idxs)
            if min_dist <= MAX_INHERIT_DISTANCE:
                tied = [j for j in explicit_idxs if abs(j - idx) == min_dist]
                if len(tied) == 1 or not vals:
                    nearest = tied[0]
                else:
                    def _fit_count(j: int) -> int:
                        lo, hi = (195, 280) if materials[j] == "P91" else (0, 235)
                        return sum(1 for v in vals if lo <= v <= hi)
                    nearest = max(tied, key=_fit_count)

        if nearest is not None:
            inherited_mat = materials[nearest]
            materials[idx] = inherited_mat
            bases[idx] = f"nearest explicit grade (row {nearest + 1}, {page.table_rows[nearest].component})"
            grounded[idx] = True
            explicit_idxs.append(idx)
        else:
            mat, is_grounded, basis = _fallback_material(page, row, idx)
            materials[idx] = mat
            bases[idx] = basis
            grounded[idx] = is_grounded

    assignments: list[MaterialAssignment] = []
    candidates: list[dict] = []
    for idx, row in enumerate(page.table_rows):
        mat, basis = materials[idx], bases[idx]
        assignments.append(MaterialAssignment(
            page_idx=page.page_idx,
            row_index=idx,
            component=row.component,
            material=mat,
            reasoning=f"Rule-based ({basis}): joint={row.joint_name}, mat={mat}",
        ))
        if not grounded[idx]:
            candidates.append({
                "page_idx": page.page_idx,
                "row_index": idx,
                "component": row.component,
                "is_weld": row.is_weld,
                "left_text": row.left_text,
                "joint_name": row.joint_name,
                "drawing_no": page.drawing_no,
                "spool_no": page.spool_no,
                "rule_based_guess": mat,
                "basis": basis,
            })

    return assignments, candidates


def _review_batch(batch: list[dict], ollama_host: str, batch_num: int) -> dict[tuple[int, int], dict]:
    rows_for_prompt = [
        {
            "id": i,
            "component": c["component"],
            "is_weld": c["is_weld"],
            "left_text": c["left_text"],
            "joint_name": c["joint_name"],
            "drawing_no": c["drawing_no"],
            "spool_no": c["spool_no"],
            "rule_based_guess": c["rule_based_guess"],
        }
        for i, c in enumerate(batch)
    ]
    prompt = MATERIAL_CANDIDATE_USER_PROMPT.format(n=len(batch), rows_json=json.dumps(rows_for_prompt, indent=2))

    resolved: dict[tuple[int, int], dict] = {}
    try:
        raw_response = _call_ollama_batch(prompt, ollama_host, num_predict=min(len(batch) * 30 + 50, 400))
        parsed = _parse_batch_response(raw_response)
        for item in parsed:
            idx = item.get("id")
            if idx is None or not (0 <= idx < len(batch)):
                continue
            mat = str(item.get("material", "")).upper().strip()
            if mat in ("P91", "P22", "P11", "CS"):
                c = batch[idx]
                resolved[(c["page_idx"], c["row_index"])] = {
                    "material": mat,
                    "reasoning": item.get("reasoning", f"LLM-confirmed: {mat}"),
                }
        logger.info(f"Material candidate batch {batch_num}: {len(resolved)} of {len(batch)} rows resolved by LLM")
    except Exception as e:
        logger.warning(
            f"Material candidate batch {batch_num} failed ({e}) — keeping rule-based guesses for this batch"
        )
    return resolved


def _review_candidates_with_llm(
    candidates: list[dict],
    ollama_host: str,
    batch_size: int = 20,
) -> dict[tuple[int, int], dict]:
    """
    Send ambiguous rows to the LLM in batches, keyed by a synthetic "id".
    Returns {(page_idx, row_index): {"material": ..., "reasoning": ...}}.
    """
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    resolved: dict[tuple[int, int], dict] = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=OLLAMA_BATCH_CONCURRENCY) as executor:
        futures = {
            executor.submit(_review_batch, batch, ollama_host, i + 1): i
            for i, batch in enumerate(batches)
        }
        completed_count = 0
        for future in as_completed(futures):
            resolved.update(future.result())
            completed_count += 1
            pct = 50 + int((completed_count / len(batches)) * 50)
            pipeline_progress.set_percentage(pct)

    return resolved


# ── LangGraph Node: Material Resolver ──────────────────────────────

def material_resolver_node(state: ReviewState) -> dict:
    """
    LangGraph node: Resolve material grade for each table row.
    Every row is resolved deterministically first; only rows whose grade
    was inferred (not read explicitly) are batched into LLM calls.
    """
    page_results = state.get("page_ocr_results", [])

    all_assignments: list[MaterialAssignment] = []
    all_candidates: list[dict] = []
    reasoning_traces: list[dict] = []

    pipeline_progress.set_percentage(0)

    with StageTracker("resolve_materials", f"pages={len(page_results)}") as tracker:
        pages_with_data = [p for p in page_results if p.table_rows]
        tracker.details += f" | pages_with_rows={len(pages_with_data)}"

        total_pages_count = max(1, len(pages_with_data))
        for idx, p in enumerate(pages_with_data):
            assignments, candidates = _resolve_page_deterministic(p)
            all_assignments.extend(assignments)
            all_candidates.extend(candidates)
            pct = int(((idx + 1) / total_pages_count) * (50 if all_candidates else 95))
            pipeline_progress.set_percentage(pct)

        tracker.details += f" | total_assignments={len(all_assignments)} | candidates={len(all_candidates)}"

        llm_candidates = all_candidates
        skipped = 0
        if len(all_candidates) > MAX_LLM_CANDIDATES_MATERIAL:
            skipped = len(all_candidates) - MAX_LLM_CANDIDATES_MATERIAL
            llm_candidates = all_candidates[:MAX_LLM_CANDIDATES_MATERIAL]
            pipeline_logger.info(
                f"Material Resolver: {len(all_candidates)} candidates exceed the "
                f"{MAX_LLM_CANDIDATES_MATERIAL}-candidate LLM budget — reviewing the first "
                f"{MAX_LLM_CANDIDATES_MATERIAL}, {skipped} keep their rule-based guess"
            )
            tracker.details += f" | llm_budget_skipped={skipped}"

        if llm_candidates:
            ollama_host = get_ollama_host()
            tracker.details += f" | ollama={ollama_host}"
            resolved = _review_candidates_with_llm(llm_candidates, ollama_host)

            assignments_by_key = {(a.page_idx, a.row_index): a for a in all_assignments}
            for key, result in resolved.items():
                a = assignments_by_key.get(key)
                if a:
                    a.material = result["material"]
                    a.reasoning = f"LLM-confirmed: {result['reasoning']}"
            tracker.details += f" | llm_resolved={len(resolved)}"

        pipeline_progress.set_percentage(100)

        for a in all_assignments:
            reasoning_traces.append({
                "stage": "material_resolver",
                "page": a.page_idx + 1,
                "row": a.row_index,
                "component": a.component,
                "material": a.material,
                "reasoning": a.reasoning,
            })

    pipeline_logger.info(
        f"Material Resolver: {len(all_assignments)} row-material assignments "
        f"across {len(pages_with_data)} pages ({len(llm_candidates)} sent to LLM, "
        f"{skipped} kept rule-based due to LLM budget)"
    )

    return {
        "material_assignments": all_assignments,
        "llm_reasoning_traces": reasoning_traces,
        "current_agent": "material_resolver",
    }
