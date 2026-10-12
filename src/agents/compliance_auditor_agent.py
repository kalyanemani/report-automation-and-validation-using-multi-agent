from __future__ import annotations

"""
QDR·AI — Agent 3: Compliance Auditor (Deterministic + LLM-on-Exception)
────────────────────────────────────────────────────────────────────────
Every reading is checked against its hardness limit deterministically
first (fast, no LLM). Readings that land comfortably inside or outside
the limit are decided by that check alone. Readings sitting close to a
limit boundary — where spec notes/exceptions (e.g. P91 weld single-
reading allowance) actually matter — are batched into a single LLM call
per document so the model reasons only about the borderline cases.
"""

import json
import logging
import re
from typing import Any

import requests

from src.agents.state import (
    ComplianceFinding,
    MaterialAssignment,
    PageOCRResult,
    ReviewState,
)
from src.config import (
    MAX_LLM_CANDIDATES_COMPLIANCE,
    OLLAMA_BATCH_CONCURRENCY,
    OLLAMA_BATCH_TIMEOUT,
    OLLAMA_MODEL,
    get_ollama_host,
)
from src.utils.logger import pipeline_logger
from src.utils.metrics import StageTracker
from src.utils import progress as pipeline_progress

logger = logging.getLogger("qdr_ai.agent.compliance_auditor")

# Readings within this many HBW of a limit are borderline and get an LLM review.
BORDERLINE_MARGIN_HBW = 15
SKIP_VALUES = (270, 180, 90, 0)  # calibration block readings, never evaluated

COMPLIANCE_AUDITOR_SYSTEM_PROMPT = """You are a QA reviewer deciding hardness compliance for borderline readings that a rule-based pass flagged as close to a limit.

Limits:
- P91 base metal: 195-248 HBW
- P91 weld / HAZ: 195-280 HBW (Note 3 allows a single reading up to 315 HBW if the rest of the weld's readings are within 195-280 HBW)
- P22 (2.25Cr-1Mo): max 235 HBW (no minimum)
- P11 (1.25Cr-0.5Mo): max 225 HBW (no minimum)
- Carbon Steel (CS): max 200 HBW (no minimum)

For every candidate you are given:
- Set "verdict" to exactly "PASS" if every reading is within its limit (accounting for Note 3 where it applies), or "FAIL" if any reading violates its limit.
- Only when "verdict" is "FAIL": set "severity" to "HOLD" if a reading is above the maximum, or "MAJOR" if a reading is below the minimum; list the specific failing values in "failing_readings"; state the limit/note violated in "limit_violated".
- When "verdict" is "PASS", leave "severity", "failing_readings", and "limit_violated" empty.
- Give a one-sentence "reasoning" that cites the specific limit or note applied.

Return exactly one object per candidate "id" — never skip one, never merge two into one object."""

COMPLIANCE_CANDIDATE_USER_PROMPT = """Review these {n} candidates and return exactly {n} objects, one per "id":

{rows_json}"""

# Forcing a JSON-array schema (not just format="json") is what actually makes the
# model emit one object per candidate — without it, it tends to answer only the
# first candidate and stop, since a single object already satisfies "valid JSON".
COMPLIANCE_BATCH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "verdict": {"type": "string"},
            "failing_readings": {"type": "array", "items": {"type": "integer"}},
            "limit_violated": {"type": "string"},
            "severity": {"type": "string"},
            "requirement_ref": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["id", "verdict", "reasoning"],
    },
}


def _call_ollama_batch(user_prompt: str, host: str, num_predict: int) -> str:
    """Call Ollama chat API for a batched candidate review (schema-constrained output)."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": COMPLIANCE_AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": COMPLIANCE_BATCH_SCHEMA,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    url = f"{host}/api/chat"
    res = requests.post(url, json=payload, timeout=OLLAMA_BATCH_TIMEOUT)
    if res.status_code == 200:
        return res.json().get("message", {}).get("content", "").strip()
    raise RuntimeError(f"Ollama returned status {res.status_code}: {res.text[:200]}")


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
            return val
        if isinstance(val, dict):
            for v in val.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    return _extract_row_objects(cleaned)


def _limits_for(
    material: str,
    is_weld: bool,
    component: str,
    page_acceptance_range: Optional[tuple[int, int]] = None,
) -> tuple[Optional[int], int]:
    """
    Returns (min_limit, max_limit) for a given row.
    P91: Weld & HAZ = (195, 280); Base Metal = (195, 248)
    P22: Max 235 HBW (no minimum)
    P11: Max 225 HBW (no minimum)
    CS: Max 200 HBW (no minimum)
    """
    mat_upper = (material or "P91").upper()
    comp_upper = (component or "").upper()
    is_weld_zone = is_weld or "WELD" in comp_upper or "HAZ" in comp_upper

    if "P91" in mat_upper or "9CR" in mat_upper:
        return (195, 280) if is_weld_zone else (195, 248)
    if "P22" in mat_upper or "2.25CR" in mat_upper or "F22" in mat_upper:
        return (None, 235)
    if "P11" in mat_upper or "1.25CR" in mat_upper or "F11" in mat_upper:
        return (None, 225)
    if "CS" in mat_upper or "CARBON" in mat_upper or "A106" in mat_upper or "WPB" in mat_upper:
        return (None, 200)
    return (None, 235)


def _evaluate_page_deterministic(
    page: PageOCRResult,
    mat_lookup: dict[int, str],
) -> tuple[list[ComplianceFinding], list[dict]]:
    """
    Rule-based compliance check for every reading on a page.
    Returns (findings, candidates) — candidates are rows with at least
    one reading within BORDERLINE_MARGIN_HBW of its limit, pass or fail.
    """
    findings: list[ComplianceFinding] = []
    candidates: list[dict] = []

    # Per-page acceptance range (from ISGEC form header, if available)
    page_acceptance = getattr(page, 'acceptance_range', None)

    for idx, row in enumerate(page.table_rows):
        material = mat_lookup.get(idx, "P91")
        reading_vals = [r.value for r in row.readings if 120 <= r.value <= 400]
        if not reading_vals:
            continue
        # A row is a calibration/reference check (not real test data) only if
        # EVERY reading in it is a suspiciously exact round number — real
        # measurements have natural scatter, so a genuine violation sitting
        # next to round numbers is never masked by this.
        if all(v in SKIP_VALUES for v in reading_vals):
            continue

        is_weld_zone = row.is_weld or "HAZ" in row.component or row.component == "Weld"
        min_limit, max_limit = _limits_for(material, row.is_weld, row.component, page_acceptance)

        # ── 700-022 §8.2: 5-reading trimming rule ─────────────────────────
        if len(reading_vals) == 5:
            sorted_vals = sorted(reading_vals)
            considered_vals = sorted_vals[1:-1] # Discard 1 lowest and 1 highest
        else:
            considered_vals = reading_vals

        avg_considered = sum(considered_vals) / len(considered_vals) if considered_vals else 0.0

        failing_high = [v for v in considered_vals if v > max_limit]
        failing_low = [v for v in considered_vals if min_limit is not None and v < min_limit]

        if not page_acceptance:
            # 700-022 Table 8 Note 2: For P91 Base Metal / HAZ (min 195 HBW), 1 reading >= 190 allowed if avg >= 195 HBW
            if material == "P91" and min_limit == 195.0:
                if len(failing_low) == 1 and failing_low[0] >= 190.0 and avg_considered >= 195.0:
                    failing_low = []

            if material == "P91" and is_weld_zone:
                # Note 3: single weld reading up to 315 HBW is acceptable if avg <= 280 HBW
                if len(failing_high) == 1 and failing_high[0] <= 315 and avg_considered <= 280.0:
                    failing_high = []
            elif material == "P91":
                # Single reading at 249-250 HBW on base metal acceptable if avg <= 248 HBW
                if len(failing_high) == 1 and failing_high[0] <= 250 and avg_considered <= 248.0:
                    failing_high = []

        margin = min(abs(v - max_limit) if min_limit is None else min(abs(v - max_limit), abs(v - min_limit)) for v in considered_vals) if considered_vals else 999.0
        is_borderline = margin <= BORDERLINE_MARGIN_HBW

        finding = None
        if failing_high:
            finding = ComplianceFinding(
                page_idx=page.page_idx,
                row_index=idx,
                component=row.component,
                material=material,
                failing_readings=failing_high,
                limit_violated=f"above maximum {max_limit} HBW",
                severity="HOLD",
                requirement_ref=f"700-022 Table 8 ({material} max {max_limit} HBW)",
                reasoning=f"Rule-based: {row.component} {material} readings {failing_high} exceed {max_limit} HBW",
            )
        elif failing_low:
            finding = ComplianceFinding(
                page_idx=page.page_idx,
                row_index=idx,
                component=row.component,
                material=material,
                failing_readings=failing_low,
                limit_violated=f"below minimum {min_limit} HBW",
                severity="MAJOR",
                requirement_ref=f"700-022 Table 8 Note 2 ({material} min {min_limit} HBW)",
                reasoning=f"Rule-based: {row.component} {material} readings {failing_low} below {min_limit} HBW",
            )
        if finding:
            findings.append(finding)

        if is_borderline:
            candidates.append({
                "page_idx": page.page_idx,
                "row_index": idx,
                "component": row.component,
                "material": material,
                "is_weld": row.is_weld,
                "joint_name": row.joint_name,
                "readings": reading_vals,
                "rule_based_verdict": "FAIL" if finding else "PASS",
                "margin": margin,
            })

    return findings, candidates


def _review_batch(batch: list[dict], ollama_host: str, batch_num: int) -> dict[tuple[int, int], dict]:
    rows_for_prompt = [
        {
            "id": i,
            "component": c["component"],
            "material": c["material"],
            "is_weld": c["is_weld"],
            "joint_name": c["joint_name"],
            "readings": c["readings"],
            "rule_based_verdict": c["rule_based_verdict"],
        }
        for i, c in enumerate(batch)
    ]
    prompt = COMPLIANCE_CANDIDATE_USER_PROMPT.format(n=len(batch), rows_json=json.dumps(rows_for_prompt, indent=2))

    resolved: dict[tuple[int, int], dict] = {}
    try:
        raw_response = _call_ollama_batch(prompt, ollama_host, num_predict=min(len(batch) * 35 + 60, 450))
        parsed = _parse_batch_response(raw_response)
        for item in parsed:
            idx = item.get("id")
            if idx is None or not (0 <= idx < len(batch)):
                continue
            c = batch[idx]
            resolved[(c["page_idx"], c["row_index"])] = item
        logger.info(f"Compliance candidate batch {batch_num}: {len(resolved)} of {len(batch)} readings reviewed")
    except Exception as e:
        logger.warning(
            f"Compliance candidate batch {batch_num} failed ({e}) — keeping rule-based verdicts for this batch"
        )
    return resolved


def _review_candidates_with_llm(
    candidates: list[dict],
    ollama_host: str,
    batch_size: int = 20,
) -> dict[tuple[int, int], dict]:
    """
    Send borderline readings to the LLM in batches, keyed by a synthetic "id".
    Returns {(page_idx, row_index): verdict_dict}.
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


# ── LangGraph Node: Compliance Auditor ─────────────────────────────

def compliance_auditor_node(state: ReviewState) -> dict:
    """
    LangGraph node: Evaluate compliance for every reading.
    Every reading is checked deterministically first; only readings sitting
    near a limit boundary are batched into LLM calls for a final verdict.
    """
    page_results = state.get("page_ocr_results", [])
    material_assignments = state.get("material_assignments", [])

    assignments_by_page: dict[int, list[MaterialAssignment]] = {}
    for a in material_assignments:
        assignments_by_page.setdefault(a.page_idx, []).append(a)

    all_findings: list[ComplianceFinding] = []
    all_candidates: list[dict] = []
    reasoning_traces = state.get("llm_reasoning_traces", []) or []

    pipeline_progress.set_percentage(0)

    with StageTracker("audit_compliance", f"pages={len(page_results)}") as tracker:
        pages_with_data = [p for p in page_results if p.table_rows]
        tracker.details += f" | pages_to_audit={len(pages_with_data)}"

        findings_by_key: dict[tuple[int, int], ComplianceFinding] = {}
        total_pages_count = max(1, len(pages_with_data))
        for idx, p in enumerate(pages_with_data):
            mat_lookup = {a.row_index: a.material for a in assignments_by_page.get(p.page_idx, [])}
            findings, candidates = _evaluate_page_deterministic(p, mat_lookup)
            for f in findings:
                findings_by_key[(f.page_idx, f.row_index)] = f
            all_candidates.extend(candidates)
            pct = int(((idx + 1) / total_pages_count) * (50 if all_candidates else 95))
            pipeline_progress.set_percentage(pct)

        tracker.details += f" | rule_based_findings={len(findings_by_key)} | candidates={len(all_candidates)}"

        llm_candidates = all_candidates
        skipped = 0
        if len(all_candidates) > MAX_LLM_CANDIDATES_COMPLIANCE:
            skipped = len(all_candidates) - MAX_LLM_CANDIDATES_COMPLIANCE
            # Closest-to-boundary readings are the truest edge cases — review those first.
            llm_candidates = sorted(all_candidates, key=lambda c: c["margin"])[:MAX_LLM_CANDIDATES_COMPLIANCE]
            pipeline_logger.info(
                f"Compliance Auditor: {len(all_candidates)} candidates exceed the "
                f"{MAX_LLM_CANDIDATES_COMPLIANCE}-candidate LLM budget — reviewing the "
                f"{MAX_LLM_CANDIDATES_COMPLIANCE} closest to their limit, {skipped} keep their rule-based verdict"
            )
            tracker.details += f" | llm_budget_skipped={skipped}"

        if llm_candidates:
            ollama_host = get_ollama_host()
            tracker.details += f" | ollama={ollama_host}"
            resolved = _review_candidates_with_llm(llm_candidates, ollama_host)
            candidates_by_key = {(c["page_idx"], c["row_index"]): c for c in llm_candidates}

            for key, verdict in resolved.items():
                page_idx, row_index = key
                candidate = candidates_by_key.get(key)
                if candidate is None:
                    continue
                llm_verdict = str(verdict.get("verdict", "")).upper()
                if llm_verdict == "PASS":
                    findings_by_key.pop(key, None)
                    continue
                page_obj = next((p for p in page_results if p.page_idx == page_idx), None)
                p_acceptance = getattr(page_obj, "acceptance_range", None) if page_obj else None
                min_lim, max_lim = _limits_for(candidate["material"], candidate["is_weld"], candidate.get("component", ""), p_acceptance)
                failing_raw = verdict.get("failing_readings")
                if failing_raw and isinstance(failing_raw, list) and all(isinstance(v, int) for v in failing_raw):
                    failing = [v for v in failing_raw if v > max_lim or v < min_lim]
                else:
                    failing = [v for v in candidate["readings"] if v > max_lim or v < min_lim]

                is_weld_zone = candidate.get("is_weld") or "HAZ" in candidate.get("component", "") or candidate.get("component") == "Weld"
                if not p_acceptance:
                    c_vals = candidate["readings"]
                    if candidate["material"] == "P91" and is_weld_zone:
                        if len(failing) == 1 and failing[0] <= 315 and failing[0] > max_lim:
                            others = [v for v in c_vals if v != failing[0]]
                            if all(195 <= v <= 280 for v in others):
                                failing = []
                    elif candidate["material"] == "P91":
                        if len(failing) == 1 and failing[0] <= 250 and failing[0] > max_lim:
                            others = [v for v in c_vals if v != failing[0]]
                            if all(195 <= v <= 248 for v in others):
                                failing = []

                if not failing:
                    # Mathematically verified that no readings violate spec limits
                    findings_by_key.pop(key, None)
                    continue

                if all(v > max_lim for v in failing):
                    limit_violated = f"above maximum {max_lim} HBW"
                    severity = "HOLD"
                    req_ref = f"DEETH-QCP-EL-003 / 700-006 Table 8 ({candidate['material']} max {max_lim} HBW)"
                elif all(v < min_lim for v in failing):
                    limit_violated = f"below minimum {min_lim} HBW"
                    severity = "MAJOR"
                    req_ref = f"700-006 Table 8 Note 2 ({candidate['material']} min {min_lim} HBW)"
                else:
                    limit_violated = f"outside allowable range {min_lim}-{max_lim} HBW"
                    severity = "HOLD"
                    req_ref = f"DEETH-QCP-EL-003 / 700-006 Table 8 ({candidate['material']})"

                findings_by_key[key] = ComplianceFinding(
                    page_idx=page_idx,
                    row_index=row_index,
                    component=candidate["component"],
                    material=candidate["material"],
                    failing_readings=failing,
                    limit_violated=limit_violated,
                    severity=severity,
                    requirement_ref=req_ref,
                    reasoning=f"LLM-confirmed: {verdict.get('reasoning', 'borderline reading reviewed')}",
                )
            tracker.details += f" | llm_reviewed={len(resolved)}"

        pipeline_progress.set_percentage(100)

        all_findings = list(findings_by_key.values())

        for f in all_findings:
            reasoning_traces.append({
                "stage": "compliance_auditor",
                "page": f.page_idx + 1,
                "row": f.row_index,
                "component": f.component,
                "material": f.material,
                "failing_readings": f.failing_readings,
                "severity": f.severity,
                "reasoning": f.reasoning,
            })

        tracker.details += f" | total_findings={len(all_findings)}"

    pipeline_logger.info(
        f"Compliance Auditor: {len(all_findings)} non-conformances "
        f"across {len(pages_with_data)} pages ({len(llm_candidates)} sent to LLM, "
        f"{skipped} kept rule-based due to LLM budget)"
    )

    return {
        "compliance_findings": all_findings,
        "llm_reasoning_traces": reasoning_traces,
        "current_agent": "compliance_auditor",
    }
