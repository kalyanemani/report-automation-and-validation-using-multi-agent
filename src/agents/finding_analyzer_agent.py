from __future__ import annotations

"""
QDR·AI — Finding Analyzer & Limit Extractor (LLM call #1 of 2)
──────────────────────────────────────────────────────────────
The pipeline's first LLM call. Given specification clauses retrieved by RAG and
a compact statistical summary of the submittal's hardness readings, it:

  1. establishes the governing hardness limits per material and zone, each with
     a citation back to the clause that states it, and
  2. groups the apparent non-conformances by metallurgical root cause.

Pipeline flow:
  ocr_extract → build_rag_context (metadata + checklist → retrieval)
  → THIS AGENT (limits + grouping)
  → Deterministic Rule Engine (applies the limits to every reading)

It does NOT compare individual readings. A submittal carries ~9,000 readings
against an 8K context (OLLAMA_ANALYZER_NUM_CTX), so that is both impossible here
and arithmetic better done reproducibly in Python.
"""

import json
import logging
import re
import time
from typing import Any, Optional

import requests

from src.agents.state import PageOCRResult, ReviewState
from src.agents.rule_engine import SKIP_CALIBRATION_VALUES
from src.agents.materials import canonicalise_limits, p_number_legend
from src.config import (
    OLLAMA_ANALYZER_NUM_CTX,
    OLLAMA_ANALYZER_TIMEOUT,
    OLLAMA_MODEL,
    STRICT_LLM_MODE,
    LLMAgentError,
    get_ollama_host,
)
from src.utils.llm_tracer import record_llm_trace
from src.utils.logger import pipeline_logger
from src.utils.metrics import StageTracker
from src.utils import progress as pipeline_progress

logger = logging.getLogger("qdr_ai.agent.finding_analyzer")

def _parse_json_robust(text: str) -> Optional[dict]:
    """Robustly parse JSON output from LLM responses."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(text[i : j + 1].strip())
        except Exception:
            try:
                return json.loads(text[i : j + 1].rstrip(", \n") + "}")
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════
# ANALYZER — LLM CALL #1 OF 2
# ══════════════════════════════════════════════════════════════════════
#
# The pipeline's first LLM call (the second writes the executive summary in
# report_synthesizer_agent.py). It does two jobs that used to be split across
# the old LLM Call #1 (Requirement Parser) and LLM Call #2 (Finding Analyzer):
#
#   1. Establish the governing hardness limits from the RAG-retrieved
#      specification clauses, citing where each number came from.
#   2. Assess the document's reading statistics against those limits and group
#      the resulting non-conformances by root cause.
#
# It does NOT compare individual readings. There are ~9,000 readings in a
# typical submittal against an 8K context, so per-reading comparison is
# physically impossible here — and it is arithmetic, which Python does
# reproducibly. `audit_rules` applies the limits this call establishes to every
# reading afterwards.
#
# Note the prompt deliberately contains NO numeric limits. Every threshold must
# come from the supplied clauses, so its output is evidence about the document
# rather than a restatement of the prompt.
SINGLE_ANALYZER_SYSTEM_PROMPT = """You are a Principal Metallurgical Quality Assurance Specialist reviewing a pressure piping hardness test submittal.

You are given:
  A. SPECIFICATION CLAUSES — retrieved from the governing project documents.
  B. REVIEW GUIDE CHECKLIST — the criteria the reviewer requires to be met.
  C. SUBMITTAL METADATA — materials, drawings, spools and any acceptance range the report states itself.
  D. READING STATISTICS — per spool and material: how many readings were taken and their observed range.

Do two things.

TASK 1 — Establish the governing limits.
For each material present in the submittal, determine the hardness acceptance limits for base_metal, haz and weld from the SPECIFICATION CLAUSES and any acceptance range stated in the submittal.
  - Specification tables often label materials by ASME P-Number rather than by grade. Section E below gives the identities. A row labelled with a material's P-Number states that material's limits — use it, and key your output by the GRADE name (P91/P22/P11/CS), not the P-Number.
  - Every number MUST be traceable to the supplied text. Do not use limits from your own knowledge of ASME B31.1 or any other standard.
  - Cite the clause, table or section for each limit in `ref`. If you cannot cite it, do not report it.
  - If no minimum is stated, use null. If no maximum is stated, use null. Never invent a bound.
  - If a clause states a single-reading exception (one outlying reading tolerated beyond the normal band), record it as `single_reading_max` / `single_reading_min` and cite the note.
  - Where a submittal-stated range conflicts with a specification clause, use the more restrictive and name both in `ref`.
  - Omit any material the supplied text does not address.

TASK 2 — Assess the readings.
Using ONLY the limits you established in TASK 1, identify which spool/material combinations in the READING STATISTICS appear non-conformant, and group them by metallurgical root cause:
  - "INSUFFICIENT_PWHT_SOAK" — observed hardness below the established minimum.
  - "UNTEMPERED_MARTENSITE_EMBRITTLEMENT" — observed hardness above the established maximum.
  - "MATERIAL_GRADE_DISCREPANCY" — observed hardness incompatible with the drawing's material callout.
Assign severity "HOLD" or "MAJOR" and a mandatory corrective action for each group.

Return ONLY a valid JSON object with exactly these two top-level keys:
{
  "limits": [
    {"material": "<GRADE>", "zone": "base_metal", "min": <number|null>, "max": <number|null>, "unit": "<as stated>", "ref": "<citation>"},
    {"material": "<GRADE>", "zone": "haz",        "min": <number|null>, "max": <number|null>, "unit": "<as stated>", "ref": "<citation>"},
    {"material": "<GRADE>", "zone": "weld",       "min": <number|null>, "max": <number|null>, "unit": "<as stated>", "ref": "<citation>", "single_reading_max": <number|null>}
  ],
  "groups": [
    {
      "spool_no": "<spool>",
      "drawing_no": "<drawing>",
      "material": "<material>",
      "affected_joints": ["<joint>"],
      "pages_affected": [<page numbers>],
      "category": "<one of the three categories above>",
      "severity": "HOLD" | "MAJOR",
      "summary": "<what was observed and which established limit it breaches>",
      "recommended_action": "<mandatory corrective action>"
    }
  ]
}

Emit one `limits` entry per material+zone you can establish and cite. `material` must be one of the grades listed in section E; `zone` must be exactly "base_metal", "haz" or "weld". Omit entries you cannot cite rather than guessing.

If the SPECIFICATION CLAUSES establish no hardness limits at all, return {"limits": [], "groups": []}."""


# JSON Schema handed to Ollama's `format` parameter.
#
# Bare `format: "json"` only guarantees *some* JSON comes back. On 2026-08-20
# this call answered with {"governingLimits": {...}, "assessedData": [...]} —
# valid JSON, entirely the wrong shape, with no material dimension at all — so
# the parser found no `limits` key and the run aborted. A schema constrains
# decoding to this structure, which removes shape drift as a failure mode
# altogether. It cannot stop a wrong *number* being emitted; the `ref`
# requirement and the strict checks address that separately.
def build_analyzer_schema(materials: list[str]) -> dict:
    """
    Per-call response schema, with `material` and `zone` constrained to enums.

    An object keyed by material name is not enough. Tested against the live
    model, an open `additionalProperties` map produced
    `{"limits": {"P91_base_metal": {"haz": {...}}}}` — the material and zone
    dimensions collapsed into one key, and the base-metal limit filed under
    `haz`. A flat array whose `material` and `zone` are enums makes both
    mistakes unrepresentable: the decoder cannot emit a token outside the
    allowed sets.

    `material` is enumerated from what OCR actually found in this document, so
    the agent also cannot report limits for a material that isn't present.
    """
    mats = [m for m in (materials or []) if m] or ["P91", "P22", "P11", "CS"]
    return {
        "type": "object",
        "properties": {
            "limits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "material": {"type": "string", "enum": mats},
                        "zone": {"type": "string", "enum": ["base_metal", "haz", "weld"]},
                        "min": {"type": ["number", "null"]},
                        "max": {"type": ["number", "null"]},
                        "single_reading_max": {"type": ["number", "null"]},
                        "single_reading_min": {"type": ["number", "null"]},
                        "unit": {"type": "string"},
                        "ref": {"type": "string"},
                    },
                    "required": ["material", "zone", "min", "max", "unit", "ref"],
                },
            },
            "groups": _GROUPS_SCHEMA,
        },
        "required": ["limits", "groups"],
    }


def limits_array_to_schema(rows: Any) -> dict:
    """
    Fold the flat [{material, zone, min, max, ...}] array into the nested
    {material: {zone: {...}}} shape the rule engine consumes.
    """
    out: dict = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        mat = str(row.get("material") or "").strip()
        zone = str(row.get("zone") or "").strip()
        if not mat or zone not in ("base_metal", "haz", "weld"):
            continue
        entry = {k: row[k] for k in ("min", "max", "unit", "ref") if k in row}
        for k in ("single_reading_max", "single_reading_min"):
            if row.get(k) is not None:
                entry[k] = row[k]
        out.setdefault(mat.upper(), {})[zone] = entry
    return out


_GROUPS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "spool_no": {"type": "string"},
            "drawing_no": {"type": "string"},
            "material": {"type": "string"},
            "affected_joints": {"type": "array", "items": {"type": "string"}},
            "pages_affected": {"type": "array", "items": {"type": "integer"}},
            "category": {
                "type": "string",
                "enum": [
                    "INSUFFICIENT_PWHT_SOAK",
                    "UNTEMPERED_MARTENSITE_EMBRITTLEMENT",
                    "MATERIAL_GRADE_DISCREPANCY",
                ],
            },
            "severity": {"type": "string", "enum": ["HOLD", "MAJOR"]},
            "summary": {"type": "string"},
            "recommended_action": {"type": "string"},
        },
        "required": ["spool_no", "material", "category", "severity", "summary"],
    },
}


def _summarize_readings_for_llm(pages: list[PageOCRResult]) -> tuple[str, dict, int]:
    """
    Compact per-spool/material reading statistics.

    Deliberately limit-free: this runs BEFORE the LLM establishes any limits, so
    it cannot pre-judge conformance. It reports only what was measured, which is
    what lets a single call both set the limits and comment on the data.
    """
    buckets: dict[tuple, dict] = {}
    total = 0

    for p in pages:
        for row in p.table_rows:
            vals = [r.value for r in row.readings if r.value and r.value not in SKIP_CALIBRATION_VALUES]
            if not vals:
                continue
            total += len(vals)
            mat = (row.material_hint or getattr(p, "material_spec", "") or "UNSPECIFIED").upper()
            zone = "weld" if row.is_weld else ("haz" if "HAZ" in row.component.upper() else "base_metal")
            key = (p.spool_no or "UNKNOWN", mat, zone)
            b = buckets.setdefault(key, {
                "count": 0, "min": 10**9, "max": -10**9,
                "pages": set(), "joints": set(), "drawing": p.drawing_no or "",
            })
            b["count"] += len(vals)
            b["min"] = min(b["min"], min(vals))
            b["max"] = max(b["max"], max(vals))
            b["pages"].add(p.page_idx + 1)
            if row.joint_name:
                b["joints"].add(row.joint_name)

    lines: list[str] = []
    for (spool, mat, zone), b in sorted(buckets.items(), key=lambda kv: -kv[1]["max"]):
        pg = sorted(b["pages"])
        lines.append(
            f"- Spool {spool} | Dwg {b['drawing']} | {mat} {zone} | "
            f"n={b['count']} | observed {b['min']}-{b['max']} | "
            f"pages {pg[:5]}{'...' if len(pg) > 5 else ''} | "
            f"joints {sorted(b['joints'])[:3]}"
        )

    # Cap the summary so the prompt stays inside the context window. Sorted by
    # highest observed reading, so the most likely non-conformances survive.
    #
    # Was 40. A real submittal produced 111 spool/material/zone groups, and 40
    # of those lines plus the clauses drove the prompt to ~3,900 tokens of a
    # 4,096 context — leaving ~187 tokens for output when the task needs
    # 600-1,000. The model then emitted a truncated, collapsed schema with
    # invented numbers because that was all that fit. 20 lines is ample for
    # establishing limits; every reading is evaluated deterministically
    # afterwards regardless of what appears here.
    MAX_LINES = 20
    truncated = len(lines) > MAX_LINES
    body = "\n".join(lines[:MAX_LINES])
    if truncated:
        body += f"\n... ({len(lines) - MAX_LINES} further spool/material groups omitted; " \
                f"all are evaluated deterministically after limits are established)"
    return body, buckets, total


def analyze_and_establish_limits_llm_call(
    pages: list[PageOCRResult],
    retrieved_context: list[dict],
    requirements: list[str],
    detected_materials: list[str],
    project_name: str = "",
) -> tuple[dict, list[dict], dict]:
    """
    The pipeline's single LLM call.

    Returns (limits_schema, grouped_clusters, trace).
    """
    t0 = time.time()
    ollama_host = get_ollama_host()

    summary_body, buckets, total_readings = _summarize_readings_for_llm(pages)

    # Clause budget. Was 8 clauses x 700 chars (~1,400 tokens); combined with
    # the reading summary that overflowed the context window and starved the
    # response. 5 x 450 keeps the highest-ranked clauses — which is where the
    # limit tables are — while leaving room to actually answer.
    clause_lines = []
    for i, c in enumerate(retrieved_context[:5], 1):
        content = (c.get("content") or "").strip().replace("\n", " ")
        src = c.get("source", "unknown")
        sec = c.get("section", "")
        clause_lines.append(f"[{i}] ({src}{' — ' + sec if sec else ''}) {content[:450]}")

    drawings = sorted({p.drawing_no for p in pages if p.drawing_no})
    spools = sorted({p.spool_no for p in pages if p.spool_no})
    stated = sorted({tuple(p.acceptance_range) for p in pages if p.acceptance_range})

    user_prompt = f"""## A. SPECIFICATION CLAUSES (retrieved)
{chr(10).join(clause_lines) if clause_lines else "NONE RETRIEVED"}

## B. REVIEW GUIDE CHECKLIST
{chr(10).join(f"- {r[:200]}" for r in requirements[:5]) if requirements else "None supplied."}

## C. SUBMITTAL METADATA
Project: {project_name or "not stated"}
Pages: {len(pages)} | Total readings: {total_readings}
Materials observed by OCR: {', '.join(detected_materials) or "NONE"}
Drawings: {', '.join(drawings[:6])}{'...' if len(drawings) > 6 else ''}
Spools: {len(spools)}
Acceptance ranges stated in the submittal itself: {', '.join(f'{lo}-{hi} HBW' for lo, hi in stated) if stated else "none stated"}

## D. READING STATISTICS (per spool / material / zone)
{summary_body}

## E. MATERIAL IDENTITIES (grade ↔ ASME P-Number)
{p_number_legend(detected_materials) or "No material grades identified."}
These are identities only — they state which P-Number label denotes which grade. They contain no hardness values. All limits must still come from section A or C.

Establish the governing limits from section A, then assess section D against them. Return the JSON object."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SINGLE_ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # Schema-constrained decoding, not bare "json" — see
        # build_analyzer_schema() for why the enums matter.
        "format": build_analyzer_schema(detected_materials),
        "stream": False,
        "options": {
            "temperature": 0.0,
            # Ollama defaults num_ctx to 4096. Nothing here overrode it, so a
            # ~3,900-token prompt left ~187 tokens of output room and this call
            # could not physically emit the schema it was asked for. 8192 gives
            # the response actual space; a 3B Q4 model at 8K context is cheap
            # on a 128 GB host (costs some prefill time, not correctness).
            "num_ctx": OLLAMA_ANALYZER_NUM_CTX,
            "num_predict": 1024,
            "num_thread": 8,
        },
    }

    # Log the prompt budget so a future overflow is visible in the log rather
    # than showing up as an inexplicably malformed response.
    _est_in = (len(SINGLE_ANALYZER_SYSTEM_PROMPT) + len(user_prompt)) // 4
    pipeline_logger.info(
        f"Analyzer prompt budget: ~{_est_in} input tokens of {OLLAMA_ANALYZER_NUM_CTX} ctx "
        f"({len(clause_lines)} clauses, {len(buckets)} reading groups) -> "
        f"~{OLLAMA_ANALYZER_NUM_CTX - _est_in} tokens available for output "
        f"(num_predict=1024)"
    )
    if _est_in + 1024 > OLLAMA_ANALYZER_NUM_CTX:
        pipeline_logger.warning(
            f"Analyzer prompt may overflow context: ~{_est_in} input + 1024 requested output "
            f"> {OLLAMA_ANALYZER_NUM_CTX} num_ctx. The response will be truncated — raise "
            f"OLLAMA_ANALYZER_NUM_CTX or reduce the clause/summary budgets."
        )

    raw_response = ""
    status = "SUCCESS"
    failure: Optional[str] = None
    limits: dict = {}
    groups: list[dict] = []

    try:
        resp = requests.post(f"{ollama_host}/api/chat", json=payload, timeout=OLLAMA_ANALYZER_TIMEOUT)
        resp.raise_for_status()
        raw_response = resp.json().get("message", {}).get("content", "").strip()
        parsed = _parse_json_robust(raw_response)

        if not isinstance(parsed, dict):
            status = "PARSE_ERROR"
            failure = "Response was not parseable as a JSON object."
        else:
            raw_limits = parsed.get("limits")
            # Schema returns a flat array of {material, zone, ...}; fold it into
            # the nested {material: {zone: {...}}} shape. A dict is still
            # accepted for robustness if a model answers in the older shape.
            if isinstance(raw_limits, list):
                raw_limits = limits_array_to_schema(raw_limits)
            if isinstance(raw_limits, dict):
                # Canonicalise keys through the P-Number mapping. An agent
                # reading a P-Number-keyed table may legitimately answer with
                # "P5A" or "P-No. 4"; left as-is those keys never match the rule
                # engine's material names and the readings would be skipped as
                # "no established limits" — a silent hole rather than a result.
                limits, renames = canonicalise_limits(raw_limits)
                if renames:
                    pipeline_logger.info(
                        f"Analyzer returned P-Number-keyed limits; canonicalised {renames}"
                    )
            raw_groups = parsed.get("groups")
            if isinstance(raw_groups, list):
                groups = [g for g in raw_groups if isinstance(g, dict)]

            if not limits:
                status = "NO_LIMITS_ESTABLISHED"
                failure = (
                    "Agent established no hardness limits from the retrieved specification "
                    "clauses. Without limits the rule engine has nothing to evaluate against."
                )
            else:
                pipeline_logger.info(
                    f"Single LLM Call: established limits for {sorted(limits.keys())} from "
                    f"{len(clause_lines)} retrieved clauses; grouped {len(groups)} non-conformance clusters."
                )
    except Exception as e:
        status = f"ERROR: {e}"
        raw_response = f"Error: {e}"
        failure = str(e)
        if not STRICT_LLM_MODE:
            pipeline_logger.warning(f"Single LLM Call error: {e}")

    if failure and not STRICT_LLM_MODE:
        # Non-strict: keep the pipeline runnable on the documented defaults, but
        # make it unmistakable in the trace that these were NOT agent-derived.
        from src.agents.requirement_parser_agent import DEFAULT_HARDNESS_RULES
        limits = json.loads(json.dumps(DEFAULT_HARDNESS_RULES))
        pipeline_logger.warning(
            "Single LLM Call failed — falling back to DEFAULT_HARDNESS_RULES. "
            "Limits in this report were NOT established from the retrieved specifications."
        )

    duration = time.time() - t0
    trace = record_llm_trace(
        call_id="call_1_analyzer",
        stage_name="Finding Analyzer & Limit Extractor (single LLM call)",
        model_name=OLLAMA_MODEL,
        system_prompt=SINGLE_ANALYZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        raw_response=raw_response,
        parsed_output={"limits": limits, "groups": groups},
        duration_s=duration,
        status=f"STRICT_FAILURE: {failure}" if (STRICT_LLM_MODE and failure) else status,
        metadata={
            "total_pages": len(pages),
            "total_readings": total_readings,
            "clauses_supplied": len(clause_lines),
            "checklist_items": len(requirements),
            "reading_groups": len(buckets),
            "materials_limited": sorted(limits.keys()),
            "groups_count": len(groups),
            "limits_source": "agent" if not failure else ("defaults" if not STRICT_LLM_MODE else "none"),
        },
    )

    if STRICT_LLM_MODE and failure:
        pipeline_logger.error(f"Single LLM Call failed under STRICT_LLM_MODE: {failure}")
        raise LLMAgentError("call_1_analyzer", failure)

    return limits, groups, trace


def finding_analyzer_node(state: ReviewState) -> dict:
    """
    LLM node #1 of 2.

    Establishes hardness limits from the RAG-retrieved clauses and groups
    non-conformances. Individual reading comparison happens next, in
    `audit_rules`, deterministically.
    """
    pipeline_progress.set_stage("analyze_findings")

    pages = state.get("page_ocr_results", [])
    retrieved_context = state.get("retrieved_context", []) or []
    requirements = state.get("requirements", []) or []
    detected_materials = state.get("detected_materials", []) or []
    proj_name = state.get("gen_params_input", {}).get("proj_name", "")

    with StageTracker(
        "analyze_findings_llm",
        f"pages={len(pages)}, clauses={len(retrieved_context)}, checklist={len(requirements)}",
    ) as st:
        limits, grouped, trace = analyze_and_establish_limits_llm_call(
            pages,
            retrieved_context=retrieved_context,
            requirements=requirements,
            detected_materials=detected_materials,
            project_name=proj_name,
        )
        st.details += f" | limits={sorted(limits.keys())}, groups={len(grouped)}"

    # COVERAGE CHECK — every material the document contains needs limits.
    #
    # The rule engine skips any material with no entry in the schema, logging
    # NEEDS_REVIEW and moving on. That is a silent hole: readings for that
    # material are never compared to anything, so a genuine violation is simply
    # absent from the report rather than flagged.
    #
    # This does NOT abort the run. Discarding a complete 80-page review because
    # one clause failed to retrieve is more brittle than useful — and the review
    # of the other materials is still valid work. Instead the gap is reported as
    # an explicit finding (see report_synthesizer_agent), so the report states
    # which materials were not evaluated rather than quietly omitting them.
    uncovered = [m for m in detected_materials if m.upper() not in {k.upper() for k in limits}]
    if uncovered:
        pipeline_logger.warning(
            f"Coverage gap: OCR observed {uncovered} but limits were established only for "
            f"{sorted(limits.keys())}. Readings for {uncovered} will NOT be evaluated; the "
            f"report will carry an explicit unevaluated-material finding for each."
        )

    existing_traces = list(state.get("llm_traces", []))
    existing_traces.append(trace)

    return {
        "hardness_rules_schema": limits,
        "grouped_findings": grouped,
        "uncovered_materials": uncovered,
        "llm_traces": existing_traces,
        "current_agent": "finding_analyzer",
    }
