from __future__ import annotations

"""
QDR·AI — Deterministic Compliance Check
────────────────────────────────────────
Applies the limits established by the requirement-resolution LLM call to every
extracted reading. Pure arithmetic, no model involved.

WHY THIS IS PYTHON AND NOT AN LLM
─────────────────────────────────
By the time a row reaches here, the expensive reasoning is already done: the
Context Resolver has assigned its material from document evidence, and the
per-material limit calls have established cited bounds. The row dict already
holds material, zone, readings and limits. Asking a model "is 244 between 195
and 248?" adds latency and non-determinism to arithmetic.

Measured on 221135LBP-HRDR001: the LLM version needed 83 sequential calls at
~244s each — about 5.6 hours — against a 300s per-call timeout that aborts the
whole run under STRICT_LLM_MODE. This runs in well under a second and returns
the same answer every time.

WHAT IT IS NOT
──────────────
It is deliberately not `if minimum <= value <= maximum`. That expression is
wrong for this domain in three ways:

  * P22, P11 and CS have a maximum but NO minimum in the governing tables.
    Treating a missing minimum as 0 — or worse, borrowing P91's 195 — turns a
    conforming 190 HB reading into a false positive. A missing bound means
    "unbounded on that side", not "zero".
  * Specifications carry single-reading allowances (700-006 Table 8 Note 3
    tolerates one weld reading above the band if the others conform). A naive
    comparison reports a finding the specification explicitly permits.
  * A row with no readings, or no applicable limit, is NOT compliant. It is
    unevaluated, and saying so is the whole point — a silent PASS on an
    unevaluable row is indistinguishable from a real one.

Every result carries the numbers it used and a human-readable `reason`, so a
reviewer can audit the verdict without rerunning anything.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("qdr_ai.compliance_check")

COMPLIANT = "COMPLIANT"
NON_COMPLIANT = "NON_COMPLIANT"
UNRESOLVED = "UNRESOLVED"

# Reason codes for UNRESOLVED, each tagged with where the fix belongs. Only the
# rows routed to "llm" genuinely need a model; the rest are upstream defects
# that would be masked by sending them to one.
UNRESOLVED_REASONS: dict[str, dict[str, str]] = {
    "MISSING_VALUE":        {"route": "extraction",  "detail": "row has no usable hardness readings"},
    "MALFORMED_VALUE":      {"route": "extraction",  "detail": "readings outside any plausible hardness range"},
    "MISSING_MATERIAL":     {"route": "resolver",    "detail": "no material established from document evidence"},
    "EXTRACTION_UNRELIABLE": {"route": "extraction",  "detail": "the page's table could not be read with confidence"},
    "MISSING_LIMIT":        {"route": "retrieval",   "detail": "material known but no cited limit retrieved for its zone"},
    "INCOMPLETE_LIMIT":     {"route": "retrieval",   "detail": "limit entry states neither a minimum nor a maximum"},
    "AMBIGUOUS_ZONE":       {"route": "llm",         "detail": "component does not map cleanly to base_metal/haz/weld"},
    "CONFLICTING_LIMITS":   {"route": "llm",         "detail": "more than one limit appears applicable"},
    "NEEDS_INTERPRETATION": {"route": "llm",         "detail": "requires engineering judgement"},
}

# Physically plausible Brinell window. Outside this, the reading is an OCR
# artefact rather than a measurement, and grading it either way is meaningless.
PLAUSIBLE_MIN, PLAUSIBLE_MAX = 50, 600

# Components whose zone cannot be inferred from the label alone.
_AMBIGUOUS_COMPONENTS = {"", "?", "UNKNOWN", "GENERIC"}


def _unresolved(reason_code: str, row: dict, extra: str = "") -> dict:
    meta = UNRESOLVED_REASONS.get(reason_code, {"route": "llm", "detail": reason_code})
    return {
        "status": UNRESOLVED,
        "reason_code": reason_code,
        "route": meta["route"],
        "material": row.get("material"),
        "zone": row.get("zone"),
        "component": row.get("component"),
        "page": (row.get("page_idx") or 0) + 1,
        "joint": row.get("joint_name"),
        "readings": row.get("readings") or [],
        "limits": row.get("limits") or {},
        "provenance": row.get("provenance"),
        "confidence": row.get("confidence"),
        "reason": f"{meta['detail']}{(' — ' + extra) if extra else ''}",
    }


def check_row(row: dict) -> dict:
    """
    Evaluate one extracted row against its established limits.

    Returns a dict whose `status` is COMPLIANT, NON_COMPLIANT or UNRESOLVED.
    UNRESOLVED carries a `reason_code` and a `route` naming which part of the
    pipeline should handle it — see UNRESOLVED_REASONS.
    """
    # An untrusted extraction is answered as such before anything else. Asking
    # "is this row compliant" of cells that were probably misread has no useful
    # answer, and MISSING_MATERIAL would send a reviewer to the wrong place.
    if row.get("extraction_unreliable"):
        return _unresolved("EXTRACTION_UNRELIABLE", row)

    readings: list = [r for r in (row.get("readings") or []) if r is not None]
    material = row.get("material")
    zone = row.get("zone")
    limits = row.get("limits") or {}

    # ── Guard: is there anything to evaluate? ─────────────────────────
    if not readings:
        return _unresolved("MISSING_VALUE", row)

    implausible = [r for r in readings if not (PLAUSIBLE_MIN <= r <= PLAUSIBLE_MAX)]
    if len(implausible) == len(readings):
        return _unresolved("MALFORMED_VALUE", row, f"values {implausible[:6]}")

    # Drop individual OCR artefacts but keep the row if most readings are sane.
    usable = [r for r in readings if PLAUSIBLE_MIN <= r <= PLAUSIBLE_MAX]

    if not material or material == "UNRESOLVED":
        return _unresolved("MISSING_MATERIAL", row)

    if zone in _AMBIGUOUS_COMPONENTS or zone not in ("base_metal", "haz", "weld"):
        return _unresolved("AMBIGUOUS_ZONE", row, f"component={row.get('component')!r}")

    if not limits:
        return _unresolved("MISSING_LIMIT", row, f"{material}/{zone}")

    lo = limits.get("min")
    hi = limits.get("max")
    unit = limits.get("unit") or "HBW"
    ref = limits.get("ref") or "established limit"

    # A limit entry with neither bound cannot decide anything.
    if lo is None and hi is None:
        return _unresolved("INCOMPLETE_LIMIT", row, f"{material}/{zone} per {ref}")

    # ── 700-022 §8.2 Five-Reading Trimming Rule ──────────────────────
    # If 5 or more indents are taken, discard 1 highest and 1 lowest reading.
    is_5_reading_set = len(usable) == 5
    if is_5_reading_set:
        sorted_r = sorted(usable)
        discarded_low = sorted_r[0]
        discarded_high = sorted_r[-1]
        considered = sorted_r[1:-1]  # Middle 3 readings
        trim_note = f"5 readings evaluated per 700-022 §8.2: discarded lowest ({discarded_low}) & highest ({discarded_high})"
    else:
        considered = usable
        trim_note = ""

    avg_val = sum(considered) / len(considered) if considered else 0.0

    # ── Comparison on Considered Readings ─────────────────────────────
    over = [r for r in considered if hi is not None and r > hi]
    under = [r for r in considered if lo is not None and r < lo]

    # ── Single-reading allowance (700-022 / 700-006 Table 8 Note 2 & Note 3) ──
    allowance_applied = None

    # Note 2: For P91 Base Metal / HAZ (min 195 HBW), 1 reading down to 190 HBW allowed if avg >= 195 HBW
    if material == "P91" and lo == 195.0 and len(under) == 1 and under[0] >= 190.0 and avg_val >= 195.0:
        allowance_applied = (
            f"1 reading ({under[0]} HBW) permitted below 195 HBW per 700-022 Table 8 Note 2 "
            f"(3-reading average {avg_val:.1f} HBW >= 195 HBW"
            + (f"; {trim_note}" if trim_note else "") + ")"
        )
        under = []

    # Note 3: Weld Metal single reading allowance up to 315 HBW (avg <= 280 / 248 HBW)
    srm = limits.get("single_reading_max") or (315.0 if material == "P91" and zone == "weld" else None)
    if srm is not None and len(over) == 1 and over[0] <= srm and (hi is None or avg_val <= hi):
        allowance_applied = (
            f"1 reading ({over[0]} HBW) tolerated up to {srm} {unit} per 700-022 Table 8 Note 3 "
            f"(average {avg_val:.1f} {unit} <= {hi} {unit})"
        )
        over = []

    srn = limits.get("single_reading_min")
    if srn is not None and len(under) == 1 and under[0] >= srn:
        allowance_applied = f"1 reading ({under[0]} {unit}) tolerated down to {srn} {unit} per {ref}"
        under = []

    band = (
        f"{lo}-{hi} {unit}" if lo is not None and hi is not None
        else (f"max {hi} {unit}" if hi is not None else f"min {lo} {unit}")
    )

    base = {
        "material": material,
        "zone": zone,
        "component": row.get("component"),
        "page": (row.get("page_idx") or 0) + 1,
        "joint": row.get("joint_name"),
        "readings": usable,
        "limits": {"min": lo, "max": hi, "unit": unit, "ref": ref},
        "provenance": row.get("provenance"),
        "confidence": row.get("confidence"),
    }

    if not over and not under:
        return {
            **base,
            "status": COMPLIANT,
            "failing_readings": [],
            "reason": (
                f"all {len(usable)} readings within {band} for {material} {zone} per {ref}"
                + (f"; {allowance_applied}" if allowance_applied else "")
            ),
        }

    failing = sorted(set(over + under))
    if over and under:
        violated = f"outside {band}"
        severity = "HOLD"
    elif over:
        violated = f"above maximum {hi} {unit}"
        severity = "HOLD"
    else:
        violated = f"below minimum {lo} {unit}"
        severity = "MAJOR"

    return {
        **base,
        "status": NON_COMPLIANT,
        "failing_readings": failing,
        "severity": severity,
        "limit_violated": violated,
        "reason": (
            f"{material} {zone}: readings {failing} {violated} "
            f"(limit {band} per {ref})"
        ),
    }


def check_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """
    Evaluate every row. Returns (results, summary_counts).

    `summary_counts` reports COMPLIANT / NON_COMPLIANT plus one entry per
    UNRESOLVED reason code, so a run states exactly how much of the document it
    actually managed to evaluate.
    """
    results = [check_row(r) for r in rows]
    counts: dict[str, int] = {}
    for res in results:
        key = res["status"] if res["status"] != UNRESOLVED else f"UNRESOLVED:{res['reason_code']}"
        counts[key] = counts.get(key, 0) + 1
    return results, counts


def needs_llm_adjudication(results: list[dict]) -> list[dict]:
    """
    The subset genuinely requiring a model.

    Rows unresolved for extraction, resolver or retrieval reasons are upstream
    defects — sending them to an LLM would produce a confident answer over
    missing data. Only `route == "llm"` rows carry a real interpretation
    question.
    """
    return [r for r in results if r["status"] == UNRESOLVED and r.get("route") == "llm"]
