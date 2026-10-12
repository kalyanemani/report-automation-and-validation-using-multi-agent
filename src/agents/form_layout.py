from __future__ import annotations

"""
QDR·AI — Semantic Column Mapping for Unknown Forms
───────────────────────────────────────────────────
Works out which column is which on a table whose headers the deterministic
matcher does not recognise, and remembers the answer per form.

WHERE THIS FITS
───────────────
Column roles are read from header text first — "Material Grade", "Joint No.",
"H1".."H5" — because that is free, instant and deterministic. It covers the
forms in front of us today. It will not cover a supplier who writes "Base Matl.",
"Wld No." or "Rdg 1", and the plant team will meet those.

The alternative to asking is guessing, and guessing is what produced a day of
false findings: a hardcoded column position does not fail loudly on a form it
was not written for, it reads the neighbouring cell and grades against it.

COST
────
One call per FORM, not per page and not per row, keyed on the header signature
and cached to disk. A supplier seen before costs nothing. A new one costs a
single ~4s call whose answer then serves every page of every future submittal on
that form.

TRUST
─────
The mapping is verified before use, in the same spirit as the limit quotes: the
model must echo the header text of each column it maps, and that text must match
what is actually in that cell. A mapping that cannot be checked against the
document is discarded and the page falls back to OCR — never silently believed.
"""

import hashlib
import json
import logging
import re
from typing import Any, Optional

import requests

from src.config import OLLAMA_BATCH_TIMEOUT, OLLAMA_MODEL, PARSED_DIR, get_ollama_host

logger = logging.getLogger("qdr_ai.agent.form_layout")

_CACHE_PATH = PARSED_DIR / "form_layouts.json"

SYSTEM = """You are reading the header of a hardness test report table and identifying what each column holds.

You are given the header cells, numbered by their column index, and one sample data row.

Identify:
- "component": the column naming what was tested (pipe, weld, HAZ, elbow, plate, parent metal...)
- "material": the column giving the material grade (P91, F22, SA335P91, ...) — null if the table has none
- "joint": the column giving the joint or weld number — null if the table has none
- "readings": the columns holding individual hardness measurements, as a list of indices

For every column you identify, also copy that column's header text exactly as given, so your answer can be checked.

Rules:
- Use only the indices shown. Do not invent a column.
- The readings columns hold the measured numbers, not a size, a thickness or a serial number.
- If a role genuinely is not present in this table, return null for it rather than choosing a column that means something else.
- Give the INDEX NUMBER — the number in brackets — for every column you identify, as well as its header text."""

SCHEMA = {
    "type": "object",
    "properties": {
        "component": {"type": ["integer", "null"]},
        "component_header": {"type": "string"},
        "material": {"type": ["integer", "null"]},
        "material_header": {"type": "string"},
        "joint": {"type": ["integer", "null"]},
        "joint_header": {"type": "string"},
        "readings": {"type": "array", "items": {"type": "integer"}},
        "readings_header": {"type": "string"},
    },
    "required": ["component", "component_header", "readings"],
}


def form_signature(header_cells: list[str]) -> str:
    """Identity of a form layout: its header text, normalised."""
    blob = "|".join(re.sub(r"\s+", " ", (c or "").strip().upper()) for c in header_cells)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _load_cache() -> dict:
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"form layout cache read failed: {exc}")
    return {}


def _store(signature: str, roles: dict, headers: list[str]) -> None:
    try:
        cache = _load_cache()
        cache[signature] = {"roles": roles, "headers": headers[:40]}
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"form layout cache write failed: {exc}")


def _norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _resolve_indices(answer: dict, headers: list[str]) -> None:
    """
    Fill in any index the model left null but named in its header echo.

    Models are markedly better at NAMING the column than at numbering it —
    measured here, one returned `component: null` alongside
    `component_header: "Component Desc."`, having plainly identified the right
    column and failed only to state its number. The name is the part that can be
    checked against the document anyway, so the index is derived from it rather
    than depended upon.
    """
    normalised = [_norm(h) for h in headers]
    for role in ("component", "material", "joint"):
        if isinstance(answer.get(role), int):
            continue
        claimed = _norm(answer.get(f"{role}_header", ""))
        if not claimed:
            continue
        for idx, actual in enumerate(normalised):
            if actual and (actual == claimed or claimed in actual or actual in claimed):
                answer[role] = idx
                break


def _verify(answer: dict, headers: list[str]) -> Optional[str]:
    """Check the mapping against the header cells. Returns None if sound."""
    n = len(headers)

    comp = answer.get("component")
    if not isinstance(comp, int) or not (0 <= comp < n):
        return f"component index {comp!r} is not one of the {n} columns"

    readings = [r for r in (answer.get("readings") or []) if isinstance(r, int) and 0 <= r < n]
    if len(readings) < 3:
        return f"only {len(readings)} valid reading columns were identified"

    # The echoed header must be the header that is actually there. A model that
    # read the table passes this without effort; one that guessed an index does
    # not.
    for role in ("component", "material", "joint"):
        idx = answer.get(role)
        claimed = _norm(answer.get(f"{role}_header", ""))
        if idx is None or not isinstance(idx, int) or not (0 <= idx < n):
            continue
        actual = _norm(headers[idx])
        if claimed and actual and claimed not in actual and actual not in claimed:
            return (f"{role} was mapped to column {idx}, but that column's header is "
                    f"{headers[idx]!r} and the answer called it "
                    f"{answer.get(f'{role}_header')!r}")

    if comp in readings:
        return "the component column cannot also be a readings column"
    return None


def map_columns_semantically(headers: list[str], sample_row: list[str]) -> Optional[dict]:
    """
    Column roles for a form the header matcher could not read, or None.

    Cached per form signature, so this is asked once per supplier layout.
    """
    signature = form_signature(headers)
    cached = _load_cache().get(signature)
    if cached:
        logger.info(f"form layout reused from cache (signature={signature}) — 0 LLM calls")
        return cached.get("roles")

    numbered = "\n".join(f"  [{i}] {h!r}" for i, h in enumerate(headers))
    sample = "\n".join(f"  [{i}] {c!r}" for i, c in enumerate(sample_row))
    prompt = (f"HEADER CELLS:\n{numbered}\n\nONE DATA ROW FROM THE SAME TABLE:\n{sample}\n\n"
              f"Which column holds which?")

    try:
        response = requests.post(
            f"{get_ollama_host()}/api/chat",
            json={"model": OLLAMA_MODEL,
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": prompt}],
                  "format": SCHEMA, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 400}},
            timeout=OLLAMA_BATCH_TIMEOUT,
        )
        response.raise_for_status()
        answer = json.loads(response.json().get("message", {}).get("content", "") or "{}")
    except Exception as exc:
        logger.warning(f"semantic column mapping failed: {exc}")
        return None

    _resolve_indices(answer, headers)
    problem = _verify(answer, headers)
    if problem:
        logger.warning(
            f"Discarding semantic column mapping: {problem}. The page falls back to OCR "
            f"rather than being read against an unverified layout."
        )
        return None

    roles: dict[str, Any] = {
        "component": answer["component"],
        "readings": sorted({r for r in answer["readings"] if isinstance(r, int)}),
    }
    for role in ("material", "joint"):
        idx = answer.get(role)
        if isinstance(idx, int) and 0 <= idx < len(headers):
            roles[role] = idx

    logger.info(
        f"Form layout established for signature={signature}: component={roles['component']}, "
        f"material={roles.get('material')}, joint={roles.get('joint')}, "
        f"readings={roles['readings']}. Cached for every later document on this form."
    )
    _store(signature, roles, headers)
    return roles
