from __future__ import annotations

"""Strict, LLM-only hardness review in bounded sequential batches."""

import hashlib
import json
import logging
import re
import time
from typing import Any, Optional

import requests

from src.agents.compliance_check import (
    NON_COMPLIANT,
    UNRESOLVED_REASONS,
    check_rows,
    needs_llm_adjudication,
)
from src.agents.context_resolver import UNRESOLVED, resolve_page_rows
from src.agents.materials import (
    MATERIAL_P_NUMBERS,
    canonical_grade,
    clause_project_conflicts,
    grades_named_in,
    project_prefix,
)
from src.agents.state import ComplianceFinding, ReviewState
from src.config import (
    LLM_AUDIT_BATCH_SIZE,
    PARSED_DIR,
    OLLAMA_ANALYZER_NUM_CTX,
    OLLAMA_BATCH_TIMEOUT,
    OLLAMA_MODEL,
    STRICT_LLM_MODE,
    get_ollama_host,
)
from src.utils.llm_tracer import record_llm_trace
from src.utils.logger import pipeline_logger
from src.utils.metrics import StageTracker
from src.utils import progress as pipeline_progress

logger = logging.getLogger("qdr_ai.agent.llm_batch_reviewer")

# ══════════════════════════════════════════════════════════════════════
# LIMIT EXTRACTION — QUOTE-FIRST, MACHINE-VERIFIABLE
# ══════════════════════════════════════════════════════════════════════
#
# Six earlier attempts with instruction-only prompts each failed differently:
# P22 got P91's 195-248 band; CS cited the P15E row; P91's stated 195 minimum
# was dropped; P91's weld ceiling came back as P22's 225; a "Weld" row was filed
# under `haz`; and one call returned no limits array at all.
#
# Telling a 3B model "do not hallucinate" does not work, and adding another
# prohibition has not worked five times running. What DOES work is making every
# claim checkable in Python:
#
#   The model must COPY the exact sentence or table row it is reading, name the
#   material that sentence is about, and only then give the numbers.
#
# Python then verifies mechanically:
#   * is `quote` genuinely present in the supplied clause text?   (no invention)
#   * does `material_in_quote` match the material we asked about?  (no misattribution)
#   * do `min` / `max` appear as numbers inside `quote`?           (no invented bounds)
#   * does `quote` mention the zone it was filed under?            (no zone confusion)
#
# A fabricated answer cannot satisfy those checks, so it is rejected rather than
# believed. Instructions are phrased positively — small models follow "do X"
# far more reliably than "do not do Y" — and a worked example is given for each
# of the three cases that previously went wrong.
LIMIT_SYSTEM = """You are reading specification clauses to find hardness acceptance limits for ONE material.

Your answer must be traceable. For every limit you report, you first COPY the exact text you read it from, then state which material that text is about, then give the numbers that appear in it.

METHOD — follow these steps in order:
1. Read the supplied clauses. Find the sentence or table row that states a hardness limit FOR THE REQUESTED MATERIAL.
2. Copy that sentence or table row into "quote", character for character, exactly as it appears. Copy enough of it to include the material name, the numbers, AND the words that say where the limit applies ("base metal", "parent metal", "HAZ", "weld"). When those words are in the heading above the row you are reading, start your quote at that heading.
3. In "material_in_quote", write the material name or P-Number label that appears in your quote (for example "P91", "Alloy 91", "P5A", "P4", "P1", "Carbon Steel").
4. In "zone", state which location the quote is about: "base_metal" for parent/base metal, "haz" for heat affected zone, "weld" for weld metal.
5. Read the numbers out of your quote:
   - A range such as "195-248 HB" gives min 195 and max 248.
   - "225 HB Max" or "shall not exceed 225 HB" gives min null and max 225.
   - "not less than 200 BHN" gives min 200 and max null.
6. Put the document and section you took it from in "ref".

RULES THAT KEEP THE ANSWER TRUE:
- Every number you report appears inside your own quote. If a number is not in your quote, it does not belong in your answer.
- Your quote is about the requested material. A quote about a different material or a different P-Number label belongs to that material, not this one — leave it out.
- A missing bound is null. Many materials have a maximum and no minimum; report that faithfully as min null.
- A stated bound is reported. If the text says "195-248", both 195 and 248 appear in your answer.
- Report one entry per zone at most. Report only the zones your quotes actually cover.
- When no supplied clause states a limit for the requested material, return {"limits": []}. That is a correct and useful answer.

WORKED EXAMPLES

Requested material: P91
Clause: "[doc — 8 Acceptance Criteria] permissible hardness values for Alloy 91 shall be: Base metal after PWHT (195-248 HB) refer Note-2."
Answer:
{"limits":[{"quote":"permissible hardness values for Alloy 91 shall be: Base metal after PWHT (195-248 HB)","material_in_quote":"Alloy 91","zone":"base_metal","min":195,"max":248,"unit":"HB","ref":"doc — 8 Acceptance Criteria"}]}

Requested material: P22
Clause: "[doc — table] | P15E | Base (After PWHT) | 195-248 HB | | P5A | Base and Weld | 225 HB Max |"
Answer:
{"limits":[{"quote":"| P5A | Base and Weld | 225 HB Max |","material_in_quote":"P5A","zone":"base_metal","min":null,"max":225,"unit":"HB","ref":"doc — table"}]}
(The P15E row states P91's limits, so it is not used here. P5A states a maximum only, so min is null.)

Requested material: CS
Clause: "[doc — 8 Acceptance Criteria] permissible hardness values for Alloy 91 shall be: Base metal after PWHT (195-248 HB)."
Answer:
{"limits":[]}
(No supplied clause states a limit for carbon steel.)"""

# Output budget for one limit call.
#
# The quote-first schema carries the evidence text inside the answer, so an
# entry is several times larger than the numbers-only version it replaced: three
# zones with ~120-character quotes is comfortably 400+ tokens. The previous
# budget of 240 cut CS off mid-string on 2026-08-20 ("...No more than one
# reading is allowed below" — end of response), and because Ollama's
# grammar-constrained decoding only guarantees valid JSON for a COMPLETED
# generation, a truncated one is unparseable and was read as "the model returned
# no limits array". The budget is sized for the schema, with a larger retry for
# clause sets that quote long table rows.
LIMIT_NUM_PREDICT = 700
LIMIT_NUM_PREDICT_RETRY = 1100

LIMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "limits": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    # Ordered so the model commits to the evidence BEFORE the
                    # numbers: autoregressive decoding means the quote is
                    # generated first and then constrains what follows.
                    "quote": {"type": "string"},
                    "material_in_quote": {"type": "string"},
                    "zone": {"type": "string", "enum": ["base_metal", "haz", "weld"]},
                    "min": {"type": ["number", "null"]},
                    "max": {"type": ["number", "null"]},
                    "single_reading_min": {"type": ["number", "null"]},
                    "single_reading_max": {"type": ["number", "null"]},
                    "unit": {"type": "string"},
                    "ref": {"type": "string"},
                },
                "required": ["quote", "material_in_quote", "zone", "min", "max", "unit", "ref"],
            },
        },
    },
    "required": ["limits"],
}

AUDIT_SYSTEM = """You are a metallurgical QA reviewer. Decide every submitted table row only from its supplied cited limits.
Return one verdict per id. Use FAIL only for a nonconformance; use PASS when compliant; use NEEDS_REVIEW if no supplied limit permits a decision. Never invent limits or citations."""
AUDIT_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "integer"}, "verdict": {"type": "string", "enum": ["PASS", "FAIL", "NEEDS_REVIEW"]},
    "failing_readings": {"type": "array", "items": {"type": "number"}},
    "severity": {"type": "string", "enum": ["HOLD", "MAJOR", "NEEDS_REVIEW"]},
    "limit_violated": {"type": "string"}, "requirement_ref": {"type": "string"},
    "reasoning": {"type": "string"},
}, "required": ["id", "verdict", "reasoning"]}}


def _chat(host: str, system: str, prompt: str, schema: dict, num_predict: int) -> str:
    payload = {"model": OLLAMA_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
               "format": schema, "stream": False,
               "options": {"temperature": 0.0, "num_ctx": OLLAMA_ANALYZER_NUM_CTX, "num_predict": num_predict}}
    response = requests.post(f"{host}/api/chat", json=payload, timeout=OLLAMA_BATCH_TIMEOUT)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()


def _json(value: str, expected: type) -> Any:
    try:
        result = json.loads(value)
        return result if isinstance(result, expected) else None
    except json.JSONDecodeError:
        return None


def build_multi_limit_schema(materials: list[str]) -> dict:
    """
    One schema covering every material, as a flat enum-constrained array.

    Replaces one call per material. The per-material loop cost ~330s on this
    document (4 calls at 63-102s each) mostly in fixed overhead — each call
    re-prefilled the same system prompt and largely the same clause text. A
    single call with `material` enumerated returns the whole table in one
    generation; measured at ~30s for a two-material prompt.

    `material` and `zone` are enums so the decoder cannot collapse the two
    dimensions into one key (observed: "P91_base_metal" under an open schema).
    """
    mats = [m for m in (materials or []) if m] or ["P91", "P22", "P11", "CS"]
    return {
        "type": "object",
        "properties": {
            "limits": {
                "type": "array",
                "maxItems": len(mats) * 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "material": {"type": "string", "enum": mats},
                        "zone": {"type": "string", "enum": ["base_metal", "haz", "weld"]},
                        "min": {"type": ["number", "null"]},
                        "max": {"type": ["number", "null"]},
                        "single_reading_min": {"type": ["number", "null"]},
                        "single_reading_max": {"type": ["number", "null"]},
                        "unit": {"type": "string"},
                        "ref": {"type": "string"},
                    },
                    "required": ["material", "zone", "min", "max", "unit", "ref"],
                },
            },
        },
        "required": ["limits"],
    }


def _limits_cache_key(guide_code: str, materials: list[str], clauses: list[str]) -> str:
    """
    Cache identity for an established limit set.

    Limits are a property of the SPECIFICATIONS, not of the submittal being
    reviewed — the same clauses yield the same limits every time. Re-deriving
    them per document is pure repeated cost, so key on the guide plus the exact
    clause text and reuse the answer.
    """
    h = hashlib.sha256()
    h.update(guide_code.encode("utf-8"))
    # The MODEL is part of the identity. A limit is what a given reader made of
    # a given clause, and readers differ: qwen2.5:3b needed the elided-quote
    # recovery to find P22's row at all, where 7b-instruct quotes it cleanly.
    # Without this, changing OLLAMA_MODEL silently reuses the previous model's
    # readings for every document reviewed against the same specifications, and
    # an upgrade appears to change nothing.
    h.update(OLLAMA_MODEL.encode("utf-8"))
    h.update("|".join(sorted(materials)).encode("utf-8"))
    for clause in sorted(clauses):
        h.update(clause.encode("utf-8"))
    return h.hexdigest()[:24]


def _load_cached_limits(key: str) -> Optional[dict]:
    try:
        path = PARSED_DIR / "limits_cache.json"
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8")).get(key)
        return entry.get("limits") if isinstance(entry, dict) else None
    except Exception as exc:
        logger.warning(f"limits cache read failed: {exc}")
        return None


def _store_cached_limits(key: str, limits: dict, materials: list[str]) -> None:
    try:
        path = PARSED_DIR / "limits_cache.json"
        cache = {}
        if path.exists():
            cache = json.loads(path.read_text(encoding="utf-8"))
        cache[key] = {"limits": limits, "materials": materials,
                      "stored_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"limits cache write failed: {exc}")


def _material_mentions(text: str, material: str) -> list[tuple[int, int]]:
    """Character spans where `text` names this material, matched whole-token."""
    upper = (text or "").upper()
    spans: list[tuple[int, int]] = []
    for alias in MATERIAL_P_NUMBERS.get(material, {}).get("aliases", ()):
        pattern = rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])"
        spans.extend(m.span() for m in re.finditer(pattern, upper))
    return sorted(set(spans))


def _material_window(content: str, material: str, budget: int = 1800,
                     pad_before: int = 500, pad_after: int = 600) -> str:
    """
    The part of a clause that actually talks about THIS material.

    A clause is selected because its full text names the material, but what was
    then sent to the model was `content[:700]` — the head of the chunk, which is
    not where the material is necessarily mentioned. On this corpus the two are
    routinely different, and the gap is exactly where the wrong answers came
    from. Measured 2026-08-20 on 241135LBP-HRDP001 section 5 (1235 characters):

        chars    0-700   P91's notes and the carbon steel line
        chars  700-830   ": 225 HB (max) ... for (P11 or equivalent)LAS material.
                          : 235 HB (max) ... for (P22 or equivalent)LAS material."

    So a P22 limit call was handed a clause selected on the strength of a P22
    sentence it could not see, ending mid-line at ": 225 HB (max)". The model
    then did the only thing available to it and answered from P91's rows — the
    observed P22 = 195-248 and CS = 195-280 misattributions. Under the
    quote-first schema the same window produces the opposite failure: no quote
    in it names P22, so every entry is rejected and P22 rows report
    MISSING_LIMIT despite the specification stating 235 HB max plainly.

    Windowing on the mention fixes both directions. Context before the mention
    is generous on purpose: these limits are stated as a heading ("values for
    Weld, HAZ & parent metal after PWHT for LAS shall be:") followed by
    per-material lines, and the heading is what identifies the zone.

    Windows around multiple mentions are merged and joined with " ... " when a
    gap is dropped, so the model can see it is not reading contiguous text.

    ON THE BUDGET
    ─────────────
    Raised from 1200 to 1800 on 2026-08-20 after it cost a limit. Section 5 of
    241135LBP-HRDP001 is 1235 characters and opens with

        "5 of 6 Weld metal/HAZ (195-280 HB) refer Note-3"

    while its only P91 mention ("For Alloy 91, if any reading...") sits near the
    END, around character 1000. Windowing on that mention trimmed the opening —
    so the weld limit, the one thing 221 weld rows needed, was the one line the
    model never saw. It returned base metal only, and every weld row reported
    MISSING_LIMIT.

    A clause states a material once and its limits throughout, so the window has
    to be generous enough to hold a whole clause of this size. These prompts run
    in ~4s on the GPU; the cost of a larger window is nothing next to the cost of
    cutting the answer out of it.
    """
    content = content or ""
    if len(content) <= budget:
        return content

    spans = _material_mentions(content, material)
    if not spans:
        # Selected by `grades_named_in` but no alias span located (multi-word
        # alias forms). Head of the chunk is as good a guess as any.
        return content[:budget]

    merged: list[list[int]] = []
    for start, end in spans:
        lo, hi = max(0, start - pad_before), min(len(content), end + pad_after)
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    # Leftover budget goes backwards from the first window, because that is
    # where the identifying context lives: these clauses state a heading
    # ("...values for Weld, HAZ & parent metal after PWHT for LAS shall be:")
    # and then one line per material, and a clause may name the material only
    # in a closing sentence while stating its numbers further up. Measured on
    # 241135LBP-HRDP001 section 5, "Alloy 91" occurs at character 1002 while
    # P91's notes run from character 0 — windowing on the mention alone would
    # have cut the notes off.
    spent = sum(hi - lo for lo, hi in merged)
    if spent < budget and merged[0][0] > 0:
        merged[0][0] = max(0, merged[0][0] - (budget - spent))

    parts: list[str] = []
    used = 0
    for lo, hi in merged:
        if used >= budget:
            break
        piece = content[lo:min(hi, lo + (budget - used))]
        used += len(piece)
        parts.append(("... " if lo > 0 else "") + piece.strip()
                     + (" ..." if hi < len(content) else ""))
    return " ".join(parts)


def _clauses_for_material(context: list[dict], material: str,
                          submittal_project: Optional[str] = None) -> list[str]:
    """
    Clauses that actually mention this material. Returns [] when none do.

    THERE IS DELIBERATELY NO FALLBACK.
    The previous version ended with:

        if not selected:
            selected = [... for x in context[:3]]   # top 3 clauses, ANY material

    Retrieval finds no P22/P11/CS-specific clause on this corpus, so those three
    calls were each handed the same P91-heavy clauses and extracted P91's
    numbers. Observed 2026-08-20: P22 came back 195-248, P11 came back 180-280,
    and CS cited the "P15E" row outright — P91's own P-Number. Every call
    reported SUCCESS because the model did exactly what it was asked.

    P22, P11 and CS have a maximum and NO minimum in the governing tables, so
    inheriting P91's 195 floor turns conforming readings into findings. Handing
    a limit call another material's clauses cannot produce a correct answer, so
    it is better to establish nothing: those rows then surface as MISSING_LIMIT
    and the report states the gap instead of inventing agreement.
    """
    # Whole-token matching, not substring. The alias lists contain short
    # designations — "5A" (P22), "P1" (CS), "P4" (P11) — and substring matching
    # made them fire on unrelated text: "P1" matches inside "P15E", "5A" inside
    # any "...5A..." run. That selected P91 clauses as P22 evidence, which is
    # how the misattributed limits got their source material in the first place.
    selected = []
    excluded_other_project = []
    for item in context:
        content = (item.get("content") or "").replace("\n", " ").strip()
        if material not in grades_named_in(content):
            continue

        # PROJECT SCOPING. The corpus holds specifications from more than one
        # project, and they disagree: 241135LBP-HRDP001 states "235 HB (max)
        # ... for (P22 or equivalent)" while 221135LBP-HRDP001 tabulates
        # "P5A | Base and Weld | 225 HB Max". For a 221135 submittal only the
        # latter governs, but retrieval returned both — so which limit a reading
        # was graded against depended on which clause the model happened to
        # read. General specifications (700-006, 700-016, ASME) carry no project
        # number and always apply.
        source = str(item.get("source") or "")
        if clause_project_conflicts(source, submittal_project):
            excluded_other_project.append(f"{source} (project {project_prefix(source)})")
            continue

        # Window on the material's own mention, not the head of the chunk —
        # see _material_window for what the head-of-chunk version cost.
        snippet = _material_window(content, material)
        selected.append(f"[{source} — {item.get('section', '')}] {snippet}")

    if excluded_other_project:
        pipeline_logger.info(
            f"{material}: excluded {len(excluded_other_project)} clause(s) from other projects "
            f"(submittal is project {submittal_project}): {excluded_other_project[:3]}"
        )
    if not selected:
        logger.warning(
            f"No in-project clause mentions {material} (or its P-Number "
            f"{MATERIAL_P_NUMBERS.get(material, {}).get('p_no', '?')}). Establishing no limits "
            f"for it rather than supplying another material's or another project's clauses."
        )
    return selected[:4]


def _norm(text: str) -> str:
    """Collapse whitespace and case so OCR spacing differences don't defeat a match."""
    return re.sub(r"\s+", " ", (text or "")).strip().upper()


def _squash(text: str) -> str:
    """
    Case and ALL whitespace removed, for asking "is this text present?".

    Collapsing runs of whitespace is not enough. The specification tables come
    out of OCR with stray spaces inside the cells — "| P15E | Base (After PWHT )
    | 195-248 HB |" has one before the bracket — and a model copying that row
    writes the tidy "Base (After PWHT)". Measured 2026-08-20: P91's base_metal
    AND weld limits, both read correctly at 195-248 and 195-280, were discarded
    as "quote not found in the supplied clauses" over that single space. Losing
    P91's limits to a typographical difference is a worse outcome than any it
    was protecting against.

    Removing whitespace entirely cannot let a fabricated quote through — the
    characters still have to be there, in order — it only stops spacing from
    deciding the answer.
    """
    return re.sub(r"\s+", "", (text or "")).upper()


def _squash_map(text: str) -> tuple[str, list[int]]:
    """
    `_squash(text)` plus, for each squashed character, its index in the original.

    Needed to answer "what does the clause say just BEFORE this quote?". The
    quote has to be located with whitespace ignored (OCR spacing differs from
    what a model copies), but the text around it must be read with its spaces
    intact — "ALLOY 91" is not a grade designation once the space is gone.
    """
    squashed: list[str] = []
    positions: list[int] = []
    for i, ch in enumerate(text or ""):
        if not ch.isspace():
            squashed.append(ch.upper())
            positions.append(i)
    return "".join(squashed), positions


def _lead_in(clauses: list[str], quote: str, width: int = 260) -> str:
    """
    The clause text immediately preceding `quote`, with source labels removed.

    Document labels are stripped because "[700-016-R2 Hardness Testing of Alloy
    91 Materials.pdf — ...]" names a grade in its FILENAME, and a filename is not
    a statement about the clause beneath it — the same reason `_quote_body`
    exists.
    """
    joined = " ".join(clauses)
    squashed, positions = _squash_map(joined)
    idx = squashed.find(_squash(quote))
    if idx == -1:
        return ""
    start = positions[idx]
    window = joined[max(0, start - width):start]
    return re.sub(r"\[[^\]]*\]", " ", window)


def _quote_body(quote: str) -> str:
    """
    The quote with its leading "[source — section]" label removed.

    Clauses are supplied to the model as "[241135LBP-HRDP001.pdf — 5 of 6] <text>"
    and models copy that label into the quote. Left in, it is matched by the
    material check: "[700-016-R2 Hardness Testing of Alloy 91 Materials.pdf — ...]"
    names Alloy 91 in its FILENAME, so any quote taken from that document would
    have satisfied "the quote names P91" no matter what the clause underneath
    said. A document's title is not a statement about the row being quoted.
    """
    return re.sub(r"^\s*\[[^\]]*\]\s*", "", quote or "")


# Words a quote must contain to justify the zone it was filed under. A table row
# reading "Base and Weld" legitimately satisfies base_metal and weld both.
_ZONE_WORDS = {
    "base_metal": ("BASE", "PARENT", "BM", "BASE METAL", "PARENT METAL"),
    "haz": ("HAZ", "HEAT AFFECTED", "HAZ1", "HAZ2"),
    "weld": ("WELD", "WM", "W.M.", "WELD METAL"),
}


def _numbers_present(entry: dict, text: str) -> bool:
    """True when every bound the entry reports appears as a number in `text`."""
    for field in ("min", "max", "single_reading_min", "single_reading_max"):
        val = entry.get(field)
        if val is None:
            continue
        as_int = int(val) if float(val).is_integer() else val
        if not re.search(rf"(?<!\d){re.escape(str(as_int))}(?!\d)", text):
            return False
    return True


def _clause_segments(clause: str) -> list[str]:
    """
    A clause split into the rows and sentences a limit can be stated in.

    Table rows arrive as one line — "| P4 | Base and Weld | 225 HB Max | | P5A |
    Base and Weld | 225 HB Max | ..." — where "| |" separates one row from the
    next, so that is the row boundary. Prose is split on sentence ends.
    """
    segments: list[str] = []
    for part in re.split(r"\|\s*\|", clause or ""):
        for sentence in re.split(r"(?<=[.;])\s+", part):
            cleaned = sentence.strip(" |")
            if len(_squash(cleaned)) >= 12:
                segments.append(cleaned)
    return segments


def _supporting_segment(entry: dict, quote: str, clauses: list[str],
                        material: str) -> Optional[str]:
    """
    The verbatim clause row that actually supports this entry, if there is one.

    Models quote a table by copying the header, the "| --- |" separator, and the
    row they mean — dropping the rows in between. Observed on this corpus
    2026-08-20:

        "| Material | Hadness on Location | Brinell Hardness(HB) | | --- | --- |
         --- | | P5A | Base and Weld | 225 HB Max |"

    Every character of that is real, but it is not CONTIGUOUS text: the P4 row
    between the separator and the P5A row has been elided, so a substring test
    fails. P11's row is the first data row, so the same elision happened to be
    contiguous and was accepted, while P22's — the second row — was rejected
    three times over. The reported value, 225 HB Max, was correct in both cases;
    P22 ended the run with no limits and 652 rows reported MISSING_LIMIT.

    Rather than loosen the presence test, narrow the claim to what the clause
    actually says: find the clause row that is ITSELF contained in the model's
    quote, names the requested material, and carries the reported numbers. That
    row becomes the quote of record. It is verbatim specification text, and it
    is text the model demonstrably read.

    This is stricter than the plain substring test, not looser. Stitching two
    rows together cannot pass it — a fabricated "| P5A | ... | 195-248 HB |"
    finds no single clause row that both names P22 and contains 195 and 248.
    """
    squashed_quote = _squash(quote)
    best: Optional[str] = None
    for clause in clauses:
        for segment in _clause_segments(clause):
            if _squash(segment) not in squashed_quote:
                continue                                  # not text the model read
            if material not in grades_named_in(_quote_body(segment)):
                continue                                  # not this material's row
            if not _numbers_present(entry, segment):
                continue                                  # not where these numbers are
            if best is None or len(_squash(segment)) > len(_squash(best)):
                best = segment
    return best


def verify_limit_entry(entry: dict, clauses: list[str], material: str) -> Optional[str]:
    """
    Verify one limit entry against its own quote. Returns None if sound, else why not.

    This is the half that makes the quote-first prompt worth having. Instructions
    alone failed six times; these checks make a fabricated answer mechanically
    detectable rather than merely discouraged:

      1. the quote must genuinely occur in the supplied clause text
         -> a quote the model composed itself cannot pass
      2. the material the quote names must be the material we asked about
         -> catches CS answered from the P15E row, P22 from P91's band
      3. every reported bound must appear as a number inside the quote
         -> catches invented limits, and catches a bound copied from a
            neighbouring row that the quote does not cover
      4. the quote must be about the zone it was filed under — either naming
         it, or sitting directly under a heading that does
         -> catches a "Weld (After PWHT)" row filed as `haz`

    A model that reads carefully passes all four without effort. One that guesses
    fails at least one.
    """
    quote = (entry.get("quote") or "").strip()
    if len(quote) < 12:
        return "quote missing or too short to verify"

    # 1. Quote must be present in what we actually supplied.
    #
    # Preferably narrowed to the single clause row that supports this entry:
    # models quote a whole table to state one row, and the row is both the
    # honest citation and the stricter test. See _supporting_segment.
    haystack = _squash(" ".join(clauses))
    supporting = _supporting_segment(entry, quote, clauses, material)
    if supporting is not None:
        # The quote of record becomes verbatim clause text. Written back so the
        # accepted limit — and the finding that cites it — shows the row the
        # verdict rests on rather than the model's paraphrase of a table.
        quote = supporting
        entry["quote"] = supporting
    elif _squash(quote) not in haystack:
        return f"quote not found in the supplied clauses: {quote[:70]!r}"

    # 2. The quote must be about the requested material.
    # The QUOTE has to name the material — not `material_in_quote`, which is
    # free text the model writes and can therefore assert anything. Accepting
    # either one made the field a bypass: a quote of the P5A row labelled
    # "Alloy 91" was accepted as P91's limit, and a quote taken from
    # "700-016-R2 Hardness Testing of Alloy 91 Materials.pdf" could be attributed
    # to P91 on the strength of the filename alone (see _quote_body). The field
    # is kept because writing it makes the model commit to an attribution before
    # it emits any numbers, but it is not evidence.
    named_in_quote = grades_named_in(_quote_body(quote))
    if material not in named_in_quote:
        others = sorted(named_in_quote)
        if others:
            # Names a DIFFERENT grade. That is another material's limit and no
            # amount of surrounding context makes it this one's.
            return (f"quote does not name {material} (it names {others}): {quote[:70]!r}")

        # Names NO grade. Specifications routinely state the material once, in a
        # heading, and then list its limits line by line beneath:
        #
        #   "permissible hardness values for Weld, HAZ & parent metal for HAS
        #    (Alloy 91) shall be: Base metal-Prior to fabrication (200-248 HB)
        #    ... Base metal- after PWHT (195-248 HB) ..."
        #
        # Quoting "Base metal- after PWHT (195-248 HB)" is the correct and
        # complete answer for P91, and requiring the grade INSIDE the quote
        # rejected it — which cost 241135LBP-HRDR001 every one of its limits and
        # left 341 graded rows with nothing to grade against.
        #
        # So look where the specification actually put the material name: the
        # text immediately above. Verifiable, and it stays strict — the lead-in
        # must name this material and ONLY this material. A heading covering
        # several grades cannot say which line belongs to which.
        lead = _lead_in(clauses, quote)
        near = grades_named_in(lead)
        if near == {material}:
            pass                    # the heading above states this material
        elif near:
            return (f"quote names no material and the text above it covers {sorted(near)}, "
                    f"so it does not establish {material}'s limit: {quote[:70]!r}")
        else:
            claimed = (entry.get("material_in_quote") or "").strip()
            return (f"quote does not name {material}, nor does the text above it"
                    + (f" — it was labelled {claimed!r}, but the quoted text is what counts"
                       if claimed else "")
                    + f": {quote[:70]!r}")

    # 3. Reported bounds must appear inside the quote.
    if not _numbers_present(entry, quote):
        stated = {f: entry.get(f) for f in ("min", "max") if entry.get(f) is not None}
        return f"reported bounds {stated} do not appear in the quote: {quote[:70]!r}"

    # 4. The quote must be about the zone it was filed under.
    zone = entry.get("zone") or ""
    words = _ZONE_WORDS.get(zone, ())
    normalised = _squash(quote)
    words = tuple(_squash(w) for w in words)
    if words and not any(w in normalised for w in words):
        # Does it name a DIFFERENT location? Then it is that location's limit
        # and this filing is wrong — this is the "Weld (After PWHT) row filed
        # as haz" case, and it stays rejected.
        elsewhere = sorted(z for z, ws in _ZONE_WORDS.items()
                           if z != zone and any(_squash(w) in normalised for w in ws))
        if elsewhere:
            return f"quote is about {elsewhere}, not {zone}: {quote[:70]!r}"

        # It names no location at all. In this corpus that is normal and
        # legitimate: 241135LBP-HRDP001 section 5 states one heading —
        # "values for Weld, HAZ & parent metal after PWHT for LAS shall be:" —
        # and then a bare line per material (": 235 HB (max) average hardness
        # reading for (P22 or equivalent)LAS material."). That line carries the
        # material and the number but no location word, so requiring one inside
        # the quote rejected a perfectly sound reading and left P22, P11 and CS
        # with no limits at all. Look at the text immediately above the quote in
        # the clause it was taken from, which is still mechanically checkable.
        idx = haystack.find(normalised)
        lead_in = haystack[max(0, idx - 190):idx] if idx != -1 else ""
        if not any(w in lead_in for w in words):
            return (f"neither the quote nor the text above it mentions the {zone} "
                    f"location: {quote[:70]!r}")

    # 5. If the quote states a RANGE, both bounds must be reported.
    #
    # The other four checks cannot catch a dropped minimum: quoting
    # "(195-248 HB)" and reporting min=None passes all of them, because the
    # quote is genuine, names P91, contains 248 and mentions base metal. That
    # exact omission recurred across attempts and is the dangerous direction —
    # losing P91's 195 floor makes every under-hardness reading PASS.
    #
    # A range in the quote is unambiguous, so hold the answer to it.
    rng = re.search(
        r"(?<!\d)(\d{2,3})\s*(?:-|–|—|~|\bto\b)\s*(\d{2,3})(?!\d)",
        quote,
    )
    if rng:
        lo, hi = int(rng.group(1)), int(rng.group(2))
        if lo < hi:  # a genuine ascending range, not a date or clause number
            got_min, got_max = entry.get("min"), entry.get("max")
            if got_min is None:
                return (f"quote states the range {lo}-{hi} but min was reported as null; "
                        f"a stated minimum must not be dropped: {quote[:70]!r}")
            if int(got_min) != lo or (got_max is not None and int(got_max) != hi):
                return (f"quote states the range {lo}-{hi} but min={got_min}/max={got_max} "
                        f"was reported: {quote[:70]!r}")

    return None


def _citation_names_other_material(ref: str, material: str) -> Optional[str]:
    """
    Detect a limit whose own citation points at a different material.

    Cross-material misattribution is the failure mode this guards: a CS limit
    citing "[P15E | Base (After PWHT) | 195-248 HB]" is quoting P91's row. The
    citation is the model's own account of where the number came from, so when it
    names another grade and not this one, the number is not this material's.
    """
    if not ref:
        return None

    # Whole-token matching via canonical_grade. Substring matching is unsafe
    # here for the same reason it was unsafe in extraction: "P1" (a CS alias)
    # occurs inside "P15E" (P91), so a CS limit citing P91's P15E row would
    # match its own alias and be accepted — the precise misattribution this
    # function exists to catch.
    named = grades_named_in(ref)

    if material in named:
        return None            # names this material — accept
    if named:
        return sorted(named)[0]  # names only other materials — reject

    # Names NO grade at all. Previously accepted, and that was the hole: the
    # P91 weld limit came back as 225 (P22's ceiling, not P91's 280) citing
    # '[241135LBP-HRDP001.pdf — 6 of 6]', and P22/P11 both cited 'Note-3 (2)',
    # which is a P91 weld provision. A citation that identifies no material
    # cannot support a material-specific limit, so it is not evidence.
    return "UNVERIFIABLE"


def _salvage_limit_entries(raw: str) -> list[dict]:
    """
    Recover the complete entries from a truncated limits response.

    Ollama decodes against the JSON schema, so a response is well-formed right
    up to the point the token budget ran out — the CS call on 2026-08-20 stopped
    inside a `ref` string, leaving two complete, valid entries stranded inside an
    unparseable document. Discarding those and failing the run threw away work
    the model had already done correctly.

    Every recovered entry still goes through `verify_limit_entry`, so salvaging
    loosens nothing: it only stops a cut-off tail from voiding the entries that
    were finished before the cut.
    """
    entries: list[dict] = []
    stack: list[int] = []
    in_string = escaped = False
    for i, ch in enumerate(raw or ""):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            obj = _json(raw[stack.pop():i + 1], dict)
            if isinstance(obj, dict) and "zone" in obj:
                entries.append(obj)
    return entries


# Heat-treatment stage a quoted limit applies to. Specifications state more than
# one band for the same zone — 241135LBP-HRDP001 gives P91 base metal as
# "Prior to fabrication (200-248 HB)" AND "after PWHT (195-248 HB)", and the
# 221135 table likewise separates "Base Material (Incoming Stage)" from
# "Base (After PWHT)". A hardness test report is submitted AFTER PWHT — the DEE
# form prints "All hardness have been verified After PWHT" on every page — so
# the post-PWHT band governs.
#
# Without this the zone was simply overwritten by whichever entry the model
# emitted last, so P91's minimum came out 195 or 200 depending on generation
# order. 200 against an after-PWHT report makes a finding of every conforming
# 195-199 reading.
_STAGE_PRE = ("PRIORTOFABRICATION", "INCOMINGSTAGE", "INCOMING", "BEFOREPWHT", "PRIORTOPWHT")
_STAGE_POST = ("AFTERPWHT", "POSTPWHT", "AFTERALLPWHT", "AFTERHEATTREATMENT")


def _stage_rank(quote: str) -> int:
    """2 = states post-PWHT, 0 = states a pre-PWHT stage, 1 = says neither."""
    q = _squash(quote)
    if any(k in q for k in _STAGE_POST):
        return 2
    if any(k in q for k in _STAGE_PRE):
        return 0
    return 1


def _accept_limits(rows_out: list, clauses: list[str], material: str) -> dict:
    """Verify each returned entry against its own quote; keep only what holds up."""
    accepted: dict[str, dict] = {}
    for r in rows_out:
        if not isinstance(r, dict) or r.get("zone") not in ("base_metal", "haz", "weld"):
            continue
        # Verify against the model's own quote. This replaces the
        # citation-string guard, which rejected valid work: a real citation is a
        # "document — section" reference and does not repeat the material name,
        # so requiring it to do so discarded P91's correct limits from
        # '[... — 8 Acceptance Criteria]'. Checking the QUOTE instead is both
        # stricter about fabrication and fair to legitimate references.
        problem = verify_limit_entry(r, clauses, material)
        if problem:
            pipeline_logger.warning(
                f"Discarding {material}/{r.get('zone')} limit "
                f"(min={r.get('min')}, max={r.get('max')}): {problem}"
            )
            continue
        entry = {k: r.get(k) for k in ("min", "max", "single_reading_min",
                                       "single_reading_max", "unit", "ref")
                 if r.get(k) is not None}
        # Keep the quote with the limit so a finding can show the exact
        # specification text its verdict rests on. `verify_limit_entry` may have
        # narrowed it to the single clause row that supports the entry.
        entry["quote"] = (r.get("quote") or "").strip()

        prior = accepted.get(r["zone"])
        if prior is not None and _stage_rank(entry["quote"]) < _stage_rank(prior.get("quote", "")):
            pipeline_logger.info(
                f"{material}/{r['zone']}: keeping the post-PWHT band "
                f"(min={prior.get('min')}, max={prior.get('max')}) over "
                f"min={entry.get('min')}/max={entry.get('max')} from "
                f"{entry['quote'][:60]!r} — the submittal reports hardness after PWHT."
            )
            continue
        accepted[r["zone"]] = entry

    # A row that states several locations states a limit for each of them.
    #
    # "| P5A | Base and Weld | 225 HB Max |" is one limit covering two zones,
    # but the model reports one entry per zone and typically returns only one of
    # them — so P22 and P11 came away with a base_metal limit and nothing for
    # weld, and every weld row of those materials was reported MISSING_LIMIT
    # against a specification that plainly covers it.
    #
    # The expansion is read off the accepted quote, which by this point is
    # verbatim clause text narrowed to a single row, so it asserts nothing the
    # specification does not. It never overwrites a zone the model reported
    # explicitly, and it is confined to short single-row quotes: a quote long
    # enough to span several rows may name a zone belonging to a DIFFERENT row.
    for zone, entry in list(accepted.items()):
        quote = _squash(entry.get("quote") or "")
        if len(quote) > 180:
            continue
        for other, words in _ZONE_WORDS.items():
            if other in accepted or other == zone:
                continue
            if any(_squash(w) in quote for w in words):
                accepted[other] = dict(entry)
                pipeline_logger.info(
                    f"{material}: applying the {zone} limit to {other} as well — its row "
                    f"states both locations: {entry.get('quote')!r}"
                )
    return accepted


# REMOVED: _row_material(row, page, fallback)
#
# It returned `fallback` — which callers set to materials[0], i.e. "P91" — for
# any row whose own hint was empty. On 221135LBP-HRDR001 that graded 111 rows
# as P91 with no supporting evidence, applying P91's 195 HBW minimum to rows
# that may be P22 (no minimum at all). Material resolution now lives in
# src/agents/context_resolver.py, which reports UNRESOLVED instead of guessing.


def llm_batch_review_node(state: ReviewState) -> dict:
    """LLM-only: establish limits per material, then review all table rows in batches."""
    pages = state.get("page_ocr_results", []) or []
    materials = state.get("detected_materials", []) or []
    context = state.get("retrieved_context", []) or []
    host = get_ollama_host()
    traces = list(state.get("llm_traces", []))
    limits: dict[str, dict] = {}
    limit_failures: list[str] = []
    attempted: list[str] = []

    # Which project's specifications govern this submittal. Taken from the
    # drawing/spool identifiers OCR read, falling back to the filename.
    submittal_project = None
    for page in pages:
        submittal_project = (project_prefix(getattr(page, "drawing_no", "") or "")
                             or project_prefix(getattr(page, "spool_no", "") or ""))
        if submittal_project:
            break
    if not submittal_project:
        submittal_project = project_prefix(state.get("submittal_filename", "") or "")
    if submittal_project:
        pipeline_logger.info(
            f"Submittal belongs to project {submittal_project}; specification clauses from "
            f"other projects will be excluded (general specs always apply)."
        )
    else:
        pipeline_logger.warning(
            "Could not determine the submittal's project number — clauses from ALL projects "
            "will be considered, and different projects state different limits."
        )

    with StageTracker("llm_batch_review", f"materials={materials}, pages={len(pages)}") as stage:
        # ── PHASE 1: establish limits — ONE call for all materials ────
        #
        # Was one call per material (4 calls, ~330s on this document). Most of
        # that was fixed overhead re-prefilling the same system prompt and
        # largely the same clause text each time. One enum-constrained call
        # returns the whole table; and because limits are a property of the
        # SPECIFICATIONS rather than the submittal, the result is cached so
        # subsequent documents reviewed against the same guide pay nothing.
        clause_set: list[str] = []
        seen: set[str] = set()
        for material in materials:
            for clause in _clauses_for_material(context, material, submittal_project):
                if clause not in seen:
                    seen.add(clause)
                    clause_set.append(clause)
        clause_set = clause_set[:6]

        cache_key = _limits_cache_key(state.get("guide_code", "") or "", materials, clause_set)
        cached = _load_cached_limits(cache_key)

        if cached:
            limits = cached
            pipeline_logger.info(
                f"Limits reused from cache (key={cache_key}) for {sorted(limits)} — 0 LLM calls. "
                f"Cached limits are keyed by guide + exact clause text, so a change in either misses."
            )
        else:
            # ONE CALL PER MATERIAL — deliberately not one call for all four.
            #
            # A single all-materials call is ~6x faster (52.8s vs ~330s) but
            # measurably wrong on this model: asked for four materials at once,
            # qwen2.5:3b established only two and attributed P91's weld band
            # (195-280, Note 3 -> 315) to CARBON STEEL, whose actual limit is
            # 200 HB Max. Per-material calls got all four right. Accuracy wins
            # here; the speed comes from the cache above instead, which makes
            # this cost apply once per specification set rather than per
            # document.
            for material in materials:
                clauses = _clauses_for_material(context, material, submittal_project)
                if not clauses:
                    # No clause mentions this material. Skip the call entirely:
                    # there is nothing to extract from, and asking anyway
                    # produced another material's limits (see
                    # _clauses_for_material).
                    pipeline_logger.warning(
                        f"Skipping limit extraction for {material}: no retrieved clause mentions it. "
                        f"Its rows will report MISSING_LIMIT rather than be graded against "
                        f"another material's band."
                    )
                    continue
                attempted.append(material)
                p_no = MATERIAL_P_NUMBERS.get(material, {}).get("p_no", "unknown")
                prompt = (
                    f"Material: {material}; ASME P-Number: {p_no}\n"
                    f"Specification tables may label this material \"P-No. {p_no}\" or \"P{p_no}\" "
                    f"rather than \"{material}\" — a row so labelled states this material's limits.\n"
                    "Extract base_metal, haz, and weld limits only where explicitly stated, with citations.\n\n"
                    "SPECIFICATION CLAUSES:\n" + "\n".join(clauses)
                )
                started = time.time()
                raw = ""
                material_limits: dict = {}
                failure: Optional[str] = None
                salvaged = False

                # Two attempts, the second with a larger output budget. A
                # truncated response is a budget problem and is worth retrying;
                # a parsed response whose entries fail verification is a reading
                # problem and is not — more tokens will not make the model quote
                # a sentence that is not in front of it, so that case breaks out
                # below with whatever survived verification.
                for budget in (LIMIT_NUM_PREDICT, LIMIT_NUM_PREDICT_RETRY):
                    try:
                        raw = _chat(host, LIMIT_SYSTEM, prompt, LIMIT_SCHEMA, budget)
                    except Exception as exc:
                        failure = f"limit call failed: {exc}"
                        continue

                    parsed = _json(raw, dict)
                    rows_out = parsed.get("limits") if isinstance(parsed, dict) else None
                    if not isinstance(rows_out, list):
                        rows_out = _salvage_limit_entries(raw)
                        if not rows_out:
                            failure = (f"limit batch did not return a limits array "
                                       f"(response unparseable, likely truncated at "
                                       f"num_predict={budget}; {len(raw)} chars returned)")
                            continue
                        salvaged = True
                        pipeline_logger.warning(
                            f"{material}: limit response was cut off at num_predict={budget}; "
                            f"recovered {len(rows_out)} complete entries from it. Each still "
                            f"has to pass quote verification."
                        )

                    material_limits = _accept_limits(rows_out, clauses, material)
                    failure = None
                    break

                if failure:
                    # DEGRADE, DO NOT ABORT.
                    #
                    # This used to raise LLMAgentError, which ended the run and
                    # wrote no report at all — one material's truncated response
                    # discarded the other materials' correct limits and every row
                    # of the submittal along with it. STRICT_LLM_MODE exists to
                    # stop hardcoded limits being substituted for real
                    # extraction; it does not require the whole review to be
                    # lost. Establishing nothing for this material is the honest
                    # outcome and is already a first-class one: its rows report
                    # MISSING_LIMIT (route=retrieval) and it appears in
                    # `uncovered_materials`, so the report states the gap.
                    traces.append(record_llm_trace(
                        f"limit_{material}", f"LLM Limit Batch — {material}", OLLAMA_MODEL,
                        LIMIT_SYSTEM, prompt, raw, {}, time.time() - started,
                        f"FAILURE: {failure}",
                        {"material": material, "limits": 0, "cache_key": cache_key,
                         "degraded": True},
                    ))
                    pipeline_logger.error(
                        f"Limit extraction failed for {material} after 2 attempts: {failure}. "
                        f"No limits established for {material} — its rows will report "
                        f"MISSING_LIMIT rather than be graded against another material's band. "
                        f"The remaining materials and the rest of the review continue."
                    )
                    limit_failures.append(material)
                    continue

                if material_limits:
                    limits[material] = material_limits
                else:
                    pipeline_logger.warning(
                        f"{material}: the model answered, but no entry survived quote "
                        f"verification, so no limit is established for it. See the discard "
                        f"reasons above — a clause stating this material's limit outside the "
                        f"supplied window is the usual cause."
                    )
                traces.append(record_llm_trace(
                    f"limit_{material}", f"LLM Limit Batch — {material}", OLLAMA_MODEL, LIMIT_SYSTEM,
                    prompt, raw, limits.get(material, {}), time.time() - started, "SUCCESS",
                    {"material": material, "limits": len(limits.get(material, {})),
                     "cache_key": cache_key, "salvaged": salvaged},
                ))

            # CROSS-MATERIAL DUPLICATE CHECK
            #
            # Distinct materials having identical bounds AND an identical
            # citation is the signature of one clause being read as evidence for
            # several materials. Observed: P22 and P11 both returned
            # min=None/max=225 citing the same "Note-3 (2)" — their values happen
            # to be right, but neither was actually read from its own row, so the
            # agreement is luck rather than evidence. Flag it loudly; do not
            # silently trust it.
            seen_fingerprints: dict[tuple, list[str]] = {}
            for mat, zones in limits.items():
                for zone, v in zones.items():
                    fp = (zone, v.get("min"), v.get("max"), str(v.get("ref") or "")[:60])
                    seen_fingerprints.setdefault(fp, []).append(mat)
            for fp, mats in seen_fingerprints.items():
                if len(mats) > 1:
                    pipeline_logger.warning(
                        f"Suspect limits: {sorted(mats)} all report {fp[0]} min={fp[1]} max={fp[2]} "
                        f"from the same citation {fp[3]!r}. Identical bounds from one citation "
                        f"across different materials suggests a single clause was reused as "
                        f"evidence for several — verify against the specification before relying "
                        f"on these findings."
                    )

            # A material we called for and got nothing usable from is not a
            # settled answer, and caching it makes the gap permanent: the key is
            # the guide plus the clause text, so an identical rerun would reuse
            # the incomplete set and never call again. Measured 2026-08-20: the
            # 18:24 run cached {P91, P11} while P22's entries had all failed
            # verification, so P22's 225 HB Max would have stayed MISSING_LIMIT
            # on every subsequent review of these specifications.
            #
            # A material SKIPPED for want of any clause (CS here) is different —
            # nothing was asked, so nothing is missing from the answer.
            incomplete = [m for m in attempted if m not in limits]
            if limits and not limit_failures and not incomplete:
                _store_cached_limits(cache_key, limits, materials)
                pipeline_logger.info(
                    f"Limits cached under key={cache_key}; subsequent documents reviewed against "
                    f"the same specification clauses will skip all {len(materials)} limit calls."
                )
            elif incomplete:
                pipeline_logger.warning(
                    f"Not caching limits: {sorted(incomplete)} produced no usable limit despite "
                    f"having clauses to read. Caching now would freeze that gap into every later "
                    f"review against these same clauses."
                )
            elif limit_failures:
                # A partial set must not be cached. The key is the guide plus the
                # clause text, neither of which changes when a call merely times
                # out or truncates, so caching now would make a transient failure
                # permanent: every later document against these specifications
                # would silently reuse an incomplete limit set and report those
                # rows MISSING_LIMIT forever.
                pipeline_logger.warning(
                    f"Not caching limits: extraction failed for {sorted(limit_failures)}. "
                    f"Caching a partial set would make a transient failure permanent for every "
                    f"later document reviewed against these clauses. The next run retries."
                )

        rows: list[dict] = []
        # Material comes from the Context Resolver: row-level evidence first,
        # then joint-scoped inheritance from explicitly-graded rows, then an
        # unambiguous page test-block designation — and UNRESOLVED when the
        # document establishes nothing.
        #
        # This replaces `_row_material`, which returned `materials[0]` for any
        # row without an explicit hint. On this submittal that silently graded
        # 111 rows as P91, applying P91's 195 HBW minimum to rows with no
        # material evidence at all — a false-positive generator. There is no
        # default here on purpose: a row nobody can grade is reported as
        # unevaluated rather than guessed.
        unresolved_rows: list[dict] = []
        unreliable_pages = 0
        for page in pages:
            # A page whose extraction is not trusted is not graded. Its rows are
            # carried through as unevaluated so the report states the gap —
            # grading them would produce findings that look exactly like real
            # ones while resting on misread cells.
            if not getattr(page, "extraction_ok", True):
                unreliable_pages += 1
                for row_index, row in enumerate(page.table_rows):
                    readings = [r.value for r in row.readings if r.value is not None]
                    if not readings:
                        continue
                    unresolved_rows.append({
                        "page_idx": page.page_idx, "row_index": row_index,
                        "component": row.component, "material": "UNRESOLVED",
                        "zone": "base_metal", "joint_name": row.joint_name,
                        "readings": readings, "provenance": "extraction_unreliable",
                        "confidence": 0.0, "limits": {},
                        "extraction_unreliable": True,
                    })
                continue
            contexts = {c.row_index: c for c in resolve_page_rows(page)}
            for row_index, row in enumerate(page.table_rows):
                readings = [reading.value for reading in row.readings if reading.value is not None]
                if not readings:
                    continue
                ctx = contexts.get(row_index)
                if ctx is None:
                    continue
                entry = {"page_idx": page.page_idx, "row_index": row_index, "component": row.component,
                         "material": ctx.material, "zone": ctx.zone, "joint_name": row.joint_name,
                         "readings": readings, "provenance": ctx.provenance, "confidence": ctx.confidence,
                         "limits": limits.get(ctx.material, {}).get(ctx.zone, {})}
                (rows if ctx.resolved else unresolved_rows).append(entry)

        pipeline_logger.info(
            f"Context resolver: {len(rows)} rows graded from document evidence, "
            f"{len(unresolved_rows)} UNRESOLVED (no material established) — the "
            f"unresolved rows are the only ones an LLM needs to adjudicate."
        )

        # ── PHASE 2: deterministic compliance check ───────────────────
        #
        # Replaces 83 sequential LLM calls that asked "is this reading within
        # these limits?" — ~244s each, ~5.6 hours, against a 300s per-call
        # timeout fatal under STRICT_LLM_MODE. Everything those calls needed
        # (material, zone, readings, cited limits) is already in the row dict by
        # this point, so the comparison is arithmetic. See
        # src/agents/compliance_check.py for why it is not a bare
        # `min <= value <= max`.
        all_rows = rows + unresolved_rows
        results, counts = check_rows(all_rows)
        pipeline_progress.set_percentage(95)

        findings: list[ComplianceFinding] = []
        for row, res in zip(all_rows, results):
            if res["status"] != NON_COMPLIANT:
                continue
            findings.append(ComplianceFinding(
                page_idx=row["page_idx"],
                row_index=row["row_index"],
                component=row.get("component") or "",
                material=res["material"],
                failing_readings=res["failing_readings"],
                limit_violated=res["limit_violated"],
                severity=res["severity"],
                # The citation the limit-resolution call established, carried
                # through unchanged so every finding traces to a real clause.
                requirement_ref=res["limits"].get("ref") or "Retrieved specification clause",
                reasoning=res["reason"],
            ))

        # Rows that genuinely need a model: an ambiguous zone or competing
        # limits. Rows unresolved for extraction / resolver / retrieval reasons
        # are upstream defects — an LLM would answer confidently over missing
        # data and hide them.
        adjudicate = needs_llm_adjudication(results)
        upstream = [r for r in results if r["status"] == "UNRESOLVED" and r.get("route") != "llm"]

        stage.details += (
            f" | limits={list(limits)}, rows={len(all_rows)}, findings={len(findings)}, "
            f"llm_adjudication={len(adjudicate)}, upstream_gaps={len(upstream)}"
            + (f", limit_extraction_failed={sorted(limit_failures)}" if limit_failures else "")
        )

    uncovered = [material for material in materials if material not in limits]
    if uncovered:
        pipeline_logger.warning(
            f"No limits established for {sorted(uncovered)} "
            + (f"(extraction failed for {sorted(limit_failures)}) " if limit_failures else "")
            + "— every row of those materials is reported unevaluated (MISSING_LIMIT) rather "
            "than graded against another material's band."
        )

    pipeline_logger.info(
        "Deterministic compliance check: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    pipeline_logger.info(
        f"Compliance: {len(findings)} findings from {len(all_rows)} rows with 0 LLM calls "
        f"(was {-(-len(all_rows) // LLM_AUDIT_BATCH_SIZE)} calls at batch size {LLM_AUDIT_BATCH_SIZE})."
    )
    if adjudicate:
        pipeline_logger.warning(
            f"{len(adjudicate)} row(s) need LLM adjudication "
            f"({-(-len(adjudicate) // LLM_AUDIT_BATCH_SIZE)} batches at size {LLM_AUDIT_BATCH_SIZE}): "
            f"{sorted({r['reason_code'] for r in adjudicate})}"
        )
    if upstream:
        by_reason: dict[str, int] = {}
        for r in upstream:
            by_reason[r["reason_code"]] = by_reason.get(r["reason_code"], 0) + 1
        pipeline_logger.warning(
            f"{len(upstream)} row(s) unevaluated for upstream reasons (not sent to an LLM): "
            + ", ".join(f"{k}={v} [fix in {UNRESOLVED_REASONS[k]['route']}]" for k, v in sorted(by_reason.items()))
        )

    return {
        "hardness_rules_schema": limits,
        "compliance_findings": findings,
        "grouped_findings": [],
        "uncovered_materials": uncovered,
        "compliance_results": results,
        "rows_needing_adjudication": adjudicate,
        "llm_traces": traces,
        "current_agent": "llm_batch_reviewer",
    }
