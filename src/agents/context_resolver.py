from __future__ import annotations

"""
QDR·AI — Context / Row Relationship Resolver
─────────────────────────────────────────────
Sits between table extraction and RAG retrieval. Decides, for every extracted
row, which material grade governs it — and says so with provenance, or admits
it does not know.

WHY THIS EXISTS
───────────────
A single page of a hardness package mixes materials. Page 13 of
221135LBP-HRDR001 carries P91 welds and a P22 pipe in one table, and only 1 of
its 12 rows states a grade explicitly. Two earlier approaches both got it
wrong, in opposite directions:

  * `llm_batch_reviewer._row_material` returned `materials[0]` for any row
    without an explicit hint — so 11 of 12 rows on page 13 became P91 by
    accident. Applying P91's 195 minimum to a P22 pipe reading of 190 produces
    a FALSE POSITIVE: P22 has no minimum at all.

  * `rule_engine._resolve_page_materials` inferred material from the spool
    prefix ("RP" → P22, "HP" → P91) and returned P22 for the entire page —
    overriding even the one row that explicitly said P91. Spool numbering is a
    naming convention, not a material designation, so this inference has no
    basis. It is not used here.

RULES
─────
1. Material comes ONLY from an actual grade designation (P91 / P22 / P11 / CS,
   or an ASME P-Number that denotes one) found in the document text.
2. Explicit row-level evidence outranks every inference and is never
   overridden.
3. Inheritance is bounded to the joint group, and only from a row whose own
   material was explicit. A Pipe row does not inherit from a Weld row.
4. A page-level designation is usable only when the page names exactly one
   material. A page naming both P91 and P22 provides no page-level answer.
5. When nothing establishes a grade, the answer is UNRESOLVED. There is no
   default. A row nobody can grade must not be silently evaluated against a
   guessed range — that is the defect this module replaces.

Every result carries `provenance` and `confidence` so a finding can state why
its row was graded as it was, and so UNRESOLVED rows can be routed to an LLM
or a human instead of disappearing into a default.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.agents.materials import canonical_grade, grades_named_in

logger = logging.getLogger("qdr_ai.context_resolver")

UNRESOLVED = "UNRESOLVED"

# Ordered strongest-first; also the order tried below.
PROVENANCE_CONFIDENCE = {
    "row_explicit": 1.00,      # row's own material_hint states the grade
    "row_left_text": 0.95,     # grade appears in the row's own label text
    "joint_explicit": 0.85,    # inherited from an explicitly-graded row, same joint
    "haz_adjacent": 0.80,      # HAZ takes the grade of the component it adjoins
    "page_single": 0.70,       # page/test block names exactly one material
    "none": 0.00,
}


@dataclass
class RowContext:
    """Resolved context for one extracted table row."""

    row_index: int
    page_idx: int
    component: str
    zone: str                  # base_metal | haz | weld
    joint_name: str
    material: str              # canonical grade, or UNRESOLVED
    provenance: str
    confidence: float
    readings: list[int]

    @property
    def resolved(self) -> bool:
        return self.material != UNRESOLVED


# Components that state nothing about where the reading was taken.
_NO_ZONE = {"", "UNKNOWN", "GENERIC", "?", "N/A", "-"}


def zone_of(row: Any) -> str:
    """
    Map a component to the zone whose limits apply, or "" when it says nothing.

    The base_metal fallback used to be unconditional, which made it an ASSERTION
    rather than a default: a row whose location could not be read was declared
    base metal and graded against the base-metal band. On 241135LBP-HRDR001
    every row arrived unlabelled, so all 361 were called base metal and weld
    readings of 249-269 were failed against a 248 ceiling that never applied to
    them.

    An unrecognised component now yields no zone, and compliance_check reports
    the row AMBIGUOUS_ZONE. A named component still maps as before — the
    fallback stays for labels like "Elbow" or "Plate", which genuinely are
    base metal.
    """
    comp = (getattr(row, "component", "") or "").upper().strip()
    if getattr(row, "is_weld", False) or any(w in comp for w in ("WELD", "WM", "W.M.", "W/M", "WLD")):
        return "weld"
    if "HAZ" in comp:
        return "haz"
    if comp in _NO_ZONE:
        return ""
    return "base_metal"


def _grades_in(text: str) -> list[str]:
    """
    Every distinct grade designation appearing in a blob of text.

    Delegates to `materials.grades_named_in`, which is the single definition of
    what counts as naming a grade. This used to be a second, weaker
    implementation: a token pass through `canonical_grade` and nothing else. It
    therefore saw none of the multi-word designations ("Alloy 91", "Carbon
    Steel", "Grade 22") and none of the separator-free ASME product
    specifications ("SA335P91", "SA182F22").

    That divergence had teeth. Every page of the ISGEC hardness report states
    its material as "SA335P91+SA234WP91", and this function returned [] for all
    of them — so no page established a grade, every row reported
    MISSING_MATERIAL, and the entire submittal graded nothing. Meanwhile
    `grades_named_in`, used elsewhere in the pipeline, read those same strings
    correctly. One question deserves one answer.

    Sorted for determinism: callers branch on `len(...) == 1` and index [0].
    """
    return sorted(grades_named_in(text))


_PLACEHOLDER_JOINTS = {"GENERAL JOINT", "GENERAL", "N/A", "-", "NONE", ""}
_JOINT_ID = re.compile(r"\bJ\s*-?\d+\b", re.IGNORECASE)


def _is_real_joint(joint: str) -> bool:
    """
    True only for an identified joint (e.g. "Joint J26").

    Extraction falls back to "General Joint" when no joint number is legible.
    Inheritance must not treat that as a grouping key — see resolve_page_rows.
    """
    j = (joint or "").strip().upper()
    if j in _PLACEHOLDER_JOINTS:
        return False
    return bool(_JOINT_ID.search(j))


def _explicit_grade(row: Any) -> tuple[Optional[str], str]:
    """Row-level evidence only: the row's own hint, then its own label text."""
    if g := canonical_grade(getattr(row, "material_hint", "") or ""):
        return g, "row_explicit"
    grades = _grades_in(getattr(row, "left_text", "") or "")
    if len(grades) == 1:
        return grades[0], "row_left_text"
    return None, "none"


def resolve_page_rows(page: Any) -> list[RowContext]:
    """
    Resolve every row on one page.

    Two passes: establish explicit grades first, then let unresolved rows
    inherit — so inheritance can only ever draw from explicit evidence, never
    from another inference.
    """
    rows = list(getattr(page, "table_rows", []) or [])
    n = len(rows)
    page_idx = getattr(page, "page_idx", 0)

    material: list[Optional[str]] = [None] * n
    provenance: list[str] = ["none"] * n

    # ── Pass 1: explicit row-level evidence ───────────────────────────
    for i, row in enumerate(rows):
        g, src = _explicit_grade(row)
        if g:
            material[i], provenance[i] = g, src

    explicit = {i for i in range(n) if provenance[i] in ("row_explicit", "row_left_text")}

    # ── Pass 2a: inherit within the joint group, from explicit rows only ──
    #
    # Only runs when the row belongs to a REAL joint. Extraction emits
    # "General Joint" as a placeholder when it cannot read a joint number, and
    # on page 13 every one of 12 rows carries that placeholder. Treating it as
    # a group turns joint-scoped inheritance into page-wide inheritance: a
    # single explicit "P91" on one HAZ2 row captured the whole page, including
    # the P22 pipe, on a page whose own test block reads DDEL-TH-P22-D3.
    #
    # A placeholder is an absence of grouping information, not a group.
    # A joint whose explicitly-graded rows DISAGREE is not one group.
    #
    # These submittals contain dissimilar joints — a P22 plate welded to P91
    # pipe — which the form labels correctly, one grade per row. The joint
    # number is printed once per group, so when the group boundary above it is
    # missed, two adjacent groups merge under a single joint number and this
    # pass then carries a grade from one group into the other.
    #
    # Measured on 221135LBP-HRDR001 (2026-08-20): the spool header that
    # separates groups is unread on 39 of 80 pages (38 of them on the OCR
    # route), producing 22 merged joints holding 119 rows. Page 22 came out as
    # nine rows under one "Joint J2", and a P91 pipe reading [224, 219, 226,
    # 216, 225, 222] inherited P22 from a plate in the NEXT group — 226 passes
    # P91's 248 and fails P22's 225, so it became a finding the specification
    # does not support. Half of every P22 verdict on that document (49 of 92)
    # rests on this inheritance rather than on a row's own grade cell.
    #
    # Nearest-row wins cannot resolve that: proximity is meaningless once the
    # boundary is gone. A row whose joint carries two grades is reported
    # UNRESOLVED, which sends it to MISSING_MATERIAL — visible and honest —
    # instead of being graded against a band picked by accident.
    joint_grades: dict[str, set[str]] = {}
    for j in explicit:
        jn = (getattr(rows[j], "joint_name", "") or "").strip()
        if jn and material[j]:
            joint_grades.setdefault(jn, set()).add(material[j])  # type: ignore[arg-type]

    # A row carrying an UNREADABLE grade cell states a grade of its own, even
    # though we cannot tell which. It must not inherit a neighbour's.
    #
    # This is the other half of the same failure. On page 35 the J48 weld cell
    # read "P94" instead of P91, so the joint appeared to hold exactly one grade
    # — the P22 of a plate that belongs to the NEXT group — and a P91 weld at
    # 245-260 was graded against P22's 225 ceiling, producing six failures the
    # specification permits to 280.
    unreadable_joints = {
        (getattr(r, "joint_name", "") or "").strip()
        for r in rows if getattr(r, "material_hint_raw", "")
    }

    for i, row in enumerate(rows):
        if material[i]:
            continue
        joint = (getattr(row, "joint_name", "") or "").strip()
        if not joint or not _is_real_joint(joint):
            continue
        if getattr(row, "material_hint_raw", ""):
            logger.info(
                f"page {page_idx + 1} {joint}: {getattr(row, 'component', '?')} has its own "
                f"Material Grade cell but it could not be read "
                f"({getattr(row, 'material_hint_raw')!r}); left UNRESOLVED rather than "
                f"inheriting another row's grade."
            )
            continue
        if len(joint_grades.get(joint, ())) > 1 or joint in unreadable_joints:
            logger.info(
                f"page {page_idx + 1} {joint}: explicit rows state "
                f"{sorted(joint_grades.get(joint, ()))} — the group boundary was probably missed, "
                f"so {getattr(row, 'component', '?')} is left UNRESOLVED rather than "
                f"inheriting one of them."
            )
            continue
        # Nearest explicitly-graded row sharing this joint.
        best: Optional[tuple[int, str]] = None
        for j in explicit:
            if (getattr(rows[j], "joint_name", "") or "").strip() != joint:
                continue
            if best is None or abs(j - i) < abs(best[0] - i):
                best = (j, material[j])  # type: ignore[arg-type]
        if best:
            material[i], provenance[i] = best[1], "joint_explicit"

    # ── Pass 2b: HAZ adjacency ────────────────────────────────────────
    # HAZ1 is heat-affected by the component above it; HAZ2 by the one below.
    # Bounded to an explicitly-graded neighbour so a HAZ cannot import a guess.
    for i, row in enumerate(rows):
        if material[i]:
            continue
        comp = (getattr(row, "component", "") or "").upper()
        if not comp.startswith("HAZ"):
            continue
        order = [i - 1, i + 1] if "HAZ1" in comp else [i + 1, i - 1]
        for j in order:
            if 0 <= j < n and j in explicit:
                material[i], provenance[i] = material[j], "haz_adjacent"  # type: ignore[assignment]
                break

    # ── Pass 3: page-level designation from the test block / material spec ──
    #
    # The test block is a material declaration for the page's test coupon
    # ("DDEL-TH-P22-D3" declares P22), so it is real evidence — not a naming
    # convention like the spool number. It applies only to rows that no
    # row-level evidence has already claimed, so an explicitly-P91 row on a P22
    # test-block page keeps its P91. That is the correct reading of page 13:
    # a P22 page containing one explicitly-P91 HAZ2 row.
    page_grades = _grades_in(
        " ".join(filter(None, [
            getattr(page, "material_spec", "") or "",
            getattr(page, "test_block_info", "") or "",
        ]))
    )
    # The designation must not contradict what the page's own rows state.
    #
    # "Exactly one grade in the header" was the whole guard, and it is not
    # enough: page 13 of 221135LBP-HRDR001 carries a P22 test block AND four
    # rows whose Material Grade column reads P91. A page holding explicit rows
    # of another grade is a MIXED page whatever its header says, and stamping
    # every unclaimed row with the header's grade there is a guess wearing the
    # clothes of evidence — one that cost 133 false findings, because P91 rows
    # judged against P22's 225 HB ceiling fail at readings P91 permits to 248.
    #
    # When the page contradicts itself, the unclaimed rows are left UNRESOLVED
    # and reported as such. "We could not establish this row's material" is a
    # true statement a reviewer can act on; "this row is P22" was not.
    explicit_grades = {material[i] for i in explicit if material[i]}
    contradicted = bool(page_grades) and bool(explicit_grades - {page_grades[0]})
    if len(page_grades) == 1 and not contradicted:
        for i in range(n):
            if not material[i]:
                material[i], provenance[i] = page_grades[0], "page_single"
    elif contradicted:
        logger.info(
            f"page {page_idx + 1}: header names {page_grades} but rows explicitly state "
            f"{sorted(explicit_grades)} — mixed page, so no page-wide material is applied. "
            f"Rows with no evidence of their own stay UNRESOLVED."
        )

    # ── Assemble ──────────────────────────────────────────────────────
    contexts: list[RowContext] = []
    for i, row in enumerate(rows):
        mat = material[i] or UNRESOLVED
        src = provenance[i] if material[i] else "none"
        contexts.append(RowContext(
            row_index=i,
            page_idx=page_idx,
            component=getattr(row, "component", "") or "",
            zone=zone_of(row),
            joint_name=getattr(row, "joint_name", "") or "",
            material=mat,
            provenance=src,
            confidence=PROVENANCE_CONFIDENCE.get(src, 0.0),
            readings=[r.value for r in getattr(row, "readings", []) if getattr(r, "value", None)],
        ))
    return contexts


def resolve_all(pages: list[Any]) -> tuple[list[RowContext], dict[str, int]]:
    """
    Resolve every row across the submittal.

    Returns (contexts, provenance_counts). The counts make the resolver's own
    reliability visible per run: a high `none` count means the document does not
    state its materials row-by-row and those rows need adjudication, rather than
    being quietly graded.
    """
    contexts: list[RowContext] = []
    for page in pages:
        contexts.extend(resolve_page_rows(page))

    counts: dict[str, int] = {}
    for c in contexts:
        counts[c.provenance] = counts.get(c.provenance, 0) + 1
    return contexts, counts


def rag_query_for(ctx: RowContext, test_method: str = "", procedure: str = "") -> str:
    """
    Contextual retrieval query for one row, per the review-planner design.

    Retrieving on "what is the hardness requirement?" cannot distinguish P22
    from P91. Naming material, component, zone, condition and procedure gives
    the retriever something to discriminate on.
    """
    from src.agents.materials import retrieval_terms
    parts = [
        f"Material: {retrieval_terms(ctx.material)}" if ctx.resolved else "Material: unspecified",
        f"Component: {ctx.component}",
        f"Zone: {ctx.zone}",
        "Condition: after PWHT",
        f"Test: {test_method or 'hardness'}",
    ]
    if procedure:
        parts.append(f"Procedure: {procedure}")
    parts.append("acceptance criteria maximum minimum permissible hardness")
    return " | ".join(parts)
