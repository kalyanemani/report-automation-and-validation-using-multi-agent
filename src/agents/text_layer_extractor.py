from __future__ import annotations

"""
QDR·AI — Text-Layer Table Extraction
─────────────────────────────────────
Reads hardness tables straight out of a PDF's text layer, for the pages that
have one, instead of rendering them to pixels and OCR-ing them back into text.

WHY THIS EXISTS
───────────────
Every extraction defect fixed on 2026-08-20 was an OCR artefact, and none of
them exist in the text layer. Measured on page 13 of 221135LBP-HRDR001:

    what the page states     OCR produced          text layer holds
    ─────────────────────    ──────────────────    ─────────────────
    P91 (4 grade cells)      "poi", "Pot"          "P91" x5, "P22" x1
    J25 / J26 / J27          all collapsed to J25  "J25", "J26", "J27"
    Haz1 x3                  "Hazt" -> row dropped "Haz1" x3
    spool RP-03-08           "P22-D3"              "RP-03-08" x3

Worse, the damage was silent: a Haz1 row that matched no component pattern was
simply dropped, and grade cells landing a row late produced findings that looked
exactly like real ones. `find_tables()` returns the same page as rows and
columns with the grade, joint and size already on their own row — no banding, no
offset fitting, no digit repair.

WHY IT IS PER PAGE, NOT PER DOCUMENT
────────────────────────────────────
221135LBP-HRDR001 is mixed: pages 1, 2, 3 and 41 carry no text at all (0 words),
while pages 13 and 80 carry 381 and 283 words. A per-document choice is wrong
either way, so the decision is made per page from what the page actually offers.
241135LBP-HRDR001 averages 2 words per page and stays on the OCR path, which is
the right tool for it.

CONTRACT
────────
Returns the same `PageOCRResult` the OCR path returns, so everything downstream —
the context resolver, compliance check, limit verification, annotator — is
untouched and cannot tell which path produced a page. When this module cannot
extract with confidence it returns None, and the caller falls back to OCR.
"""

import logging
import re
from typing import Any, Optional

import fitz

from src.agents.form_layout import map_columns_semantically
from src.agents.materials import canonical_grade
from src.agents.state import PageOCRResult, Reading, TableRow

logger = logging.getLogger("qdr_ai.agent.text_layer")

# A page needs a usable amount of text before its layer is worth trusting. Below
# this it is a scan with a stray caption or a stamp, and OCR is the honest route.
MIN_WORDS = 40

# Column roles, read from the table's own header cells rather than assumed from
# pixel positions. The header names its columns; every vendor form states them
# somewhere, which is what makes this survive a supplier nobody has coded for.
# Synonyms are cheap and the LLM fallback is not, so this list is deliberately
# generous: every header wording handled here is a supplier that never needs a
# model call. The fallback exists for wordings nobody anticipated, not for
# ordinary variation like "Matl." or "Rdg".
_ROLE_HEADERS: list[tuple[str, re.Pattern]] = [
    ("component", re.compile(
        r"item\s*descript|descript|component|location|element|part", re.I)),
    ("material", re.compile(
        r"material\s*grade|^\s*material|^\s*grade|mat'?l|matl|base\s*mat", re.I)),
    ("joint", re.compile(r"joint|^\s*jt|wld\s*no|weld\s*no", re.I)),
    ("size", re.compile(r"size|thk|thickness|inch|dia|nps", re.I)),
]

# Individual reading columns: "H1".."H5", "AVG". AVG is kept because the
# specifications grade it explicitly — 241135LBP-HRDP001 states limits for the
# "average hardness reading" — and because the OCR path already includes it, so
# excluding it here would make the two paths disagree for reasons unrelated to
# extraction quality.
_READING_HEADER = re.compile(
    r"^\s*(?:H\s*\d{1,2}|R\s*\d{1,2}|RDG\.?\s*\d{0,2}|READING\s*\d{0,2}"
    r"|AVG\.?|AVERAGE|MEAN|BHN|HB|HBW)\s*$", re.I)

_HARDNESS_HINT = re.compile(r"hardness|brinell|\bhb\b|\bbhn\b", re.I)


def _cells(row: Any) -> list[str]:
    return [(c or "").replace("\n", " ").strip() for c in row]


def _column_roles(rows: list[list[str]]) -> dict[str, Any]:
    """
    Map column index -> role by reading the header cells.

    Header text is spread over several rows on these forms ("Hardness (HB)",
    then "Hardness value at 0°", then "H1 H2 H3 H4 H5 AVG"), so every row above
    the first data row is scanned and the roles accumulated.
    """
    roles: dict[str, Any] = {"readings": []}
    for row in rows[:8]:
        for idx, text in enumerate(_cells(row)):
            if not text:
                continue
            if _READING_HEADER.match(text):
                if idx not in roles["readings"]:
                    roles["readings"].append(idx)
                continue
            for role, pattern in _ROLE_HEADERS:
                if role not in roles and pattern.search(text):
                    roles[role] = idx
    roles["readings"].sort()
    return roles


def _plausible_reading(text: str) -> Optional[int]:
    """A hardness value, or None. Deliberately narrow: cells hold sizes and notes too."""
    t = (text or "").strip()
    if not re.fullmatch(r"\d{2,3}", t):
        return None
    value = int(t)
    return value if 50 <= value <= 600 else None


def _select_table(page: fitz.Page) -> Optional[tuple[Any, list[list[str]], dict[str, Any]]]:
    """
    The table on this page that holds hardness readings, with its column roles.

    A page can carry several tables (title blocks, dimension legends). The one
    wanted is the one whose header mentions hardness AND that has several
    reading columns and a description column — anything less cannot be graded
    and is better left to the fallback than guessed at.
    """
    try:
        found = page.find_tables()
    except Exception as exc:                                    # pragma: no cover
        logger.debug(f"find_tables failed on page {page.number + 1}: {exc}")
        return None

    best = None
    for table in getattr(found, "tables", []) or []:
        try:
            rows = table.extract()
        except Exception:
            continue
        if not rows:
            continue
        header_blob = " ".join(" ".join(_cells(r)) for r in rows[:8])
        if not _HARDNESS_HINT.search(header_blob):
            continue
        roles = _column_roles(rows)
        if "component" not in roles or len(roles["readings"]) < 3:
            # The header matcher does not recognise this layout. Rather than
            # guess a column — which is how a hardcoded position fails, silently
            # and confidently — ask what the columns are, once per form, and
            # verify the answer against the header text. See form_layout.
            roles = _semantic_roles(rows) or roles
            if "component" not in roles or len(roles.get("readings", [])) < 3:
                continue
        score = len(roles["readings"]) + len(rows)
        if best is None or score > best[0]:
            best = (score, table, rows, roles)
    return (best[1], best[2], best[3]) if best else None


def _semantic_roles(rows: list[list[str]]) -> Optional[dict]:
    """Column roles for an unrecognised layout, asked once per form and cached."""
    header_idx = None
    for i, row in enumerate(rows[:8]):
        if sum(1 for c in _cells(row) if c) >= 3:
            header_idx = i
            break
    if header_idx is None:
        return None
    headers = _cells(rows[header_idx])

    sample = None
    for row in rows[header_idx + 1:]:
        cells = _cells(row)
        if sum(1 for c in cells if re.fullmatch(r"\d{2,3}", c or "")) >= 3:
            sample = cells
            break
    if sample is None:
        return None

    roles = map_columns_semantically(headers, sample)
    if not roles:
        return None
    return {"component": roles["component"], "readings": list(roles["readings"]),
            **({"material": roles["material"]} if "material" in roles else {}),
            **({"joint": roles["joint"]} if "joint" in roles else {})}


def _cell_rect(table: Any, row_idx: int, col_idx: int) -> Optional[fitz.Rect]:
    """Bounding box of one cell, when the table finder exposes it."""
    try:
        cell = table.rows[row_idx].cells[col_idx]
        return fitz.Rect(cell) if cell else None
    except Exception:
        return None


def extract_page_text_layer(
    page: fitz.Page,
    page_idx: int,
    report_format: str,
    full_text: str,
    drawing_no: str,
    spool_no: str,
    test_block_info: str,
    acceptance_range: Optional[tuple[int, int]],
    material_spec: str,
    has_astm: bool,
    detect_component,
    spool_header_re: re.Pattern,
) -> Optional[PageOCRResult]:
    """
    Build a PageOCRResult from the page's text layer, or None to fall back.

    Header/identifier parsing is passed in rather than duplicated, so a page
    reports the same drawing number, test block and acceptance range no matter
    which path read its table.
    """
    selected = _select_table(page)
    if selected is None:
        return None
    table, rows, roles = selected

    c_component = roles["component"]
    c_material = roles.get("material")
    c_joint = roles.get("joint")
    reading_cols = roles["readings"]

    table_rows: list[TableRow] = []
    group_start_flags: list[bool] = []
    pending_group_start = True
    current_joint = "General Joint"
    current_spool = spool_no

    for r_idx, raw in enumerate(rows):
        cells = _cells(raw)
        if c_component >= len(cells):
            continue
        label = cells[c_component]

        # A spool header opens a new joint group, exactly as on the OCR path:
        # the joint number is printed once per group, on the weld row, so the
        # rows above it would otherwise inherit the previous group's joint.
        #
        # Here the header occupies a row of its OWN — it has no readings, so it
        # is skipped below and never becomes a TableRow. The OCR path could find
        # it in a data row's text because banding merged the two; this path must
        # remember it and attach it to the next row that IS emitted.
        header_match = spool_header_re.search(label)
        if header_match:
            current_spool = header_match.group(1).upper().replace(".", "-")
            current_joint = "General Joint"
            pending_group_start = True

        component, is_weld = detect_component(label)
        if component is None:
            continue

        readings: list[Reading] = []
        for c_idx in reading_cols:
            if c_idx >= len(cells):
                continue
            value = _plausible_reading(cells[c_idx])
            if value is None:
                continue
            rect = _cell_rect(table, r_idx, c_idx)
            readings.append(Reading(
                value=value,
                x=int(rect.x0) if rect else 0,
                y=int(rect.y0) if rect else 0,
                w=int(rect.width) if rect else 0,
                h=int(rect.height) if rect else 0,
                conf=100,          # the text layer is not a guess
            ))
        if not readings:
            continue

        material_hint = ""
        if c_material is not None and c_material < len(cells):
            material_hint = canonical_grade(cells[c_material]) or ""

        if c_joint is not None and c_joint < len(cells):
            joint_cell = cells[c_joint].strip()
            m = re.fullmatch(r"J?\s*-?\s*(\d{1,3})", joint_cell, re.I)
            if m:
                current_joint = f"Joint J{int(m.group(1))}"

        row_rect = _cell_rect(table, r_idx, c_component)
        group_start_flags.append(pending_group_start)
        pending_group_start = False
        table_rows.append(TableRow(
            ry=int(row_rect.y0) if row_rect else r_idx,
            component=component,
            is_weld=is_weld,
            left_text=" ".join(x for x in cells[:max(1, c_component + 3)] if x),
            joint_name=current_joint,
            readings=readings,
            material_hint=material_hint,
            spool_name=current_spool,
        ))

    if not table_rows:
        return None

    # One joint number applies to its whole group — see the OCR path for why the
    # cell is printed only on the weld row.
    starts = [i for i, flag in enumerate(group_start_flags) if flag]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    for lo, hi in zip(starts, starts[1:] + [len(table_rows)]):
        joints = {r.joint_name for r in table_rows[lo:hi] if r.joint_name != "General Joint"}
        if len(joints) == 1:
            joint = joints.pop()
            for row in table_rows[lo:hi]:
                row.joint_name = joint

    logger.info(
        f"page {page_idx + 1}: read from the text layer — {len(table_rows)} rows, "
        f"{sum(len(r.readings) for r in table_rows)} readings, "
        f"columns component={c_component} material={c_material} joint={c_joint} "
        f"readings={reading_cols}"
    )
    return PageOCRResult(
        page_idx=page_idx,
        full_text=full_text,
        drawing_no=drawing_no,
        spool_no=spool_no,
        table_rows=table_rows,
        test_block_info=test_block_info,
        report_format=report_format,
        has_astm_prefix=has_astm,
        acceptance_range=acceptance_range,
        material_spec=material_spec,
    )


def page_has_text_layer(page: fitz.Page) -> bool:
    """Whether this page offers enough text to be worth reading directly."""
    try:
        return len(page.get_text("words") or []) >= MIN_WORDS
    except Exception:
        return False
