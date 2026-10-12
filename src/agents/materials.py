from __future__ import annotations

"""
QDR·AI — Material Grade ↔ ASME P-Number Mapping
────────────────────────────────────────────────
The submittals name materials by grade designation (P91, P22, P11, carbon
steel). The governing specifications frequently tabulate hardness limits by
ASME BPVC Section IX **P-Number group** instead:

    | Material | Hardness on Location | Brinell Hardness(HB) |
    | P4       | Base and Weld        | 225 HB Max           |
    | P5A      | Base and Weld        | 225 HB Max           |

`P4` there is not "P-11 mis-typed" — it is P-Number 4, the group that contains
1.25Cr-0.5Mo (P11). Nothing in that table contains the literal string "P11", so
a retrieval query for "P11 hardness limit" matches it only by luck, and an
agent asked to extract "P11's limit" from it has to already know the grouping.

This module supplies that mapping so:
  * retrieval queries can carry both vocabularies, and
  * the analyzer prompt can state the identities, and
  * limits returned keyed by either vocabulary canonicalise to one form.

Supplying the P-Number identity is NOT the same as supplying the answer. The
identity is a fixed materials-engineering fact; the limits still have to be
read out of the retrieved specification text.
"""

import re
from typing import Iterable, Optional

# Canonical grade → P-Number group and the terms that denote it in spec text.
MATERIAL_P_NUMBERS: dict[str, dict] = {
    "P91": {
        "p_no": "15E",
        "composition": "9Cr-1Mo-V",
        "common_name": "Alloy 91",
        # Terms that, appearing in spec text, denote this material.
        "aliases": (
            "P91", "F91", "T91", "WP91", "SA-335 P91", "SA-182 F91", "SA-234 WP91",
            "SA-387 P91", "GRADE 91", "GR. 91", "ALLOY 91", "9CR-1MO-V", "9CR",
            "P-NO. 15E", "P-NO 15E", "PNO 15E", "P15E", "P 15E", "15E",
        ),
    },
    "P22": {
        "p_no": "5A",
        "composition": "2.25Cr-1Mo",
        "common_name": "Grade 22",
        "aliases": (
            "P22", "F22", "T22", "WP22", "SA-335 P22", "SA-182 F22",
            "GRADE 22", "GR. 22", "2.25CR-1MO", "2-1/4CR-1MO", "2.25CR",
            "P-NO. 5A", "P-NO 5A", "PNO 5A", "P5A", "P 5A", "5A",
        ),
    },
    "P11": {
        "p_no": "4",
        "composition": "1.25Cr-0.5Mo",
        "common_name": "Grade 11",
        "aliases": (
            "P11", "F11", "T11", "WP11", "SA-335 P11", "SA-182 F11",
            "GRADE 11", "GR. 11", "1.25CR-0.5MO", "1-1/4CR-1/2MO", "1.25CR",
            "P-NO. 4", "P-NO 4", "PNO 4", "P4", "P 4",
        ),
    },
    "CS": {
        "p_no": "1",
        "composition": "carbon steel",
        "common_name": "Carbon Steel",
        "aliases": (
            "CS", "CARBON STEEL", "A106", "SA-106", "A105", "SA-105",
            "WPB", "A234 WPB", "SA-234 WPB", "GR. B", "GRADE B",
            "P-NO. 1", "P-NO 1", "PNO 1", "P1", "P 1",
        ),
    },
}

# Longest aliases first so "P-NO. 15E" wins over "15E", and "2.25CR-1MO" over
# "2.25CR". Without this, a short alias can claim text that belongs to another
# material — notably bare "P1" inside "P11", which is why matching is
# whole-token below rather than substring.
_ALIAS_TO_GRADE: list[tuple[str, str]] = sorted(
    ((alias.upper(), grade) for grade, meta in MATERIAL_P_NUMBERS.items() for alias in meta["aliases"]),
    key=lambda kv: -len(kv[0]),
)


def canonical_grade(token: str) -> Optional[str]:
    """
    Map any grade designation or P-Number term to its canonical grade key.

    Returns None when nothing matches, rather than guessing — a wrong grade
    selects the wrong hardness band and silently changes pass/fail verdicts.

        canonical_grade("P5A")       -> "P22"
        canonical_grade("P-No. 15E") -> "P91"
        canonical_grade("SA-335 P11")-> "P11"
        canonical_grade("Inconel")   -> None
    """
    if not token:
        return None
    t = token.upper().strip()

    # Exact canonical key.
    if t in MATERIAL_P_NUMBERS:
        return t

    # Whole-token alias match. Padding with spaces and requiring a non-alnum
    # boundary keeps "P1" from matching inside "P11" / "P15E".
    padded = f" {t} "
    for alias, grade in _ALIAS_TO_GRADE:
        if alias == t:
            return grade
        a = f" {alias} "
        if a in padded:
            return grade
        # Alias appearing as a delimited token inside a longer string.
        idx = t.find(alias)
        while idx != -1:
            before = t[idx - 1] if idx > 0 else " "
            after_i = idx + len(alias)
            after = t[after_i] if after_i < len(t) else " "
            if not before.isalnum() and not after.isalnum():
                return grade
            idx = t.find(alias, idx + 1)
    return None


def grades_named_in(text: str) -> set[str]:
    """
    Every grade a block of text names, matched safely.

    Two passes, because neither alone is sufficient:

      * single tokens via `canonical_grade` — catches "P91", "P5A", "SA-335"
        with whole-token boundaries, so "P1" cannot fire inside "P15E";
      * multi-word aliases via word-boundary regex — catches "Alloy 91",
        "Grade 22", "Carbon Steel", "2.25Cr-1Mo", which a token-by-token pass
        splits apart and therefore misses entirely.

    Plain substring matching over the whole text is NOT used: the alias lists
    contain short designations ("5A", "P1", "P4") that fire on unrelated runs of
    characters, which is how P91 clauses were selected as P22 evidence.
    """
    if not text:
        return set()
    found: set[str] = set()

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-/]*", text):
        g = canonical_grade(token)
        if g:
            found.add(g)

    upper = text.upper()
    for grade, meta in MATERIAL_P_NUMBERS.items():
        if grade in found:
            continue
        for alias in meta["aliases"]:
            if " " not in alias:
                continue  # single tokens already covered above
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])", upper):
                found.add(grade)
                break

    # ASME product specifications written WITHOUT separators.
    #
    # The ISGEC hardness report states its material as "SA335P91+SA234WP91" or
    # "SA182F22+SA234WP91" — no space, no hyphen. Whole-token matching cannot
    # see the grade there: in "SA335P91" the character before "P91" is "5", so
    # the boundary test fails, exactly as it is designed to for "P1" inside
    # "P15E". The two cases look identical to a token matcher and are opposites
    # in meaning.
    #
    # Anchoring on the product-specification prefix distinguishes them. "SA335"
    # followed by "P91" IS a grade statement; "P15E" is not "P1" followed by
    # anything. Measured 2026-08-20 on 241135LBP-HRDR001: every ISGEC page
    # reported its material this way, so no page resolved a grade and every row
    # came back MISSING_MATERIAL — the whole document graded nothing.
    for m in re.finditer(r"(?:SA|A)-?\s?\d{3}\s*-?\s*((?:W?P|F|T)\s*-?\s*\d{1,3}[A-Z]?)", upper):
        grade = canonical_grade(m.group(1).replace(" ", "").replace("-", ""))
        if grade:
            found.add(grade)
    return found


_PROJECT_PREFIX = re.compile(r"\b(\d{6})LBP\b", re.IGNORECASE)


def project_prefix(text: str) -> Optional[str]:
    """
    The project number embedded in a drawing, spool or document identifier.

    "221135LBP-HRDR001" -> "221135"      (the submittal under review)
    "221135LBP-HRDP001" -> "221135"      (that project's specification)
    "241135LBP-HRDP001" -> "241135"      (a DIFFERENT project's specification)
    "700-016-R2 ..."    -> None          (a general, project-independent spec)

    This matters because the corpus holds specifications from more than one
    project, and they state DIFFERENT limits for the same material. Measured on
    2026-08-20: 241135LBP-HRDP001 states "235 HB (max) ... for (P22 or
    equivalent)" while 221135LBP-HRDP001 tabulates "P5A | Base and Weld |
    225 HB Max". Retrieval returned both for a 221135 submittal, so the limit a
    row was graded against depended on which clause the model happened to read.
    """
    if not text:
        return None
    m = _PROJECT_PREFIX.search(text)
    return m.group(1) if m else None


def clause_project_conflicts(source: str, submittal_project: Optional[str]) -> bool:
    """
    True when a retrieved clause belongs to a different project than the submittal.

    A clause with no project number (700-006, 700-016, ASME) is general and
    always applicable, so it never conflicts. A clause carrying a different
    project's number is governed by that project's contract, not this one.
    """
    if not submittal_project:
        return False          # unknown submittal project — cannot judge, allow
    clause_project = project_prefix(source or "")
    if clause_project is None:
        return False          # general specification — applies
    return clause_project != submittal_project


def canonicalise_limits(limits: dict) -> tuple[dict, dict[str, str]]:
    """
    Rewrite a limits schema so its keys are canonical grades.

    An agent reading a P-Number table may legitimately return limits keyed
    "P5A" or "P-No. 4". Left alone those keys never match the rule engine's
    material names, so the readings would be skipped as "no established
    limits" — a silent hole. Returns (canonicalised, renames_applied).
    """
    out: dict = {}
    renames: dict[str, str] = {}
    for key, value in (limits or {}).items():
        if not isinstance(value, dict):
            continue
        grade = canonical_grade(str(key))
        if grade is None:
            # Preserve unknown keys untouched: dropping them would hide that
            # the agent reported a material we could not interpret.
            out[str(key).upper()] = value
            continue
        if grade != str(key).upper():
            renames[str(key)] = grade
        if grade in out and isinstance(out[grade], dict):
            out[grade].update(value)
        else:
            out[grade] = value
    return out, renames


def retrieval_terms(grade: str) -> str:
    """
    Both-vocabulary query fragment for one grade.

    A query naming only "P22" cannot match a table row labelled "P5A", which is
    where the limits actually live in this corpus.
    """
    meta = MATERIAL_P_NUMBERS.get(grade.upper())
    if not meta:
        return grade
    return (
        f"{grade} {meta['common_name']} {meta['composition']} "
        f"P-No. {meta['p_no']} P{meta['p_no']}"
    )


def p_number_legend(grades: Iterable[str]) -> str:
    """
    Human-readable identity legend for the analyzer prompt.

    States only the grade ↔ P-Number ↔ composition identities, never any
    hardness value, so the agent can read a P-Number-keyed table without being
    handed the limits it is supposed to extract.
    """
    lines = []
    for g in grades:
        meta = MATERIAL_P_NUMBERS.get(str(g).upper())
        if meta:
            lines.append(
                f"- {str(g).upper()} ({meta['common_name']}, {meta['composition']}) "
                f"is ASME P-Number {meta['p_no']}; specification tables may label it "
                f"\"P-No. {meta['p_no']}\" or \"P{meta['p_no']}\"."
            )
    return "\n".join(lines)
