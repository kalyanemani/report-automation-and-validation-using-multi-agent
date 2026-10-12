from __future__ import annotations

"""
QDR·AI — Documented Default Hardness Limits
────────────────────────────────────────────
What remains of the former "LLM Call #1: Requirement Parser Agent".

That agent asked the model to return the hardness limits, but its own system
prompt listed those limits verbatim, so a successful call only ever echoed its
input — proven on 2026-08-20, when one run using this constant and another
using a successful LLM parse produced byte-identical findings. Establishing
limits is now part of the analyzer's job (see finding_analyzer_agent.py), which
reads them out of the RAG-retrieved clauses with a citation for each.

The constant below survives as the DOCUMENTED DEFAULT used only when
STRICT_LLM_MODE is off and the analyzer fails. It is not authoritative: the
governing specifications tabulate limits by ASME P-Number, and at least one
value here (P22 max 235) may be more permissive than the corpus states
(the P5A row reads 225 HB Max). Treat it as a legacy fallback, not as truth.
"""

DEFAULT_HARDNESS_RULES = {
    "P91": {
        "base_metal": {"min": 195, "max": 248, "unit": "HBW", "ref": "700-022 / DEETH-QCP-EL-003"},
        "haz": {
            "min": 195,
            "max": 248,
            "single_reading_max": 250,
            "unit": "HBW",
            "ref": "700-022 / DEETH-QCP-EL-003 / ASME B31.1",
        },
        "weld": {
            "min": 195,
            "max": 280,
            "single_reading_max": 315,
            "unit": "HBW",
            "ref": "700-006 Table 8 Note 2 & Note 3",
        },
    },
    "P22": {
        "base_metal": {"min": None, "max": 235, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
        "haz": {"min": None, "max": 235, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
        "weld": {"min": None, "max": 235, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
    },
    "P11": {
        "base_metal": {"min": None, "max": 225, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
        "haz": {"min": None, "max": 225, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
        "weld": {"min": None, "max": 225, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
    },
    "CS": {
        "base_metal": {"min": None, "max": 200, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
        "haz": {"min": None, "max": 200, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
        "weld": {"min": None, "max": 200, "unit": "HBW", "ref": "241135LBP-HRDP001 Section II / ASME B31.1"},
    },
}
