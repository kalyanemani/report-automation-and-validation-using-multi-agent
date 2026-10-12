from __future__ import annotations

"""
QDR·AI — LangGraph State Schema
──────────────────────────────────
Typed state for the multi-agent review workflow.
All agents read from and write to this shared state.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


# ── OCR Extractor Output Types ──────────────────────────────────────

@dataclass
class Reading:
    """A single hardness reading extracted from OCR."""
    value: int
    x: int
    y: int
    w: int
    h: int
    conf: int = 0


@dataclass
class TableRow:
    """A parsed table row from OCR spatial extraction."""
    ry: int                        # Y-coordinate of the row
    component: str                 # "Pipe", "HAZ1", "Weld", "HAZ2", "Elbow", etc.
    is_weld: bool                  # True if component is weld metal
    left_text: str                 # Raw left-column text
    joint_name: str                # "Joint J1", "Joint J27", "General Joint"
    readings: list[Reading] = field(default_factory=list)
    material_hint: str = ""        # Row-level material detected from OCR (e.g. "P91", "F22")
    # A Material Grade cell that holds something grade-shaped which could NOT be
    # read as a grade — "P94" for P91, say. Not a material, but not nothing
    # either: it is evidence that this row states a grade of its own, which
    # stops a neighbour's grade being inherited over the top of it.
    material_hint_raw: str = ""
    spool_name: str = ""           # Specific spool section (e.g. "HP-18-02", "RP-MULTI-02")


@dataclass
class PageOCRResult:
    """Complete OCR extraction result for a single PDF page."""
    page_idx: int
    full_text: str
    drawing_no: str                # "221135LBP-HP17", "1-191735LBP-HP2008"
    spool_no: str                  # "HP-17-01", "RP-03-02"
    table_rows: list[TableRow] = field(default_factory=list)
    test_block_info: str = ""      # "DDEL-TH-P91-D3"
    report_format: str = "DEE_TABULAR"  # "DEE_TABULAR" | "ISGEC_SINGLE" | "GOLDEN_PASS"
    has_astm_prefix: bool = False  # True if SA-335/SA-234/ASTM found
    # Per-page acceptance range parsed from the form header (e.g. "195 ~ 280 HB")
    # (min_hb, max_hb) or None if not explicitly stated on this page
    acceptance_range: Optional[tuple[int, int]] = None
    # Material specification as printed on the form (e.g. "SA335P91+SA234WP91 TYPE 1")
    material_spec: str = ""
    # Whether this page's table was read with enough confidence to grade against.
    # False means the extraction itself is suspect — its rows are reported
    # unevaluated rather than judged, because a misread page produces findings
    # indistinguishable from real ones. See ocr_extractor._assess_extraction.
    extraction_ok: bool = True
    extraction_warnings: list[str] = field(default_factory=list)


# ── Material Resolver Output Types ──────────────────────────────────

@dataclass
class MaterialAssignment:
    """LLM-resolved material assignment for a single table row."""
    page_idx: int
    row_index: int
    component: str
    material: str                  # "P91", "P22", "P11", "CS"
    reasoning: str                 # LLM's metallurgical reasoning


# ── Compliance Auditor Output Types ─────────────────────────────────

@dataclass
class ComplianceFinding:
    """LLM-evaluated compliance finding for a table row."""
    page_idx: int
    row_index: int
    component: str
    material: str
    failing_readings: list[int]
    limit_violated: str            # "above maximum 225 HBW" | "below minimum 195 HBW"
    severity: str                  # "HOLD" | "MAJOR"
    requirement_ref: str           # "700-006 Table 8 Note 2"
    reasoning: str                 # LLM's compliance reasoning


class EvidenceDict(TypedDict, total=False):
    text: str
    bbox: list[float]  # [x1, y1, x2, y2] in PDF coordinate space


class ContextDict(TypedDict, total=False):
    material: str
    joint: str
    location: str
    drawing: str
    spool: str


class RequirementDict(TypedDict, total=False):
    minimum: Optional[float]
    maximum: Optional[float]
    unit: str
    source: str
    status: str  # ESTABLISHED | PARTIALLY_ESTABLISHED | NOT_ESTABLISHED


class ValidationDict(TypedDict, total=False):
    status: str  # FAIL | PASS | NEEDS_REVIEW | NOT_APPLICABLE | WARNING
    reason: str


class CriticVerificationDict(TypedDict, total=False):
    evidence_supported: bool
    requirement_supported: bool
    calculation_correct: bool
    page_correct: bool
    final_decision: str  # APPROVE | REJECT | NEEDS_REVIEW


class AnnotationDict(TypedDict, total=False):
    type: str  # highlight_and_comment | highlight_only | warning_callout


class Finding(TypedDict, total=False):
    """A single compliance finding (conforming to 5-layer structured finding contract)."""
    # ── Structured 5-Layer Contract Schema ──
    finding_id: str                              # e.g. "F-014"
    page: int                                    # 1-indexed page number
    evidence: Any                                # EvidenceDict or legacy str
    context: ContextDict                         # Material, joint, location context
    requirement: Any                            # RequirementDict or legacy str
    validation: ValidationDict                   # Deterministic validation status & reason
    annotation: AnnotationDict                   # Annotation directive
    confidence_score: int                        # Evidence quality score (0 to 9)
    confidence_level: str                        # "HIGH" | "MEDIUM" | "LOW"
    critic_verification: CriticVerificationDict  # Stage 2 Critic verification result

    # ── Legacy / Flat UI Compatibility Fields ──
    id: int
    section: str
    type: str          # HOLD | MAJOR | MINOR | INFO | NEEDS_REVIEW
    comment: str
    action: str
    closeout: str      # Required before approval | Required before next revision | Advisory
    checklist_item: Optional[str]
    target_term: Optional[str]
    box: Optional[tuple[int, int, int, int]]
    bbox: Optional[list[float]]


class GenParamCheck(TypedDict, total=False):
    """A document control parameter check."""
    param: str
    expected: str
    found: str
    status: str        # PRESENT & MATCHES | PRESENT — MISMATCH | MISSING | NOT VERIFIABLE
    comment: Optional[str]


class ReviewState(TypedDict, total=False):
    """
    Shared state for the review workflow.
    """
    # ── Input ────────────────────────────────────────────────────────
    submittal_text: str
    submittal_filename: str
    submittal_filepath: str

    # Review guide / discipline
    guide_code: str
    guide_name: str
    cat_code: str
    cat_label: str

    # Review metadata
    supplier: str
    doc_no: str
    revision: str
    reviewer: str

    # General document control params (input)
    gen_params_input: dict[str, str]

    # ── Classification ──────────────────────────────
    doc_type: str              # supplier_submittal | review_guide | specification | etc.
    test_method: str           # Rockwell HRC | Brinell HBW | etc.
    material_grade: str        # Alloy 91 | Grade 92 | etc.

    # ── Requirements ────────────────────────────────
    requirements: list[str]    # Extracted checklist items from review guide
    requirement_sources: list[dict[str, str]]  # {requirement, source_doc, section, page}

    # ── Retrieved context ────────────────────────────────────────────
    detected_materials: list[str]   # Material grades OCR actually observed (drives RAG queries)
    retrieved_criteria: list[str]   # Top-k criteria from vector search
    retrieved_context: list[dict]   # Full retrieval results with metadata

    # ── Agent Pipeline Intermediates ─────────────────────────────────
    page_ocr_results: list[PageOCRResult]         # OCR & Table extraction output
    hardness_rules_schema: dict                   # Analyzer LLM: limits established from retrieved clauses
    uncovered_materials: list[str]                 # Materials present but with no established limits (unevaluated)
    compliance_results: list[dict]                 # Per-row deterministic verdicts (status/reason/limits used)
    rows_needing_adjudication: list[dict]          # Rows whose UNRESOLVED reason genuinely needs an LLM
    raw_rule_failures: list[dict]                 # Python Rule Engine: Deterministic numerical failures
    grouped_findings: list[dict]                  # Analyzer LLM: grouped & analyzed defect patterns
    critic_verifications: list[dict]              # Report synthesizer: QA Critic validation results
    material_assignments: list[MaterialAssignment] # Legacy/compat output
    compliance_findings: list[ComplianceFinding]   # Legacy/compat output

    # ── Compliance Review ───────────────────────────────────────────
    gen_param_checks: list[GenParamCheck]
    findings: list[Finding]
    visual_observations: list[dict]

    # ── Gap Analysis ────────────────────────────────────────────────
    gaps: list[dict[str, str]]  # {layer, description, severity, evidence}
    missing_fields: list[str]
    ambiguities: list[str]

    # ── Final Report ────────────────────────────────────────────────
    disposition: str    # APPROVED | APPROVED WITH COMMENTS | REVISE AND RESUBMIT | REJECTED
    score: int          # 0–100
    verdict: str        # One-sentence status
    summary: str        # 3–4 sentence summary

    # ── Diagnostics & Latency Metrics ────────────────────────────────
    pipeline_metrics: list[dict]
    llm_traces: list[dict]
    llm_reasoning_traces: list[dict]  # Full LLM reasoning for auditability

    # ── Agent communication ──────────────────────────────────────────
    messages: Annotated[list, add_messages]
    current_agent: str
    errors: list[str]
