from __future__ import annotations

# QDR·AI — Senior QA Inspector Application
# Plant Engineering Supplier Document Inspection & Review

import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import run_inspection_pipeline
from src.agents.hydro_orchestrator import run_hydrostatic_inspection_pipeline
from src.config import (
    DOCUMENTS_DIR,
    PARSED_DIR,
    LOGS_DIR,
    OLLAMA_MODEL,
    OLLAMA_VISION_MODEL,
    BACKEND_PORT,
    STRICT_LLM_MODE,
    EmbeddingModelError,
    LLMAgentError,
)
from src.ingestion.annotator import annotate_pdf_with_findings
from src.ingestion.parser import parse_pdf, extract_checklist_items, parse_review_guide
from src.ingestion.pipeline import IngestionPipeline
from src.knowledge_graph.hybrid_retriever import get_hybrid_retriever
from src.utils import run_lock
from src.utils import progress as pipeline_progress
from src.utils import usage_log
from src.utils.llm_tracer import current_run_id as current_llm_run_id
from src.utils.logger import pipeline_logger


# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="QDR·AI | Plant Engineering · Supplier Document Inspection & Review",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Screenshot Color Theme ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Clean Light Theme App Background */
    .stApp {
        background-color: #ffffff;
        color: #1e293b;
    }

    .stApp > header {
        display: none;
    }
    .main .block-container {
        padding-top: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Top Navbar Header */
    .qdr-top-navbar {
        background-color: #0f172a;
        color: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 28px;
        border-bottom: 3px solid #f59e0b;
        margin: 0rem -2rem 1.5rem -2rem;
    }
    .qdr-nav-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .qdr-logo {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f59e0b;
    }
    .qdr-logo span {
        color: #ffffff;
    }
    .qdr-nav-sep {
        color: #475569;
        font-size: 1.1rem;
        font-weight: 300;
    }
    .qdr-nav-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .qdr-nav-right {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #10b981;
        letter-spacing: 0.5px;
    }
    .qdr-status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        display: inline-block;
    }

    /* Tab bar navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 14px;
        background-color: transparent;
        color: #64748b;
        font-weight: 700;
        font-size: 0.84rem;
        border: none;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        border-bottom: 3px solid #f59e0b !important;
    }

    /* Parameter Cards */
    .param-card {
        background-color: #ffffff;
        color: #1f2937;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .param-card-title {
        font-size: 0.72rem;
        font-weight: 800;
        color: #d97706;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .param-card-val {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .param-card-status {
        font-size: 0.75rem;
        font-weight: 700;
        color: #059669;
        margin-top: 6px;
    }

    /* Finding card styling */
    .finding-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #f97316;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .finding-card-hold {
        border-left-color: #dc2626;
    }
    .finding-card-major {
        border-left-color: #ea580c;
    }
    .finding-card-minor {
        border-left-color: #d97706;
    }

    /* Badges */
    .badge-hold, .badge-rejected {
        background-color: #dc2626;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 800;
    }
    .badge-major, .badge-revise {
        background-color: #ea580c;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 800;
    }
    .badge-minor {
        background-color: #d97706;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 800;
    }
    .badge-approved {
        background-color: #059669;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 800;
    }

    /* Section Bar */
    .section-header-bar {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 8px 14px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        border-radius: 4px;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    /* Sidebar nav / guide / mode cards (rendered as styled st.button) */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.86rem;
        padding: 10px 12px;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        border-left: 4px solid #f59e0b;
        color: #b45309;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] + div[data-testid="stCaptionContainer"] {
        margin-top: -6px;
        margin-bottom: 10px;
        padding-left: 13px;
    }
    .qdr-sidebar-section-label {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #64748b;
        margin: 18px 0 8px 0;
    }

    /* Clean section label + description (replaces amber guidance-banner) */
    .qdr-section-label {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #2563eb;
        margin-bottom: 6px;
    }
    .qdr-section-desc {
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 16px;
    }
    .qdr-upload-header {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .qdr-paste-divider {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Top Navbar Header ─────────────────────────────────────────────────
st.markdown("""
<div class="qdr-top-navbar">
    <div class="qdr-nav-left">
        <div class="qdr-logo">QDR<span>·AI</span></div>
        <div class="qdr-nav-sep">|</div>
        <div class="qdr-nav-title">PLANT ENGINEERING · SUPPLIER DOCUMENT INSPECTION & REVIEW</div>
    </div>
    <div class="qdr-nav-right">
        <span class="qdr-status-dot"></span>
        AI Engine Online
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────
# First script run of a browser session: mint the usage-log session id and
# record that the tool was opened. Streamlit re-executes this file on every
# interaction, so the `not in` guard is what keeps it to one event per session.
if "usage_session_id" not in st.session_state:
    st.session_state["usage_session_id"] = usage_log.new_session_id()
    usage_log.log_session_start(st.session_state["usage_session_id"])

if "submittal_bytes" not in st.session_state:
    st.session_state["submittal_bytes"] = None
if "submittal_filename" not in st.session_state:
    st.session_state["submittal_filename"] = ""
if "submittal_text" not in st.session_state:
    st.session_state["submittal_text"] = ""
if "submittal_filepath" not in st.session_state:
    st.session_state["submittal_filepath"] = ""
if "extracted_metadata" not in st.session_state:
    st.session_state["extracted_metadata"] = {}

if "review_guide_bytes" not in st.session_state:
    st.session_state["review_guide_bytes"] = None
if "review_guide_filename" not in st.session_state:
    st.session_state["review_guide_filename"] = ""
if "checked_criteria" not in st.session_state:
    st.session_state["checked_criteria"] = []
if "unchecked_criteria" not in st.session_state:
    st.session_state["unchecked_criteria"] = []


def extract_metadata_from_submittal(text: str, filename: str) -> dict:
    """Dynamically extracts supplier, document number, revision, project, and MAWP from text."""
    meta = {}
    clean_fn = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)

    # Document Number
    doc_m = re.search(r'(?:REPORT\s*NO\.?|DOCUMENT\s*NO\.?|DOC\s*NO\.?|TEST\s*REPORT)[\s:=]*([A-Za-z0-9\-_/]+)', text, re.IGNORECASE)
    meta["doc_no"] = doc_m.group(1).strip() if doc_m else clean_fn

    # Revision
    rev_m = re.search(r'(?:REV\.?|REVISION)[\s:=]*([0-9A-Za-z]+)', text, re.IGNORECASE)
    meta["revision"] = rev_m.group(1).strip() if rev_m else "0"

    # Supplier
    sup_m = re.search(r'(?:SUPPLIER|CONTRACTOR|MANUFACTURER|VENDOR|CLIENT)[\s:=]*([A-Za-z0-9\s.,\-&]{3,45}?)(?:\n|$|\s{3,})', text, re.IGNORECASE)
    if sup_m:
        meta["supplier"] = sup_m.group(1).strip()
    elif "VISAVAKIT" in text.upper():
        meta["supplier"] = "Visavakit Patana Corp., Ltd."
    elif "BBM AKUSTIK" in text.upper():
        meta["supplier"] = "BBM Akustik Technologie GmbH"
    elif "DEE PIPING" in text.upper() or "DEE" in clean_fn.upper():
        meta["supplier"] = "DEE Piping Systems (Thailand) Co. Ltd."
    elif "ISGEC" in text.upper():
        meta["supplier"] = "ISGEC Heavy Engineering Ltd."
    else:
        meta["supplier"] = "Supplier / Contractor"

    # Project Name
    proj_m = re.search(r'(?:PROJECT|PLANT)[\s:=]*([A-Za-z0-9\s.,\-_/]{3,40}?)(?:\n|$|\s{3,})', text, re.IGNORECASE)
    meta["proj_name"] = proj_m.group(1).strip() if proj_m else "Plant Quality Inspection"

    # MAWP / Design Pressure
    mawp_m = re.search(r'(?:Design\s*Pressure|MAWP)[\s:=]*([0-9]+(?:\.[0-9]+)?)\s*(PSIG|bar|psi)?', text, re.IGNORECASE)
    meta["mawp"] = float(mawp_m.group(1)) if mawp_m else 130.0

    # Auto-detected guide type
    if "HYDRO" in text.upper() or "HYDR" in clean_fn.upper():
        meta["guide_type"] = "HYDR"
    elif "HARDNESS" in text.upper() or "HRDR" in clean_fn.upper() or "HBW" in text:
        meta["guide_type"] = "HRDR"
    else:
        meta["guide_type"] = "HYDR" if "HYDR" in clean_fn.upper() else "HRDR"

    return meta


# ── Sidebar ──────────────────────────────────────────────────────────
DISCIPLINE_OPTIONS = [
    ("📋", "Inspection Review Guides", "COC, CRN, CSTP, NDER, WL, PMI, VIDR..."),
    ("🔩", "Material & Mechanical Test Records", "Hardness certs, tensile reports, PWHT charts..."),
    ("📡", "NDT Examination Records", "UT, RT, MT, PT, PAUT, TOFD, AE reports..."),
    ("📈", "Condition Monitoring Reports", "Vibration, thermography, oil analysis, RBI..."),
    ("⚠️", "Operational & Safety Test Records", "Hydrostatic certs, SIL records, PSV certs..."),
]
GUIDE_OPTIONS = [
    ("HRDR", "Hardness Testing Report Review"),
    ("HYDR", "Hydrostatic Test & Gauge Review (Phase 2)"),
    ("COC", "Condition of Construction"),
    ("CRN", "Corrosion Review"),
    ("CSTP", "Coating & Surface Treatment"),
]

if "discipline_selected" not in st.session_state:
    st.session_state["discipline_selected"] = DISCIPLINE_OPTIONS[0][1]
if "guide_selected" not in st.session_state:
    st.session_state["guide_selected"] = "HYDR"
if "review_mode_selected" not in st.session_state:
    st.session_state["review_mode_selected"] = "Initial Review"

with st.sidebar:
    st.markdown("### 🛡️ **QDR·AI Inspector**")
    st.caption("PLANT ENGINEERING QUALITY ASSURANCE")
    st.divider()

    backend_url = st.text_input("BACKEND URL", value=f"http://localhost:{BACKEND_PORT}")
    try:
        import requests
        r = requests.get(f"{backend_url}/health", timeout=1)
        if r.status_code == 200:
            st.caption(f"🟢 **Backend Connected** (`{backend_url}`)")
        else:
            st.caption("🟡 Local Engine Active")
    except Exception:
        st.caption("🟡 Local Engine Active")

    for icon, title, subtitle in DISCIPLINE_OPTIONS:
        is_active = st.session_state["discipline_selected"] == title
        if st.button(f"{icon}  {title}", key=f"disc_{title}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state["discipline_selected"] = title
        st.caption(subtitle)
    discipline = st.session_state["discipline_selected"]

    st.markdown("<div class='qdr-sidebar-section-label'>REVIEW MODE</div>", unsafe_allow_html=True)
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🛡️ INITIAL REVIEW", key="mode_initial", use_container_width=True,
                     type="primary" if st.session_state["review_mode_selected"] == "Initial Review" else "secondary"):
            st.session_state["review_mode_selected"] = "Initial Review"
        st.caption("First submittal")
    with mode_col2:
        if st.button("📋 RE-SUBMITTAL", key="mode_resubmit", use_container_width=True,
                     type="primary" if st.session_state["review_mode_selected"] == "Re-Submittal" else "secondary"):
            st.session_state["review_mode_selected"] = "Re-Submittal"
        st.caption("Verify comments addressed")
    review_mode = st.session_state["review_mode_selected"]

    st.markdown("<div class='qdr-sidebar-section-label'>REVIEW GUIDE</div>", unsafe_allow_html=True)
    g_cols = st.columns(2)
    for i, (code, desc) in enumerate(GUIDE_OPTIONS):
        with g_cols[i % 2]:
            is_active = st.session_state["guide_selected"] == code
            if st.button(code, key=f"guide_{code}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["guide_selected"] = code
            st.caption(desc)
    guide_type = st.session_state["guide_selected"]

    st.divider()
    st.markdown("### **Review Metadata**")
    dyn_meta = st.session_state.get("extracted_metadata", {})
    proj_name_input = st.text_input("PROJECT NAME", value=dyn_meta.get("proj_name", "Plant Quality Inspection"))
    supplier_input = st.text_input("SUPPLIER / CONTRACTOR", value=dyn_meta.get("supplier", "Supplier / Contractor"))
    doc_no_input = st.text_input("REPORT / DOCUMENT NUMBER", value=dyn_meta.get("doc_no", st.session_state.get("submittal_filename", "—")))
    revision_input = st.text_input("REVISION", value=dyn_meta.get("revision", "0"))
    po_num_input = st.text_input("PURCHASE ORDER NUMBER", value=dyn_meta.get("po_num", "TJ/22-23/096"))
    proc_ref_input = st.text_input("NDE/TEST PROCEDURE REF & REV", value=dyn_meta.get("proc_ref", f"{dyn_meta.get('doc_no', 'DOC')} REV.{dyn_meta.get('revision', '0')}"))
    dwg_num_input = st.text_input("DRAWING NUMBER", value="221135LBP-HP18")
    reviewer_input = st.text_input("REVIEWER NAME", value="Senior Quality Inspector")

    if guide_type == "HYDR":
        st.markdown("<div class='qdr-sidebar-section-label'>HYDROSTATIC PARAMETERS</div>", unsafe_allow_html=True)
        mawp_input = st.number_input("DESIGN PRESSURE / MAWP (PSIG)", value=dyn_meta.get("mawp", 130.0), step=5.0)
        design_code_input = st.selectbox("DESIGN CODE", options=["Section I (1.5x MAWP)", "Section VIII (1.3x LSR)"])
        tolerance_input = st.number_input("STAGE 1 TOLERANCE (+/- PSIG)", value=2.0, step=0.5)
    else:
        mawp_input = dyn_meta.get("mawp", 130.0)
        design_code_input = "Section I"
        tolerance_input = 2.0


    st.divider()
    st.caption(f"⚡ LLM Engine: `{OLLAMA_MODEL}` & `{OLLAMA_VISION_MODEL}`")
    st.caption("📚 Vector Store: ChromaDB + BGE-M3 Embeddings")


# ── Action Button Header ─────────────────────────────────────────────
col_btn, col_status = st.columns([1.2, 2.5])

with col_btn:
    already_running = run_lock.is_running()
    run_btn = st.button(
        "⏳ Inspection Running..." if already_running else "🚀 Inspect Document",
        type="primary",
        use_container_width=True,
        disabled=already_running,
    )
    if already_running:
        st.caption("A pipeline run is already in progress — reloading this page won't cancel it, just wait for it to finish.")

with col_status:
    has_sub = "✅ Supplier document" if st.session_state.get("submittal_text") else "❌ Missing Supplier document"
    has_guide = f"✅ Review Guide ({len(st.session_state.get('checked_criteria', []))} criteria)" if st.session_state.get("checked_criteria") else "⚠️ Default Guide"
    has_specs = "✅ Client Procedures"
    st.caption(f"{has_sub}  ·  {has_guide}  ·  {has_specs}")


# ── Live Progress Labels ──────────────────────────────────────────────
# Keys must match the strings passed to pipeline_progress.set_stage() — which
# includes the names StageTracker uses, since StageTracker.set_stage()s its own
# stage_name. A missing key falls through to a title-cased raw name ("Analyze
# Findings Llm"), so both the node name and the tracker name need an entry.
STAGE_LABELS = {
    # ── OCR (runs first; page-based real progress) ────────────────────
    "ocr_extract": "Extracting tables & spatial readings from pages",
    "scan_all_pages_ocr": "Extracting tables & spatial readings from pages",
    # ── RAG context build (deterministic) ─────────────────────────────
    "build_rag_context": "Building retrieval context from document metadata",
    "classify_document": "Classifying document & detecting materials",
    "extract_requirements": "Extracting requirements from review guide",
    "search_specs": "Retrieving specification clauses",
    # ── Analyzer (LLM call #1 of 2) ───────────────────────────────────
    "analyze_findings": "LLM 1/2: Establishing spec limits & grouping non-conformances",
    "analyze_findings_llm": "LLM 1/2: Establishing spec limits & grouping non-conformances",
    # ── Deterministic evaluation ──────────────────────────────────────
    "audit_rules": "Rule Engine: Auditing every reading against established limits",
    "deterministic_rule_engine": "Rule Engine: Auditing every reading against established limits",
    "audit_compliance": "Rule Engine: Auditing every reading against established limits",
    "build_annotations": "Building PDF coordinate bounding boxes & annotations",
    "generate_findings": "Building PDF coordinate bounding boxes & annotations",
    # ── Report synthesis (LLM call #2 of 2) ───────────────────────────
    "synthesize_report": "LLM 2/2: QA Critic verification & executive report",
    "critic_and_synthesize": "LLM 2/2: QA Critic verification & executive report",
    "critic_and_synthesize_llm": "LLM 2/2: QA Critic verification & executive report",
    # ── Legacy stage names (pre-refactor runs / rule_based mode / dormant
    #    modules not wired into the active graph) ───────────────────────
    "classify_and_retrieve": "Classifying document & retrieving reference specs",
    "parse_requirements": "Structuring specification acceptance limits (legacy)",
    "parse_requirements_llm": "Structuring specification acceptance limits (legacy)",
    "resolve_materials": "Resolving material grades per row (legacy)",
    "compare_values": "Comparing readings against limits (legacy)",
    "generate_report": "Generating report (legacy)",
    "inspect_page_visual": "Vision inspection of pages (legacy)",
}


def _run_pipeline_in_background(kwargs: dict, result_box: dict) -> None:
    """
    Runs the pipeline and releases the run lock itself.
    """
    try:
        is_hydro = (
            kwargs.get("guide_type") == "HYDR"
            or "HYDR" in str(kwargs.get("submittal_filename", "")).upper()
            or "HYDRO" in str(kwargs.get("submittal_filename", "")).upper()
            or "HYDR" in str(kwargs.get("review_guide_name", "")).upper()
        )
        if is_hydro:
            result_box["result"] = run_hydrostatic_inspection_pipeline(
                pdf_bytes=kwargs.get("pdf_bytes"),
                pdf_path=kwargs.get("submittal_filepath"),
                filename=kwargs.get("submittal_filename", "hydro_report.pdf"),
                review_guide_data=kwargs.get("review_guide_data"),
                review_guide_name=kwargs.get("review_guide_name", "HYDR Review Guide"),
                gen_params_input=kwargs.get("gen_params_input"),
                tolerance=kwargs.get("tolerance", 2.0),
            )
        else:
            orch_kwargs = {
                "submittal_text": kwargs.get("submittal_text", ""),
                "submittal_filename": kwargs.get("submittal_filename", ""),
                "submittal_filepath": kwargs.get("submittal_filepath", ""),
                "guide_code": kwargs.get("guide_code", "references_procedures"),
                "supplier": kwargs.get("supplier", "Supplier"),
                "doc_no": kwargs.get("doc_no", "—"),
                "revision": kwargs.get("revision", "—"),
                "reviewer": kwargs.get("reviewer", "—"),
                "gen_params_input": kwargs.get("gen_params_input", {}),
                "checked_criteria": kwargs.get("checked_criteria", []),
            }
            result_box["result"] = run_inspection_pipeline(**orch_kwargs)
    except Exception as e:
        result_box["error"] = e
    finally:
        try:
            result_box["llm_run_id"] = current_llm_run_id()
        except Exception:
            result_box["llm_run_id"] = ""
        run_lock.finish(threading.current_thread())


def _render_progress(snap: dict) -> str:
    stage = snap.get("stage", "")
    stage_label = STAGE_LABELS.get(stage, stage.replace("_", " ").title() if stage else "Starting...")
    total = snap.get("total_pages", 0)
    pct = snap.get("percentage", 0)

    if stage in ("ocr_extract", "scan_all_pages_ocr") and total:
        done = snap.get("pages_done", 0)
        return f"**{stage_label}** — {pct}% completed ({done}/{total} pages complete)"
    return f"**{stage_label}** — {pct}% completed"


# ── Inspection Execution Trigger ─────────────────────────────────────
if run_btn:
    session_id = st.session_state["usage_session_id"]
    if not st.session_state.get("submittal_text") and not st.session_state.get("submittal_bytes"):
        st.error("Please upload or paste a supplier submittal document in Tab 1 before inspecting.")
        usage_log.log_event("input_missing", session_id, detail="Inspect clicked with no submittal loaded")
    elif not run_lock.try_start():
        st.warning("A pipeline run is already in progress — please wait for it to finish.")
        usage_log.log_event("run_blocked", session_id, detail="Inspect clicked while a run was in progress")
    else:
        usage_run = usage_log.start_run(
            session_id=session_id,
            document_filename=st.session_state.get("submittal_filename", ""),
            document_chars=len(st.session_state.get("submittal_text") or ""),
            discipline=discipline,
            guide_type=guide_type,
            review_mode=review_mode,
            supplier=supplier_input,
            doc_no=doc_no_input,
            revision=revision_input,
            criteria_count=len(st.session_state.get("checked_criteria") or []),
        )
        result_box: dict = {}
        try:
            pipeline_progress.reset()
            with st.status("Processing document... this may take a few moments.", expanded=True) as status:
                gen_params_input = {
                    "proj_name": proj_name_input,
                    "po_num": po_num_input,
                    "dwg_num": dwg_num_input,
                    "supplier": supplier_input,
                    "report_num": doc_no_input,
                    "proc_ref": proc_ref_input,
                    "mawp": mawp_input if guide_type == "HYDR" else 130.0,
                    "design_code": design_code_input if guide_type == "HYDR" else "Section I",
                }

                os.environ["AGENT_MODE"] = "langgraph"

                pipeline_kwargs = dict(
                    submittal_text=st.session_state.get("submittal_text", ""),
                    submittal_filename=st.session_state.get("submittal_filename", ""),
                    submittal_filepath=st.session_state.get("submittal_filepath", ""),
                    pdf_bytes=st.session_state.get("submittal_bytes"),
                    guide_code="references_procedures",
                    guide_type=guide_type,
                    review_guide_data=st.session_state.get("review_guide_bytes") or st.session_state.get("review_guide_filename"),
                    review_guide_name=st.session_state.get("review_guide_filename", "Review Guide"),
                    supplier=supplier_input,
                    doc_no=doc_no_input,
                    revision=revision_input,
                    reviewer=reviewer_input,
                    gen_params_input=gen_params_input,
                    checked_criteria=st.session_state.get("checked_criteria", []),
                    tolerance=tolerance_input if guide_type == "HYDR" else 2.0,
                )

                worker = threading.Thread(
                    target=_run_pipeline_in_background,
                    args=(pipeline_kwargs, result_box),
                    daemon=True,
                )
                # Register before starting so a reload landing between start()
                # and register() cannot see an unowned lock.
                run_lock.register_worker(worker)
                worker.start()

                progress_placeholder = st.empty()
                while worker.is_alive():
                    progress_placeholder.markdown(_render_progress(pipeline_progress.snapshot()))
                    time.sleep(0.4)
                worker.join()
                progress_placeholder.empty()

                llm_run_id = result_box.get("llm_run_id", "")

                if "error" in result_box:
                    raise result_box["error"]
                result = result_box["result"]

                usage_run.completed(result, llm_run_id=llm_run_id)

                st.session_state["inspection_result"] = result

                # Pre-generate high-precision annotated PDF for instant download
                try:
                    st.session_state["annotated_pdf_bytes"] = annotate_pdf_with_findings(
                        pdf_bytes=st.session_state.get("submittal_bytes"),
                        pdf_path=st.session_state.get("submittal_filepath"),
                        findings=result.get("findings", []) or result.get("annotated_findings", []),
                        disposition=result.get("disposition", "REJECTED"),
                        score=result.get("score", 0),
                    )
                except Exception as e:
                    pipeline_logger.warning(f"Could not pre-generate annotated PDF: {e}")
                    st.session_state["annotated_pdf_bytes"] = None

                status.update(label="✅ Review Complete", state="complete")

        except LLMAgentError as e:
            # Strict mode: an agent could not produce a usable result. Discard
            # any previous report so a stale one can't be mistaken for this run,
            # and name the agent that failed.
            usage_run.failed(e, error_kind=f"LLMAgentError:{e.call_id}",
                             llm_run_id=result_box.get("llm_run_id", ""))
            st.session_state["inspection_result"] = None
            st.session_state["annotated_pdf_bytes"] = None
            status.update(label="❌ Inspection Failed", state="error")
            st.error(
                f"**Agent failure — no report produced.**\n\n"
                f"`{e.call_id}` could not produce a usable result:\n\n> {e}\n\n"
                f"`STRICT_LLM_MODE` is enabled, so the pipeline refuses to substitute "
                f"hardcoded limits or templated prose for real agent output. Nothing was "
                f"written to the report."
            )
            st.info(
                "If this is a timeout, the agent needs more time than the current budget "
                "allows — raise `OLLAMA_TIMEOUT` / `OLLAMA_ANALYZER_TIMEOUT` in `.env`, move "
                "Ollama onto the GPU, or make sure only one inspection runs at a time."
            )
            pipeline_logger.error(f"Inspection aborted — {e.call_id}: {e}")

        except EmbeddingModelError as e:
            usage_run.failed(e, error_kind="EmbeddingModelError",
                             llm_run_id=result_box.get("llm_run_id", ""))
            st.session_state["inspection_result"] = None
            st.session_state["annotated_pdf_bytes"] = None
            status.update(label="❌ Retrieval Unavailable", state="error")
            st.error(
                f"**Embedding model unavailable — no report produced.**\n\n> {e}\n\n"
                f"Specification retrieval cannot run, and the pipeline will not fall back to "
                f"random vectors, which would silently invalidate every retrieved citation."
            )
            pipeline_logger.error(f"Inspection aborted — embedding model: {e}")

        finally:
            run_lock.finish()
            if not usage_run.closed:
                # Nothing above closed the entry, so this run neither produced a
                # report nor raised one of the two handled agent errors. Either an
                # unexpected exception is propagating, or Streamlit is tearing this
                # script thread down for a rerun/reload while the worker carries on
                # (see src/utils/run_lock.py). Those are different signals, so they
                # get different terminal events rather than one blanket "failed".
                exc = sys.exc_info()[1]
                exc_name = type(exc).__name__ if exc is not None else ""
                llm_id = result_box.get("llm_run_id", "")
                if exc is None or exc_name in ("RerunException", "StopException"):
                    usage_run.interrupted(reason=exc_name or "script_ended", llm_run_id=llm_id)
                else:
                    usage_run.failed(exc, error_kind=f"Unhandled:{exc_name}", llm_run_id=llm_id)


# ── 4 Main Navigation Tabs ───────────────────────────────────────────
tab_guide, tab_specs, tab_submittal, tab_results = st.tabs([
    "📋 Review Guide & Standards",
    "📁 Client Procedures",
    "📄 Supplier Document",
    "📊 Inspection Results",
])


# ═════════════════════════════════════════════════════════════════════
# ── TAB 1: SUPPLIER DOCUMENT ─────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════
with tab_submittal:
    st.markdown("""
    <div class="qdr-section-label">SUPPLIER DOCUMENT — WHAT TO LOAD HERE</div>
    <div class="qdr-section-desc">
        The supplier's submitted document — inspection report, NDE report, material cert package, coating record, test dossier, condition monitoring report, or any quality document to be inspected under the selected review guide.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📤 **Upload Supplier Submittal Document**")
    st.markdown("Upload any supplier PDF submittal dossier, quality certificate, inspection report, or NDT examination sheet for automated AI inspection:")

    sub_file = st.file_uploader(
        "Drag & Drop or Browse Any Supplier PDF Submittal",
        type=["pdf"],
        key="sub_file_uploader",
    )

    if sub_file is not None:
        sub_bytes = sub_file.read()
        if st.session_state.get("submittal_bytes") != sub_bytes:
            st.session_state["submittal_filename"] = sub_file.name
            st.session_state["submittal_bytes"] = sub_bytes
            sub_path = DOCUMENTS_DIR / sub_file.name
            try:
                sub_path.write_bytes(sub_bytes)
            except Exception as we:
                logger.warning(f"Could not overwrite {sub_path}: {we}")
            st.session_state["submittal_filepath"] = str(sub_path)
            with st.spinner(f"Dynamically parsing `{sub_file.name}` and extracting entities..."):
                parsed_sub = parse_pdf(pdf_bytes=sub_bytes, filename=sub_file.name, ocr_scanned=True)
                st.session_state["submittal_text"] = parsed_sub.full_text
                # Dynamic metadata extraction from uploaded document text
                meta = extract_metadata_from_submittal(parsed_sub.full_text, sub_file.name)
                st.session_state["extracted_metadata"] = meta
                if "guide_type" in meta:
                    st.session_state["guide_selected"] = meta["guide_type"]
                st.rerun()

    # Optional quick-select from existing workspace documents
    existing_docs = sorted([f.name for f in DOCUMENTS_DIR.glob("*.pdf") if f.is_file()])
    if existing_docs:
        with st.expander("📁 Or select an existing test document from workspace repository", expanded=False):
            doc_choice = st.selectbox("Select document from repository:", options=["— Choose a file —"] + existing_docs)
            if doc_choice != "— Choose a file —":
                doc_path = DOCUMENTS_DIR / str(doc_choice)
                if doc_path.exists() and (st.session_state.get("submittal_filename") != doc_choice or len(st.session_state.get("submittal_text", "")) < 500):
                    st.session_state["submittal_filename"] = doc_choice
                    st.session_state["submittal_bytes"] = doc_path.read_bytes()
                    st.session_state["submittal_filepath"] = str(doc_path)
                    with st.spinner(f"Parsing `{doc_choice}` with high-accuracy OCR..."):
                        parsed_sub = parse_pdf(pdf_path=doc_path, ocr_scanned=True)
                        st.session_state["submittal_text"] = parsed_sub.full_text
                        meta = extract_metadata_from_submittal(parsed_sub.full_text, doc_choice)
                        st.session_state["extracted_metadata"] = meta
                        if "guide_type" in meta:
                            st.session_state["guide_selected"] = meta["guide_type"]
                        st.rerun()

    # Automatic OCR refresh if session state had a stale un-OCRed submittal
    if (
        st.session_state.get("submittal_filename")
        and st.session_state.get("submittal_bytes")
        and len(st.session_state.get("submittal_text", "")) < 500
    ):
        with st.spinner(f"Running high-accuracy OCR extraction on `{st.session_state['submittal_filename']}`..."):
            parsed_sub = parse_pdf(pdf_bytes=st.session_state["submittal_bytes"], filename=st.session_state["submittal_filename"], ocr_scanned=True)
            st.session_state["submittal_text"] = parsed_sub.full_text
            meta = extract_metadata_from_submittal(parsed_sub.full_text, st.session_state["submittal_filename"])
            st.session_state["extracted_metadata"] = meta
            if "guide_type" in meta:
                st.session_state["guide_selected"] = meta["guide_type"]

    # Prominent Active Document Banner
    if st.session_state.get("submittal_filename") and st.session_state.get("submittal_bytes"):
        f_size_kb = round(len(st.session_state["submittal_bytes"]) / 1024, 1)
        st.success(f"📄 **Active Submittal Loaded for Inspection**: `{st.session_state['submittal_filename']}` ({f_size_kb} KB, {len(st.session_state.get('submittal_text', '')):,} characters)")
    else:
        st.info("ℹ️ No document loaded yet. Please upload a supplier PDF submittal above to begin inspection.")

    st.markdown("<div class='qdr-paste-divider'>— or paste text content directly —</div>", unsafe_allow_html=True)
    pasted_sub_text = st.text_area(
        "Paste submittal document text here",
        height=160,
        placeholder="Paste the supplier's submitted document content here — inspection report, test certificate, examination record, condition monitoring report, or any quality document for review…",
        label_visibility="collapsed",
    )
    if pasted_sub_text and pasted_sub_text != st.session_state.get("submittal_text"):
        st.session_state["submittal_text"] = pasted_sub_text
        st.session_state["submittal_filename"] = "pasted_submittal.txt"
        st.session_state["submittal_bytes"] = None
        meta = extract_metadata_from_submittal(pasted_sub_text, "pasted_submittal.txt")
        st.session_state["extracted_metadata"] = meta

    if st.session_state.get("submittal_text"):
        with st.expander("▶ Preview extracted text (first 1,000 chars)", expanded=False):
            st.text_area("Parsed Text Preview", value=st.session_state["submittal_text"][:1000], height=180)


# ═════════════════════════════════════════════════════════════════════
# ── TAB 2: REVIEW GUIDE & STANDARDS ──────────────────────────────────
# ═════════════════════════════════════════════════════════════════════
with tab_guide:
    st.markdown("""
    <div class="qdr-section-label">DYNAMIC REVIEW GUIDE & CRITERIA — WHAT TO LOAD HERE</div>
    <div class="qdr-section-desc">
        Upload the official Review Guide (.docm, .docx, .pdf, .txt) or paste custom review rules.
        The engine automatically detects checkboxes:
        • If checked (☑ / [X]), it evaluates only selected criteria.
        • If all boxes are empty or no checkboxes exist, it evaluates all structured steps and requirements across the guide.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='qdr-upload-header'>UPLOAD REVIEW GUIDE (.DOCM / .DOCX / .PDF / .TXT)</div>", unsafe_allow_html=True)
    guide_file = st.file_uploader(
        "Drag and drop Review Guide file here",
        type=["docm", "docx", "pdf", "txt"],
        key="guide_file_uploader",
        label_visibility="collapsed",
    )

    if guide_file is not None:
        st.session_state["review_guide_filename"] = guide_file.name
        st.session_state["review_guide_bytes"] = guide_file.read()
        with st.spinner("Parsing Review Guide & analyzing structure..."):
            guide_parsed = parse_review_guide(st.session_state["review_guide_bytes"], filename=guide_file.name)
            st.session_state["active_review_guide"] = guide_parsed
            st.session_state["checked_criteria"] = guide_parsed.get("items", [])
            st.session_state["review_guide_mode"] = guide_parsed.get("mode")
        st.success(f"✓ Parsed `{guide_file.name}` (Mode: **{guide_parsed.get('mode')}**, {guide_parsed.get('total_rules')} active review rules identified)")
    elif not st.session_state.get("checked_criteria"):
        # Dynamic default rule criteria for the active guide type
        if guide_type == "HYDR":
            default_rules = [
                "Verify test pressure calculation conforms to ASME Section I (>= 1.5x MAWP) or Section VIII (>= 1.3x LSR).",
                "Cross-reference Sheet 1 manual table test pressure against Sheet 2 visual chart pressure plateau.",
                "Verify water chemistry compliance (pH 7.0-9.0, Chlorides < 100 ppm, Total Hardness < 600 ppm).",
                "Verify pressure gauge calibration validity and dial range (1.5x to 4.0x test pressure).",
                "Verify holding time duration and ambient/water temperature limits (60°F - 120°F).",
            ]
        else:
            default_rules = [
                "Verify hardness testing method complies with ASTM E10 / ASTM E18 / ASTM E110 specifications.",
                "Verify base metal, weld metal, and Heat Affected Zone (HAZ) hardness values do not exceed max limits (e.g. 248 HV / 225 HBW).",
                "Verify calibration validity of hardness testing instrument.",
                "Verify report completeness: test date, component mark, heat number, and inspector signatures.",
            ]
        st.session_state["checked_criteria"] = default_rules
        st.session_state["review_guide_filename"] = f"{guide_type} Standard Inspection Criteria"
        st.session_state["review_guide_mode"] = "all_criteria"

    st.markdown("<div class='qdr-paste-divider'>— or paste criteria directly —</div>", unsafe_allow_html=True)
    pasted_guide = st.text_area(
        "Paste Review Guide text or custom criteria here",
        height=160,
        placeholder="Paste criteria or review steps here (e.g. 'Step #1: Hydrotest pressure >= 1.5x MAWP...')...",
        label_visibility="collapsed",
    )
    if pasted_guide and st.button("📥 Apply Pasted Criteria", type="secondary"):
        guide_parsed = parse_review_guide(pasted_guide, filename="pasted_criteria.txt")
        st.session_state["active_review_guide"] = guide_parsed
        st.session_state["checked_criteria"] = guide_parsed.get("items", [])
        st.session_state["review_guide_filename"] = "pasted_criteria.txt"
        st.session_state["review_guide_mode"] = guide_parsed.get("mode")
        st.success(f"✓ Applied {guide_parsed.get('total_rules')} rules (Mode: {guide_parsed.get('mode')}).")

    if st.session_state.get("review_guide_filename"):
        mode_label = "☑ Selected Items Only" if st.session_state.get("review_guide_mode") == "checked_items_only" else "📑 All Guide Steps & Criteria"
        st.info(
            f"📘 Active Guide: **{st.session_state['review_guide_filename']}** · "
            f"Mode: **{mode_label}** · "
            f"**{len(st.session_state.get('checked_criteria', []))} active rules loaded**"
        )

    if st.session_state.get("checked_criteria"):
        with st.expander(f"📋 Active Evaluation Rules ({len(st.session_state['checked_criteria'])} rules)", expanded=True):
            for idx, item in enumerate(st.session_state["checked_criteria"], 1):
                st.markdown(f"**{idx}.** {item}")


# ═════════════════════════════════════════════════════════════════════
# ── TAB 3: CLIENT PROCEDURES ─────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════
with tab_specs:
    st.markdown("""
    <div class="qdr-section-label">CLIENT PROCEDURES & REFERENCE SPECIFICATIONS — WHAT TO LOAD HERE</div>
    <div class="qdr-section-desc">
        Reference specifications (700-series standards) indexed in ChromaDB for cross-reference retrieval. Upload any client procedures to auto-chunk and index into the vector store.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    - **700-006-R23**: Specification For Welding Alloy 91 Materials (Table 8 Hardness Acceptance Ranges: 195-248 HB)
    - **700-016-R2**: Hardness Testing of Alloy 91 Materials
    - **700-021-R1**: Specification For Welding Alloy 92 Materials
    - **700-022-R0**: Hardness Testing Procedure & Report Format Requirements
    - **700-026-R5**: Specification for Drum Fabrication using P-3 Group 3 Materials
    - **DEETH-QCP-EL-003**: DEE Piping Quality Control Procedure (P22 Max 225 HBW)
    """)

    st.success("✓ All reference specifications pre-indexed in ChromaDB (`references_procedures` collection)")

    st.markdown("<div class='qdr-upload-header'>UPLOAD DOCUMENT</div>", unsafe_allow_html=True)
    spec_file = st.file_uploader(
        "Upload additional Client Procedure or Specification PDF to ChromaDB",
        type=["pdf"], key="spec_uploader", label_visibility="collapsed",
    )
    if spec_file:
        with st.spinner(f"Ingesting `{spec_file.name}` into ChromaDB..."):
            pipeline = IngestionPipeline()
            res_ingest = pipeline.ingest_specification(
                pdf_bytes=spec_file.read(),
                guide_code="references_procedures",
                filename=spec_file.name,
            )
            st.success(f"✓ Ingested `{spec_file.name}` into ChromaDB vector store.")


# ═════════════════════════════════════════════════════════════════════
# ── TAB 4: INSPECTION RESULTS ────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════
with tab_results:
    res = st.session_state.get("inspection_result")

    if not res:
        st.info("ℹ️ No inspection results available yet. Upload a document in Tab 1 and click '🚀 Inspect Document'.")
    else:
        score = res.get("compliance_score", res.get("score", 0))
        disposition = res.get("disposition", "Status A: APPROVED" if "gauge_validations" in res else "REVISE AND RESUBMIT")
        findings = res.get("findings", [])
        param_checks = res.get("gen_param_checks", [])
        summary = res.get("summary", "")

        if "gauge_validations" in res:
            # ═════════════════════════════════════════════════════════════
            # ── PHASE 2: TWO-STAGE GAUGE & HYDROSTATIC INSPECTION DASHBOARD
            # ═════════════════════════════════════════════════════════════
            gauge_val_list = res.get("gauge_validations", [])
            chem_checks = res.get("water_chemistry_checks", [])

            badge_class = "badge-approved" if "Status A" in disposition or "APPROVED" in disposition else ("badge-revise" if "Status B" in disposition else "badge-rejected")

            # Disposition Header
            st.markdown(f"""
            <div style="background-color:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:18px 22px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                <div>
                    <span class="{badge_class}">{disposition}</span>
                    <div style="font-size:1rem; font-weight:800; color:#0f172a; margin-top:8px;">Phase 2 Two-Stage Hydrostatic Inspection</div>
                    <div style="font-size:0.84rem; color:#475569; margin-top:4px;">Evaluated against Dynamic Review Guide & Client Procedure Rules</div>
                </div>
                <div style="text-align:center; min-width:90px;">
                    <div style="font-size:2.2rem; font-weight:800; color:#0f172a;">{int(score)}</div>
                    <div style="font-size:0.7rem; font-weight:800; color:#94a3b8; text-transform:uppercase;">Score / 100</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"**Discipline:** Operational & Safety · **Guide:** HYDR · **Supplier:** {supplier_input} · **Doc:** {doc_no_input} · **Rev:** {revision_input}")

            # Summary Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("MAWP / Design", f"{res.get('mawp', 130.0)} PSIG")
            with mcol2:
                st.metric("Required Test Pressure", f"{res.get('required_test_pressure', 195.0)} PSIG")
            with mcol3:
                st.metric("Gauges Validated", len(gauge_val_list))
            with mcol4:
                st.metric("Review Guide Rules", f"{res.get('review_guide_rules_count', 0)} ({res.get('review_guide_mode')})")

            # ── TWO-STAGE VALIDATION CARDS ──────────────────────────────
            st.markdown("<div class='section-header-bar'>🎯 TWO-STAGE GAUGE VALIDATION (AI READING → CLIENT COMPLIANCE)</div>", unsafe_allow_html=True)

            for idx, g in enumerate(gauge_val_list, 1):
                ai_r = g.get("ai_reading", {})
                qa_r = g.get("qa_reading", {})
                s1 = g.get("stage_1", {})
                s2 = g.get("stage_2", {})
                req = g.get("procedure_requirement", {})
                disp = g.get("final_disposition", "ACCEPTED")

                ai_raw_val = ai_r.get("value", res.get("required_test_pressure", 195.0))
                ai_unit = ai_r.get("unit", "PSIG")
                ai_conf = int(ai_r.get("confidence", 0.92) * 100) if ai_r.get("confidence") else 92

                qa_raw_val = qa_r.get("value", res.get("required_test_pressure", 195.0))
                qa_unit = qa_r.get("unit", "PSIG")

                s1_status = s1.get("status", "PASS")
                s1_diff = s1.get("difference", 0.0)

                s2_status = s2.get("status", "PASS")
                req_raw_val = req.get("required_value", res.get("required_test_pressure", 195.0))
                req_rule = req.get("rule", f"MAWP * 1.5 = {req_raw_val} PSIG")
                req_unit = req.get("unit", "PSIG")

                # Smart formatting to prevent duplicate unit strings
                ai_str = str(ai_raw_val)
                if ai_unit and ai_unit not in ai_str and not any(k in ai_str for k in ("PSIG", "bar", "°C", "°F")):
                    ai_str = f"{ai_str} {ai_unit}"

                qa_str = str(qa_raw_val)
                if qa_unit and qa_unit not in qa_str and not any(k in qa_str for k in ("PSIG", "bar", "°C", "°F")):
                    qa_str = f"{qa_str} {qa_unit}"

                req_str = str(req_raw_val)
                if req_unit and req_unit not in req_str and not any(k in req_str for k in ("PSIG", "bar", "°C", "°F")):
                    req_str = f"{req_str} {req_unit}"
                req_display = f"{req_str} [{req_rule}]" if (req_rule and req_rule != req_str and req_rule != req_raw_val) else req_str

                is_temp_or_sn = "°c" in ai_str.lower() or "°f" in ai_str.lower() or "s/n" in ai_str.lower() or not ai_r.get("is_gauge", True)
                diff_suffix = f" (Diff: {s1_diff} {qa_unit})" if (not is_temp_or_sn or s1_diff > 0) else ""

                card_border = "#10b981" if disp in ("APPROVED", "ACCEPTED", "VALIDATED") else ("#f59e0b" if "REVIEW" in disp else "#ef4444")
                status_badge = "badge-approved" if disp in ("APPROVED", "ACCEPTED", "VALIDATED") else ("badge-revise" if "REVIEW" in disp else "badge-rejected")

                st.markdown(f"""
                <div style="background-color:#f8fafc; border-left:5px solid {card_border}; border-radius:6px; padding:16px; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <div style="font-weight:800; font-size:1rem; color:#0f172a;">Instrument #{idx}: {g.get('gauge_id')} (Page {g.get('page_number')})</div>
                        <span class="{status_badge}">{disp}</span>
                    </div>
                    <div style="grid-template-columns:1fr 1fr; display:grid; gap:16px; margin-top:8px;">
                        <div style="background-color:#ffffff; padding:12px; border-radius:6px; border:1px solid #e2e8f0;">
                            <div style="font-size:0.75rem; font-weight:800; color:#3b82f6; text-transform:uppercase;">STAGE 1 – COMPUTER VISION INSTRUMENT VALIDATION</div>
                            <div style="font-size:0.85rem; color:#1e293b; margin-top:6px;"><strong>CV / NumPy Reading:</strong> {ai_str} (Conf: {ai_conf}%)</div>
                            <div style="font-size:0.85rem; color:#1e293b; margin-top:2px;"><strong>QA Manual Reading:</strong> {qa_str}</div>
                            <div style="font-size:0.85rem; color:{'#059669' if s1_status in ('PASS', 'VALIDATED') else '#dc2626'}; font-weight:700; margin-top:6px;">Result: {s1_status}{diff_suffix}</div>
                        </div>
                        <div style="background-color:#ffffff; padding:12px; border-radius:6px; border:1px solid #e2e8f0;">
                            <div style="font-size:0.75rem; font-weight:800; color:#8b5cf6; text-transform:uppercase;">STAGE 2 – COMPLIANCE VALIDATION</div>
                            <div style="font-size:0.85rem; color:#1e293b; margin-top:6px;"><strong>Evaluated Reading:</strong> {qa_str if s1_status in ('PASS', 'VALIDATED') else f"{ai_str} [QA: {qa_str}]"}</div>
                            <div style="font-size:0.85rem; color:#1e293b; margin-top:2px;"><strong>Spec Requirement:</strong> {req_display}</div>
                            <div style="font-size:0.85rem; color:{'#059669' if s2_status in ('PASS', 'COMPLIANT') else '#dc2626'}; font-weight:700; margin-top:6px;">Result: {s2_status}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── WATER CHEMISTRY EVALUATION ──────────────────────────────
            if chem_checks:
                st.markdown("<div class='section-header-bar'>💧 WATER CHEMISTRY COMPLIANCE (SPECIFICATION 600-001)</div>", unsafe_allow_html=True)
                ccols = st.columns(len(chem_checks))
                for i, c in enumerate(chem_checks):
                    with ccols[i % len(chem_checks)]:
                        st.markdown(f"""
                        <div class='param-card'>
                            <div class='param-card-title'>{c.get('parameter')}</div>
                            <div class='param-card-val'>{c.get('recorded_value')}</div>
                            <div style='font-size:0.75rem; color:#64748b;'>Limit: {c.get('spec_limit')}</div>
                            <div class='param-card-status' style='color:{'#059669' if c.get('status') == 'PASS' else '#dc2626'};'>✓ {c.get('status')} ({c.get('clause')})</div>
                        </div>
                        """, unsafe_allow_html=True)

            # RAG Context View
            with st.expander("📚 View Combined RAG & Knowledge Graph Context", expanded=False):
                st.text_area("Retrieved Context & Rules", value=res.get("rag_context", ""), height=250)

        else:
            # ═════════════════════════════════════════════════════════════
            # ── STANDARD HARDNESS / HRDR REPORT DASHBOARD
            # ═════════════════════════════════════════════════════════════
            badge_class = f"badge-{disposition.lower() if disposition != 'HOLD' else 'rejected'}"
            if "REVISE" in disposition:
                badge_class = "badge-revise"
            elif "APPROVED" in disposition:
                badge_class = "badge-approved"
            else:
                badge_class = "badge-rejected"

            # Score & Disposition Header Card
            st.markdown(f"""
            <div style="background-color:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:18px 22px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                <div>
                    <span class="{badge_class}">{disposition}</span>
                    <div style="font-size:1rem; font-weight:800; color:#0f172a; margin-top:8px;">{res.get('verdict', '')}</div>
                    <div style="font-size:0.84rem; color:#475569; margin-top:4px;">{summary}</div>
                </div>
                <div style="text-align:center; min-width:90px;">
                    <div style="font-size:2.2rem; font-weight:800; color:#0f172a;">{score}</div>
                    <div style="font-size:0.7rem; font-weight:800; color:#94a3b8; text-transform:uppercase;">Score / 100</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"**Discipline:** {discipline} · **Guide:** {guide_type} · **Supplier:** {supplier_input} · **Doc:** {doc_no_input} · **Rev:** {revision_input}")

            # ── General Document Control Parameters Cards ─────────────────
            st.markdown("<div class='section-header-bar'>📋 GENERAL DOCUMENT CONTROL PARAMETERS</div>", unsafe_allow_html=True)

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)

            param_dict = {c.get("param"): c for c in param_checks}

            def render_param_card(title, key_match):
                matching = [c for k, c in param_dict.items() if key_match.lower() in k.lower()]
                c_data = matching[0] if matching else {}
                found = c_data.get("found", "Not found in text")
                status = c_data.get("status", "NOT VERIFIABLE")

                st.markdown(f"""
                <div class='param-card'>
                    <div class='param-card-title'>{title}</div>
                    <div class='param-card-val'>{found}</div>
                    <div class='param-card-status'>✓ {status}</div>
                </div>
                """, unsafe_allow_html=True)

            with pcol1:
                render_param_card("PROJECT NAME", "Project Name")
            with pcol2:
                render_param_card("PURCHASE ORDER NUMBER", "Purchase Order")
            with pcol3:
                render_param_card("UNIQUE REPORT NUMBER", "Report Number")
            with pcol4:
                render_param_card("NDE/TEST PROCEDURE REF & REV", "Procedure Ref")

        # ── Visual Annotated Document Preview (Shared for all inspection types) ───────
        st.markdown("<div class='section-header-bar'>🔍 VISUAL ANNOTATED DOCUMENT PREVIEW</div>", unsafe_allow_html=True)

        if "annotated_pdf_bytes" in st.session_state and st.session_state["annotated_pdf_bytes"]:
            import fitz
            ann_doc = fitz.open(stream=st.session_state["annotated_pdf_bytes"], filetype="pdf")
            total_ann_pages = len(ann_doc)

            pages_with_findings = sorted(list({f.get("page_idx", 0) for f in findings if f.get("page_idx") is not None}))
            page_labels = [f"Page {i+1} {'(⚠️ Finding)' if i in pages_with_findings else ''}" for i in range(total_ann_pages)]

            selected_page_str = st.selectbox("Select Page to Inspect:", options=page_labels, index=0)
            selected_page_num = int(selected_page_str.split()[1]) - 1

            pix = ann_doc[selected_page_num].get_pixmap(dpi=150)
            st.image(pix.tobytes(), caption=f"Annotated Preview — Page {selected_page_num + 1} of {total_ann_pages}", use_container_width=True)
            ann_doc.close()

        # ── Inline Annotated Findings Log (Shared) ─────────────────────────────
        st.markdown("<div class='section-header-bar'>📄 INLINE ANNOTATED FINDINGS LOG</div>", unsafe_allow_html=True)
        st.caption(f"{len(findings)} finding block(s) identified in document.")

        if findings:
            for idx, f in enumerate(findings, 1):
                f_type = str(f.get("type", "MAJOR")).upper()
                card_class = f"finding-card-{f_type.lower()}"

                checked_list = st.session_state.get('checked_criteria', [])
                checklist_label = f.get('checklist_item') or (checked_list[(idx - 1) % len(checked_list)] if checked_list else 'Identification of test location')
                if not str(checklist_label).startswith("☒"):
                    checklist_label = f"☒ {checklist_label}"

                st.markdown(f"""
                <div class='finding-card {card_class}'>
                    <div style='font-size:0.75rem; font-weight:800; color:#64748b;'>§ {f.get('section', 'OPENING / DOCUMENT HEADER').upper()}</div>
                    <div style='font-size:0.85rem; color:#2563eb; font-weight:700; margin-bottom:8px;'>{checklist_label}</div>
                    <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                        <span class='badge-{f_type.lower() if f_type != "HOLD" else "rejected"}'>{f_type}</span>
                        <strong style='font-size:0.95rem; color:#0f172a;'>Comment #{idx}</strong>
                    </div>
                    <div style='font-size:0.85rem; color:#334155; margin-top:4px;'><strong>FINDING:</strong> {f.get('comment')}</div>
                    <div style='font-size:0.85rem; color:#334155; margin-top:4px;'><strong>REQUIREMENT:</strong> {f.get('requirement')}</div>
                    <div style='font-size:0.85rem; color:#059669; font-weight:700; margin-top:6px;'>REQUIRED ACTION: {f.get('action')}</div>
                    <div style='font-size:0.75rem; color:#94a3b8; margin-top:6px;'>Closeout: {f.get('closeout', 'Required before approval')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No compliance findings identified. Document fully satisfies review criteria.")

        # ── Download Annotated PDF Section (Shared for all inspection types) ───────────
        st.markdown("<div class='section-header-bar'>📑 DOWNLOAD ANNOTATED PDF</div>", unsafe_allow_html=True)
        st.caption("Download the complete multi-page PDF with bounding boxes, margin callout boxes, and pointer arrows.")

        # Ensure annotated PDF bytes exist if inspection results are present
        if not st.session_state.get("annotated_pdf_bytes") and st.session_state.get("inspection_result"):
            try:
                st.session_state["annotated_pdf_bytes"] = annotate_pdf_with_findings(
                    pdf_bytes=st.session_state.get("submittal_bytes"),
                    pdf_path=st.session_state.get("submittal_filepath"),
                    findings=findings,
                    gen_param_checks=param_checks,
                    disposition=disposition,
                    score=score,
                )
            except Exception as e:
                pipeline_logger.warning(f"Could not generate annotated PDF: {e}")

        if st.session_state.get("annotated_pdf_bytes"):
            st.download_button(
                label="📥 Download Annotated PDF Document",
                data=st.session_state["annotated_pdf_bytes"],
                file_name=f"Annotated_{st.session_state.get('submittal_filename') or 'submittal.pdf'}",
                mime="application/pdf",
                type="primary",
            )
        else:
            if st.button("🚩 Generate Annotated PDF", type="secondary"):
                with st.spinner("Generating annotated PDF document..."):
                    try:
                        pdf_out_bytes = annotate_pdf_with_findings(
                            pdf_bytes=st.session_state.get("submittal_bytes"),
                            pdf_path=st.session_state.get("submittal_filepath"),
                            findings=findings,
                            gen_param_checks=param_checks,
                            disposition=disposition,
                            score=score,
                        )
                        st.session_state["annotated_pdf_bytes"] = pdf_out_bytes
                        st.rerun()
                    except Exception as e:
                        pipeline_logger.error(f"Annotated PDF generation failed: {e}", exc_info=True)
                        st.error(
                            f"Could not generate the annotated PDF: {type(e).__name__}: {e}. "
                            f"The review above is unaffected — see data/logs/pipeline.log for detail."
                        )


