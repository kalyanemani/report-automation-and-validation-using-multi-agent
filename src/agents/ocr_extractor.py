from __future__ import annotations

"""
QDR·AI — Agent 1: OCR Extractor (Deterministic)
─────────────────────────────────────────────────
Extracts structured table data from each PDF page using Tesseract OCR
with spatial coordinate awareness. No LLM involved — pure deterministic
text extraction and spatial grouping.

Produces a PageOCRResult per page containing:
  - Drawing number, spool number, test block info
  - Structured table rows with component labels and hardness readings
  - Bounding box coordinates for each reading token
"""

import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import pymupdf as fitz  # PyMuPDF 1.28+: `fitz` is deprecated
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from src.agents.state import PageOCRResult, Reading, ReviewState, TableRow
from src.agents.text_layer_extractor import extract_page_text_layer, page_has_text_layer
from src.agents.materials import canonical_grade
from src.config import TESSERACT_CMD, PARSED_DIR, LOGS_DIR
from src.utils.metrics import StageTracker
from src.utils.logger import pipeline_logger
from src.utils import progress as pipeline_progress

logger = logging.getLogger("qdr_ai.agent.ocr_extractor")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ── Component Detection ──────────────────────────────────────────────

COMPONENT_PATTERNS = [
    ("Weld",           re.compile(r"\b(?:weld|wm|w\.m\.|w/m|wld|weld\d+|weld\s*pot|weld\s*metal|weld\s*zone|butt\s*weld|buttweld|bw|fw|shop\s*weld|field\s*weld|weld\s*joint|weld\s*seam|fillet\s*weld)\b", re.I)),
    # The suffix digit is the first thing OCR loses on these scans: "Haz1"
    # comes back as "Hazt" (the 1 read as a t), and neither `haz\s*1` nor the
    # `\bhaz\b` catch-all matches it — "hazt" has no word boundary after "haz".
    # Every Haz1 row on page 13 of 221135LBP-HRDR001 therefore matched no
    # component at all and was dropped from the table: 3 of 15 rows on that page,
    # and the readings with them. Tolerating the usual 1/2 confusions costs
    # nothing — no other component label begins "haz".
    ("HAZ1",           re.compile(r"\bhaz\s*[1lit!|]\b", re.I)),
    ("HAZ2",           re.compile(r"\bhaz\s*[2z]\b", re.I)),
    ("HAZ",            re.compile(r"\bhaz\b(?!\s*[12lit!|z])", re.I)),
    ("Pipe",           re.compile(r"\bpipe\b", re.I)),
    ("Elbow",          re.compile(r"\belbow\b", re.I)),
    ("Plate",          re.compile(r"\bplate\b", re.I)),
    ("Tee",            re.compile(r"\btee\b", re.I)),
    ("Reducer",        re.compile(r"\breducer\b", re.I)),
    ("Thermowell",     re.compile(r"\bthermowell\b", re.I)),
    ("End Cap",        re.compile(r"\bcap\b", re.I)),
    ("Half Coupling",  re.compile(r"\bcoupling\b", re.I)),
]


def _detect_component(text: str) -> tuple[Optional[str], bool]:
    """Detect component type from row text. Returns (component, is_weld)."""
    for name, pattern in COMPONENT_PATTERNS:
        if pattern.search(text):
            return name, (name == "Weld")
    return None, False


# OCR routinely renders digits as visually similar letters in short tokens:
# J27 -> "Jaz", J1 -> "Jl", J0 -> "JO". A joint number lost to one substituted
# glyph costs the row its joint identity, which in turn defeats joint-scoped
# material inheritance — so tolerate the common confusions.
_OCR_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0",
                                  "I": "1", "l": "1", "|": "1",
                                  "Z": "2", "z": "2", "S": "5", "s": "5",
                                  "B": "8", "G": "6", "T": "7", "A": "4"})


def _detect_joint(text: str) -> str:
    """
    Extract joint name (e.g. 'Joint J27') from row text.

    Tries an exact match first, then a glyph-corrected pass so an OCR slip in
    the number does not discard the joint entirely.
    """
    m = re.search(r"\bJ\d{1,3}[A-Z]?\b", text, re.I)
    if m:
        return f"Joint {m.group(0).upper()}"

    # Glyph-corrected retry: a J followed by 1-3 chars that are digits once the
    # common OCR substitutions are undone.
    for cand in re.findall(r"\bJ([A-Za-z0-9|]{1,3})\b", text):
        fixed = cand.translate(_OCR_DIGIT_FIXES)
        if fixed.isdigit():
            return f"Joint J{int(fixed)}"
    return "General Joint"


def _detect_report_format(full_text: str) -> str:
    """Classify the report format from full page text."""
    upper = full_text.upper()
    if any(k in upper for k in ("REQUEST CUM REPORT", "REQUEST NO", "ISGEC", "MISC. TEST", "ALPHA DUR", "BASE MATEL", "BASE METAL")):
        return "ISGEC_SINGLE"
    if "GOLDEN PASS" in upper or "1-191735LBP" in upper:
        return "GOLDEN_PASS"
    return "DEE_TABULAR"


# ── ISGEC Component Detection (broader than DEE_TABULAR) ─────────────
# ISGEC forms use different labels: "Parent Metal", "PM", "BM", "Sockolet",
# "Swage Nipple", "Branch", "Header", etc. plus the standard ones.

ISGEC_COMPONENT_PATTERNS = [
    ("Weld",           re.compile(r"\b(?:weld|wm|w\.m\.|w/m|wld|weld\d+|weld\s*pot|weld\s*metal|weld\s*zone|butt\s*weld|buttweld|bw|fw|shop\s*weld|field\s*weld|weld\s*joint|weld\s*seam|fillet\s*weld)\b", re.I)),
    # Seam- and butt-weld identifiers, which is how the ISGEC form names the
    # test location: rows read "SW-03(3+1-2)", "SW-04(1-2+3)". These ARE weld
    # measurements — the form's own stated acceptance range, 195-280 HB, is the
    # weld/HAZ band — but nothing matched them, so every row fell through to the
    # "Pipe" default below and was graded against base metal's 248 HB ceiling.
    ("Weld",           re.compile(r"\b(?:SW|BW|FW|LW)\s*-?\s*\d{1,3}\b", re.I)),
    ("HAZ1",           re.compile(r"\bhaz\s*[-/]?\s*1\b", re.I)),
    ("HAZ2",           re.compile(r"\bhaz\s*[-/]?\s*2\b", re.I)),
    ("HAZ",            re.compile(r"\bhaz\b(?!\s*[-/]?\s*[12])", re.I)),
    ("Pipe",           re.compile(r"\b(?:pipe|parent\s*metal|pm|bm|base\s*metal|header)\b", re.I)),
    ("Elbow",          re.compile(r"\belbow\b", re.I)),
    ("Tee",            re.compile(r"\btee\b", re.I)),
    ("Reducer",        re.compile(r"\breducer\b", re.I)),
    ("Plate",          re.compile(r"\bplate\b", re.I)),
    ("Thermowell",     re.compile(r"\bthermowell\b", re.I)),
    ("End Cap",        re.compile(r"\b(?:cap|end\s*cap)\b", re.I)),
    ("Half Coupling",  re.compile(r"\b(?:coupling|half\s*coupling|sockolet|swage|nipple|branch)\b", re.I)),
]


def _detect_isgec_component(text: str) -> tuple[Optional[str], bool]:
    """Detect component type from ISGEC row text. Returns (component, is_weld)."""
    for name, pattern in ISGEC_COMPONENT_PATTERNS:
        if pattern.search(text):
            return name, (name == "Weld")
    return None, False


# ── ISGEC Header Metadata Extraction ──────────────────────────────────

def _parse_isgec_acceptance_range(text: str) -> Optional[tuple[int, int]]:
    """
    Parse acceptance range from ISGEC form header.
    Handles multiple formats:
      - "Required Max. Hardness :195 ~ 280 HB"   → (195, 280)
      - "Required Max. Hardness :235 HB"         → (130, 235)
      - "Required Max. Hardness :MAX 280 HB"      → (130, 280)
      - "Acceptance Range: 195 - 280 HBW"         → (195, 280)
      - "195~280 HB"                               → (195, 280)
      - "MIN 195 HB"                               → (195, 400)
    """
    upper = text.upper()

    # Pattern 1 (range with Hardness keyword): "Required Max. Hardness :195 ~ 280 HB" or "195 - 280 HBW"
    m = re.search(
        r'(?:REQUIRED\s*)?(?:MAX\.?\s*)?HARDNESS\s*:?\s*'
        r'(\d{2,3})\s*[~\-–—/to]+\s*(\d{2,3})\s*(?:HB|HBW|BHN)?',
        upper,
    )
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if 100 <= lo < hi <= 400:
            return (lo, hi)

    # Pattern 2 (standalone range with HB suffix, no Hardness keyword)
    m = re.search(
        r'(\d{3})\s*[~\-–—]+\s*(\d{3})\s*(?:HB|HBW|BHN)',
        upper,
    )
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if 100 <= lo < hi <= 400:
            return (lo, hi)

    # Pattern 3 (MAX-only): "Required Max. Hardness :235 HB" or "HARDNESS : MAX 280 HB"
    m = re.search(
        r'(?:REQUIRED\s*)?(?:MAX\.?\s*)?HARDNESS\s*:?\s*(?:MAX\.?\s*)?(\d{2,3})\s*(?:HB|HBW|BHN)',
        upper,
    )
    if m:
        max_val = int(m.group(1))
        if 100 <= max_val <= 400:
            mat_header = _parse_isgec_material_spec(text).upper()
            is_p91 = bool(re.search(r'\b(?:P91|F91|WP91|9CR)\b', mat_header)) or max_val >= 248
            min_val = 195 if is_p91 else None
            return (min_val, max_val)

    # Pattern 4 (MIN-only): "MIN 195 HB"
    m = re.search(r'MIN\.?\s*(\d{2,3})\s*(?:HB|HBW|BHN)', upper)
    if m:
        return (int(m.group(1)), 400)  # default maximum 400

    return None


def _sanitize_ocr_reading(tok_str: str) -> Optional[int]:
    """
    Sanitize and correct common OCR digit recognition errors on dot-matrix/monospace forms.
    In piping fabrication (P91, P22, P11, CS), hardness values are physically 100-350 HBW.
    Common OCR character substitutions:
    - '1' with slanted top or serif misread as '4' (e.g. '477' -> 177, '475' -> 175, '470' -> 170, '479' -> 179, '492' -> 192).
    - '1' with horizontal noise misread as '7' (e.g. '784' -> 184, '793' -> 193, '789' -> 189).
    """
    cleaned = tok_str.strip()
    if not re.match(r"^\d{3}$", cleaned):
        return None
    
    val = int(cleaned)
    if 100 <= val <= 350:
        return val
    
    # 4xx misread: in pressure piping, 400-499 is not a valid hardness reading; leading 4 is an OCR error for 1
    if 400 <= val <= 499:
        corrected = int("1" + cleaned[1:])
        if 100 <= corrected <= 350:
            return corrected
            
    # 7xx misread: in pressure piping, 700-799 is an OCR error for 1xx
    if 700 <= val <= 799:
        corrected = int("1" + cleaned[1:])
        if 100 <= corrected <= 350:
            return corrected

    return None


def _parse_isgec_material_spec(text: str) -> str:
    """
    Parse material specification from ISGEC form header.
    Examples: "SA335P91+SA234WP91 TYPE 1", "SA182F22+SA234WP91"
    """
    # Look for Material field in OCR text
    m = re.search(
        r'Material\s*:?\s*([A-Z0-9+\-\s]+(?:TYPE\s*\d)?)',
        text,
        re.I,
    )
    if m:
        return m.group(1).strip()
    return ""


def _parse_isgec_spool_joints(text: str) -> tuple[str, list[str]]:
    """
    Parse Spool No. and Joint No. list from ISGEC form header.
    Example: "241135-00-HP-001-03/1 ,SW-15(1-5+2), SW-13(1-4+2)"
    Returns (spool_no, [joint_ids])
    """
    # Extract spool number
    spool_m = re.search(
        r'Spool\s*No\.?/?\s*Joint\s*No\.?\s*'
        r'([\d]+[\-][\d]+[\-](?:HP|RP|CSP|SIL)[\-][\d]+[\-][\d]+)',
        text, re.I,
    )
    spool_no = spool_m.group(1).upper() if spool_m else ""

    # If we didn't match, try simpler pattern
    if not spool_no:
        spool_m = re.search(
            r'(?:HP|RP|CSP|SIL)[\-]\d{1,3}[\-]\d{1,3}',
            text, re.I,
        )
        spool_no = spool_m.group(0).upper() if spool_m else "Submittal Spool"

    # Extract SW- joint identifiers
    joints = re.findall(r'SW[-]?\d+[A-Z]?', text, re.I)
    joints = [j.upper() for j in joints]

    return spool_no, joints


# ── ISGEC Image Preprocessing ────────────────────────────────────────

def _preprocess_isgec_image(img: Image.Image) -> Image.Image:
    """
    Preprocess a scanned ISGEC form image for high-precision OCR:
    1. Convert to grayscale numpy array.
    2. Detect and subtract vertical & horizontal grid cell borders using
       morphological operations so lines don't fuse with digits (e.g. '| 209 |' -> '209').
    3. Enhance contrast and sharpness for crisp character separation.
    """
    img_gray = np.array(img.convert("L"))

    try:
        # Adaptive threshold (inverted: text & lines are white)
        thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )

        # 1. Detect horizontal lines (kernel size 45x1)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # 2. Detect vertical lines (kernel size 1x45)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

        # Combine and dilate lines slightly
        grid_lines = cv2.add(h_lines, v_lines)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid_lines_dilated = cv2.dilate(grid_lines, dilate_kernel, iterations=1)

        # Subtract grid lines by setting line pixels to white (255)
        cleaned_gray = img_gray.copy()
        cleaned_gray[grid_lines_dilated > 0] = 255
        pil_cleaned = Image.fromarray(cleaned_gray)
    except Exception as e:
        logger.debug(f"Grid line removal fallback: {e}")
        pil_cleaned = Image.fromarray(img_gray)

    # Boost contrast & sharpness
    enhanced = ImageEnhance.Contrast(pil_cleaned).enhance(1.8)
    sharpened = ImageEnhance.Sharpness(enhanced).enhance(1.4)
    return sharpened


# ── ISGEC Table Body Extraction ──────────────────────────────────────

# Where the page header ends and the readings table begins. The test-block
# declaration lives above this line; everything below is row data.
_TABLE_HEADER_MARKERS = ("ITEM DESCRIPTION", "HARDNESS (HB)", "HARDNESS(HB)", "(SPOOL NO.)")


def _extract_test_block(full_text: str) -> str:
    """
    The full "Test Block Sr. No." declaration, as written on the page.

    This used to be `re.search(r"DDEL-TH-[A-Z0-9-]+", full_text)` — a token
    match that could only ever return ONE of the two things this field says.
    Page 13 of 221135LBP-HRDR001 declares:

        Test Block Sr. No. :-   P91 SAMPLE MARKED A1
                                DDEL-TH-P22-D3

    and the regex returned "DDEL-TH-P22-D3", silently discarding "P91 SAMPLE
    MARKED A1". Downstream, `resolve_page_rows` inherits a page-wide material
    only when the page names EXACTLY ONE grade — a guard specifically meant to
    stop a mixed page from being stamped. Dropping one of the two grades
    satisfied that guard artificially, so every row without its own evidence was
    stamped P22 on a page whose Material Grade column reads P91 for nearly every
    joint. Those rows were then graded against P22's 225 HB ceiling instead of
    P91's 248, turning conforming 226-248 readings into findings: 133 of the 151
    findings in the 19:01 run.

    The same regex failed in the opposite direction on page 1, which declares
    "P91 SAMPLE MARKED A1" and no DDEL-TH line at all: it returned "", so a page
    that plainly states its material inherited nothing and its rows reported
    MISSING_MATERIAL.

    Reading the field to the start of the table returns what the page actually
    says — one grade, two grades, or none — and lets the existing guard do its
    job on true information.
    """
    match = re.search(r"Test\s*Block[^:]{0,20}:?\s*-?\s*", full_text or "", re.I)
    if not match:
        return ""
    window = full_text[match.end():match.end() + 260]
    upper = window.upper()
    cut = len(window)
    for marker in _TABLE_HEADER_MARKERS:
        idx = upper.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return " ".join(window[:cut].split()).strip()


def _extract_isgec_single(
    page: fitz.Page,
    page_idx: int,
    dpi: int = 300,
) -> PageOCRResult:
    """
    Extract structured data from an ISGEC 'Request Cum Report of Misc. Test'
    scanned form. Handles multiple sub-formats at 300 DPI with grid line removal.
    """
    # Render at 300 DPI for high-precision OCR
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes()))

    # Scale factor to map pixel coords back to page coordinates
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height

    # Preprocess for cleaner OCR on bordered/shaded cells
    preprocessed = _preprocess_isgec_image(img)

    # Run OCR on preprocessed image
    data = pytesseract.image_to_data(
        preprocessed,
        output_type=pytesseract.Output.DICT,
        config="--psm 6 --oem 3",
    )

    # Also get full text from the original (color) image for metadata
    full_text = pytesseract.image_to_string(
        img, config="--psm 6 --oem 3"
    ).strip()

    # ── Parse header metadata ────────────────────────────────────────
    acceptance_range = _parse_isgec_acceptance_range(full_text)
    material_spec = _parse_isgec_material_spec(full_text)
    spool_no, joint_ids = _parse_isgec_spool_joints(full_text)

    # Drawing number (HP/RP prefix)
    drawing_m = re.search(
        r'\b(?:241135LBP-|1-191735LBP-)?(HP|RP|CSP|SIL)[A-Z0-9-]*\b',
        full_text, re.I,
    )
    drawing_no = drawing_m.group(0).upper() if drawing_m else ""

    # ASTM/ASME prefix detection
    has_astm = bool(re.search(
        r'\b(?:SA[\-]?335|SA[\-]?234|SA[\-]?182|SA[\-]?387|ASTM|ASME)\b',
        full_text, re.I,
    ))

    # Test block info — the whole declaration; see _extract_test_block.
    test_block_info = _extract_test_block(full_text)

    # ── Dynamically find table body region ────────────────────────────
    # Instead of hardcoded y thresholds, detect where the table starts by
    # looking for table header keywords: "Sl.No", "Description", "Location",
    # "Hardness", "Reading", "Result", "Remarks"
    # Fallback to y > 40% of image height if no header found.
    img_height = pix.height
    img_width = pix.width

    table_start_y = int(img_height * 0.40)  # fallback: 40% down
    table_end_y = int(img_height * 0.88)    # fallback: 88% down

    table_header_kws = [
        re.compile(r'\bsl\.?\s*no\b', re.I),
        re.compile(r'\bdescription\b', re.I),
        re.compile(r'\blocation\b', re.I),
        re.compile(r'\breading\b', re.I),
        re.compile(r'\bresult\b', re.I),
        re.compile(r'\bidentification\b', re.I),
    ]

    # Scan tokens for table header row
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        y = data['top'][i]
        if not txt or int(data['conf'][i]) < 30:
            continue
        for kw in table_header_kws:
            if kw.search(txt) and y > img_height * 0.25:
                # Found a table header keyword — table starts just below it
                table_start_y = y + 15
                break

    # Detect table end by looking for signature-area keywords
    sig_kws = re.compile(r'\b(?:NDT\s*LAB|SIGNATURE|STAMP|WITNESSED|CLIENT|APPROVED)\b', re.I)
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        y = data['top'][i]
        if not txt or int(data['conf'][i]) < 30:
            continue
        if sig_kws.search(txt) and y > img_height * 0.60:
            table_end_y = y - 10
            break

    # ── Group tokens by row y-coordinate within table body ───────────
    # Use wider y-tolerance (18px) for ISGEC's taller bordered rows
    ROW_Y_TOLERANCE = 18

    rows: dict[int, list[tuple]] = {}
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        conf = int(data['conf'][i])
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        if not txt or conf < 25:
            continue
        if y < table_start_y or y > table_end_y:
            continue

        matched = False
        for ry in rows:
            if abs(ry - y) < ROW_Y_TOLERANCE:
                rows[ry].append((txt, x, y, w, h, conf))
                matched = True
                break
        if not matched:
            rows[y] = [(txt, x, y, w, h, conf)]

    # ── Detect column boundaries dynamically ─────────────────────────
    # ISGEC forms vary in column widths. Instead of hardcoded x thresholds,
    # find where 3-digit numbers cluster (that's the readings area).
    all_reading_xs = []
    for ry, tokens in rows.items():
        for tok, x, y, w, h, conf in tokens:
            if re.match(r'^\d{3}$', tok) and 100 <= int(tok) <= 400:
                all_reading_xs.append(x)

    # Readings x-boundary: left edge of the readings region
    if all_reading_xs:
        readings_x_start = max(0, min(all_reading_xs) - 20)
    else:
        # Fallback: readings typically start at ~55% of image width for ISGEC
        readings_x_start = int(img_width * 0.55)

    # ── Parse rows into structured TableRow objects ──────────────────
    table_rows: list[TableRow] = []

    for ry in sorted(rows.keys()):
        tokens = sorted(rows[ry], key=lambda item: item[1])

        # Split into label tokens (left of readings) and reading tokens
        label_tokens = [tok for tok in tokens if tok[1] < readings_x_start]
        reading_tokens = [tok for tok in tokens if tok[1] >= readings_x_start]

        label_str = " ".join(t[0] for t in label_tokens)

        # Detect component using the broader ISGEC pattern set
        component, is_weld = _detect_isgec_component(label_str)

        # If no component detected from label, check if the row has readings
        # and try to infer from context (serial number rows, etc.)
        if component is None:
            # Check if there are 3-digit hardness readings in this row
            has_readings = any(
                re.match(r'^\d{3}$', t[0]) and 100 <= int(t[0]) <= 400
                for t in reading_tokens
            )
            if not has_readings:
                continue
            # A row with readings but no legible location.
            #
            # This used to be called "Pipe", which the zone mapper turns into
            # base_metal — so an unreadable label silently became a base-metal
            # CLAIM. On 241135LBP-HRDR001 that applied to all 361 rows and
            # graded weld readings of 249-269 against base metal's 195-248 band,
            # producing 194 findings the specification does not support.
            #
            # "Unknown" maps to no zone, so the row is reported AMBIGUOUS_ZONE
            # and routed for adjudication rather than judged on a guess.
            component = "Unknown"
            is_weld = False

        # Extract joint name from label text
        joint_name = _detect_joint(label_str)
        # If no J-number found, try SW- numbers
        if joint_name == "General Joint":
            sw_m = re.search(r'SW[-]?\d+[A-Z]?', label_str, re.I)
            if sw_m:
                joint_name = f"Joint {sw_m.group(0).upper()}"

        # Extract hardness readings (3-digit numbers in the readings columns)
        readings = []
        for tok, rx, ry_pos, rw, rh, rconf in reading_tokens:
            val = _sanitize_ocr_reading(tok)
            if val is not None:
                # Convert pixel coordinates to PDF page coordinates
                readings.append(Reading(
                    value=val,
                    x=int(rx * scale_x),
                    y=int(ry_pos * scale_y),
                    w=int(rw * scale_x),
                    h=int(rh * scale_y),
                    conf=rconf,
                ))

        if readings:  # Only add rows that have actual readings
            table_rows.append(TableRow(
                ry=ry,
                component=component,
                is_weld=is_weld,
                left_text=label_str,
                joint_name=joint_name,
                readings=readings,
            ))

    logger.info(
        f"  ISGEC page {page_idx + 1}: {len(table_rows)} rows, "
        f"{sum(len(r.readings) for r in table_rows)} readings, "
        f"acceptance={acceptance_range}, material={material_spec[:30]}"
    )

    return PageOCRResult(
        page_idx=page_idx,
        full_text=full_text,
        drawing_no=drawing_no,
        spool_no=spool_no,
        table_rows=table_rows,
        test_block_info=test_block_info,
        report_format="ISGEC_SINGLE",
        has_astm_prefix=has_astm,
        acceptance_range=acceptance_range,
        material_spec=material_spec,
    )


# ── Single Page OCR Extraction ────────────────────────────────────────

# A printed spool designation, e.g. "RP-03-08" / "RP.03-08" / "HP-18-02". Shared
# by the OCR and text-layer paths so a group boundary is the same thing to both.
#
# The two numeric groups are required, and a match glued to a preceding token is
# refused: the older pattern matched "P22-D3" INSIDE the test-block string
# "DDEL-TH-P22-D3", so every row on page 13 reported a test-coupon identifier as
# its spool.
_SPOOL_HEADER_RE = re.compile(
    r"(?<![A-Z0-9])((?:HP|RP|CSP|SIL|P11|P22|CS)[-.]\d{2}[-.]\d{2})", re.I
)


def _parse_page_header(full_text: str, report_format: str) -> dict:
    """
    The identifiers printed on a page: drawing, spool, test block, material.

    Shared by both extraction paths so a page reports the same header no matter
    how its table was read — otherwise project scoping and material resolution
    would depend on which route the page happened to take.
    """
    drawing_m = re.search(
        r"\b(?:221135LBP-|1-191735LBP-)?(HP|RP|CSP|SIL)[A-Z0-9-]*\b", full_text, re.I
    )
    spool_m = re.search(
        r"\b(?:\d-)?(?:HP-\d{2}-\d{2}|RP-\d{2}-\d{2}|HP-\d+|RP-\d+|221135[A-Z0-9-]+)\b",
        full_text, re.I,
    )
    is_isgec = report_format == "ISGEC_SINGLE"
    return {
        "drawing_no": drawing_m.group(0).upper() if drawing_m else "",
        "spool_no": spool_m.group(0).upper() if spool_m else "Submittal Spool",
        "test_block_info": _extract_test_block(full_text),
        "has_astm": bool(re.search(
            r"\b(?:SA-?335|SA-?234|SA-?182|SA-?387|ASTM|ASME)\b", full_text, re.I)),
        "acceptance_range": _parse_isgec_acceptance_range(full_text) if is_isgec else None,
        "material_spec": _parse_isgec_material_spec(full_text) if is_isgec else "",
    }


# Column boundaries of the DEE_TABULAR report, in 150-DPI pixels. Measured on
# 221135LBP-HRDR001: description x 43-129, Material Grade x 219, Joint No.
# x 288, Size x 342-378, hardness values x 430+.
# These remain only as the FALLBACK for a page whose header cannot be read.
# Anything derived from the page itself takes precedence — see _derive_columns.
_READING_X_MIN = 430   # hardness value columns start here
_DESC_X_MAX = 150      # "Item Description (Spool No.)" column ends here
_GRADE_X = (180, 260)  # "Material Grade" cell
_JOINT_X = (260, 320)  # "Joint No." cell

# Header words that name each column. A form states what its columns are; these
# are the words to look for, not the positions to assume.
_HEADER_ANCHORS: list[tuple[str, re.Pattern]] = [
    ("desc", re.compile(r"^(?:ITEM|DESCRIPTION|COMPONENT|LOCATION|SPOOL)$", re.I)),
    ("material", re.compile(r"^(?:MATERIAL|GRADE)$", re.I)),
    ("joint", re.compile(r"^(?:JOINT|JT)$", re.I)),
    ("size", re.compile(r"^(?:SIZE|THK|THICKNESS|\(INCH\)|INCH)$", re.I)),
    ("reading", re.compile(r"^(?:H\s?\d{1,2}|AVG\.?|HARDNESS)$", re.I)),
]


def _derive_columns(tokens: list[dict]) -> dict:
    """
    Column boundaries read from the page's own header words.

    The constants above describe ONE vendor's form at ONE resolution. Any other
    supplier — or the same supplier at a different DPI — has its columns
    somewhere else, and a hardcoded boundary does not fail loudly there: it
    silently reads the wrong cell and produces findings that look real. That is
    the failure mode this exists to remove.

    The form itself says where its columns are. "Material", "Grade", "Joint",
    "Size", "H1".."H5" are printed above the data, so their x-positions define
    the boundaries, and each boundary is placed midway between neighbouring
    anchors. Where a header word is missing the corresponding fallback constant
    is kept, so a partially-legible header degrades one column at a time rather
    than all at once.
    """
    anchors: dict[str, list[int]] = {}
    for tok in tokens:
        word = (tok.get("text") or "").strip().strip(".:()")
        for role, pattern in _HEADER_ANCHORS:
            if pattern.fullmatch(word):
                anchors.setdefault(role, []).append(int(tok["x"]))
                break

    def mid(a: str, b: str) -> Optional[int]:
        if a in anchors and b in anchors:
            left, right = min(anchors[a]), min(anchors[b])
            if left < right:
                return (left + right) // 2
        return None

    cols = {
        "desc_max": _DESC_X_MAX,
        "grade": _GRADE_X,
        "joint": _JOINT_X,
        "reading_min": _READING_X_MIN,
        "source": "fallback constants",
    }

    desc_material = mid("desc", "material")
    material_joint = mid("material", "joint")
    joint_size = mid("joint", "size")
    reading_start = None
    if "reading" in anchors:
        first_reading = min(anchors["reading"])
        prior = [min(anchors[r]) for r in ("size", "joint", "material")
                 if r in anchors and min(anchors[r]) < first_reading]
        reading_start = ((max(prior) + first_reading) // 2) if prior else first_reading - 10

    derived = []
    if desc_material is not None:
        cols["desc_max"] = desc_material
        derived.append("desc")
    if desc_material is not None and material_joint is not None:
        cols["grade"] = (desc_material, material_joint)
        derived.append("material")
    if material_joint is not None:
        upper = joint_size if joint_size is not None else material_joint + 60
        cols["joint"] = (material_joint, upper)
        derived.append("joint")
    if reading_start is not None:
        cols["reading_min"] = reading_start
        derived.append("readings")
    if derived:
        cols["source"] = "page header (" + ", ".join(derived) + ")"
    return cols


# Digit confusions specific to the Material Grade cell. The grades that can
# appear there use only the digits 1, 2, 4, 5 and 9 (P91, P22, P11, P4, P5A,
# P15E), so a letter standing in for a digit is unambiguous here in a way it is
# not elsewhere: "O" is a 9, never a 0. Measured on page 13, "P91" came back as
# "poi" and "Pot" — three of the four grade cells on the page were lost to this,
# and with them the material of every row that inherits from them.
_GRADE_OCR_FIXES = str.maketrans({
    "O": "9", "Q": "9", "D": "9",
    "I": "1", "L": "1", "T": "1", "|": "1", "!": "1",
    "S": "5", "Z": "2", "G": "6", "B": "8",
})


def _grade_from_cell(text: str) -> Optional[str]:
    """The grade a Material Grade cell states, tolerating OCR digit damage."""
    if grade := canonical_grade(text):
        return grade
    token = (text or "").strip().upper()
    # Only repair things already shaped like a P-designation, so "CS" is never
    # mangled into "C5" by the digit map.
    if re.fullmatch(r"P[A-Z0-9]{1,3}", token):
        return canonical_grade("P" + token[1:].translate(_GRADE_OCR_FIXES))
    return None


def _joint_from_cell(text: str) -> Optional[str]:
    """
    The joint a Joint No. cell states, with or without its "J".

    The leading J is routinely lost on these scans — page 13's J26 and J27 both
    arrive as bare "26" and "27" — and `_detect_joint`, which works on the row's
    whole label text, cannot tell a bare joint number from a pipe size. Read out
    of the Joint No. COLUMN there is no ambiguity: whatever is in that cell is
    the joint. Without this every row on the page kept the last J-prefixed joint
    seen (J25), so three joint groups merged into one and one group's material
    inheritance spilled into the other two.
    """
    token = (text or "").strip().upper().replace(".", "")
    m = re.fullmatch(r"J?\s*(\d{1,3})", token.translate(_OCR_DIGIT_FIXES))
    return f"Joint J{m.group(1)}" if m else None


def _fit_column_offset(tokens: list[dict], band_ys: list[float], pitch: float) -> int:
    """
    The vertical offset that best aligns one column's tokens to the row bands.

    Fitted globally over the column rather than resolved per token, because a
    per-token nearest-band decision is a coin flip exactly when it matters: the
    description labels sit half a row pitch from their band (13px of a 26px
    pitch), so each one is equidistant from two bands and ties break
    arbitrarily. Scoring one offset against every token at once has no ties —
    the correct offset drives total distance to nearly zero while its
    neighbours cost half a pitch per token.
    """
    if not tokens or not band_ys:
        return 0
    best_off, best_cost = 0, None
    for off in range(-int(pitch), int(pitch) + 1):
        cost = 0.0
        for tok in tokens:
            y = tok["y"] + off
            cost += min(abs(b - y) for b in band_ys)
        if best_cost is None or cost < best_cost:
            best_off, best_cost = off, cost
    return best_off


def _band_rows_by_readings(tokens: list[dict],
                           cols: Optional[dict] = None) -> Optional[list[list[dict]]]:
    """
    Group tokens into table rows using the hardness columns as the skeleton.

    WHY THE READINGS AND NOT THE LABELS
    ───────────────────────────────────
    Clustering every token together by proximity assumes one row occupies one
    horizontal band. In this report it does not. Measured on page 13 of
    221135LBP-HRDR001 (150 DPI, 26px row pitch):

        row label   "Pipe"@531  "Hazt"@557  "Weld"@584  "Haz2"@610
        grade/joint             "P91"@597   "J25"@597
        readings    223@544     227@570     245@597     225@623

    The description labels are drawn 13px ABOVE their own readings; the Material
    Grade, Joint No. and Size cells are drawn ON them. One tolerance cannot
    serve both, and the old nearest-anchor pass paired each reading band with
    the label 13px below it — the NEXT row's label. Every row on the page came
    out shifted: the readings of the Haz1 row arrived labelled "Weld", and
    "P91 J25" landed on a Haz2 row.

    That shift is what put P91 material on the wrong rows, collapsed J26 and J27
    into J25, and made "Haz1" disappear entirely — and downstream it is why P91
    readings were graded against P22's 225 HB ceiling.

    The readings are the reliable skeleton: six to twenty-four per row, tightly
    aligned, and unambiguous. So rows are defined by them, and each label column
    is then fitted to those rows with its own offset — which self-corrects to 0
    on a page whose columns are aligned, so a well-formed report is unaffected.

    Returns None when the page has too few readings to establish a skeleton, in
    which case the caller keeps the previous proximity clustering.
    """
    cols = cols or {}
    reading_min = cols.get("reading_min", _READING_X_MIN)
    desc_max = cols.get("desc_max", _DESC_X_MAX)
    reads = [t for t in tokens
             if t["x"] >= reading_min and _sanitize_ocr_reading(t["text"]) is not None]
    if len(reads) < 6:
        return None

    reads.sort(key=lambda t: t["y"])
    bands: list[list[dict]] = [[reads[0]]]
    for tok in reads[1:]:
        if abs(tok["y"] - bands[-1][0]["y"]) <= max(8.0, tok["h"] * 0.8):
            bands[-1].append(tok)
        else:
            bands.append([tok])
    if len(bands) < 3:
        return None

    band_ys = [float(b[0]["y"]) for b in bands]
    gaps = sorted(band_ys[i + 1] - band_ys[i] for i in range(len(band_ys) - 1))
    pitch = gaps[len(gaps) // 2] or 20.0

    columns = (
        [t for t in tokens if t["x"] < desc_max],
        [t for t in tokens if desc_max <= t["x"] < reading_min],
        [t for t in tokens if t["x"] >= reading_min
         and _sanitize_ocr_reading(t["text"]) is None],
    )
    # The offset is fitted on the TABLE BODY only. Column-header text
    # ("Item Description", "Material Grade"), the remark line and the signature
    # block all fall in these x ranges but belong to no row, and including them
    # dragged the fit away from the true offset — the P91/J25 cells then missed
    # their weld row by one band, which is the very error being corrected here.
    body_lo, body_hi = band_ys[0] - pitch, band_ys[-1] + pitch
    lines: list[list[dict]] = [list(b) for b in bands]
    for column in columns:
        body = [t for t in column if body_lo <= t["y"] <= body_hi]
        offset = _fit_column_offset(body, band_ys, pitch)
        for tok in column:
            y = tok["y"] + offset
            k = min(range(len(band_ys)), key=lambda i: abs(band_ys[i] - y))
            # Beyond a little over half a pitch the token belongs to no row —
            # page headers, the remark line and signature blocks land here.
            if abs(band_ys[k] - y) <= pitch * 0.6:
                lines[k].append(tok)
    return lines


def extract_page_ocr(doc: fitz.Document, page_idx: int) -> PageOCRResult:
    """
    Extract one page's table, by whichever route the page itself supports.

    The choice is made per PAGE from evidence, not per document and not from the
    filename. 221135LBP-HRDR001 is mixed — pages 1, 2, 3 and 41 carry no text at
    all while pages 13 and 80 carry 381 and 283 words — so a per-document
    decision is wrong either way.

    Order of preference:
      1. the page's own text layer, when it has one and its table can be read
         with confidence (exact values, no OCR artefacts, ~50ms)
      2. OCR of a rendered image (this module's original behaviour), which is
         the right and only tool for a genuine scan such as 241135LBP-HRDR001
         at 2 words per page

    Both routes return the same PageOCRResult, so nothing downstream can tell
    them apart.
    """
    page = doc[page_idx]

    # ── Route 1: the page's own text layer ───────────────────────────
    if page_has_text_layer(page):
        page_text = page.get_text("text") or ""
        report_format = _detect_report_format(page_text)
        header = _parse_page_header(page_text, report_format)
        detector = (_detect_isgec_component if report_format == "ISGEC_SINGLE"
                    else _detect_component)
        result = extract_page_text_layer(
            page, page_idx, report_format, page_text,
            header["drawing_no"], header["spool_no"], header["test_block_info"],
            header["acceptance_range"], header["material_spec"], header["has_astm"],
            detector, _SPOOL_HEADER_RE,
        )
        if result is not None:
            return result
        logger.info(
            f"  page {page_idx + 1}: has a text layer but no table could be read from it "
            f"with confidence; falling back to OCR."
        )

    # ── Route 2: render and OCR ──────────────────────────────────────
    # Quick format detection: get raw text first to decide which extractor to use
    pix_quick = page.get_pixmap(dpi=100)  # low-res for fast format detection
    img_quick = Image.open(io.BytesIO(pix_quick.tobytes()))
    quick_text = pytesseract.image_to_string(
        img_quick, config="--psm 6 --oem 3"
    ).strip()

    report_format = _detect_report_format(quick_text)

    # Route to ISGEC-specific extractor
    if report_format == "ISGEC_SINGLE":
        return _extract_isgec_single(page, page_idx)

    # ── DEE_TABULAR / GOLDEN_PASS extraction (improved grid/spatial clustering) ──
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes()))
    # Preprocess to eliminate grid lines and contrast noise
    cleaned_img = _preprocess_isgec_image(img)
    data = pytesseract.image_to_data(cleaned_img, output_type=pytesseract.Output.DICT)

    # Second pass on the UNPROCESSED image, used only for the label column.
    #
    # The grid-line removal that makes the numeric columns legible also erodes
    # thin glyphs in the label column. Measured on page 13 of
    # 221135LBP-HRDR001: after preprocessing, not one "J" survives — the joint
    # numbers arrive as bare digits ("27" at x=288) or vanish entirely, so every
    # row is labelled "General Joint" and joint-scoped material inheritance has
    # nothing to work with. The same page OCR'd without preprocessing yields
    # "J25", "J26", "Jaz" at that same x-position.
    #
    # So: keep the cleaned pass for readings, and merge in label-region tokens
    # the cleaned pass lost. Costs one extra OCR pass per page.
    # Crop to the label column before the second pass. Only x < 430 is ever
    # merged, so OCRing the full page here wastes roughly three quarters of the
    # work — the readings columns are already covered by the cleaned pass.
    _LABEL_COL_PX = 470  # a little wider than the merge window, for context
    _raw_crop = img.crop((0, 0, min(_LABEL_COL_PX, img.width), img.height))
    raw_data = pytesseract.image_to_data(_raw_crop, output_type=pytesseract.Output.DICT)

    # Build full text from OCR tokens
    full_text = " ".join(
        data["text"][i] for i in range(len(data["text"])) if data["text"][i].strip()
    )

    # Extract drawing number
    drawing_m = re.search(
        r"\b(?:221135LBP-|1-191735LBP-)?(HP|RP|CSP|SIL)[A-Z0-9-]*\b", full_text, re.I
    )
    drawing_no = drawing_m.group(0).upper() if drawing_m else ""

    # Extract spool number
    spool_m = re.search(
        r"\b(?:\d-)?(?:HP-\d{2}-\d{2}|RP-\d{2}-\d{2}|HP-\d+|RP-\d+|221135[A-Z0-9-]+)\b",
        full_text, re.I,
    )
    spool_no = spool_m.group(0).upper() if spool_m else "Submittal Spool"

    # Extract test block info — the whole declaration; see _extract_test_block.
    test_block_info = _extract_test_block(full_text)

    # Check for ASTM/ASME prefix
    has_astm = bool(re.search(
        r"\b(?:SA-?335|SA-?234|SA-?182|SA-?387|ASTM|ASME)\b", full_text, re.I
    ))

    # Collect valid tokens (filtering out table border line tokens)
    tokens = []
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        conf = int(data["conf"][i])
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        if y > 350 and conf > 20 and t and not re.match(r"^[|\[\]_.\-–—]+$", t):
            tokens.append({"text": t, "x": x, "y": y, "w": w, "h": h, "conf": conf})

    # Merge label-column tokens recovered from the unprocessed image. Only the
    # label region (x < LABEL_COL_MAX) is considered, and only tokens that do
    # not already have a counterpart at roughly the same position — so this
    # restores what preprocessing destroyed without disturbing the readings.
    _MERGE_X_MAX = 430
    for i in range(len(raw_data["text"])):
        t = raw_data["text"][i].strip()
        conf = int(raw_data["conf"][i])
        x, y = raw_data["left"][i], raw_data["top"][i]
        w, h = raw_data["width"][i], raw_data["height"][i]
        if not (y > 350 and conf > 20 and t) or x >= _MERGE_X_MAX:
            continue
        if re.match(r"^[|\[\]_.\-–—]+$", t):
            continue
        # Skip if the cleaned pass already produced a token here.
        if any(abs(e["x"] - x) < 25 and abs(e["y"] - y) < 12 for e in tokens):
            continue
        tokens.append({"text": t, "x": x, "y": y, "w": w, "h": h, "conf": conf})

    tokens.sort(key=lambda tok: tok["y"])

    # Rows come from the readings columns; each label column is then fitted to
    # them. See _band_rows_by_readings for the measurements behind this — in
    # short, the description labels and the grade/joint cells sit at DIFFERENT
    # heights within the same row, so no single proximity rule places both.
    cols = _derive_columns(tokens)
    logger.debug(f"  page {page_idx + 1}: columns from {cols['source']} — "
                 f"desc<{cols['desc_max']} grade{cols['grade']} joint{cols['joint']} "
                 f"readings>={cols['reading_min']}")
    lines = _band_rows_by_readings(tokens, cols)
    if lines is None:
        # Too few readings to establish a skeleton (a cover sheet, a page whose
        # table failed to OCR). Fall back to proximity clustering: anchor on the
        # line's first token, pick the NEAREST line, and scale tolerance with
        # glyph height.
        lines = []
        anchors: list[float] = []
        for tok in tokens:
            tol = max(12.0, tok["h"] * 0.9)
            best_i, best_d = -1, tol + 1.0
            for i, anchor in enumerate(anchors):
                d = abs(tok["y"] - anchor)
                if d <= tol and d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0:
                lines[best_i].append(tok)
            else:
                lines.append([tok])
                anchors.append(float(tok["y"]))

    lines.sort(key=lambda l: min(t["y"] for t in l))

    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height

    # Parse rows into structured TableRow objects
    table_rows: list[TableRow] = []
    group_start_flags: list[bool] = []
    current_joint = "General Joint"
    current_spool = spool_no
    for line in lines:
        line.sort(key=lambda item: item["x"])
        left_tokens = [item["text"] for item in line if item["x"] < cols["reading_min"]]
        left_str = " ".join(left_tokens)

        # Spool header, e.g. "RP-03-08" / "RP.03-08" / "HP-18-02".
        #
        # The old pattern was `\b((?:HP|RP|P11|P22|CS|SIL|CSP)[A-Z0-9-]*)\b`,
        # which matched "P22-D3" INSIDE the test-block string "DDEL-TH-P22-D3" —
        # the hyphen before "P22" is a word boundary. Every row on page 13 then
        # reported spool "P22-D3" instead of "RP-03-08", so the spool column
        # carried a test-coupon identifier and named a material that had nothing
        # to do with the rows. Requiring the two numeric groups of a real spool
        # designation, and refusing a match glued to a preceding token, keeps it
        # to actual spool numbers.
        #
        # The header no longer has to be alone on its line either: now that rows
        # band correctly, "RP.03-08" shares a band with the first component of
        # its group, and the old `not _detect_component(...)` guard would have
        # discarded it.
        spool_header_m = _SPOOL_HEADER_RE.search(left_str)
        starts_group = bool(spool_header_m)
        if spool_header_m:
            current_spool = spool_header_m.group(1).upper().replace(".", "-")
            # A printed spool header opens a new joint group. Without clearing
            # the running joint here, the first rows of a group — the ones drawn
            # ABOVE its weld row, which is where the joint number is printed —
            # keep the PREVIOUS group's joint and inherit that group's material.
            # On page 13 that gave the third group's Reducer and Haz1 rows the
            # P22 of the second group's pipe, and P22's 225 HB ceiling then
            # failed readings that P91 permits to 248.
            current_joint = "General Joint"

        component, is_weld = _detect_component(left_str)
        if component is None:
            continue

        # Joint No. read from its own column first — see _joint_from_cell.
        row_joint = "General Joint"
        for item in line:
            if cols["joint"][0] <= item["x"] < cols["joint"][1]:
                if joint := _joint_from_cell(item["text"]):
                    row_joint = joint
                    break
        if row_joint == "General Joint":
            row_joint = _detect_joint(left_str)
        if row_joint != "General Joint":
            current_joint = row_joint
        # NOTE: this used to read `elif is_weld: current_joint = row_joint`,
        # which assigned "General Joint" back into current_joint whenever a weld
        # row had no legible joint number — wiping the running joint context for
        # every row after it. A row with no readable joint number should inherit
        # the joint in force, not destroy it.

        # Extract hardness readings (3-digit numbers in data columns)
        readings = []
        for tok in line:
            if tok["x"] >= cols["reading_min"]:
                val = _sanitize_ocr_reading(tok["text"])
                if val is not None:
                    readings.append(Reading(
                        value=val,
                        x=int(round(tok["x"] * scale_x)),
                        y=int(round(tok["y"] * scale_y)),
                        w=int(round(tok["w"] * scale_x)),
                        h=int(round(tok["h"] * scale_y)),
                        conf=tok["conf"],
                    ))

        # Row-level material grade detection from the row's own label text.
        #
        # Uses the shared canonicaliser rather than substring matching. The old
        # chain tested `"CS" in left_upper`, which fires on any label containing
        # those two letters, and `"P11" in left_upper`, which fires inside
        # "P110" — while recognising no ASME P-Number designation at all, so a
        # label reading "P5A" or "P-No. 4" yielded no material. canonical_grade
        # matches on whole tokens and understands both vocabularies.
        # Material Grade read from its own column first, so an OCR-damaged
        # "poi" is recovered as P91 and a grade cannot be picked up from a
        # neighbouring column's text.
        material_hint = ""
        material_hint_raw = ""
        for item in line:
            if cols["grade"][0] <= item["x"] < cols["grade"][1]:
                if g := _grade_from_cell(item["text"]):
                    material_hint = g
                    break
                # Grade-shaped but unreadable. Measured on page 35 of
                # 221135LBP-HRDR001: the J48 weld cell OCR'd as "P94" — a 1 read
                # as a 4 — and repairing it is not safe, because "P94" is one
                # edit from both P91 and P4 (which is P11). Recording that the
                # cell EXISTS is safe, and it is what stops the row inheriting a
                # neighbouring group's grade as though it stated nothing.
                if re.fullmatch(r"P[A-Z0-9]{1,3}", (item["text"] or "").strip().upper()):
                    material_hint_raw = item["text"].strip()
        if not material_hint:
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-/]*", left_str):
                g = canonical_grade(token)
                if g:
                    material_hint = g
                    break

        ry_avg = int(round(sum(tok["y"] for tok in line) / len(line)))
        group_start_flags.append(starts_group)
        table_rows.append(TableRow(
            ry=ry_avg,
            component=component,
            is_weld=is_weld,
            left_text=left_str,
            joint_name=row_joint if row_joint != "General Joint" else current_joint,
            readings=readings,
            material_hint=material_hint,
            material_hint_raw=material_hint_raw,
            spool_name=current_spool,
        ))

    # The Joint No. cell is merged across the rows of its group and printed
    # once, on the weld row in the middle. Rows above it therefore carry no
    # joint of their own, and joint-scoped material inheritance — the mechanism
    # that gives a Pipe or Haz row the grade stated on its weld row — cannot
    # reach them. The group's own boundaries are printed on the page as the
    # spool header, so use them: within one group, a single joint number applies
    # to every row. A group showing more than one joint is left alone rather
    # than forced into agreement.
    starts = [i for i, flag in enumerate(group_start_flags) if flag]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    for lo, hi in zip(starts, starts[1:] + [len(table_rows)]):
        joints = {r.joint_name for r in table_rows[lo:hi] if r.joint_name != "General Joint"}
        if len(joints) == 1:
            joint = joints.pop()
            for row in table_rows[lo:hi]:
                row.joint_name = joint

    return PageOCRResult(
        page_idx=page_idx,
        full_text=full_text,
        drawing_no=drawing_no,
        spool_no=spool_no,
        table_rows=table_rows,
        test_block_info=test_block_info,
        report_format=report_format,
        has_astm_prefix=has_astm,
    )


# ── LangGraph Node: OCR Extractor ────────────────────────────────────

# Physically plausible Brinell window, matching compliance_check. Outside it a
# number is an extraction artefact rather than a measurement.
_PLAUSIBLE = (50, 600)


def _assess_extraction(result: PageOCRResult, page: "fitz.Page") -> None:
    """
    Decide whether this page's table was read well enough to grade against.

    Sets `extraction_ok` / `extraction_warnings` on the result. This is the gate
    that keeps a misread page out of the report entirely, and it exists because
    of how the failures actually looked on 2026-08-20: page 13's grade cells
    landed one row late and produced findings that were indistinguishable from
    genuine ones — same shape, same confidence, same wording. Nothing downstream
    could have told them apart, and no reviewer would have known to look.

    "This page could not be read reliably" is a statement a reviewer can act on.
    A confident wrong answer is not. So the checks below are deliberately about
    the EXTRACTION, not the metallurgy: whether the numbers are physically
    possible, whether a page that clearly contains a table produced rows, and
    whether the rows are internally consistent.
    """
    warnings: list[str] = []
    rows = result.table_rows
    values = [r.value for row in rows for r in row.readings if r.value is not None]

    # 1. A page that plainly holds a hardness table but yielded nothing.
    text = (result.full_text or "").upper()
    looks_like_table = ("HARDNESS" in text or "BHN" in text) and sum(
        c.isdigit() for c in text) > 40
    if not rows and looks_like_table:
        warnings.append("page appears to contain a hardness table but no rows were extracted")

    # 2. Values outside anything a Brinell test can produce.
    if values:
        implausible = [v for v in values if not (_PLAUSIBLE[0] <= v <= _PLAUSIBLE[1])]
        share = len(implausible) / len(values)
        if share > 0.20:
            warnings.append(
                f"{len(implausible)} of {len(values)} readings ({share:.0%}) are outside "
                f"{_PLAUSIBLE[0]}-{_PLAUSIBLE[1]} HB — the readings columns were probably misread")

    # 3. Rows that carry a single reading where the rest of the page carries a
    #    full set. On these forms every row is measured the same number of
    #    times, so an odd row out means cells were lost.
    if len(rows) >= 4:
        counts = [len(r.readings) for r in rows]
        typical = max(counts)
        # A FRAGMENT is a row holding almost nothing — one or two stray numbers
        # where the page holds sets of six or more. Rows holding *fewer* readings
        # than the fullest row are NOT fragments: the DEE form carries four
        # measurement blocks (0°, 90°, 180°, 270°) selected by diameter, printed
        # on the form as "For Dia. <=6" / >6" & <=12" / >12"", so a small-bore row
        # legitimately has 6 readings on a page whose large-bore rows have 24.
        #
        # Comparing every row to the fullest one flagged page 13 of
        # 221135LBP-HRDR001 — a page verified row-for-row against the paper — and
        # excluded 84 correctly-read rows from grading. The threshold now scales
        # with the page but never rises above a genuine fragment.
        floor = max(3, typical // 4)
        starved = [c for c in counts if c < floor]
        if typical >= 6 and len(starved) > len(rows) * 0.25:
            warnings.append(
                f"{len(starved)} of {len(rows)} rows hold fewer than {floor} readings "
                f"where the fullest holds {typical} — cells were probably dropped")

    result.extraction_warnings = warnings
    result.extraction_ok = not warnings
    if warnings:
        pipeline_logger.warning(
            f"Page {result.page_idx + 1}: extraction not trusted — " + "; ".join(warnings)
            + ". Its rows are reported unevaluated rather than graded."
        )


def ocr_extractor_node(state: ReviewState) -> dict:
    """
    LangGraph node: Extract structured OCR data from ALL pages of the PDF.
    Runs in parallel using ThreadPoolExecutor for speed.
    """
    filepath = state.get("submittal_filepath", "")
    if not filepath or not Path(filepath).exists():
        pipeline_logger.warning("No valid PDF filepath for OCR extraction")
        return {"page_ocr_results": [], "current_agent": "ocr_extractor"}

    with StageTracker("ocr_extract", f"filepath={filepath}") as tracker:
        doc = fitz.open(filepath)
        total_pages = len(doc)
        tracker.details += f" | total_pages={total_pages}"
        pipeline_progress.set_total_pages(total_pages)

        def _extract(p_idx: int) -> PageOCRResult:
            pipeline_progress.start_page(p_idx)
            try:
                return extract_page_ocr(doc, p_idx)
            except Exception as e:
                logger.warning(f"OCR extraction error on page {p_idx}: {e}")
                return PageOCRResult(
                    page_idx=p_idx, full_text="", drawing_no="", spool_no=""
                )
            finally:
                pipeline_progress.finish_page(p_idx)

        max_ocr_workers = min(12, max(4, (os.cpu_count() or 8)))
        with ThreadPoolExecutor(max_workers=max_ocr_workers) as pool:
            results = list(pool.map(_extract, range(total_pages)))

        # Gate every page before anything downstream sees it & collect PyMuPDF trace metadata
        page_details = []
        total_digital_chars = 0
        total_embedded_images = 0
        gauge_candidate_images = 0

        for res in results:
            p_idx = res.page_idx
            page_obj = doc[p_idx]
            try:
                _assess_extraction(res, page_obj)
            except Exception as exc:
                logger.debug(f"extraction assessment failed on page {res.page_idx + 1}: {exc}")

            # PyMuPDF digital text & image extraction metrics
            try:
                digital_text = page_obj.get_text("text").strip()
                digital_len = len(digital_text)
                total_digital_chars += digital_len
                raw_imgs = page_obj.get_images(full=True)
                page_imgs = []
                for img_info in raw_imgs:
                    xref = img_info[0]
                    try:
                        base_img = doc.extract_image(xref)
                        w = base_img.get("width", 0)
                        h = base_img.get("height", 0)
                        ext = base_img.get("ext", "png")
                        is_gauge_candidate = w >= 120 and h >= 120
                        if is_gauge_candidate:
                            gauge_candidate_images += 1
                        page_imgs.append({
                            "xref": xref,
                            "width": w,
                            "height": h,
                            "ext": ext,
                            "is_gauge_candidate": is_gauge_candidate,
                        })
                    except Exception:
                        pass
                total_embedded_images += len(page_imgs)

                page_details.append({
                    "page_number": p_idx + 1,
                    "is_scanned": digital_len < 50,
                    "digital_chars": digital_len,
                    "ocr_chars": len(res.full_text),
                    "embedded_images_count": len(page_imgs),
                    "images": page_imgs,
                    "drawing_no": res.drawing_no,
                    "spool_no": res.spool_no,
                    "test_block_info": res.test_block_info,
                    "report_format": res.report_format,
                    "table_rows_count": len(res.table_rows),
                    "components": list(set(r.component for r in res.table_rows if r.component)),
                    "readings_count": sum(len(r.readings) for r in res.table_rows),
                    "extraction_ok": res.extraction_ok,
                    "extraction_warnings": res.extraction_warnings,
                    "ocr_text_snippet": res.full_text[:400].strip(),
                })
            except Exception as e:
                logger.debug(f"Failed to gather page trace on page {p_idx+1}: {e}")

        doc.close()

        pages_with_data = sum(1 for r in results if r.table_rows)
        unreliable = [r.page_idx + 1 for r in results if not r.extraction_ok]
        tracker.details += f" | pages_with_table_data={pages_with_data}"
        if unreliable:
            tracker.details += f", extraction_unreliable={len(unreliable)}"
            pipeline_logger.warning(
                f"{len(unreliable)} page(s) could not be read reliably and will not be graded: "
                f"{unreliable[:12]}{' ...' if len(unreliable) > 12 else ''}"
            )

        # Persist structured OCR and PyMuPDF trace log
        try:
            ocr_trace_payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "filepath": str(filepath),
                "filename": Path(filepath).name,
                "total_pages": total_pages,
                "pymupdf_summary": {
                    "pages_with_digital_text": sum(1 for p in page_details if not p.get("is_scanned")),
                    "total_digital_chars": total_digital_chars,
                    "total_embedded_images": total_embedded_images,
                    "gauge_candidate_images": gauge_candidate_images,
                },
                "ocr_summary": {
                    "total_ocr_chars": sum(len(r.full_text) for r in results),
                    "pages_with_table_data": pages_with_data,
                    "total_table_rows": sum(len(r.table_rows) for r in results),
                    "total_readings": sum(sum(len(row.readings) for row in r.table_rows) for r in results),
                    "unreliable_pages": unreliable,
                },
                "pages": page_details,
            }
            trace_file = PARSED_DIR / "ocr_traces_latest.json"
            trace_file.write_text(json.dumps(ocr_trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to write ocr_traces_latest.json: {exc}")

    pipeline_logger.info(
        f"OCR & PyMuPDF Extractor: {total_pages} pages scanned | "
        f"PyMuPDF images: {total_embedded_images} (gauges: {gauge_candidate_images}) | "
        f"OCR: {pages_with_data} pages with table data, "
        f"{sum(len(r.table_rows) for r in results)} total rows, "
        f"{sum(sum(len(row.readings) for row in r.table_rows) for r in results)} readings extracted"
    )

    return {
        "page_ocr_results": results,
        "current_agent": "ocr_extractor",
    }
