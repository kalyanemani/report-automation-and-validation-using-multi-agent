from __future__ import annotations

"""
QDR·AI — Document Parser (v2)
──────────────────────────────
Layout-aware PDF parsing with table preservation and OCR for scanned pages.

Pipeline:
  PDF → classify pages (digital/scanned) → extract (PyMuPDF/OCR) →
        strip page headers → Markdown output with tables preserved
"""

import pymupdf as fitz  # PyMuPDF 1.28+: `fitz` is deprecated
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import TESSERACT_CMD, DOC_TYPES

logger = logging.getLogger("qdr_ai.parser")


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class ParsedPage:
    """Represents a single parsed page from a PDF."""
    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    is_scanned: bool = False
    char_count: int = 0


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""
    filename: str
    total_pages: int
    total_chars: int
    pages: list[ParsedPage]
    doc_type: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def all_tables(self) -> list[list[list[str]]]:
        tables = []
        for p in self.pages:
            tables.extend(p.tables)
        return tables


# ── Checkbox Detection ───────────────────────────────────────────────

CHECKED_SYMBOLS = (
    "☑", "✓", "✔", "☒", "✗", "✘", "⊠",
    "[X]", "[x]", "(X)", "(x)", "[✓]", "[✔]", "[✗]",
)

UNCHECKED_SYMBOLS = (
    "☐", "[ ]", "( )", "▫", "◽",
)


def extract_checked_lines(text: str) -> list[str]:
    """Extract lines containing checked checkbox symbols."""
    checked = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and any(sym in stripped for sym in CHECKED_SYMBOLS):
            if stripped not in seen:
                seen.add(stripped)
                checked.append(stripped)
    return checked


def extract_checklist_items(text: str) -> tuple[list[str], list[str]]:
    """
    Extract both checked (☒) and unchecked (☐) items from text.
    Returns (checked_lines, unchecked_lines).
    """
    checked = []
    unchecked = []
    seen_checked: set[str] = set()
    seen_unchecked: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        
        is_checked = any(sym in stripped for sym in CHECKED_SYMBOLS)
        is_unchecked = any(sym in stripped for sym in UNCHECKED_SYMBOLS)

        if is_checked:
            if stripped not in seen_checked:
                seen_checked.add(stripped)
                checked.append(stripped)
        elif is_unchecked:
            if stripped not in seen_unchecked:
                seen_unchecked.add(stripped)
                unchecked.append(stripped)

    return checked, unchecked


# ── Page Header Detection & Stripping ────────────────────────────────
# Engineering docs have repeating headers like:
#   Specification: 700-006  |  Revision 23  |  Date: 1/15/26  |  Page X of Y
#   TITLE: SPECIFICATION FOR WELDING ALLOY 91 MATERIALS

# Patterns that indicate header/footer lines to strip
HEADER_PATTERNS = [
    re.compile(r"^Specification:\s*\d+", re.IGNORECASE),
    re.compile(r"^Revision\s+\d+", re.IGNORECASE),
    re.compile(r"^Date:\s*\d+/\d+/\d+", re.IGNORECASE),
    re.compile(r"^Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"^TITLE:\s+", re.IGNORECASE),
    re.compile(r"^Approved\s+by:", re.IGNORECASE),
    re.compile(r"^Doc\s*No\.", re.IGNORECASE),
    re.compile(r"^Rev\.\s*\d+", re.IGNORECASE),
    # Visavakit Patana / Contractor Header Patterns
    re.compile(r"^VISAVAKIT\s+PATANA\s+CORP\.", re.IGNORECASE),
    re.compile(r"^HYDROSTATIC\s+TEST\s+PROCEDURE", re.IGNORECASE),
    re.compile(r"^Page\s*:\s*\d+\s*/\s*\d+", re.IGNORECASE),
    re.compile(r"^Issued\s+date\s*:\s*.*", re.IGNORECASE),
    re.compile(r"^Rev\s+No\.\s*:\s*\d+", re.IGNORECASE),
    re.compile(r"^Table\s+of\s+Contents", re.IGNORECASE),
    re.compile(r"^Item\s+Content\s+Page", re.IGNORECASE),
]


def strip_page_headers(text: str) -> str:
    """Remove repeating page header/footer lines from extracted text."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        is_header = any(p.match(stripped) for p in HEADER_PATTERNS)
        if not is_header:
            cleaned.append(line)
    return "\n".join(cleaned)


def normalize_engineering_headings(text: str) -> str:
    """
    Join section numbers and titles that were split across lines.
    Example:
      '6. \\nHYDROSTATIC TESTING' -> '6. HYDROSTATIC TESTING'
      '6.4 \\nWater Chemistry Analysis:' -> '6.4 Water Chemistry Analysis:'
    """
    # Pattern: Line with only number like '1.' or '6.4' followed by newline and uppercase/capitalized title
    pattern = re.compile(r"^((?:[A-Z]\.)?\d+(?:\.\d+)*\.?)\s*\n([A-Za-z].*)$", re.MULTILINE)
    return pattern.sub(r"\1 \2", text)



# ── PDF Classification ──────────────────────────────────────────────

def is_scanned_page(page: fitz.Page, min_chars: int = 100) -> bool:
    """
    Determine if a page is scanned (image-based) vs digital-born.
    Checks digital character count, image count, and layout density.
    """
    text = page.get_text("text").strip()
    if len(text) < min_chars:
        return True
    
    # Check if page is dominated by large embedded images (e.g. scanned forms with small digital stamps)
    imgs = page.get_images()
    if imgs and len(text) < 250:
        return True
        
    return False


# ── PyMuPDF Text Extraction ────────────────────────────────────────

def extract_text_pymupdf(page: fitz.Page) -> str:
    """Extract text from a digital-born PDF page."""
    return page.get_text("text")


# ── OCR for Scanned Pages ───────────────────────────────────────────

def ocr_page(page: fitz.Page, dpi: int = 150) -> str:
    """
    Render PDF page to image and OCR with Tesseract.
    Optimized for high-speed layout and table preservation.
    """
    try:
        import pytesseract
        from PIL import Image, ImageEnhance
        import numpy as np

        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        # Render page to image at optimal 150 DPI for fast OCR
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Fast Pass Strategy 1: Plain image
        text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
        if len(text.strip()) > 25:
            return text.strip()

        # Fallback Strategy 2: Grayscale + contrast enhancement
        gray = img.convert("L")
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        text_contrast = pytesseract.image_to_string(enhanced, config="--psm 6 --oem 3")
        if len(text_contrast.strip()) > len(text.strip()):
            return text_contrast.strip()

        return text.strip()

    except ImportError:
        logger.error(
            "pytesseract not installed. Install with: pip install pytesseract"
        )
        return ""
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def ocr_page_with_word_positions(page: fitz.Page, dpi: int = 300) -> list[dict]:
    """
    OCR with word-level bounding boxes (for PDF annotation).
    Returns list of {word, x0, y0, x1, y1, conf}.
    """
    try:
        import pytesseract
        from PIL import Image

        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        scale_x = page.rect.width / pix.width
        scale_y = page.rect.height / pix.height

        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT,
            config="--psm 6 --oem 3",
        )

        words = []
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            conf_val = data["conf"][i]
            conf = int(conf_val) if str(conf_val).lstrip("-").isdigit() else 0
            if not txt or conf < 20:
                continue
            words.append({
                "word": txt,
                "x0": data["left"][i] * scale_x,
                "y0": data["top"][i] * scale_y,
                "x1": (data["left"][i] + data["width"][i]) * scale_x,
                "y1": (data["top"][i] + data["height"][i]) * scale_y,
                "conf": conf,
            })
        return words

    except ImportError:
        return []
    except Exception as e:
        logger.error(f"OCR word extraction failed: {e}")
        return []


# ── Table Extraction ────────────────────────────────────────────────

def extract_tables_from_page(
    pdf_path: str, page_number: int
) -> list[list[list[str]]]:
    """Extract tables from a PDF page using pdfplumber."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                tables = page.extract_tables()
                if tables:
                    cleaned = []
                    for table in tables:
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [
                                (cell or "").strip() for cell in row
                            ]
                            # Skip rows that are all empty
                            if any(c for c in cleaned_row):
                                cleaned_table.append(cleaned_row)
                        if cleaned_table:
                            cleaned.append(cleaned_table)
                    return cleaned
    except ImportError:
        logger.warning("pdfplumber not installed — table extraction skipped")
    except Exception as e:
        logger.warning(f"Table extraction failed on page {page_number}: {e}")
    return []


def extract_tables_from_bytes(
    pdf_bytes: bytes, page_number: int
) -> list[list[list[str]]]:
    """Extract tables from PDF bytes (for uploaded files)."""
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                tables = page.extract_tables()
                if tables:
                    cleaned = []
                    for table in tables:
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [
                                (cell or "").strip() for cell in row
                            ]
                            if any(c for c in cleaned_row):
                                cleaned_table.append(cleaned_row)
                        if cleaned_table:
                            cleaned.append(cleaned_table)
                    return cleaned
    except Exception as e:
        logger.warning(f"Table extraction from bytes failed: {e}")
    return []


def table_to_markdown(table: list[list[str]]) -> str:
    """Convert a table (list of rows) to Markdown format."""
    if not table or not table[0]:
        return ""
    lines = []
    header = table[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in table[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")
    return "\n".join(lines)


def is_header_table(table: list[list[str]]) -> bool:
    """
    Detect if a table is a page header/metadata table
    (e.g., Specification/Revision/Date/Page blocks).
    These should be stripped, not chunked.
    """
    flat = " ".join(
        cell for row in table for cell in row if cell
    ).lower()
    header_keywords = [
        "specification:", "revision", "page", "approved by",
        "date:", "title:", "doc no",
    ]
    matches = sum(1 for kw in header_keywords if kw in flat)
    return matches >= 3  # at least 3 header keywords → it's a header table


DOC_CACHE_DIR = Path("data/parsed/doc_cache")


def _get_doc_cache_path(pdf_path: str | Path | None, pdf_bytes: bytes | None, ocr_scanned: bool, extract_tables: bool, strip_headers: bool) -> Optional[Path]:
    try:
        import hashlib
        DOC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if pdf_bytes:
            chash = hashlib.md5(pdf_bytes).hexdigest()
            raw_k = f"bytes_{chash}_{ocr_scanned}_{extract_tables}_{strip_headers}"
        elif pdf_path:
            p = Path(pdf_path)
            stat = p.stat()
            raw_k = f"file_{p.name}_{stat.st_size}_{stat.st_mtime}_{ocr_scanned}_{extract_tables}_{strip_headers}"
        else:
            return None
        key = hashlib.md5(raw_k.encode("utf-8")).hexdigest()
        return DOC_CACHE_DIR / f"{key}.json"
    except Exception:
        return None


def parse_pdf(
    pdf_path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    filename: str = "document.pdf",
    extract_tables: bool = True,
    strip_headers: bool = True,
    ocr_scanned: bool = True,
    use_cache: bool = True,
) -> ParsedDocument:
    """
    Parse a PDF document with layout awareness, persistent caching, and OCR support.

    Args:
        pdf_path: path to PDF file on disk
        pdf_bytes: raw PDF bytes (e.g., from upload)
        filename: display name
        extract_tables: whether to extract tables via pdfplumber
        strip_headers: whether to strip repeating page headers
        ocr_scanned: whether to OCR scanned pages
        use_cache: whether to load/save parsed document cache

    Returns:
        ParsedDocument with per-page text, tables, and metadata
    """
    import json
    cache_path = _get_doc_cache_path(pdf_path, pdf_bytes, ocr_scanned, extract_tables, strip_headers) if use_cache else None
    if cache_path and cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            pages = [
                ParsedPage(
                    page_number=p["page_number"],
                    text=p["text"],
                    tables=p.get("tables", []),
                    is_scanned=p.get("is_scanned", False),
                    char_count=p.get("char_count", len(p["text"])),
                )
                for p in cached_data["pages"]
            ]
            cached_doc = ParsedDocument(
                filename=cached_data.get("filename", filename),
                total_pages=cached_data["total_pages"],
                total_chars=cached_data["total_chars"],
                pages=pages,
                doc_type=cached_data.get("doc_type", "unknown"),
                metadata=cached_data.get("metadata", {}),
            )
            logger.info(f"Loaded {filename} from disk cache in <10ms ({cached_doc.total_pages} pages, {cached_doc.total_chars:,} chars)")
            return cached_doc
        except Exception as ce:
            logger.warning(f"Could not load doc cache: {ce}")

    if pdf_bytes:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elif pdf_path:
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        doc = fitz.open(str(pdf_path))
    else:
        raise ValueError("Either pdf_path or pdf_bytes must be provided")

    total_pages = len(doc)
    total_chars = 0
    pages: list[ParsedPage] = []

    logger.info(f"Parsing {filename}: {total_pages} pages")

    # Fast C-accelerated table extraction via PyMuPDF find_tables()
    batch_tables_by_page = {}
    if extract_tables:
        try:
            for p_idx in range(total_pages):
                fitz_p = doc[p_idx]
                tabs = fitz_p.find_tables()
                if tabs and tabs.tables:
                    cleaned_page_tables = []
                    for t in tabs.tables:
                        raw_t = t.extract()
                        cleaned_t = []
                        for r in raw_t:
                            cleaned_r = [(c or "").strip() for c in r]
                            if any(c for c in cleaned_r):
                                cleaned_t.append(cleaned_r)
                        if cleaned_t and not is_header_table(cleaned_t):
                            cleaned_page_tables.append(cleaned_t)
                    if cleaned_page_tables:
                        batch_tables_by_page[p_idx] = cleaned_page_tables
        except Exception as e:
            logger.warning(f"PyMuPDF find_tables warning: {e}")

    for page_num in range(total_pages):
        page = doc[page_num]
        scanned = is_scanned_page(page)

        # Extract text
        if scanned and ocr_scanned:
            text = ocr_page(page)
            logger.info(
                f"  Page {page_num + 1}: SCANNED → OCR ({len(text)} chars)"
            )
        elif scanned:
            text = ""
            logger.info(f"  Page {page_num + 1}: SCANNED (OCR disabled)")
        else:
            text = extract_text_pymupdf(page)
            if ocr_scanned and len(text.strip()) < 150:
                ocr_t = ocr_page(page)
                if len(ocr_t.strip()) > len(text.strip()) + 40:
                    text = ocr_t
                    logger.info(f"  Page {page_num + 1}: AUGMENTED WITH OCR ({len(text)} chars)")

        # Strip page headers and normalize headings
        if text:
            if strip_headers:
                text = strip_page_headers(text)
            text = normalize_engineering_headings(text)

        # Retrieve pre-extracted tables
        page_tables = batch_tables_by_page.get(page_num, [])
        if page_tables:
            logger.info(
                f"  Page {page_num + 1}: "
                f"{len(page_tables)} content table(s)"
            )

        char_count = len(text)
        total_chars += char_count
        pages.append(ParsedPage(
            page_number=page_num + 1,
            text=text,
            tables=page_tables,
            is_scanned=scanned,
            char_count=char_count,
        ))

    doc.close()

    parsed = ParsedDocument(
        filename=filename,
        total_pages=total_pages,
        total_chars=total_chars,
        pages=pages,
    )

    scanned_count = sum(1 for p in pages if p.is_scanned)
    table_count = sum(len(p.tables) for p in pages)
    logger.info(
        f"Parsed {filename}: {total_pages} pages, {total_chars} chars, "
        f"{table_count} tables, {scanned_count} scanned pages"
    )

    if cache_path and use_cache:
        try:
            cache_payload = {
                "filename": filename,
                "total_pages": total_pages,
                "total_chars": total_chars,
                "doc_type": parsed.doc_type,
                "metadata": parsed.metadata,
                "pages": [
                    {
                        "page_number": p.page_number,
                        "text": p.text,
                        "tables": p.tables,
                        "is_scanned": p.is_scanned,
                        "char_count": p.char_count,
                    }
                    for p in pages
                ]
            }
            cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")
            logger.info(f"Saved {filename} ({total_chars:,} chars) to persistent cache.")
        except Exception as we:
            logger.warning(f"Could not write doc cache: {we}")

    return parsed


def parse_pdf_to_markdown(
    pdf_path: str | Path | None = None,
    pdf_bytes: bytes | None = None,
    filename: str = "document.pdf",
) -> str:
    """Parse PDF and return Markdown with tables preserved."""
    parsed = parse_pdf(
        pdf_path=pdf_path, pdf_bytes=pdf_bytes, filename=filename
    )
    sections = []
    for page in parsed.pages:
        sections.append(f"<!-- Page {page.page_number} -->")
        if page.text:
            sections.append(page.text)
        for table in page.tables:
            md_table = table_to_markdown(table)
            if md_table:
                sections.append(f"\n{md_table}\n")
    return "\n\n".join(sections)


# ── Dynamic Review Guide & Custom Criteria Parser ────────────────────

def parse_review_guide(
    input_data: str | Path | bytes,
    filename: str = "review_guide.txt",
) -> dict:
    """
    Parse an uploaded review guide (.docm, .docx, .pdf, .txt) or pasted criteria text.

    Supports:
      1. Checked checkboxes (☑, [X]) -> extracts only selected criteria
      2. All checkboxes empty (☐, [ ]) -> extracts all structured steps & requirements
      3. No checkboxes -> extracts all numbered steps, bullet points, and rules from the guide
    """
    import io
    import zipfile
    import xml.etree.ElementTree as ET

    full_text = ""
    file_lower = filename.lower()

    # Determine input type and extract raw text
    if isinstance(input_data, bytes):
        if file_lower.endswith((".docm", ".docx")):
            try:
                with zipfile.ZipFile(io.BytesIO(input_data)) as z:
                    xml_content = z.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    texts = [elem.text for elem in tree.iter() if elem.text]
                    full_text = "\n".join(texts)
            except Exception as e:
                logger.warning(f"Error reading docx bytes: {e}")
                full_text = input_data.decode("utf-8", errors="ignore")
        elif file_lower.endswith(".pdf"):
            parsed_doc = parse_pdf(pdf_bytes=input_data, filename=filename)
            full_text = parsed_doc.full_text
        else:
            full_text = input_data.decode("utf-8", errors="ignore")

    elif isinstance(input_data, (str, Path)) and os.path.exists(str(input_data)):
        path = Path(input_data)
        filename = path.name
        file_lower = filename.lower()
        if file_lower.endswith((".docm", ".docx")):
            try:
                with zipfile.ZipFile(str(path)) as z:
                    xml_content = z.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    texts = [elem.text for elem in tree.iter() if elem.text]
                    full_text = "\n".join(texts)
            except Exception as e:
                logger.warning(f"Error reading docx file: {e}")
                full_text = path.read_text(encoding="utf-8", errors="ignore")
        elif file_lower.endswith(".pdf"):
            parsed_doc = parse_pdf(pdf_path=path, filename=filename)
            full_text = parsed_doc.full_text
        else:
            full_text = path.read_text(encoding="utf-8", errors="ignore")

    elif isinstance(input_data, str):
        full_text = input_data

    # Extract checked and unchecked checklist items
    checked_items, unchecked_items = extract_checklist_items(full_text)

    # Determine mode & active rules
    if checked_items:
        mode = "checked_items_only"
        active_items = checked_items
    else:
        # No checkboxes OR all checkboxes empty -> Extract all structured steps & requirements
        mode = "all_criteria"
        active_items = []
        seen = set()

        for line in full_text.splitlines():
            stripped = line.strip()
            # Clean leading unchecked boxes or bullet markers
            cleaned = stripped
            for sym in UNCHECKED_SYMBOLS:
                if cleaned.startswith(sym):
                    cleaned = cleaned[len(sym):].strip()
            for b in ("- ", "* ", "• ", "? ", "✓ "):
                if cleaned.startswith(b):
                    cleaned = cleaned[len(b):].strip()

            if len(cleaned) > 5 and cleaned not in seen:
                # Include numbered steps, bullet requirements, or procedural statements
                if (
                    re.match(r"^(?:Step\s*#?\d+|[0-9]+[\.\):]|\bReview\b|\bVerify\b|\bShall\b|\bMust\b|\bRecord\b|\bHydrostatic\b|\bWater\b|\bPressure\b)", cleaned, re.IGNORECASE)
                    or stripped.startswith(("-", "*", "•", "?", "☐", "[ ]"))
                ):
                    seen.add(cleaned)
                    active_items.append(cleaned)

        if not active_items:
            # Fallback to non-empty lines with meaningful content
            active_items = [
                line.strip() for line in full_text.splitlines()
                if len(line.strip()) > 15
            ]

    logger.info(
        f"Parsed Review Guide [{filename}]: mode={mode}, "
        f"active_items={len(active_items)}, "
        f"checked={len(checked_items)}, unchecked={len(unchecked_items)}"
    )

    return {
        "filename": filename,
        "mode": mode,
        "raw_text": full_text,
        "items": active_items,
        "checked_count": len(checked_items),
        "unchecked_count": len(unchecked_items),
        "total_rules": len(active_items),
    }

