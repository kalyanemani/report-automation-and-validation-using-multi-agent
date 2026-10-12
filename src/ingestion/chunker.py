from __future__ import annotations

"""
QDR·AI — Multi-Strategy Chunker (v2)
──────────────────────────────────────
Domain-aware chunking for engineering specs, procedures, and test reports.

Strategies applied based on document content:
  1. Hierarchical Section-Based: Splits at numbered heading boundaries
     (1.0, 2.1, 3.3.1 etc.) — primary strategy for specs/procedures
  2. Atomic Table Preservation: Tables are never split mid-row
  3. Parent-Child: Fine-grained children for retrieval, parents for LLM
  4. Metadata Enrichment: Material grades, code references, section paths
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.config import CHUNK_SIZE_CHILD, CHUNK_SIZE_PARENT, CHUNK_OVERLAP
from src.ingestion.parser import ParsedDocument, ParsedPage, table_to_markdown

logger = logging.getLogger("qdr_ai.chunker")


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single chunk of text with metadata for vector storage."""
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    parent_chunk_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        """Rough token count (~0.75 words per token)."""
        return int(len(self.content.split()) / 0.75)


# ── Engineering Heading Detection ────────────────────────────────────
# Engineering specs use numbered headings like:
#   1.  SCOPE
#   2.  REFERENCE DOCUMENTS
#   3.1 Filler Metal Requirements
#   4.1.1  Prior to Start of Fabrication
#   A.1  Appendix Section

ENGINEERING_HEADING_RE = re.compile(
    r"^"
    r"(?:"
    r"(?:[A-Z]\.)??"            # Optional appendix letter: A.
    r"\d+\.?"                   # Main number: 1 or 1.
    r"(?:\d+\.?)*"              # Sub-numbers: .1.2.3
    r")"
    r"\s+"                      # Whitespace after number
    r"(?!/\s*\d+)"              # DO NOT match page number fractions like '2 / 13'
    r"(\S.*)$",                 # Heading text (at least 1 non-space char)
    re.MULTILINE,
)

# Also catch ALL-CAPS section titles without numbers
CAPS_HEADING_RE = re.compile(
    r"^[A-Z][A-Z\s&/\-]{4,}$", re.MULTILINE
)


def detect_engineering_sections(
    text: str,
) -> list[tuple[str, str, str]]:
    """
    Split text into (section_number, heading_text, body) tuples.
    Handles numbered engineering headings (1.0, 2.1, 3.3.1 etc.).

    Returns list of (section_id, section_title, section_body) tuples.
    """
    lines = text.split("\n")
    sections: list[tuple[str, str, str]] = []
    current_id = ""
    current_title = "Document Header"
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Ignore page fractions
        if re.match(r"^\d+\s*/\s*\d+$", stripped):
            continue

        # Check for numbered heading
        m = ENGINEERING_HEADING_RE.match(stripped)
        if m and len(stripped) < 150:
            # Save previous section
            if current_body or current_id:
                body_text = "\n".join(current_body).strip()
                if body_text:
                    sections.append(
                        (current_id, current_title, body_text)
                    )

            # Parse new heading
            # Extract section number (everything before the text)
            parts = stripped.split(None, 1)
            current_id = parts[0].rstrip(".")
            current_title = parts[1] if len(parts) > 1 else stripped
            current_body = []
            continue

        # Check for ALL-CAPS heading (only if no number and short)
        if (
            CAPS_HEADING_RE.match(stripped)
            and len(stripped) < 80
            and len(stripped) > 4
            and not stripped.startswith(("NOTE", "TABLE", "FIGURE", "PAGE", "REV"))
        ):
            if current_body or current_id:
                body_text = "\n".join(current_body).strip()
                if body_text:
                    sections.append(
                        (current_id, current_title, body_text)
                    )
            current_id = ""
            current_title = stripped.title()
            current_body = []
            continue

        current_body.append(line)

    # Save last section
    body_text = "\n".join(current_body).strip()
    if body_text:
        sections.append((current_id, current_title, body_text))

    return sections


# ── Metadata Extraction ─────────────────────────────────────────────

# Material grades commonly referenced in engineering docs
MATERIAL_GRADES_RE = re.compile(
    r"\b(?:Alloy|Grade|P|T|F|WP|SA-)\s*(\d{1,3}[A-Z]?)\b",
    re.IGNORECASE,
)

# ASME / ASTM / AWS code references
CODE_REFS_RE = re.compile(
    r"\b((?:ASME|ASTM|AWS|API|ANSI|NACE)\s+"
    r"[A-Z]?\d+(?:[.-]\d+)*(?:\s+Section\s+[IVXLC]+)?)\b",
    re.IGNORECASE,
)


def extract_metadata_from_text(text: str) -> dict:
    """Extract engineering-specific metadata, topics, and test types from chunk text."""
    metadata: dict = {}
    lower_text = text.lower()

    # Material grades
    grades = list(set(MATERIAL_GRADES_RE.findall(text)))
    if grades:
        metadata["material_grades"] = ",".join(sorted(grades))

    # Code references
    codes = list(set(CODE_REFS_RE.findall(text)))
    if codes:
        metadata["applicable_codes"] = ",".join(sorted(codes))

    # Test Type identification
    is_hydro = any(k in lower_text for k in ("hydrostatic", "hydrotest", "hydp", "pressure test", "test pressure", "mawp", "psig", "water chemistry", "vpci"))
    is_hard = any(k in lower_text for k in ("hardness", "brinell", "vickers", "rockwell", "hrc", "hrb", "hld", "hrdr", "uci probe", "heat treat"))
    if is_hydro and is_hard:
        metadata["test_type"] = "general_spec"
    elif is_hydro:
        metadata["test_type"] = "hydrostatic"
    elif is_hard:
        metadata["test_type"] = "hardness"

    # Topic classification
    topics = []
    if any(k in lower_text for k in ("chloride", "sulphate", "sulfate", "hardness", "conductivity", "ph", "potable water", "demineralized water", "water chemistry", "cfu/ml", "ppm")):
        topics.append("water_chemistry")
    if any(k in lower_text for k in ("test pressure", "mawp", "design pressure", "stress ratio", "lsr", "multiplier", "1.5", "1.3")):
        topics.append("test_pressure")
    if any(k in lower_text for k in ("gauge", "pressure gauge", "dead-weight", "calibration", "dial range", "full scale")):
        topics.append("gauge_calibration")
    if any(k in lower_text for k in ("holding time", "holding period", "duration", "soak time", "minutes")):
        topics.append("holding_time")
    if any(k in lower_text for k in ("inhibitor", "vpci", "protexol", "cortec", "corrosion inhibitor", "concentration")):
        topics.append("corrosion_inhibitor")
    if any(k in lower_text for k in ("acceptance criteria", "visual inspection", "leakage", "pressure drop", "distortion")):
        topics.append("acceptance_criteria")
    if any(k in lower_text for k in ("pwht", "post weld heat treat", "soaking temperature", "cooling rate")):
        topics.append("pwht")

    if topics:
        metadata["topics"] = ",".join(topics)

    return metadata


def extract_doc_metadata(text: str, filename: str) -> dict:
    """Extract document-level metadata from the full text."""
    metadata: dict = {"filename": filename}

    # Try to extract document number
    doc_num_match = re.search(
        r"(?:Specification|Doc\s*No\.?|Document\s*No\.?)[\s:]*"
        r"([A-Z0-9\-]+)",
        text[:2000],
        re.IGNORECASE,
    )
    if doc_num_match:
        metadata["doc_number"] = doc_num_match.group(1)

    # Try to extract revision
    rev_match = re.search(
        r"(?:Revision|Rev\.?)\s*(\d+|[A-Z])",
        text[:2000],
        re.IGNORECASE,
    )
    if rev_match:
        metadata["revision"] = rev_match.group(1)

    # Try to extract title
    title_match = re.search(
        r"TITLE:\s*(.+?)(?:\n|$)",
        text[:3000],
        re.IGNORECASE,
    )
    if title_match:
        metadata["doc_title"] = title_match.group(1).strip()

    return metadata


# ── Markdown Table Detection ────────────────────────────────────────

TABLE_LINE_RE = re.compile(r"^\|.*\|$")
TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")


def extract_table_blocks(text: str) -> list[tuple[int, int, str]]:
    """Find Markdown table blocks in text."""
    lines = text.split("\n")
    blocks: list[tuple[int, int, str]] = []
    in_table = False
    table_start = 0
    table_lines: list[str] = []

    for i, line in enumerate(lines):
        if TABLE_LINE_RE.match(line.strip()):
            if not in_table:
                in_table = True
                table_start = i
                table_lines = []
            table_lines.append(line)
        else:
            if in_table:
                blocks.append((
                    table_start, i - 1, "\n".join(table_lines),
                ))
                in_table = False
                table_lines = []

    if in_table:
        blocks.append(
            (table_start, len(lines) - 1, "\n".join(table_lines))
        )
    return blocks


# ── Chunking Strategies ─────────────────────────────────────────────

def chunk_by_engineering_sections(
    text: str,
    doc_metadata: dict,
    max_tokens: int = CHUNK_SIZE_CHILD,
) -> list[Chunk]:
    """
    Strategy 1: Hierarchical section-based chunking.
    Splits at numbered engineering heading boundaries.
    Keeps each section atomic if under max_tokens.
    """
    sections = detect_engineering_sections(text)
    chunks: list[Chunk] = []

    for sec_id, sec_title, sec_body in sections:
        # Build section label
        section_label = (
            f"{sec_id} {sec_title}" if sec_id else sec_title
        )
        content = f"{section_label}\n{sec_body}".strip()
        word_count = len(content.split())

        # Extract per-chunk metadata
        chunk_meta = extract_metadata_from_text(content)

        if word_count <= max_tokens:
            # Section fits in one chunk
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                metadata={
                    **doc_metadata,
                    **chunk_meta,
                    "section": section_label,
                    "section_id": sec_id,
                    "chunk_type": "text",
                    "strategy": "engineering_section",
                },
            )
            chunks.append(chunk)
        else:
            # Split large sections by sub-paragraphs
            paragraphs = re.split(r"\n\s*\n", sec_body)
            current = section_label + "\n"
            part_num = 0

            for para in paragraphs:
                if (
                    len((current + para).split()) > max_tokens
                    and current.strip() != section_label
                ):
                    part_num += 1
                    para_meta = extract_metadata_from_text(current)
                    chunk = Chunk(
                        chunk_id=str(uuid.uuid4()),
                        content=current.strip(),
                        metadata={
                            **doc_metadata,
                            **para_meta,
                            "section": section_label,
                            "section_id": sec_id,
                            "chunk_type": "text",
                            "strategy": "section_split",
                            "part": str(part_num),
                        },
                    )
                    chunks.append(chunk)
                    current = section_label + "\n" + para + "\n\n"
                else:
                    current += para + "\n\n"

            if current.strip() and current.strip() != section_label:
                part_num += 1
                para_meta = extract_metadata_from_text(current)
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    content=current.strip(),
                    metadata={
                        **doc_metadata,
                        **para_meta,
                        "section": section_label,
                        "section_id": sec_id,
                        "chunk_type": "text",
                        "strategy": "section_split",
                        "part": str(part_num),
                    },
                )
                chunks.append(chunk)

    logger.info(
        f"Section chunking: {len(sections)} sections -> {len(chunks)} chunks"
    )
    return chunks


def chunk_tables_atomic(
    parsed_doc: ParsedDocument,
    doc_metadata: dict,
) -> list[Chunk]:
    """
    Strategy 2: Atomic table preservation.
    Each content table becomes one chunk (header tables already filtered).
    Large tables are split by row groups with headers always prepended.
    """
    chunks: list[Chunk] = []

    for page in parsed_doc.pages:
        for table in page.tables:
            md_table = table_to_markdown(table)
            if not md_table or len(md_table.strip()) < 20:
                continue

            table_meta = extract_metadata_from_text(md_table)
            word_count = len(md_table.split())

            if word_count <= CHUNK_SIZE_PARENT:
                # Entire table fits
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    content=md_table,
                    metadata={
                        **doc_metadata,
                        **table_meta,
                        "chunk_type": "table",
                        "strategy": "table_atomic",
                        "page_number": str(page.page_number),
                        "table_rows": str(len(table) - 1),
                    },
                )
                chunks.append(chunk)
            else:
                # Split by row groups, always prepend header
                lines = md_table.split("\n")
                header = lines[:2]  # header + separator
                data_rows = lines[2:]
                group_size = max(
                    5, CHUNK_SIZE_CHILD // 20
                )

                for i in range(0, len(data_rows), group_size):
                    group = data_rows[i : i + group_size]
                    content = "\n".join(header + group)
                    chunk_meta = extract_metadata_from_text(content)
                    chunk = Chunk(
                        chunk_id=str(uuid.uuid4()),
                        content=content,
                        metadata={
                            **doc_metadata,
                            **chunk_meta,
                            "chunk_type": "table",
                            "strategy": "table_split",
                            "page_number": str(page.page_number),
                            "table_rows": str(len(group)),
                            "table_part": str(i // group_size + 1),
                        },
                    )
                    chunks.append(chunk)

    logger.info(
        f"Table chunking: {sum(len(p.tables) for p in parsed_doc.pages)} "
        f"tables -> {len(chunks)} chunks"
    )
    return chunks


def create_parent_child_chunks(
    child_chunks: list[Chunk],
    parent_size: int = CHUNK_SIZE_PARENT,
) -> list[Chunk]:
    """
    Strategy 3: Parent-child chunking.
    Group child chunks into larger parent chunks for LLM context.
    """
    if not child_chunks:
        return []

    parents: list[Chunk] = []
    current_children: list[Chunk] = []
    current_content = ""

    for child in child_chunks:
        candidate = current_content + "\n\n" + child.content
        if (
            len(candidate.split()) > parent_size
            and current_children
        ):
            parent = Chunk(
                chunk_id=str(uuid.uuid4()),
                content=current_content.strip(),
                metadata={
                    **current_children[0].metadata,
                    "chunk_type": "parent",
                    "strategy": "parent",
                    "child_count": str(len(current_children)),
                },
                children_ids=[c.chunk_id for c in current_children],
            )
            parents.append(parent)
            for c in current_children:
                c.parent_chunk_id = parent.chunk_id

            current_children = [child]
            current_content = child.content
        else:
            current_children.append(child)
            current_content = candidate

    if current_children:
        parent = Chunk(
            chunk_id=str(uuid.uuid4()),
            content=current_content.strip(),
            metadata={
                **current_children[0].metadata,
                "chunk_type": "parent",
                "strategy": "parent",
                "child_count": str(len(current_children)),
            },
            children_ids=[c.chunk_id for c in current_children],
        )
        parents.append(parent)
        for c in current_children:
            c.parent_chunk_id = parent.chunk_id

    logger.info(
        f"Parent-child: {len(child_chunks)} children -> "
        f"{len(parents)} parents"
    )
    return parents


# ── Main Chunking Pipeline ───────────────────────────────────────────

def chunk_document(
    parsed_doc: ParsedDocument,
    doc_metadata: Optional[dict] = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Main chunking pipeline — applies multi-strategy chunking.

    Returns:
        (child_chunks, parent_chunks)
        - child_chunks: fine-grained for retrieval
        - parent_chunks: larger context for LLM
    """
    if doc_metadata is None:
        doc_metadata = {}

    # Add document-level metadata
    doc_meta = extract_doc_metadata(
        parsed_doc.full_text, parsed_doc.filename
    )
    base_meta: dict = {
        "doc_id": doc_meta.get(
            "doc_number", parsed_doc.filename
        ),
        "doc_type": parsed_doc.doc_type,
        **doc_meta,
        **doc_metadata,
    }

    # Ensure all metadata values are strings (ChromaDB requirement)
    base_meta = {
        k: str(v) if v is not None else ""
        for k, v in base_meta.items()
    }

    child_chunks: list[Chunk] = []

    # 1. Extract and chunk tables atomically
    table_chunks = chunk_tables_atomic(parsed_doc, base_meta)
    child_chunks.extend(table_chunks)

    # 2. Chunk text by engineering sections
    full_text = parsed_doc.full_text

    # Remove markdown table blocks from text to avoid double-chunking
    for block in extract_table_blocks(full_text):
        full_text = full_text.replace(block[2], "")

    section_chunks = chunk_by_engineering_sections(
        full_text, base_meta
    )
    child_chunks.extend(section_chunks)

    # 3. Add page numbers to chunks that don't have them
    for chunk in child_chunks:
        if "page_number" not in chunk.metadata:
            for page in parsed_doc.pages:
                if chunk.content[:80] in page.text:
                    chunk.metadata["page_number"] = str(
                        page.page_number
                    )
                    break

    # 4. Ensure all metadata values are strings
    for chunk in child_chunks:
        chunk.metadata = {
            k: str(v) if v is not None else ""
            for k, v in chunk.metadata.items()
        }

    # 5. Create parent chunks
    parent_chunks = create_parent_child_chunks(child_chunks)

    # Stringify parent metadata too
    for chunk in parent_chunks:
        chunk.metadata = {
            k: str(v) if v is not None else ""
            for k, v in chunk.metadata.items()
        }

    logger.info(
        f"Chunked {parsed_doc.filename}: "
        f"{len(child_chunks)} children, {len(parent_chunks)} parents"
    )
    return child_chunks, parent_chunks
