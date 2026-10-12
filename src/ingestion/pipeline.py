"""
QDR·AI — Document Ingestion Pipeline
──────────────────────────────────────
End-to-end pipeline: PDF → Parse → Chunk → Embed → Store

This is the main entry point for ingesting documents.
Combines the parser, chunker, and vector store manager.
"""

import logging
from pathlib import Path
from typing import Optional

from src.ingestion.parser import parse_pdf, ParsedDocument, extract_checked_lines, extract_checklist_items
from src.ingestion.chunker import chunk_document, Chunk
from src.vectorstore.store import VectorStoreManager

logger = logging.getLogger("qdr_ai.pipeline")


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline()

        # Ingest a Review Guide
        result = pipeline.ingest_review_guide(
            pdf_bytes=uploaded_file,
            guide_code="HRD",
            filename="Review_Guide_HRDR.pdf",
        )

        # Ingest a Supplier Submittal
        result = pipeline.ingest_submittal(
            pdf_bytes=uploaded_file,
            guide_code="HRD",
            filename="Supplier_HRDR_Report.pdf",
        )

        # Search
        results = pipeline.search("HRD", "hardness acceptance criteria")
    """

    def __init__(self, store_manager: Optional[VectorStoreManager] = None):
        self.store = store_manager or VectorStoreManager()

    def ingest_review_guide(
        self,
        pdf_bytes: bytes,
        guide_code: str,
        filename: str = "review_guide.pdf",
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest a Review Guide PDF.
        Parses, chunks, embeds, and stores in vector DB.

        Returns summary dict with stats.
        """
        logger.info(f"═══ Ingesting Review Guide: {filename} [{guide_code}] ═══")

        # 1. Parse
        parsed = parse_pdf(pdf_bytes=pdf_bytes, filename=filename)
        parsed.doc_type = "review_guide"

        # 2. Extract checked & unchecked checklist lines
        checked_lines, unchecked_lines = extract_checklist_items(parsed.full_text)
        logger.info(f"Found {len(checked_lines)} checked criteria, {len(unchecked_lines)} unchecked criteria")

        # 3. Chunk
        metadata = {
            "doc_type": "review_guide",
            "guide_code": guide_code,
            "folder": "Folder 1",
            **(extra_metadata or {}),
        }
        child_chunks, parent_chunks = chunk_document(parsed, metadata)

        # 4. Embed & Store
        store_result = self.store.ingest_chunks(guide_code, child_chunks, parent_chunks)

        result = {
            **store_result,
            "filename": filename,
            "total_pages": parsed.total_pages,
            "total_chars": parsed.total_chars,
            "checked_lines": checked_lines,
            "unchecked_lines": unchecked_lines,
            "checked_count": len(checked_lines),
            "child_chunks": len(child_chunks),
            "parent_chunks": len(parent_chunks),
            "text": parsed.full_text,  # for frontend preview
        }

        logger.info(
            f"═══ Review Guide ingested: {len(child_chunks)} children, "
            f"{len(parent_chunks)} parents, {len(checked_lines)} criteria ═══"
        )
        return result

    def ingest_submittal(
        self,
        pdf_bytes: bytes,
        guide_code: str,
        filename: str = "submittal.pdf",
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest a Supplier Submittal PDF.
        Parses and returns structured content (not stored in vector DB —
        submittals are inspected, not searched).
        """
        logger.info(f"═══ Parsing Submittal: {filename} ═══")

        parsed = parse_pdf(pdf_bytes=pdf_bytes, filename=filename)
        parsed.doc_type = "supplier_submittal"

        result = {
            "filename": filename,
            "total_pages": parsed.total_pages,
            "total_chars": parsed.total_chars,
            "tables_found": sum(len(p.tables) for p in parsed.pages),
            "scanned_pages": sum(1 for p in parsed.pages if p.is_scanned),
            "text": parsed.full_text,
        }

        logger.info(
            f"═══ Submittal parsed: {parsed.total_pages} pages, "
            f"{parsed.total_chars} chars ═══"
        )
        return result

    def ingest_specification(
        self,
        pdf_bytes: bytes,
        guide_code: str,
        spec_name: str = "specification",
        filename: str = "specification.pdf",
        extra_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest a Specification / Code / Procedure PDF.
        These are stored in the same vector DB as the review guide
        to enable cross-reference retrieval.
        """
        logger.info(f"═══ Ingesting Specification: {filename} ═══")

        parsed = parse_pdf(pdf_bytes=pdf_bytes, filename=filename)
        parsed.doc_type = "specification"

        metadata = {
            "doc_type": "specification",
            "guide_code": guide_code,
            "spec_name": spec_name,
            "folder": "Folder 2",
            **(extra_metadata or {}),
        }
        child_chunks, parent_chunks = chunk_document(parsed, metadata)

        # Store in the SAME collection as the review guide
        # so cross-referencing works during retrieval
        store_result = self.store.ingest_chunks(
            f"{guide_code}_specs", child_chunks, parent_chunks
        )

        return {
            **store_result,
            "filename": filename,
            "total_pages": parsed.total_pages,
            "total_chars": parsed.total_chars,
        }

    def search(
        self,
        guide_code: str,
        query: str,
        k: int = 15,
        return_parents: bool = True,
    ) -> list[dict]:
        """Search the vector store for relevant chunks."""
        return self.store.search(
            guide_code=guide_code,
            query=query,
            k=k,
            return_parents=return_parents,
        )

    def list_stores(self) -> dict:
        """List all loaded vector stores."""
        return self.store.list_stores()
