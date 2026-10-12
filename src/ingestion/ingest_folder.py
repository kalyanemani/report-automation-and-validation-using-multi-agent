from __future__ import annotations

"""
QDR·AI — Batch Folder Ingestion Script
────────────────────────────────────────
Ingests all PDFs from a folder into ChromaDB with proper chunking.

Usage:
    python -m src.ingestion.ingest_folder <folder_path> [--collection NAME]

Example:
    python -m src.ingestion.ingest_folder "C:/path/to/References and Procedures"
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_document
from src.vectorstore.store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qdr_ai.ingest")


def ingest_folder(
    folder_path: str,
    collection_name: str = "references_procedures",
    doc_type: str = "specification",
) -> dict:
    """
    Ingest all PDF files from a folder into ChromaDB.

    Args:
        folder_path: path to folder containing PDFs
        collection_name: ChromaDB collection name prefix
        doc_type: document type label for metadata

    Returns:
        Summary dict with stats per document
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {folder}")

    logger.info(f"{'='*70}")
    logger.info(f"BATCH INGESTION: {len(pdf_files)} PDFs from {folder.name}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"{'='*70}")

    store = VectorStoreManager()
    results = []
    total_chunks = 0
    total_tables = 0
    total_scanned = 0
    all_children = []
    all_parents = []
    start_time = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n{'─'*60}")
        logger.info(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        logger.info(f"{'─'*60}")

        try:
            # 1. Parse
            parsed = parse_pdf(
                pdf_path=pdf_path,
                extract_tables=True,
                strip_headers=True,
                ocr_scanned=True,
            )
            parsed.doc_type = doc_type

            scanned_pages = sum(
                1 for p in parsed.pages if p.is_scanned
            )
            table_count = sum(len(p.tables) for p in parsed.pages)

            # 2. Chunk
            doc_metadata = {
                "doc_type": doc_type,
                "folder": "Folder 2 - References and Procedures",
                "source_file": pdf_path.name,
            }
            child_chunks, parent_chunks = chunk_document(
                parsed, doc_metadata
            )

            # 3. Store in ChromaDB
            # Short safe collection name (keep total name + suffix under 55 chars)
            clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem[:20]).strip("_").lower()
            doc_collection = f"{collection_name[:15]}_{clean_stem}"
            store_result = store.ingest_chunks(
                doc_collection, child_chunks, parent_chunks
            )

            # Accumulate for combined collection
            all_children.extend(child_chunks)
            all_parents.extend(parent_chunks)

            result = {
                "filename": pdf_path.name,
                "pages": parsed.total_pages,
                "chars": parsed.total_chars,
                "tables": table_count,
                "scanned_pages": scanned_pages,
                "child_chunks": len(child_chunks),
                "parent_chunks": len(parent_chunks),
                "collection": doc_collection,
                "status": "OK",
            }
            results.append(result)

            total_chunks += len(child_chunks)
            total_tables += table_count
            total_scanned += scanned_pages

            logger.info(
                f"  -> {len(child_chunks)} children, "
                f"{len(parent_chunks)} parents, "
                f"{table_count} tables, "
                f"{scanned_pages} scanned pages"
            )

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({
                "filename": pdf_path.name,
                "status": "ERROR",
                "error": str(e),
            })

    # Also store ALL chunks in a combined collection for cross-doc search
    logger.info(f"\n{'='*70}")
    logger.info("Creating combined collection for cross-document search...")

    if all_children:
        store.ingest_chunks(
            collection_name, all_children, all_parents
        )
        logger.info(
            f"Combined collection [{collection_name}]: "
            f"{len(all_children)} children, {len(all_parents)} parents"
        )

    elapsed = time.time() - start_time

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("INGESTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Documents:     {len(pdf_files)}")
    logger.info(f"Total chunks:  {total_chunks}")
    logger.info(f"Total tables:  {total_tables}")
    logger.info(f"Scanned pages: {total_scanned}")
    logger.info(f"Time elapsed:  {elapsed:.1f}s")
    logger.info(f"{'='*70}")

    # Per-document summary table
    logger.info("\nPer-document summary:")
    logger.info(
        f"{'File':<55} {'Pages':>5} {'Chunks':>7} "
        f"{'Tables':>7} {'OCR':>5} {'Status':>8}"
    )
    logger.info("-" * 90)
    for r in results:
        if r["status"] == "OK":
            logger.info(
                f"{r['filename'][:54]:<55} "
                f"{r['pages']:>5} "
                f"{r['child_chunks']:>7} "
                f"{r['tables']:>7} "
                f"{r['scanned_pages']:>5} "
                f"{'OK':>8}"
            )
        else:
            logger.info(
                f"{r['filename'][:54]:<55} "
                f"{'':>5} {'':>7} {'':>7} {'':>5} "
                f"{'ERROR':>8}"
            )

    return {
        "documents": len(pdf_files),
        "total_chunks": total_chunks,
        "total_tables": total_tables,
        "scanned_pages": total_scanned,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }


def test_search(collection: str = "references_procedures"):
    """Run sample searches to verify ingestion quality."""
    store = VectorStoreManager()

    test_queries = [
        "hardness acceptance criteria for Alloy 91",
        "PWHT post weld heat treatment temperature requirements",
        "filler metal requirements for welding",
        "calibration requirements for hardness testing equipment",
        "specimen preparation for hardness testing",
    ]

    logger.info(f"\n{'='*70}")
    logger.info("SEARCH QUALITY TEST")
    logger.info(f"{'='*70}")

    for query in test_queries:
        results = store.search(collection, query, k=3)
        logger.info(f"\nQuery: '{query}'")
        if results:
            for i, r in enumerate(results, 1):
                content_preview = r["content"][:120].replace("\n", " ")
                logger.info(
                    f"  #{i} [score={r['score']:.4f}] "
                    f"src={r['metadata'].get('source_file', '?')[:30]} "
                    f"sec={r['metadata'].get('section', '?')[:30]}"
                )
                logger.info(f"      {content_preview}...")
        else:
            logger.info("  No results found")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDF folder into ChromaDB"
    )
    parser.add_argument(
        "folder", help="Path to folder containing PDFs"
    )
    parser.add_argument(
        "--collection", default="references_procedures",
        help="ChromaDB collection name prefix",
    )
    parser.add_argument(
        "--test-search", action="store_true",
        help="Run sample searches after ingestion",
    )
    args = parser.parse_args()

    result = ingest_folder(args.folder, args.collection)

    if args.test_search:
        test_search(args.collection)
