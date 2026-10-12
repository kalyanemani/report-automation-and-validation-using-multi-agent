import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Force UTF-8 stdout/stderr encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import chunk_document
from src.vectorstore.store import VectorStoreManager

console = Console()

def inspect_document_chunks(pdf_path: str, max_show: int = 5):
    """Parse a document and print sample chunks with metadata."""
    console.print(f"\n[bold cyan]Parsing document:[/] {Path(pdf_path).name}")
    parsed = parse_pdf(pdf_path=pdf_path, extract_tables=True, strip_headers=True)
    
    doc_metadata = {"source_file": Path(pdf_path).name}
    child_chunks, parent_chunks = chunk_document(parsed, doc_metadata)
    
    console.print(f"[bold green]Total Child Chunks:[/] {len(child_chunks)} | [bold green]Parent Chunks:[/] {len(parent_chunks)}\n")
    
    # Filter table chunks vs text chunks
    table_chunks = [c for c in child_chunks if c.metadata.get("chunk_type") == "table"]
    section_chunks = [c for c in child_chunks if c.metadata.get("chunk_type") == "text"]
    
    console.print(f"[bold yellow]Section Chunks:[/] {len(section_chunks)} | [bold yellow]Table Chunks:[/] {len(table_chunks)}\n")
    
    console.print("[bold underline magenta]Sample Section Chunks:[/]")
    for i, c in enumerate(section_chunks[:max_show], 1):
        console.print(Panel(
            f"[bold white]{c.content[:400]}[/]...",
            title=f"Chunk #{i} | Section: {c.metadata.get('section', 'N/A')} | Strategy: {c.metadata.get('strategy')}",
            subtitle=f"Meta: Grades=[cyan]{c.metadata.get('material_grades', 'None')}[/] Codes=[cyan]{c.metadata.get('applicable_codes', 'None')}[/]"
        ))

    if table_chunks:
        console.print("\n[bold underline magenta]Sample Atomic Table Chunks:[/]")
        for i, c in enumerate(table_chunks[:max_show], 1):
            console.print(Panel(
                f"[green]{c.content}[/]",
                title=f"Table Chunk #{i} | Page: {c.metadata.get('page_number', '?')} | Rows: {c.metadata.get('table_rows', '?')}",
                subtitle=f"Strategy: {c.metadata.get('strategy')}"
            ))

def test_vector_search(query: str, collection: str = "references_procedures", k: int = 3):
    """Run a search query against ChromaDB and display matched chunks."""
    console.print(f"\n[bold cyan]Querying Vector DB:[/] '{query}' (Top {k})")
    store = VectorStoreManager()
    results = store.search(collection, query, k=k, return_parents=True)
    
    if not results:
        console.print("[red]No chunks found for this query.[/]")
        return

    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", style="bold cyan", width=6)
    table.add_column("Score", style="magenta", width=8)
    table.add_column("Doc / Section", style="green", width=30)
    table.add_column("Chunk Snippet", style="white")

    for i, r in enumerate(results, 1):
        src = r['metadata'].get('source_file', 'Unknown')
        sec = r['metadata'].get('section', 'N/A')
        snippet = r['content'][:150].replace('\n', ' ')
        table.add_row(str(i), f"{r['score']:.4f}", f"{src}\n[yellow]{sec}[/]", f"{snippet}...")

    console.print(table)


if __name__ == "__main__":
    sample_pdf = r"C:\Users\krishkal\Downloads\Document Review Guides\Document Review Guides\2. References and Procedures\700-006-R23 Specification For Welding Alloy 91 Materials.pdf"
    
    # 1. Inspect chunks of a spec document
    inspect_document_chunks(sample_pdf, max_show=3)
    
    # 2. Test vector DB similarity search
    test_vector_search("hardness testing requirement for Alloy 91 weld", k=3)
    test_vector_search("PWHT post weld heat treatment temperature", k=3)
