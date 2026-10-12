import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.vectorstore.store import VectorStoreManager

console = Console()

CHECKLIST_ITEMS = [
    {
        "id": "1",
        "query": "Identification of each hardness test location 700-022 Section 10.1.2",
        "label": "Identification of each hardness test location [700-022, 10.1.2]",
    },
    {
        "id": "2",
        "query": "Quantity of individual readings at each hardness test location number of impressions readings",
        "label": "Quantity of individual readings at each hardness test location",
    },
    {
        "id": "3",
        "query": "Values of individual readings at each hardness test location reported values",
        "label": "Values of individual readings at each hardness test location",
    },
    {
        "id": "4",
        "query": "Acceptable average value at each hardness test location average hardness acceptance criteria",
        "label": "Acceptable average value at each hardness test location",
    },
]


def test_checklist_retrieval():
    console.print("\n[bold magenta]======================================================================[/]")
    console.print("[bold cyan]SEARCH TEST: 4 Specific Hardness Review Guide Items (BGE-M3 Embeddings)[/]")
    console.print("[bold magenta]======================================================================[/]\n")

    store = VectorStoreManager()

    for item in CHECKLIST_ITEMS:
        console.print(f"[bold yellow]Checklist Item #{item['id']}:[/] [bold white]{item['label']}[/]")
        results = store.search(
            guide_code="references_procedures",
            query=item["query"],
            k=3,
            return_parents=True,
        )

        if not results:
            console.print("  [red]No matching chunks found.[/]\n")
            continue

        table = Table(show_header=True, header_style="bold green")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Score", style="magenta", width=8)
        table.add_column("Source Document", style="yellow", width=32)
        table.add_column("Section / Header", style="green", width=25)
        table.add_column("Chunk Preview / Citation", style="white")

        for rank, r in enumerate(results, 1):
            src = r["metadata"].get("source_file", "Unknown")
            sec = r["metadata"].get("section", "N/A")
            snippet = r["content"][:180].replace("\n", " ").strip()
            table.add_row(
                f"#{rank}",
                f"{r['score']:.4f}",
                src[:31],
                sec[:24],
                f"{snippet}...",
            )

        console.print(table)
        console.print()


if __name__ == "__main__":
    test_checklist_retrieval()
