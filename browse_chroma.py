import sys
import chromadb
from pathlib import Path
from src.config import CHROMA_PERSIST_DIR

# Set UTF-8 encoding for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

def inspect_collection(col_name: str, max_items: int = 5):
    try:
        col = client.get_collection(col_name)
    except Exception as e:
        print(f"Collection '{col_name}' not found: {e}")
        return

    count = col.count()
    print("\n" + "=" * 80)
    print(f"COLLECTION: {col_name} (Total Items: {count})")
    print("=" * 80)

    data = col.get(limit=max_items, include=["metadatas", "documents"])
    for i, (cid, doc, meta) in enumerate(zip(data["ids"], data["documents"], data["metadatas"]), 1):
        src = meta.get("filename") or meta.get("source_file") or meta.get("source") or "Unknown"
        page = meta.get("page_number") or meta.get("page") or "1"
        sec = meta.get("section") or meta.get("title") or "General"
        
        print(f"\n--- Item #{i} [ID: {cid[:8]}...] ---")
        print(f"Source File  : {src}")
        print(f"Page Number  : {page}")
        print(f"Section      : {sec}")
        print(f"Metadata     : {meta}")
        print(f"Content Snippet:")
        lines = [line.strip() for line in doc.split("\n") if line.strip()]
        for line in lines[:5]:
            print(f"   {line}")
        if len(lines) > 5:
            print("   ...")

if __name__ == "__main__":
    inspect_collection("qdr_references_procedures_parents", max_items=3)
    inspect_collection("qdr_references_procedures_children", max_items=4)
