import pymupdf as fitz  # PyMuPDF 1.28+: `fitz` is deprecated
import pdfplumber
import re
import os

folder = r'C:\Users\krishkal\Downloads\Document Review Guides\Document Review Guides\2. References and Procedures'

for fname in ['700-006-R23 Specification For Welding Alloy 91 Materials.pdf',
              '700-022-R0 Hardness Testing Procedure.pdf',
              '221135LBP-HRDP001.pdf']:
    path = os.path.join(folder, fname)
    doc = fitz.open(path)
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"FILE: {fname}")
    print(f"Pages: {len(doc)}")
    
    for pn in range(min(3, len(doc))):
        text = doc[pn].get_text('text')
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        print(f"\n--- Page {pn+1} (first 30 lines) ---")
        for line in lines[:30]:
            print(f"  {line[:120]}")
    
    # Check for tables
    try:
        with pdfplumber.open(path) as pdf:
            total_tables = 0
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    total_tables += len(tables)
                    if total_tables <= 2:
                        t = tables[0]
                        print(f"\n--- Table on page {i+1} ({len(t)} rows) ---")
                        for row in t[:3]:
                            cleaned = [str(c)[:30] if c else "" for c in row]
                            print(f"  {cleaned}")
            print(f"\nTotal tables found: {total_tables}")
    except Exception as e:
        print(f"pdfplumber error: {e}")
    
    doc.close()
