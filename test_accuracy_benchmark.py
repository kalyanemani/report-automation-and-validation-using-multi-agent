import fitz
from pathlib import Path
from src.agents.orchestrator import run_inspection_pipeline

submittals = [
    r"data/documents/221135LBP-HRDR001 R0 - Submittal.pdf",
    r"C:\Users\krishkal\Downloads\Document Review Guides\Document Review Guides\3. Supplier Submittals\241135LBP-HRDR001 R0 - Submittal.pdf",
    r"C:\Users\krishkal\Downloads\Document Review Guides\Document Review Guides\3. Supplier Submittals\221400-038SIL-HRDR090 R0 - Submittal.pdf"
]

for s_path in submittals:
    p = Path(s_path)
    if p.exists():
        doc = fitz.open(str(p))
        num_p = len(doc)
        txt = "\n".join([doc[i].get_text() for i in range(num_p)])
        doc.close()
        print("\n======================================================")
        print(f"TESTING: {p.name} ({num_p} Pages)")
        print("======================================================")
        res = run_inspection_pipeline(
            submittal_text=txt,
            submittal_filename=p.name,
            submittal_filepath=str(p),
            guide_code="references_procedures"
        )
        print(f"Score: {res.get('score')} | Disposition: {res.get('disposition')}")
        findings = res.get("findings", [])
        print(f"Total Findings: {len(findings)}")
        for f in findings[:6]:
            t = f.get("type")
            pidx = f.get("page_idx", 0) + 1
            cmt = f.get("comment", "")[:100]
            print(f"  [{t}] Page {pidx}: {cmt}...")
