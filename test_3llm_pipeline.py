import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import run_inspection_pipeline
from src.ingestion.parser import parse_pdf

pdf_path = PROJECT_ROOT / "data" / "documents" / "221135LBP-HRDR001 R0 - Submittal.pdf"
print(f"Reading submittal: {pdf_path.name}")
parsed = parse_pdf(pdf_path=pdf_path, ocr_scanned=False)

gen_params_input = {
    "proj_name": "EL SAUZ",
    "po_num": "TJ/22-23/096",
    "dwg_num": "221135LBP-HP18",
    "supplier": "DEE Piping Systems (Thailand) Co. Ltd.",
    "report_num": "DEE-HRD-ELSAUZ-00001",
    "proc_ref": "DEETH-QCP-EL-003 REV.1",
}

print("Running 3-LLM Multi-Agent Pipeline...")
result = run_inspection_pipeline(
    submittal_text=parsed.full_text,
    submittal_filename=pdf_path.name,
    submittal_filepath=str(pdf_path),
    guide_code="references_procedures",
    supplier=gen_params_input["supplier"],
    doc_no=gen_params_input["report_num"],
    revision="0",
    reviewer="Senior QA Inspector",
    gen_params_input=gen_params_input,
)

print("\n=================== FINAL 3-LLM PIPELINE RESULTS ===================")
print(f"Compliance Score: {result.get('score')}/100")
print(f"Disposition: {result.get('disposition')}")
print(f"Verdict: {result.get('verdict')}")
print(f"\nExecutive Summary:\n{result.get('summary')}")
print(f"\nTotal Findings: {len(result.get('findings', []))}")
for f in result.get("findings", []):
    print(f"  - [{f.get('type')}] {f.get('section')}: {f.get('comment')} (Evidence: {f.get('evidence')})")
