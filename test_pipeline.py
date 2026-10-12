import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.orchestrator import run_inspection_pipeline

sample_submittal = """
QUALITY CONTROL HARDNESS TEST REPORT
Supplier: DEE Piping Systems Co., Ltd.
Report No: HTR-2024-0033 | Rev: 01 | Date: 12-Oct-2024
Project: EL SAUZ Power Plant | PO No: PO-99482
Component: Main Steam Line Elbow (P91)
Material Grade: ASTM A335 Grade P91 (Alloy 91)
Heat Number: HT-99482-A | PWHT Status: Post-PWHT

HARDNESS READINGS (Rockwell HRC):
Location 1 (Base Metal): 215 HRC, 218 HRC, 214 HRC (Avg: 215.6 HRC)
Location 2 (HAZ): 225 HRC, 228 HRC, 220 HRC (Avg: 224.3 HRC)
Location 3 (Weld Metal): 235 HRC, 240 HRC, 238 HRC (Avg: 237.6 HRC)

Testing Equipment: Equotip Portable Hardness Tester
Equipment Serial: EQ-8842 (Calibration Date: 15-Jan-2023 - Expired)
Technician: J. Doe (Level II)
"""

result = run_inspection_pipeline(
    submittal_text=sample_submittal,
    doc_no="HTR-2024-0033",
    revision="01",
    supplier="DEE Piping Systems",
    guide_code="references_procedures"
)

print("\n=== INSPECTION PIPELINE RESULT ===")
print("Score:", result["score"])
print("Disposition:", result["disposition"])
print("Verdict:", result["verdict"])
print("Findings Count:", len(result["findings"]))
for f in result["findings"]:
    print(f"  [{f['type']}] Section: {f['section']} -> {f['comment'][:100]}")
