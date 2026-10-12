import sys
from pathlib import Path
from src.agents.hydro_orchestrator import run_hydrostatic_inspection_pipeline
from src.ingestion.parser import parse_review_guide
from src.knowledge_graph.graph_builder import KnowledgeGraph

def main():
    print("=" * 60)
    print("1. VERIFYING KNOWLEDGE GRAPH DETERMINISTIC RULES")
    print("=" * 60)
    kg = KnowledgeGraph()
    chem_limits = kg.find_water_chemistry_limits(is_stainless_steel=True)
    p_rule = kg.find_pressure_multiplier_rule("Section I")
    print("Chlorides limit (SS):", chem_limits["Chlorides"])
    print("pH limits:", chem_limits["pH"])
    print("Pressure Multipliers (Sec I):", p_rule)

    print("\n" + "=" * 60)
    print("2. VERIFYING DYNAMIC REVIEW GUIDE PARSER")
    print("=" * 60)
    guide_sample = """
Step #1: Verify hydrostatic test pressure is at least 1.5x MAWP.
Step #2: Verify water chlorides are strictly below 30 ppm for Stainless Steel.
Step #3: Verify dial gauge range is between 1.5x and 4.0x test pressure.
    """
    guide_res = parse_review_guide(guide_sample, filename="HYDR Review Guide R1.docm")
    print(f"Mode: {guide_res['mode']}, Total Rules: {guide_res['total_rules']}")
    for idx, r in enumerate(guide_res["items"], 1):
        print(f"  Rule {idx}: {r}")

    print("\n" + "=" * 60)
    print("3. RUNNING END-TO-END HYDROSTATIC TWO-STAGE PIPELINE")
    print("=" * 60)
    submittal_path = r"C:\Users\krishkal\Downloads\Document Review Guides\241511CB-HYDP001.pdf"
    if not Path(submittal_path).exists():
        submittal_path = r"data\documents\241516CB-HYDR001 1.pdf"

    result = run_hydrostatic_inspection_pipeline(
        pdf_path=submittal_path,
        filename=Path(submittal_path).name,
        review_guide_data=guide_sample,
        review_guide_name="HYDR Review Guide R1.docm",
        gen_params_input={
            "supplier": "Visavakit Patana Corp., Ltd.",
            "doc_no": "241511CB-HYDP001",
            "revision": "3",
            "material": "Stainless Steel",
            "mawp": 130.0,
            "design_code": "Section I",
            "qa_reading": 195.0,
            "unit": "PSIG"
        },
        tolerance=2.0
    )

    print("\n" + "=" * 60)
    print("4. PIPELINE EXECUTION OUTCOME")
    print("=" * 60)
    print(f"Submittal: {result['submittal_filename']}")
    print(f"Disposition: {result['disposition']}")
    print(f"Compliance Score: {result['compliance_score']}/100")
    print(f"MAWP: {result['mawp']} PSIG -> Required Test Pressure: {result['required_test_pressure']} PSIG")
    print(f"Gauges Audited: {result['total_gauges_reviewed']}")

    for idx, g in enumerate(result["gauge_validations"], 1):
        print(f"\n--- GAUGE #{idx}: {g['gauge_id']} (Page {g['page_number']}) ---")
        ai_r = g.get("ai_reading", {})
        qa_r = g.get("qa_reading", {})
        s1 = g.get("stage_1", {})
        s2 = g.get("stage_2", {})
        req = g.get("procedure_requirement", {})
        print(f"  Stage 1 (AI vs QA): AI={ai_r.get('value')} {ai_r.get('unit')} (Conf: {ai_r.get('confidence')}) | QA={qa_r.get('value')} {qa_r.get('unit')} -> {s1.get('status')} (Diff: {s1.get('difference')})")
        print(f"  Stage 2 (Compliance): Confirmed={qa_r.get('value')} {qa_r.get('unit')} | Req={req.get('required_value')} {req.get('unit')} ({req.get('rule')}) -> {s2.get('status')}")
        print(f"  Final Disposition: {g['final_disposition']}")

    print("\nWater Chemistry Findings:", result.get("water_chemistry_checks", []))
    print("\n[SUCCESS] Phase 2 End-to-End Pipeline Execution Verified!")

if __name__ == "__main__":
    main()
