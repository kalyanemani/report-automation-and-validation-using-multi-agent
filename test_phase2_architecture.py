from __future__ import annotations

"""
QDR·AI — Phase 2 Architecture Verification Test Suite
──────────────────────────────────────────────────────
Tests:
  1. Dynamic Review Guide & Custom Criteria Parser (docm, pasted text, checked/unchecked/no boxes)
  2. Knowledge Graph Engine (NetworkX entities, limits, multipliers, formulas)
  3. Hybrid Retriever (Vector + Graph + Review Guide)
  4. Two-Stage Validation (AI vs QA reading -> Client procedure compliance)
  5. Loop Engineering Controller (Iterative validation, retry limit, escalation)
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.parser import parse_review_guide, extract_checklist_items
from src.knowledge_graph.graph_builder import KnowledgeGraph, get_knowledge_graph
from src.knowledge_graph.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.agents.two_stage_validator import (
    TwoStageValidator,
    ValidationStatus,
    FailureCategory,
)
from src.agents.loop_controller import LoopController, MAX_RETRIES


class TestPhase2Architecture(unittest.TestCase):

    def test_1_dynamic_review_guide_no_checkboxes(self):
        """Test review guide with no checkboxes (should extract all numbered steps/rules)."""
        guide_text = """
        Step #1: Review the report to determine if any items are missing.
        - Cover sheet or header with Project Information.
        - Hydrostatic test procedure (HYDP), with revision.
        - Verify test pressure calculation (MAWP x 1.5 for Section I).
        - Record of corrosion inhibitor added (0.25% Protexol or VpCI 649).
        Step #2: Check pressure gauge calibration report covering test date.
        """
        result = parse_review_guide(guide_text, "no_checkbox_guide.txt")
        self.assertEqual(result["mode"], "all_criteria")
        self.assertGreater(result["total_rules"], 3)
        self.assertTrue(any("Verify test pressure" in item for item in result["items"]))

    def test_2_dynamic_review_guide_with_checked_boxes(self):
        """Test review guide with checked boxes (should extract only checked items)."""
        guide_text = """
        ☑ Verify the test pressure calculation (MAWP x 1.5).
        ☐ Verify water chemistry pH between 7.0 and 9.0.
        ☑ Pressure gauge calibration must cover test date.
        """
        result = parse_review_guide(guide_text, "checked_guide.txt")
        self.assertEqual(result["mode"], "checked_items_only")
        self.assertEqual(result["checked_count"], 2)
        self.assertEqual(len(result["items"]), 2)
        self.assertTrue(any("test pressure calculation" in item for item in result["items"]))

    def test_3_knowledge_graph_rules(self):
        """Test Knowledge Graph deterministic limit lookups."""
        kg = get_knowledge_graph()

        # Water chemistry for Stainless Steel
        chem_ss = kg.find_water_chemistry_limits(is_stainless_steel=True)
        self.assertEqual(chem_ss["Chlorides"]["max"], 30)
        self.assertEqual(chem_ss["pH"]["min"], 7.0)
        self.assertEqual(chem_ss["pH"]["max"], 9.0)

        # Water chemistry for Carbon Steel
        chem_cs = kg.find_water_chemistry_limits(is_stainless_steel=False)
        self.assertEqual(chem_cs["Chlorides"]["max"], 100)

        # Test Pressure Multipliers
        p_sec_i = kg.find_pressure_multiplier_rule("Section I")
        self.assertEqual(p_sec_i["multiplier"], 1.5)
        p_sec_viii = kg.find_pressure_multiplier_rule("Section VIII")
        self.assertEqual(p_sec_viii["multiplier"], 1.3)

        # Hardness Limits
        p91_rules = kg.find_rules_for_material("P91", test_type="Hardness")
        self.assertTrue(len(p91_rules) > 0)
        self.assertEqual(p91_rules[0]["max_limit"], 248)

    def test_4_stage_1_pass_stage_2_pass(self):
        """Stage 1: AI=195, QA=195 -> PASS. Stage 2: Required=195 -> PASS (ACCEPTED)."""
        validator = TwoStageValidator()
        ai = {"value": 195.0, "unit": "PSIG", "confidence": 0.95}
        qa = {"value": 195.0, "unit": "PSIG", "source": "manual_qa"}
        req = {"required_value": 195.0, "rule": "MAWP * 1.5", "tolerance": 0.0}

        report = validator.evaluate_gauge(
            page_number=1, gauge_id="G-01", ai_reading=ai, qa_reading=qa, procedure_requirement=req
        )
        self.assertEqual(report.stage_1["status"], "PASS")
        self.assertEqual(report.stage_2["status"], "PASS")
        self.assertEqual(report.final_disposition, "ACCEPTED")
        self.assertEqual(report.failure_category, "NONE")

    def test_5_stage_1_pass_stage_2_fail(self):
        """Stage 1: AI=195, QA=195 -> PASS. Stage 2: Required=200 -> FAIL (NON_CONFORMANCE)."""
        validator = TwoStageValidator()
        ai = {"value": 195.0, "unit": "PSIG", "confidence": 0.95}
        qa = {"value": 195.0, "unit": "PSIG", "source": "manual_qa"}
        req = {"required_value": 200.0, "rule": "MAWP * 1.5 = 200 PSIG", "tolerance": 0.0}

        report = validator.evaluate_gauge(
            page_number=1, gauge_id="G-02", ai_reading=ai, qa_reading=qa, procedure_requirement=req
        )
        self.assertEqual(report.stage_1["status"], "PASS")
        self.assertEqual(report.stage_2["status"], "FAIL")
        self.assertEqual(report.final_disposition, "NON_CONFORMANCE")
        self.assertEqual(report.failure_category, FailureCategory.PROCEDURE_NON_COMPLIANCE.value)

    def test_6_stage_1_fail_skips_stage_2(self):
        """Stage 1: AI=180, QA=195 -> FAIL (EXTRACTION_MISMATCH). Stage 2 must be NOT_EVALUATED."""
        validator = TwoStageValidator()
        ai = {"value": 180.0, "unit": "PSIG", "confidence": 0.90}
        qa = {"value": 195.0, "unit": "PSIG", "source": "manual_qa"}
        req = {"required_value": 195.0, "rule": "MAWP * 1.5"}

        report = validator.evaluate_gauge(
            page_number=1, gauge_id="G-03", ai_reading=ai, qa_reading=qa, procedure_requirement=req
        )
        self.assertEqual(report.stage_1["status"], "FAIL")
        self.assertEqual(report.stage_2["status"], "NOT_EVALUATED")
        self.assertEqual(report.final_disposition, "EXTRACTION_MISMATCH")
        self.assertEqual(report.failure_category, FailureCategory.AI_READING_MISMATCH.value)

    def test_7_loop_controller_max_retries_escalation(self):
        """Test Loop Controller retry limit and escalation."""
        controller = LoopController(max_retries=3)
        # Ambiguous AI value that causes low confidence / unclear result
        ai_unclear = {"value": None, "unit": "PSIG", "confidence": 0.3}
        qa_val = {"value": 195.0, "unit": "PSIG"}

        report, trace = controller.run_validation_loop(
            page_number=2,
            gauge_id="G-04",
            ai_reading=ai_unclear,
            qa_reading=qa_val,
        )
        self.assertEqual(trace.attempts, 3)
        self.assertTrue(trace.human_escalation)
        self.assertEqual(report.final_disposition, "HUMAN_REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
