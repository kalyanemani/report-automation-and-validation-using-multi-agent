from __future__ import annotations

"""
QDR·AI — Knowledge Graph Engine (Graph Engineering Layer)
──────────────────────────────────────────────────────────
Deterministic entity-relationship knowledge graph for engineering specifications,
client procedures, material limits, and inspection requirements.

Schema:
  Nodes:
    - Spec (600-001-R49, 241511CB-HYDP001, 700-006, ASME Sec I, etc.)
    - Material (P91, P22, P11, Carbon Steel, Stainless Steel, etc.)
    - TestType (Hydrostatic Test, Hardness Test)
    - Parameter (Pressure, Hold Time, Chlorides, pH, Hardness, Gauge Range, etc.)
    - Requirement / Rule (1.5x MAWP, < 30 ppm, 248 HBW max, etc.)

  Edges:
    - (Material)-[:REQUIRES_TEST]->(TestType)
    - (TestType)-[:GOVERNED_BY]->(Spec)
    - (Spec)-[:SPECIFIES_RULE]->(Rule)
    - (Rule)-[:APPLIES_TO]->(Parameter)
    - (Rule)-[:TARGETS_MATERIAL]->(Material)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.config import DATA_DIR

logger = logging.getLogger("qdr_ai.knowledge_graph")

GRAPH_PERSIST_PATH = DATA_DIR / "knowledge_graph.json"


class KnowledgeGraph:
    """NetworkX-based Knowledge Graph for Engineering Specs & Inspection Rules."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or GRAPH_PERSIST_PATH
        self.graph = nx.DiGraph()
        self._build_default_graph()

    def _build_default_graph(self) -> None:
        """Populate initial domain knowledge from Nooter/Eriksen and Visavakit Patana specs."""
        self.graph.clear()

        # ── 1. Specification Nodes ──────────────────────────────────
        specs = [
            ("SPEC:600-001", {"name": "600-001-R49", "title": "General Specification for Shop Fabricated HRSG Pressure Part Components", "org": "Nooter/Eriksen", "rev": "49"}),
            ("SPEC:241511CB-HYDP001", {"name": "241511CB-HYDP001", "title": "Hydrostatic Test Procedure", "org": "Visavakit Patana", "rev": "3"}),
            ("SPEC:700-006", {"name": "700-006-R23", "title": "Specification for Welding Alloy 91 Materials", "org": "Nooter/Eriksen", "rev": "23"}),
            ("SPEC:700-016", {"name": "700-016-R2", "title": "Hardness Testing of Alloy 91 Materials", "org": "Nooter/Eriksen", "rev": "2"}),
            ("SPEC:700-022", {"name": "700-022-R0", "title": "Hardness Testing Procedure", "org": "Nooter/Eriksen", "rev": "0"}),
            ("SPEC:ASME_SEC_I", {"name": "ASME BPVC Section I", "title": "Rules for Construction of Power Boilers (PG-99 / PW-54)", "org": "ASME"}),
            ("SPEC:ASME_SEC_VIII", {"name": "ASME BPVC Section VIII", "title": "Rules for Construction of Pressure Vessels (UG-99)", "org": "ASME"}),
            ("SPEC:ASME_B31_1", {"name": "ASME B31.1", "title": "Power Piping Hydrostatic Test", "org": "ASME"}),
        ]
        for node_id, data in specs:
            self.graph.add_node(node_id, node_type="Specification", **data)

        # ── 2. Test Type Nodes ──────────────────────────────────────
        test_types = [
            ("TEST:HYDROSTATIC", {"name": "Hydrostatic Test", "category": "Pressure Testing"}),
            ("TEST:HARDNESS", {"name": "Hardness Test", "category": "Mechanical / NDE"}),
            ("TEST:PWHT", {"name": "Post Weld Heat Treatment", "category": "Thermal Processing"}),
        ]
        for node_id, data in test_types:
            self.graph.add_node(node_id, node_type="TestType", **data)

        # ── 3. Material Nodes ───────────────────────────────────────
        materials = [
            ("MAT:P91", {"name": "P91 / Grade 91", "p_number": "5B", "group": "2", "uns": "K90901", "is_alloy": True}),
            ("MAT:P92", {"name": "P92 / Grade 92", "p_number": "5C", "group": "1", "is_alloy": True}),
            ("MAT:P22", {"name": "P22 / Grade 22", "p_number": "5A", "group": "1", "is_alloy": True}),
            ("MAT:P11", {"name": "P11 / Grade 11", "p_number": "4", "group": "1", "is_alloy": True}),
            ("MAT:CS", {"name": "Carbon Steel", "p_number": "1", "group": "1", "is_alloy": False}),
            ("MAT:SS", {"name": "Stainless Steel (Austenitic)", "p_number": "8", "group": "1", "is_alloy": True}),
            ("MAT:P3G3", {"name": "P-3 Group 3 Materials", "p_number": "3", "group": "3", "is_alloy": True}),
        ]
        for node_id, data in materials:
            self.graph.add_node(node_id, node_type="Material", **data)

        # Connect Materials to Test Types
        for mat_id, _ in materials:
            self.graph.add_edge(mat_id, "TEST:HYDROSTATIC", relation="requires_test")
            self.graph.add_edge(mat_id, "TEST:HARDNESS", relation="requires_test")

        # ── 4. Hydrostatic Requirements & Rules ───────────────────────
        hydro_rules = [
            # Water Chemistry
            ("RULE:HYDRO_WATER_CHLORIDES_GEN", {
                "clause": "6.4.1", "spec": "600-001", "parameter": "Chlorides",
                "limit_operator": "<", "limit_value": 100, "unit": "ppm",
                "scope": "General Carbon / Low Alloy Steels",
                "description": "Chlorides in hydrotest water shall be < 100 ppm."
            }),
            ("RULE:HYDRO_WATER_CHLORIDES_SS", {
                "clause": "6.4.1 / 4.0", "spec": "600-001 & 241511CB-HYDP001", "parameter": "Chlorides",
                "limit_operator": "<", "limit_value": 30, "unit": "ppm",
                "scope": "Stainless Steel",
                "description": "For stainless steel components, chloride limit is < 30 ppm."
            }),
            ("RULE:HYDRO_WATER_PH", {
                "clause": "6.4.1 / 4.0", "spec": "600-001 & 241511CB-HYDP001", "parameter": "pH",
                "limit_min": 7.0, "limit_max": 9.0, "unit": "pH",
                "description": "Hydrotest water pH shall be 7.0 to 9.0 prior to adding inhibitors."
            }),
            ("RULE:HYDRO_WATER_SULPHATES", {
                "clause": "6.4.1 / 4.0", "spec": "600-001 & 241511CB-HYDP001", "parameter": "Sulphates",
                "limit_operator": "<", "limit_value": 15, "unit": "ppm",
                "description": "Sulphates in hydrotest water shall be < 15 ppm."
            }),
            ("RULE:HYDRO_WATER_HARDNESS", {
                "clause": "6.4.1 / 4.0", "spec": "600-001 & 241511CB-HYDP001", "parameter": "Total Hardness",
                "limit_operator": "<", "limit_value": 600, "unit": "ppm",
                "description": "Total Hardness of hydrotest water shall be < 600 ppm."
            }),
            ("RULE:HYDRO_WATER_CONDUCTIVITY", {
                "clause": "6.4.1", "spec": "600-001", "parameter": "Cation Conductivity",
                "limit_operator": "<", "limit_value": 0.1, "unit": "uS/cm",
                "scope": "Demineralized Water",
                "description": "Cation conductivity < 0.1 uS/cm for demineralized water."
            }),
            ("RULE:HYDRO_WATER_BACTERIA", {
                "clause": "4.0", "spec": "241511CB-HYDP001", "parameter": "Bacteria Sulfur",
                "limit_operator": "<=", "limit_value": 1000, "unit": "CFU/ml",
                "description": "Sulfur Reducing Bacteria shall be <= 1000 CFU/ml."
            }),
            # Corrosion Inhibitors
            ("RULE:HYDRO_INHIBITOR", {
                "clause": "6.5.4", "spec": "600-001", "parameter": "Corrosion Inhibitor",
                "options": ["Cortec VpCI 649BD", "Cortec VpCI 649", "Protexol 0.25%"],
                "description": "Corrosion inhibitor type and concentration must be recorded (e.g. 0.25% Protexol or VpCI 649)."
            }),
            # Test Pressure Multipliers
            ("RULE:HYDRO_PRESSURE_SEC_I", {
                "clause": "6.1 / PG-99", "spec": "ASME Section I & B31.1", "parameter": "Test Pressure Multiplier",
                "multiplier": 1.5, "base": "MAWP", "formula": "MAWP * 1.5",
                "description": "Hydrostatic test pressure = 1.5 x MAWP for ASME Section I & B31.1."
            }),
            ("RULE:HYDRO_PRESSURE_SEC_VIII", {
                "clause": "6.1 / UG-99", "spec": "ASME Section VIII", "parameter": "Test Pressure Multiplier",
                "multiplier": 1.3, "base": "MAWP * LSR", "formula": "MAWP * 1.3 * LSR",
                "description": "Hydrostatic test pressure = 1.3 x MAWP x Lowest Stress Ratio (LSR) for Section VIII."
            }),
            # Gauge Requirements
            ("RULE:HYDRO_GAUGE_RANGE", {
                "clause": "4.2", "spec": "241511CB-HYDP001", "parameter": "Gauge Dial Range",
                "multiplier_min": 1.5, "multiplier_max": 4.0,
                "description": "Dial pressure gauges used in testing shall have dials graduated over 1.5 to 4.0 times the test pressure (preferably 2x)."
            }),
            ("RULE:HYDRO_GAUGE_CALIBRATION", {
                "clause": "4.1 / Step #1", "spec": "241511CB-HYDP001 & Review Guide", "parameter": "Calibration Validity",
                "description": "Pressure gauges must be calibrated with a valid calibration date covering the hydrostatic test date."
            }),
            ("RULE:HYDRO_HOLDING_TIME", {
                "clause": "7.3", "spec": "241511CB-HYDP001", "parameter": "Holding Time",
                "min_minutes": 10, "recommended_minutes": 30,
                "description": "Test pressure shall be maintained for a minimum of 10 to 30 minutes before visual inspection for leaks."
            }),
        ]

        for rule_id, data in hydro_rules:
            self.graph.add_node(rule_id, node_type="Rule", test_type="Hydrostatic", **data)
            self.graph.add_edge("TEST:HYDROSTATIC", rule_id, relation="governed_by_rule")

        # Connect specific rules to materials and specs
        self.graph.add_edge("SPEC:600-001", "RULE:HYDRO_WATER_CHLORIDES_GEN", relation="specifies")
        self.graph.add_edge("SPEC:600-001", "RULE:HYDRO_WATER_CHLORIDES_SS", relation="specifies")
        self.graph.add_edge("SPEC:600-001", "RULE:HYDRO_WATER_PH", relation="specifies")
        self.graph.add_edge("SPEC:600-001", "RULE:HYDRO_WATER_SULPHATES", relation="specifies")
        self.graph.add_edge("SPEC:600-001", "RULE:HYDRO_WATER_HARDNESS", relation="specifies")
        self.graph.add_edge("SPEC:600-001", "RULE:HYDRO_INHIBITOR", relation="specifies")
        self.graph.add_edge("SPEC:241511CB-HYDP001", "RULE:HYDRO_GAUGE_RANGE", relation="specifies")
        self.graph.add_edge("SPEC:241511CB-HYDP001", "RULE:HYDRO_GAUGE_CALIBRATION", relation="specifies")
        self.graph.add_edge("SPEC:241511CB-HYDP001", "RULE:HYDRO_HOLDING_TIME", relation="specifies")
        self.graph.add_edge("MAT:SS", "RULE:HYDRO_WATER_CHLORIDES_SS", relation="stricter_limit")

        # ── 5. Hardness Requirements & Limits ─────────────────────────
        hardness_rules = [
            ("RULE:HARDNESS_P91", {
                "material": "P91", "max_limit": 248, "unit": "HBW",
                "alt_limits": {"HV10": 250, "HRC": 25},
                "spec": "700-006 / 700-016", "clause": "Clause 8.1",
                "method": "UCI or Brinell", "min_readings": 5, "discard_extremes": True,
                "description": "P91 maximum allowable hardness is 248 HBW (250 HV10 / 25 HRC)."
            }),
            ("RULE:HARDNESS_P22", {
                "material": "P22", "max_limit": 225, "unit": "HBW",
                "spec": "600-001 / NACE MR0175", "clause": "Section 4",
                "description": "P22 maximum allowable hardness is 225 HBW."
            }),
            ("RULE:HARDNESS_P11", {
                "material": "P11", "max_limit": 225, "unit": "HBW",
                "spec": "600-001", "clause": "Section 4",
                "description": "P11 maximum allowable hardness is 225 HBW."
            }),
            ("RULE:HARDNESS_CS", {
                "material": "CS", "max_limit": 200, "unit": "HBW",
                "spec": "600-001", "clause": "Section 4",
                "description": "Carbon steel maximum allowable hardness is 200 HBW."
            }),
            ("RULE:HARDNESS_P3G3", {
                "material": "P3 Group 3", "max_limit": 225, "unit": "HBW",
                "spec": "700-026", "clause": "Section 4",
                "description": "P3 Group 3 material maximum allowable hardness is 225 HBW."
            }),
        ]

        for rule_id, data in hardness_rules:
            self.graph.add_node(rule_id, node_type="Rule", test_type="Hardness", **data)
            self.graph.add_edge("TEST:HARDNESS", rule_id, relation="governed_by_rule")

        self.graph.add_edge("MAT:P91", "RULE:HARDNESS_P91", relation="has_hardness_limit")
        self.graph.add_edge("MAT:P22", "RULE:HARDNESS_P22", relation="has_hardness_limit")
        self.graph.add_edge("MAT:P11", "RULE:HARDNESS_P11", relation="has_hardness_limit")
        self.graph.add_edge("MAT:CS", "RULE:HARDNESS_CS", relation="has_hardness_limit")
        self.graph.add_edge("MAT:P3G3", "RULE:HARDNESS_P3G3", relation="has_hardness_limit")

        logger.info(
            f"KnowledgeGraph initialized with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )
        self.save()

    # ── Persistence ─────────────────────────────────────────────────

    def save(self) -> None:
        """Serialize graph to JSON."""
        try:
            data = nx.node_link_data(self.graph)
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save KnowledgeGraph: {e}")

    def load(self) -> bool:
        """Load graph from JSON if present."""
        if not self.persist_path.exists():
            return False
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
            return True
        except Exception as e:
            logger.warning(f"Failed to load KnowledgeGraph: {e}")
            return False

    # ── Graph Query API ─────────────────────────────────────────────

    def find_rules_for_material(self, material_name: str, test_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find all applicable rules for a given material grade and test type."""
        mat_key = material_name.upper().replace(" ", "").replace("GRADE", "").replace("ALLOY", "")
        matching_nodes = [
            n for n in self.graph.nodes
            if self.graph.nodes[n].get("node_type") == "Material"
            and any(k in n for k in (mat_key, material_name.upper()))
        ]
        if not matching_nodes:
            # Fallback search by aliases
            for n in self.graph.nodes:
                if self.graph.nodes[n].get("node_type") == "Material":
                    node_name = self.graph.nodes[n].get("name", "").upper()
                    if material_name.upper() in node_name or mat_key in node_name:
                        matching_nodes.append(n)

        rules = []
        for mat_node in matching_nodes:
            for neighbor in self.graph.neighbors(mat_node):
                node_data = self.graph.nodes[neighbor]
                if node_data.get("node_type") == "Rule":
                    if not test_type or node_data.get("test_type", "").lower() == test_type.lower():
                        rules.append({**node_data, "rule_id": neighbor})

        return rules

    def find_water_chemistry_limits(self, is_stainless_steel: bool = False) -> Dict[str, Any]:
        """Get water chemistry parameter limits."""
        limits = {
            "pH": {"min": 7.0, "max": 9.0, "unit": "pH"},
            "Chlorides": {"max": 30 if is_stainless_steel else 100, "unit": "ppm", "scope": "Stainless Steel" if is_stainless_steel else "Carbon/Low Alloy Steel"},
            "Sulphates": {"max": 15, "unit": "ppm"},
            "Total Hardness": {"max": 600, "unit": "ppm"},
            "Cation Conductivity": {"max": 0.1, "unit": "uS/cm"},
            "Bacteria Sulfur": {"max": 1000, "unit": "CFU/ml"},
        }
        return limits

    def find_pressure_multiplier_rule(self, code: str = "Section I") -> Dict[str, Any]:
        """Get test pressure multiplier rule for ASME Section I, VIII, or B31.1."""
        code_upper = code.upper()
        if "VIII" in code_upper:
            return {
                "code": "ASME Section VIII",
                "multiplier": 1.3,
                "formula": "MAWP * 1.3 * LSR",
                "notes": "Lowest Stress Ratio (LSR) applies.",
                "clause": "UG-99",
            }
        else:
            return {
                "code": "ASME Section I / B31.1",
                "multiplier": 1.5,
                "formula": "MAWP * 1.5",
                "notes": "Standard boiler pressure parts test pressure.",
                "clause": "PG-99 / PW-54",
            }

    def find_gauge_requirements(self) -> Dict[str, Any]:
        """Get gauge calibration & dial range requirements."""
        return {
            "dial_range": {
                "multiplier_min": 1.5,
                "multiplier_max": 4.0,
                "recommended": 2.0,
                "rule": "Dial range shall be 1.5x to 4.0x test pressure.",
            },
            "calibration": {
                "rule": "Gauge calibration date must be valid and cover the test date.",
                "standard": "Dead-weight tester or standard calibrated master gauge.",
            },
        }

    def get_summary_context(self, test_type: str = "Hydrostatic", material: Optional[str] = None) -> str:
        """Produce a consolidated prompt-ready text summary of relevant graph rules."""
        lines = [f"=== KNOWLEDGE GRAPH SPECIFICATION RULES [{test_type.upper()}] ==="]
        if test_type.lower() in ("hydrostatic", "hydr"):
            is_ss = bool(material and ("SS" in material.upper() or "STAINLESS" in material.upper()))
            chem = self.find_water_chemistry_limits(is_stainless_steel=is_ss)
            lines.append("1. WATER CHEMISTRY LIMITS:")
            for param, d in chem.items():
                if "min" in d and "max" in d:
                    lines.append(f"   - {param}: {d['min']} to {d['max']} {d.get('unit', '')}")
                elif "max" in d:
                    lines.append(f"   - {param}: < {d['max']} {d.get('unit', '')} ({d.get('scope', 'Standard')})")

            p_rule_i = self.find_pressure_multiplier_rule("Section I")
            p_rule_viii = self.find_pressure_multiplier_rule("Section VIII")
            lines.append(f"2. TEST PRESSURE MULTIPLIERS:")
            lines.append(f"   - ASME Section I / B31.1: {p_rule_i['formula']} ({p_rule_i['clause']})")
            lines.append(f"   - ASME Section VIII: {p_rule_viii['formula']} ({p_rule_viii['clause']})")

            gauge = self.find_gauge_requirements()
            lines.append(f"3. GAUGE & HOLDING TIME REQUIREMENTS:")
            lines.append(f"   - Dial Range: {gauge['dial_range']['rule']}")
            lines.append(f"   - Calibration: {gauge['calibration']['rule']}")
            lines.append(f"   - Holding Time: Min 10-30 minutes before inspection (600-001 / 241511CB-HYDP001)")
            lines.append(f"   - Inhibitor: Cortec VpCI 649 or 0.25% Protexol recorded (600-001 Sec 6.5.4)")

        elif test_type.lower() in ("hardness", "hrdr"):
            lines.append("1. HARDNESS LIMITS:")
            for mat in ["P91", "P22", "P11", "CS", "P3 Group 3"]:
                rules = self.find_rules_for_material(mat, test_type="Hardness")
                for r in rules:
                    lines.append(f"   - {mat}: Max {r.get('max_limit')} {r.get('unit', 'HBW')} ({r.get('spec', '')})")
            lines.append("2. READINGS METHOD:")
            lines.append("   - Min 5 readings per location; discard min & max; average remaining 3.")

        return "\n".join(lines)


# Global singleton
_kg_instance: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance
