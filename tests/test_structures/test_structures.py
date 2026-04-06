# tests/test_structures/test_structures.py
"""End-to-end integration tests for the structures module."""
import math
import numpy as np
import pytest

import aerisplane as ap
import aerisplane.aero as aero
import aerisplane.weights as wts
import aerisplane.structures as struc
from aerisplane.structures.result import StructureResult, WingStructureResult


class TestAnalyze:
    def test_returns_structure_result(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        assert isinstance(result, StructureResult)

    def test_only_wings_with_spars_included(self, simple_aircraft, cruise_condition):
        # simple_aircraft has rect_wing (with spar) and htail (no spar)
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wr_wing in result.wings:
            assert wr_wing.wing_name == "rect_wing"

    def test_design_load_factor_at_least_525(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr,
                               n_limit=3.5, safety_factor=1.5)
        assert result.design_load_factor >= 3.5 * 1.5

    def test_tip_deflection_finite_and_positive(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wing_result in result.wings:
            assert wing_result.tip_deflection > 0.0
            assert np.isfinite(wing_result.tip_deflection)

    def test_tip_deflection_ratio_reasonable(self, simple_aircraft, cruise_condition):
        # Tip deflection / semispan should be < 50% for a stiff CF spar
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wing_result in result.wings:
            assert wing_result.tip_deflection_ratio < 0.50

    def test_safe_for_well_designed_wing(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr,
                               n_limit=3.5, safety_factor=1.5)
        assert result.is_safe

    def test_report_runs_and_is_nonempty(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        report = result.report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_all_margins_are_finite(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wr_wing in result.wings:
            assert np.isfinite(wr_wing.bending_margin)
            assert np.isfinite(wr_wing.shear_margin)
            assert np.isfinite(wr_wing.buckling_margin)
