"""Tests for fuselage-wing interference models."""
import numpy as np
import pytest

from aerisplane.aero.library.interference import (
    CDA_junction,
    wing_carryover_lift_factor,
    wing_carryover_drag_factor,
    total_junction_drag,
)


class TestJunctionDrag:
    def test_typical_rc_aircraft(self):
        cda = CDA_junction(wing_root_thickness=0.03, fuselage_radius=0.04, Re_root=3e5)
        assert 1e-5 < cda < 1e-3, f"CDA_junction={cda:.6f} out of expected range"

    def test_zero_thickness_zero_drag(self):
        cda = CDA_junction(wing_root_thickness=0.0, fuselage_radius=0.04, Re_root=3e5)
        assert cda == pytest.approx(0.0, abs=1e-12)

    def test_scales_with_thickness(self):
        cda1 = CDA_junction(wing_root_thickness=0.02, fuselage_radius=0.04, Re_root=3e5)
        cda2 = CDA_junction(wing_root_thickness=0.04, fuselage_radius=0.04, Re_root=3e5)
        ratio = cda2 / cda1
        assert 1.5 < ratio < 2.5, f"Scaling ratio={ratio:.2f}, expected ~2 (linear in t)"

    def test_fillet_reduces_drag(self):
        cda_no = CDA_junction(wing_root_thickness=0.03, fuselage_radius=0.04, Re_root=3e5, fillet_radius=0.0)
        cda_yes = CDA_junction(wing_root_thickness=0.03, fuselage_radius=0.04, Re_root=3e5, fillet_radius=0.01)
        assert cda_yes < cda_no


class TestCarryoverFactors:
    def test_lift_factor_no_fuselage(self):
        assert wing_carryover_lift_factor(0.0, 2.0) == pytest.approx(1.0)

    def test_lift_factor_typical_rc(self):
        k = wing_carryover_lift_factor(0.11, 2.4)
        assert 1.0 < k < 1.01

    def test_lift_factor_large_fuselage(self):
        k = wing_carryover_lift_factor(0.4, 2.0)
        assert 1.01 < k < 1.03

    def test_drag_factor_no_fuselage(self):
        assert wing_carryover_drag_factor(0.0, 2.0) == pytest.approx(1.0)

    def test_drag_factor_typical_rc(self):
        k = wing_carryover_drag_factor(0.11, 2.4)
        assert 1.0 < k < 1.01


class TestTotalJunctionDrag:
    def test_no_fuselages(self):
        import aerisplane as ap
        aircraft = ap.Aircraft(name="nofuse", wings=[
            ap.Wing(name="w", xsecs=[
                ap.WingXSec(xyz_le=[0, 0, 0], chord=0.25),
                ap.WingXSec(xyz_le=[0, 1, 0], chord=0.25),
            ], symmetric=True)
        ])
        cond = ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=4.0)
        assert total_junction_drag(aircraft, cond) == pytest.approx(0.0)

    def test_with_fuselage_positive(self):
        import aerisplane as ap
        fuse = ap.Fuselage(name="f", xsecs=[
            ap.FuselageXSec(x=0.0, radius=0.01),
            ap.FuselageXSec(x=0.1, radius=0.05),
            ap.FuselageXSec(x=0.5, radius=0.05),
            ap.FuselageXSec(x=0.8, radius=0.01),
        ])
        aircraft = ap.Aircraft(name="withfuse", wings=[
            ap.Wing(name="w", xsecs=[
                ap.WingXSec(xyz_le=[0.1, 0, 0], chord=0.25),
                ap.WingXSec(xyz_le=[0.1, 1, 0], chord=0.25),
            ], symmetric=True)
        ], fuselages=[fuse])
        cond = ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=4.0)
        D = total_junction_drag(aircraft, cond)
        assert D > 0.0
