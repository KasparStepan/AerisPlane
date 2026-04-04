# tests/test_structures/test_beam.py
"""Tests for the Euler-Bernoulli wing beam solver."""
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg
from aerisplane.structures.beam import BeamResult, WingBeam


class TestBeamResult:
    def test_tip_deflection_property(self):
        y = np.array([0.0, 0.5, 1.0])
        br = BeamResult(
            y=y,
            V=np.array([10.0, 5.0, 0.0]),
            M=np.array([5.0, 2.5, 0.0]),
            theta=np.array([0.0, 0.001, 0.003]),
            delta=np.array([0.0, 0.0005, 0.002]),
            EI=np.ones(3) * 1000.0,
            GJ=np.ones(3) * 500.0,
        )
        assert br.tip_deflection == pytest.approx(0.002)
        assert br.root_bending_moment == pytest.approx(5.0)
        assert br.root_shear_force == pytest.approx(10.0)


class TestWingBeamStations:
    def test_builds_correct_number_of_stations(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=30)
        assert len(wb.y) == 30

    def test_y_goes_from_root_to_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=20)
        assert wb.y[0] == pytest.approx(0.0)
        assert wb.y[-1] == pytest.approx(0.75)

    def test_EI_positive_everywhere(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=20)
        assert np.all(wb.EI > 0)

    def test_GJ_positive_everywhere(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=20)
        assert np.all(wb.GJ > 0)


class TestWingBeamSolve:
    def test_shear_zero_at_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.V[-1] == pytest.approx(0.0, abs=1e-10)

    def test_moment_zero_at_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.M[-1] == pytest.approx(0.0, abs=1e-10)

    def test_deflection_zero_at_root(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.delta[0] == pytest.approx(0.0, abs=1e-10)

    def test_deflection_positive_at_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.tip_deflection > 0.0

    def test_shear_maximum_at_root(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=1.0, inertia_relief=False)
        assert result.V[0] == np.max(result.V)

    def test_moment_maximum_at_root(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=1.0, inertia_relief=False)
        assert result.M[0] == np.max(result.M)

    def test_higher_load_gives_larger_deflection(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        r1 = wb.solve(total_lift=10.0, load_factor=1.0, inertia_relief=False)
        r2 = wb.solve(total_lift=20.0, load_factor=1.0, inertia_relief=False)
        assert r2.tip_deflection > r1.tip_deflection

    def test_inertia_relief_reduces_deflection(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        r_no = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        r_ir = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=True)
        assert r_no.tip_deflection >= r_ir.tip_deflection

    def test_uniform_beam_matches_analytic(self):
        """Root bending moment of elliptic distribution on b=1m, L=40N."""
        cf_spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
        )
        naca0012 = ap.Airfoil.from_naca("0012")
        wing = ap.Wing(
            name="uniform",
            xsecs=[
                ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.2,
                            airfoil=naca0012, spar=cf_spar),
                ap.WingXSec(xyz_le=[0.0, 1.0, 0.0], chord=0.2,
                            airfoil=naca0012, spar=cf_spar),
            ],
            symmetric=False,
        )
        L_total = 40.0
        wb = WingBeam(wing, n_stations=200)
        result = wb.solve(total_lift=L_total, load_factor=1.0, inertia_relief=False)

        b = 1.0
        # M_root = integral_0^b y * q(y) dy = q_0 * b^2 / 3
        # where q_0 = 4*(L_total/2)/(pi*b) (solver divides total_lift by 2 for semi-span)
        M_root_analytic = 2.0 * L_total * b / (3.0 * np.pi)

        assert result.root_bending_moment == pytest.approx(M_root_analytic, rel=0.02)
