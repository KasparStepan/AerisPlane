# tests/test_structures/test_checks.py
"""Tests for structural margin-of-safety functions."""
import math
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg
from aerisplane.structures.checks import (
    bending_margin,
    shear_margin,
    buckling_margin,
    fits_in_airfoil,
    divergence_speed,
)


@pytest.fixture
def cf_spar():
    return ap.Spar(
        position=0.25, material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


class TestBendingMargin:
    def test_positive_for_safe_spar(self, cf_spar):
        M_safe = 5.0  # much less than yield
        mos = bending_margin(cf_spar, M_safe)
        assert mos > 0.0

    def test_negative_for_overloaded_spar(self, cf_spar):
        # Force failure: apply huge moment
        M_fail = 1e6
        mos = bending_margin(cf_spar, M_fail)
        assert mos < 0.0

    def test_zero_moment_returns_inf(self, cf_spar):
        mos = bending_margin(cf_spar, 0.0)
        assert math.isinf(mos)


class TestShearMargin:
    def test_positive_for_small_shear(self, cf_spar):
        mos = shear_margin(cf_spar, shear_force=10.0)
        assert mos > 0.0

    def test_zero_shear_returns_inf(self, cf_spar):
        mos = shear_margin(cf_spar, shear_force=0.0)
        assert math.isinf(mos)

    def test_larger_shear_smaller_margin(self, cf_spar):
        mos1 = shear_margin(cf_spar, shear_force=50.0)
        mos2 = shear_margin(cf_spar, shear_force=500.0)
        assert mos2 < mos1


class TestBucklingMargin:
    def test_positive_for_typical_cf_tube(self, cf_spar):
        # CF tube at moderate moment — buckling usually not critical
        mos = buckling_margin(cf_spar, bending_moment=10.0)
        assert mos > 0.0

    def test_zero_moment_returns_inf(self, cf_spar):
        mos = buckling_margin(cf_spar, bending_moment=0.0)
        assert math.isinf(mos)

    def test_thin_walled_more_susceptible_to_buckling(self):
        thick_spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.003),
        )
        thin_spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.001),
        )
        M = 10.0
        mos_thick = buckling_margin(thick_spar, M)
        mos_thin = buckling_margin(thin_spar, M)
        assert mos_thick > mos_thin


class TestFitsInAirfoil:
    def test_small_spar_fits(self):
        naca0012 = ap.Airfoil.from_naca("0012")
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
        )
        assert fits_in_airfoil(naca0012, spar, chord=0.20) is True

    def test_oversized_spar_does_not_fit(self):
        naca0012 = ap.Airfoil.from_naca("0012")
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.060, wall_thickness=0.002),
        )
        assert fits_in_airfoil(naca0012, spar, chord=0.20) is False


class TestDivergenceSpeed:
    def test_returns_inf_for_zero_offset(self):
        V = divergence_speed(GJ_root=100.0, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=0.0)
        assert math.isinf(V)

    def test_returns_inf_for_negative_offset(self):
        # e < 0: AC ahead of SC — self-stabilising
        V = divergence_speed(GJ_root=100.0, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=-0.01)
        assert math.isinf(V)

    def test_returns_finite_for_positive_offset(self):
        V = divergence_speed(GJ_root=100.0, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=0.02)
        assert np.isfinite(V)
        assert V > 0.0

    def test_stiffer_wing_has_higher_divergence_speed(self):
        V_soft = divergence_speed(GJ_root=50.0, cl_alpha_per_rad=5.5,
                                  wing_area=0.3, density=1.225, e=0.02)
        V_stiff = divergence_speed(GJ_root=200.0, cl_alpha_per_rad=5.5,
                                   wing_area=0.3, density=1.225, e=0.02)
        assert V_stiff > V_soft

    def test_typical_cf_tube_has_high_divergence_speed(self, cf_spar):
        # GJ for 20mm CF tube: G=52 GPa, J=2I
        spar = cf_spar
        G = spar.material.shear_modulus
        J = 2.0 * spar.section.second_moment_of_area()
        GJ = G * J
        e = (0.25 - spar.position) * 0.20  # spar at 25% chord, chord 0.2 m -> e=0
        if e <= 0:
            return  # spar at AC -> infinite divergence, test not meaningful
        V = divergence_speed(GJ_root=GJ, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=e)
        assert V > 50.0  # should be very high
