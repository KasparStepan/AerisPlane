# tests/test_structures/test_section.py
"""Tests for airfoil-geometry-based section properties."""
import numpy as np
import pytest
import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg
from aerisplane.structures.section import (
    airfoil_spar_height,
    skin_second_moment_of_area,
    effective_EI,
    spar_fits_in_airfoil,
)


class TestAirfoilSparHeight:
    def test_naca0012_at_quarterchord(self, naca0012):
        # NACA 0012: t/c = 12%, height at 25% chord ≈ 11-12% of chord
        h = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        assert 0.018 < h < 0.026  # 9-13% of 0.20m

    def test_scales_with_chord(self, naca0012):
        h1 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        h2 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.40)
        assert h2 == pytest.approx(2.0 * h1, rel=0.01)

    def test_height_decreases_toward_te(self, naca0012):
        h_25 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        h_50 = airfoil_spar_height(naca0012, spar_position=0.50, chord=0.20)
        assert h_25 > h_50  # thicker near leading quarter

    def test_thicker_airfoil_gives_larger_height(self, naca0012):
        naca0018 = ap.Airfoil.from_naca("0018")
        h_12 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        h_18 = airfoil_spar_height(naca0018, spar_position=0.25, chord=0.20)
        assert h_18 > h_12


class TestSkinSecondMomentOfArea:
    def test_positive_for_valid_airfoil(self, naca0012, petg_skin):
        I = skin_second_moment_of_area(naca0012, chord=0.20,
                                       skin_thickness=petg_skin.thickness)
        assert I > 0.0

    def test_zero_for_zero_thickness(self, naca0012):
        I = skin_second_moment_of_area(naca0012, chord=0.20, skin_thickness=0.0)
        assert I == pytest.approx(0.0)

    def test_scales_with_chord_to_third(self, naca0012, petg_skin):
        # I ~ chord³: y ~ chord, ds ~ chord → y²·ds ~ chord³
        I1 = skin_second_moment_of_area(naca0012, chord=0.10,
                                        skin_thickness=petg_skin.thickness)
        I2 = skin_second_moment_of_area(naca0012, chord=0.20,
                                        skin_thickness=petg_skin.thickness)
        assert I2 == pytest.approx(8.0 * I1, rel=0.02)

    def test_scales_linearly_with_thickness(self, naca0012):
        I1 = skin_second_moment_of_area(naca0012, chord=0.20, skin_thickness=0.001)
        I2 = skin_second_moment_of_area(naca0012, chord=0.20, skin_thickness=0.002)
        assert I2 == pytest.approx(2.0 * I1, rel=1e-6)


class TestEffectiveEI:
    def test_spar_only_equals_E_times_I(self, naca0012, cf_spar):
        EI = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=None)
        expected = cf_spar.material.E * cf_spar.section.second_moment_of_area()
        assert EI == pytest.approx(expected, rel=1e-9)

    def test_with_skin_larger_than_spar_only(self, naca0012, cf_spar, petg_skin):
        EI_bare = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=None)
        EI_skin = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=petg_skin)
        assert EI_skin > EI_bare

    def test_stiffer_skin_gives_higher_EI(self, naca0012, cf_spar):
        cf_skin = ap.Skin(material=carbon_fiber_tube, thickness=0.5e-3)
        petg_skin_local = ap.Skin(material=petg, thickness=0.5e-3)
        EI_petg = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=petg_skin_local)
        EI_cf = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=cf_skin)
        assert EI_cf > EI_petg


class TestSparFitsInAirfoil:
    def test_small_spar_fits(self, naca0012):
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
        )
        assert spar_fits_in_airfoil(naca0012, spar, chord=0.20) is True

    def test_oversized_spar_does_not_fit(self, naca0012):
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.050, wall_thickness=0.002),
        )
        assert spar_fits_in_airfoil(naca0012, spar, chord=0.20) is False

    def test_exactly_at_limit_fits(self, naca0012):
        h = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=h * 0.99, wall_thickness=0.001),
        )
        assert spar_fits_in_airfoil(naca0012, spar, chord=0.20) is True
