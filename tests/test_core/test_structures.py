"""Tests for aerisplane.core.structures — Material, TubeSection, Spar, Skin."""

import numpy as np
import pytest

import aerisplane as ap


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


class TestMaterial:
    """Tests for Material dataclass and shear-modulus computation."""

    def test_shear_modulus_auto_computed(self, carbon_fiber: ap.Material) -> None:
        """G should equal E / (2 * (1 + nu)) when not explicitly provided."""
        expected_G = carbon_fiber.E / (2.0 * (1.0 + carbon_fiber.poisson_ratio))
        assert carbon_fiber.shear_modulus == pytest.approx(expected_G, rel=1e-6)

    def test_shear_modulus_explicit_override(self) -> None:
        """An explicit shear_modulus should override the auto-computation."""
        explicit_G = 25e9
        mat = ap.Material(
            name="Custom",
            density=1500.0,
            E=70e9,
            yield_strength=500e6,
            poisson_ratio=0.1,
            shear_modulus=explicit_G,
        )
        assert mat.shear_modulus == pytest.approx(explicit_G, rel=1e-6)


# ---------------------------------------------------------------------------
# TubeSection
# ---------------------------------------------------------------------------


class TestTubeSection:
    """Tests for TubeSection geometry calculations."""

    @pytest.fixture()
    def tube(self) -> ap.TubeSection:
        return ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002)

    def test_inner_diameter(self, tube: ap.TubeSection) -> None:
        """ID = OD - 2 * wall."""
        assert tube.inner_diameter() == pytest.approx(0.016, rel=1e-6)

    def test_area(self, tube: ap.TubeSection) -> None:
        """A = pi/4 * (OD^2 - ID^2)."""
        expected_area = np.pi / 4.0 * (0.020**2 - 0.016**2)
        assert tube.area() == pytest.approx(expected_area, rel=1e-6)

    def test_second_moment_of_area(self, tube: ap.TubeSection) -> None:
        """I = pi/64 * (OD^4 - ID^4)."""
        expected_I = np.pi / 64.0 * (0.020**4 - 0.016**4)
        assert tube.second_moment_of_area() == pytest.approx(expected_I, rel=1e-6)

    def test_section_modulus(self, tube: ap.TubeSection) -> None:
        """S = I / (OD / 2)."""
        expected_I = np.pi / 64.0 * (0.020**4 - 0.016**4)
        expected_S = expected_I / (0.020 / 2.0)
        assert tube.section_modulus() == pytest.approx(expected_S, rel=1e-6)


# ---------------------------------------------------------------------------
# Spar
# ---------------------------------------------------------------------------


class TestSpar:
    """Tests for Spar derived quantities (mass, stress, margin)."""

    def test_mass_per_length(self, cf_spar: ap.Spar) -> None:
        """mass_per_length = section.area * material.density."""
        expected = cf_spar.section.area() * cf_spar.material.density
        assert cf_spar.mass_per_length() == pytest.approx(expected, rel=1e-6)

    def test_max_bending_stress(self, cf_spar: ap.Spar) -> None:
        """sigma = M / S for a known bending moment."""
        moment = 10.0  # N*m
        expected_stress = moment / cf_spar.section.section_modulus()
        assert cf_spar.max_bending_stress(moment) == pytest.approx(
            expected_stress, rel=1e-6
        )

    def test_margin_of_safety_positive(self, cf_spar: ap.Spar) -> None:
        """Margin should be positive when stress is below yield."""
        small_moment = 0.1  # N*m — well below failure
        margin = cf_spar.margin_of_safety(small_moment)
        assert margin > 0.0

    def test_margin_of_safety_negative(self, cf_spar: ap.Spar) -> None:
        """Margin should be negative when stress exceeds yield."""
        # Pick a moment large enough to exceed yield strength.
        # stress = M / S  =>  M = stress * S
        S = cf_spar.section.section_modulus()
        moment_at_yield = cf_spar.material.yield_strength * S
        excessive_moment = moment_at_yield * 2.0  # twice the yield moment
        margin = cf_spar.margin_of_safety(excessive_moment)
        assert margin < 0.0


# ---------------------------------------------------------------------------
# Skin
# ---------------------------------------------------------------------------


class TestSkin:
    """Tests for Skin areal mass density."""

    def test_mass_per_area(self, carbon_fiber: ap.Material) -> None:
        """mass_per_area = thickness * density."""
        thickness = 0.0005  # 0.5 mm
        skin = ap.Skin(material=carbon_fiber, thickness=thickness)
        expected = thickness * carbon_fiber.density
        assert skin.mass_per_area() == pytest.approx(expected, rel=1e-6)
