"""Tests for Wing and WingXSec geometry calculations."""

import numpy as np
import pytest

import aerisplane as ap


# ---------------------------------------------------------------------------
# 1. Rectangular wing (uses rectangular_wing fixture from conftest.py)
# ---------------------------------------------------------------------------
class TestRectangularWing:
    """Rectangular wing: chord=0.2m, semispan=0.75m, symmetric."""

    def test_span(self, rectangular_wing):
        assert rectangular_wing.span() == pytest.approx(1.5, rel=1e-3)

    def test_area(self, rectangular_wing):
        assert rectangular_wing.area() == pytest.approx(0.3, rel=1e-3)

    def test_aspect_ratio(self, rectangular_wing):
        assert rectangular_wing.aspect_ratio() == pytest.approx(7.5, rel=1e-3)

    def test_mac(self, rectangular_wing):
        assert rectangular_wing.mean_aerodynamic_chord() == pytest.approx(0.2, rel=1e-3)

    def test_taper_ratio(self, rectangular_wing):
        assert rectangular_wing.taper_ratio() == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Tapered wing (uses simple_wing fixture from conftest.py)
# ---------------------------------------------------------------------------
class TestTaperedWing:
    """Tapered wing: root=0.3m, tip=0.15m, semispan=0.8m, symmetric."""

    def test_span(self, simple_wing):
        assert simple_wing.span() == pytest.approx(1.6, rel=1e-3)

    def test_area(self, simple_wing):
        # Trapezoid: (0.3 + 0.15) / 2 * 0.8 = 0.18 per side, 0.36 total
        assert simple_wing.area() == pytest.approx(0.36, rel=1e-3)

    def test_taper_ratio(self, simple_wing):
        assert simple_wing.taper_ratio() == pytest.approx(0.5, rel=1e-3)


# ---------------------------------------------------------------------------
# 3. Non-symmetric wing (e.g., vertical tail)
# ---------------------------------------------------------------------------
class TestNonSymmetricWing:
    """Non-symmetric wing: span equals semispan, area is single side only."""

    @pytest.fixture
    def vtail(self):
        """Vertical tail: 0.15m chord, 0.3m semispan, non-symmetric."""
        return ap.Wing(
            name="vtail",
            xsecs=[
                ap.WingXSec(xyz_le=[0.6, 0, 0], chord=0.15),
                ap.WingXSec(xyz_le=[0.6, 0.3, 0], chord=0.15),
            ],
            symmetric=False,
        )

    def test_span_equals_semispan(self, vtail):
        assert vtail.span() == pytest.approx(vtail.semispan(), rel=1e-3)

    def test_area_single_side(self, vtail):
        # Single panel: 0.15 * 0.3 = 0.045 m^2 (no doubling)
        assert vtail.area() == pytest.approx(0.045, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. Sweep angle
# ---------------------------------------------------------------------------
class TestSweep:
    """Wing with swept tip should give positive leading-edge sweep."""

    @pytest.fixture
    def swept_wing(self):
        return ap.Wing(
            name="swept",
            xsecs=[
                ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2),
                ap.WingXSec(xyz_le=[0.1, 0.5, 0], chord=0.2),
            ],
            symmetric=True,
        )

    def test_positive_sweep_le(self, swept_wing):
        assert swept_wing.sweep_le() > 0.0


# ---------------------------------------------------------------------------
# 5. Dihedral
# ---------------------------------------------------------------------------
class TestDihedral:
    """Wing with z-offset tip should give positive dihedral."""

    @pytest.fixture
    def dihedral_wing(self):
        return ap.Wing(
            name="dihedral",
            xsecs=[
                ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2),
                ap.WingXSec(xyz_le=[0, 0.5, 0.05], chord=0.2),
            ],
            symmetric=True,
        )

    def test_positive_dihedral(self, dihedral_wing):
        assert dihedral_wing.dihedral() > 0.0


# ---------------------------------------------------------------------------
# 6. WingXSec defaults
# ---------------------------------------------------------------------------
class TestWingXSecDefaults:
    """WingXSec with no airfoil should default to naca0012."""

    def test_default_airfoil_is_naca0012(self):
        xsec = ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2)
        assert xsec.airfoil is not None
        assert xsec.airfoil.name == "naca0012"


# ---------------------------------------------------------------------------
# 7. MAC leading-edge position
# ---------------------------------------------------------------------------
class TestMACLePosition:
    """MAC LE should return a 3-element array."""

    def test_mac_le_is_3_element_array(self, rectangular_wing):
        mac_le = rectangular_wing.mean_aerodynamic_chord_le()
        assert isinstance(mac_le, np.ndarray)
        assert mac_le.shape == (3,)


# ---------------------------------------------------------------------------
# 8. Aerodynamic center
# ---------------------------------------------------------------------------
class TestAerodynamicCenter:
    """AC should be MAC_LE + [0.25 * MAC, 0, 0]."""

    def test_ac_equals_mac_le_plus_quarter_mac(self, rectangular_wing):
        mac_le = rectangular_wing.mean_aerodynamic_chord_le()
        mac = rectangular_wing.mean_aerodynamic_chord()
        ac = rectangular_wing.aerodynamic_center()

        expected = mac_le + np.array([0.25 * mac, 0.0, 0.0])

        np.testing.assert_allclose(ac, expected, rtol=1e-3)
