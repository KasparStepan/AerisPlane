"""Tests for Wing and WingXSec geometry calculations."""

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.core.wing import Wing, WingXSec


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


# ---------------------------------------------------------------------------
# Task 4: Wing area types, volume, twist, chord utilities
# ---------------------------------------------------------------------------

class TestWingWettedArea:
    def test_wetted_area_greater_than_planform(self, simple_wing):
        assert simple_wing.area("wetted") > simple_wing.area("planform")

    def test_wetted_area_less_than_three_times_planform(self, simple_wing):
        assert simple_wing.area("wetted") < 3 * simple_wing.area("planform")

    def test_bad_type_raises(self, simple_wing):
        with pytest.raises(ValueError):
            simple_wing.area("volume")


class TestWingProjectedArea:
    def test_xy_area_rectangular_wing(self, rectangular_wing):
        # Rectangular wing with no sweep, no dihedral → XY == planform
        assert rectangular_wing.area("xy") == pytest.approx(
            rectangular_wing.area("planform"), rel=1e-3
        )

    def test_xz_area_zero_for_flat_wing(self, rectangular_wing):
        # Flat wing (no dihedral) → XZ projected area is zero
        assert rectangular_wing.area("xz") == pytest.approx(0.0, abs=1e-10)


class TestWingVolume:
    def test_volume_positive(self, simple_wing):
        assert simple_wing.volume() > 0.0

    def test_volume_scales_with_chord(self):
        # Double chord on a rectangular symmetric wing → volume * 4 (chord^2 effect)
        af = ap.Airfoil.from_naca("0012")
        wing1 = Wing(
            xsecs=[
                WingXSec(xyz_le=[0, 0, 0], chord=0.1, airfoil=af),
                WingXSec(xyz_le=[0, 0.5, 0], chord=0.1, airfoil=af),
            ],
            symmetric=True,
        )
        wing2 = Wing(
            xsecs=[
                WingXSec(xyz_le=[0, 0, 0], chord=0.2, airfoil=af),
                WingXSec(xyz_le=[0, 0.5, 0], chord=0.2, airfoil=af),
            ],
            symmetric=True,
        )
        assert wing2.volume() == pytest.approx(4.0 * wing1.volume(), rel=1e-3)


class TestWingMeanTwistAngle:
    def test_no_twist_returns_zero(self, rectangular_wing):
        assert rectangular_wing.mean_twist_angle() == pytest.approx(0.0)

    def test_uniform_twist_returns_that_angle(self):
        af = ap.Airfoil.from_naca("0012")
        wing = Wing(
            xsecs=[
                WingXSec(xyz_le=[0, 0, 0], chord=0.2, twist=5.0, airfoil=af),
                WingXSec(xyz_le=[0, 0.75, 0], chord=0.2, twist=5.0, airfoil=af),
            ],
            symmetric=True,
        )
        assert wing.mean_twist_angle() == pytest.approx(5.0, rel=1e-3)


class TestWingMeanGeometricChord:
    def test_rectangular_mgc_equals_chord(self, rectangular_wing):
        assert rectangular_wing.mean_geometric_chord() == pytest.approx(0.2, rel=1e-3)

    def test_mgc_equals_area_over_span(self, simple_wing):
        expected = simple_wing.area() / simple_wing.span()
        assert simple_wing.mean_geometric_chord() == pytest.approx(expected, rel=1e-9)


class TestWingXSecArea:
    def test_xsec_area_positive(self):
        af = ap.Airfoil.from_naca("0012")
        xsec = WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af)
        assert xsec.xsec_area() > 0.0

    def test_xsec_area_scales_with_chord_squared(self):
        af = ap.Airfoil.from_naca("0012")
        xsec1 = WingXSec(xyz_le=[0, 0, 0], chord=0.1, airfoil=af)
        xsec2 = WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af)
        assert xsec2.xsec_area() == pytest.approx(9.0 * xsec1.xsec_area(), rel=1e-6)


# ---------------------------------------------------------------------------
# Task 5: control surface utilities, symmetry check, translate
# ---------------------------------------------------------------------------

class TestWingControlSurfaceArea:
    @pytest.fixture
    def wing_with_aileron(self):
        af = ap.Airfoil.from_naca("0012")
        cs = ap.ControlSurface(
            name="aileron",
            span_start=0.5,
            span_end=0.9,
            chord_fraction=0.25,
            symmetric=False,
        )
        return Wing(
            xsecs=[
                WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af),
                WingXSec(xyz_le=[0.02, 0.75, 0], chord=0.15, airfoil=af),
            ],
            symmetric=True,
            control_surfaces=[cs],
        )

    def test_cs_area_is_positive(self, wing_with_aileron):
        assert wing_with_aileron.control_surface_area() > 0.0

    def test_cs_area_less_than_wing_area(self, wing_with_aileron):
        assert wing_with_aileron.control_surface_area() < wing_with_aileron.area()

    def test_filter_by_name(self, wing_with_aileron):
        assert wing_with_aileron.control_surface_area(by_name="aileron") == \
               pytest.approx(wing_with_aileron.control_surface_area(), rel=1e-9)

    def test_filter_nonexistent_returns_zero(self, wing_with_aileron):
        assert wing_with_aileron.control_surface_area(by_name="elevator") == \
               pytest.approx(0.0)

    def test_no_control_surfaces_returns_zero(self, rectangular_wing):
        assert rectangular_wing.control_surface_area() == pytest.approx(0.0)


class TestWingGetControlSurfaceNames:
    def test_empty_wing(self, rectangular_wing):
        assert rectangular_wing.get_control_surface_names() == []

    def test_returns_names(self):
        af = ap.Airfoil.from_naca("0012")
        cs1 = ap.ControlSurface(name="aileron", span_start=0.6, span_end=0.9,
                                 chord_fraction=0.25, symmetric=False)
        cs2 = ap.ControlSurface(name="flap", span_start=0.0, span_end=0.6,
                                 chord_fraction=0.30, symmetric=True)
        wing = Wing(
            xsecs=[WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af),
                   WingXSec(xyz_le=[0, 0.75, 0], chord=0.15, airfoil=af)],
            control_surfaces=[cs1, cs2],
        )
        assert set(wing.get_control_surface_names()) == {"aileron", "flap"}


class TestWingIsEntirelySymmetric:
    def test_symmetric_no_cs(self, rectangular_wing):
        assert rectangular_wing.is_entirely_symmetric() is True

    def test_asymmetric_wing(self):
        af = ap.Airfoil.from_naca("0012")
        wing = Wing(
            xsecs=[WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af),
                   WingXSec(xyz_le=[0, 0.75, 0], chord=0.15, airfoil=af)],
            symmetric=False,
        )
        assert wing.is_entirely_symmetric() is False

    def test_symmetric_with_anti_symmetric_cs(self):
        af = ap.Airfoil.from_naca("0012")
        cs = ap.ControlSurface(name="aileron", span_start=0.5, span_end=0.9,
                                chord_fraction=0.25, symmetric=False)
        wing = Wing(
            xsecs=[WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af),
                   WingXSec(xyz_le=[0, 0.75, 0], chord=0.15, airfoil=af)],
            symmetric=True,
            control_surfaces=[cs],
        )
        assert wing.is_entirely_symmetric() is False

    def test_symmetric_with_symmetric_cs(self):
        af = ap.Airfoil.from_naca("0012")
        cs = ap.ControlSurface(name="flap", span_start=0.0, span_end=1.0,
                                chord_fraction=0.30, symmetric=True)
        wing = Wing(
            xsecs=[WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af),
                   WingXSec(xyz_le=[0, 0.75, 0], chord=0.15, airfoil=af)],
            symmetric=True,
            control_surfaces=[cs],
        )
        assert wing.is_entirely_symmetric() is True


class TestWingTranslate:
    def test_translate_moves_xsec_le(self, rectangular_wing):
        moved = rectangular_wing.translate(np.array([1.0, 0.0, 0.5]))
        assert moved.xsecs[0].xyz_le[0] == pytest.approx(1.0)
        assert moved.xsecs[0].xyz_le[2] == pytest.approx(0.5)

    def test_translate_does_not_mutate_original(self, rectangular_wing):
        _ = rectangular_wing.translate(np.array([1.0, 0.0, 0.0]))
        assert rectangular_wing.xsecs[0].xyz_le[0] == pytest.approx(0.0)


class TestWingXSecTranslate:
    def test_translate_moves_le(self):
        af = ap.Airfoil.from_naca("0012")
        xsec = WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.3, airfoil=af)
        moved = xsec.translate(np.array([0.5, 0.1, -0.1]))
        assert moved.xyz_le == pytest.approx([0.5, 0.1, -0.1])
        assert xsec.xyz_le == pytest.approx([0.0, 0.0, 0.0])
