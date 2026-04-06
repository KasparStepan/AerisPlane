"""Tests for Fuselage and FuselageXSec from aerisplane.core.fuselage."""

import numpy as np
import pytest

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

import aerisplane as ap
from aerisplane.core.fuselage import Fuselage, FuselageXSec


# ---------------------------------------------------------------------------
# FuselageXSec geometry
# ---------------------------------------------------------------------------

class TestFuselageXSecArea:
    """Circle area = pi * r^2."""

    def test_circle_area(self):
        xsec = FuselageXSec(x=0.0, radius=0.06)
        expected_area = np.pi * 0.06**2
        assert xsec.area() == pytest.approx(expected_area)

    def test_zero_radius(self):
        xsec = FuselageXSec(x=0.0, radius=0.0)
        assert xsec.area() == pytest.approx(0.0)


class TestFuselageXSecPerimeter:
    """Circle perimeter = 2 * pi * r."""

    def test_circle_perimeter(self):
        xsec = FuselageXSec(x=0.0, radius=0.06)
        expected_perimeter = 2.0 * np.pi * 0.06
        assert xsec.perimeter() == pytest.approx(expected_perimeter)

    def test_zero_radius(self):
        xsec = FuselageXSec(x=0.0, radius=0.0)
        assert xsec.perimeter() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fuselage (uses simple_fuselage fixture from conftest)
# ---------------------------------------------------------------------------

class TestFuselageLength:
    """Length = last x - first x."""

    def test_length(self, simple_fuselage):
        # xsecs at x = 0.0, 0.15, 0.70, 0.95
        assert simple_fuselage.length() == pytest.approx(0.95)


class TestFuselageVolume:
    """Volume by trapezoidal integration of cross-section areas."""

    def test_volume(self, simple_fuselage):
        # Manual trapezoidal integration:
        # stations:  x = [0.0, 0.15, 0.70, 0.95]
        # radii:     r = [0.02, 0.06, 0.06, 0.02]
        # areas:     A = [pi*0.02^2, pi*0.06^2, pi*0.06^2, pi*0.02^2]
        areas = np.array([
            np.pi * 0.02**2,
            np.pi * 0.06**2,
            np.pi * 0.06**2,
            np.pi * 0.02**2,
        ])
        x_stations = np.array([0.0, 0.15, 0.70, 0.95])
        expected_volume = float(_trapz(areas, x_stations))
        assert simple_fuselage.volume() == pytest.approx(expected_volume, rel=1e-10)


class TestFuselageWettedArea:
    """Wetted area by trapezoidal integration of perimeters."""

    def test_wetted_area(self, simple_fuselage):
        perimeters = np.array([
            2.0 * np.pi * 0.02,
            2.0 * np.pi * 0.06,
            2.0 * np.pi * 0.06,
            2.0 * np.pi * 0.02,
        ])
        x_stations = np.array([0.0, 0.15, 0.70, 0.95])
        expected_wetted = float(_trapz(perimeters, x_stations))
        assert simple_fuselage.wetted_area() == pytest.approx(expected_wetted, rel=1e-10)


class TestFinenessRatio:
    """Fineness ratio = length / max_diameter."""

    def test_fineness_ratio(self, simple_fuselage):
        # max radius = 0.06 -> max diameter = 0.12
        expected = 0.95 / 0.12
        assert simple_fuselage.fineness_ratio() == pytest.approx(expected, rel=1e-6)


class TestEmptyFuselage:
    """Fuselage with no cross-sections should return zeros."""

    def test_empty_length(self):
        fus = Fuselage(name="empty")
        assert fus.length() == 0.0

    def test_empty_volume(self):
        fus = Fuselage(name="empty")
        assert fus.volume() == 0.0

    def test_empty_wetted_area(self):
        fus = Fuselage(name="empty")
        assert fus.wetted_area() == 0.0
