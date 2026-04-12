"""Tests for Fuselage and FuselageXSec from aerisplane.core.fuselage."""

import numpy as np
import pytest

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

import aerisplane as ap
from aerisplane.core.fuselage import Fuselage, FuselageXSec


# ---------------------------------------------------------------------------
# FuselageXSec geometry (original backward-compat tests)
# ---------------------------------------------------------------------------

class TestFuselageXSecArea:
    """Circle area = pi * r^2."""

    def test_circle_area(self):
        xsec = FuselageXSec(x=0.0, radius=0.06)
        expected_area = np.pi * 0.06**2
        assert xsec.area() == pytest.approx(expected_area, rel=1e-2)

    def test_zero_radius(self):
        xsec = FuselageXSec(x=0.0, radius=0.0)
        assert xsec.area() == pytest.approx(0.0)


class TestFuselageXSecPerimeter:
    """Circle perimeter = 2 * pi * r."""

    def test_circle_perimeter(self):
        xsec = FuselageXSec(x=0.0, radius=0.06)
        expected_perimeter = 2.0 * np.pi * 0.06
        assert xsec.perimeter() == pytest.approx(expected_perimeter, rel=1e-2)

    def test_zero_radius(self):
        xsec = FuselageXSec(x=0.0, radius=0.0)
        assert xsec.perimeter() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FuselageXSec superellipse (new tests)
# ---------------------------------------------------------------------------

class TestFuselageXSecSuperellipseArea:
    def test_circle_area(self):
        # shape=2 + equal width/height → circle area = π r²
        xsec = FuselageXSec(x=0.0, width=0.2, height=0.2, shape=2.0)
        assert xsec.area() == pytest.approx(np.pi * 0.1 ** 2, rel=1e-2)

    def test_diamond_area(self):
        # shape=1 → diamond, area = width * height / 2
        xsec = FuselageXSec(x=0.0, width=0.2, height=0.2, shape=1.0)
        assert xsec.area() == pytest.approx(0.5 * 0.2 * 0.2, rel=1e-2)

    def test_square_area(self):
        # shape=1000 → nearly square, area ≈ width * height
        xsec = FuselageXSec(x=0.0, width=0.2, height=0.2, shape=1000.0)
        assert xsec.area() == pytest.approx(0.2 * 0.2, rel=1e-2)

    def test_backward_compat_radius(self):
        # Old API: FuselageXSec(x, radius=r) must still work
        xsec = FuselageXSec(x=0.0, radius=0.06)
        assert xsec.width == pytest.approx(0.12)
        assert xsec.height == pytest.approx(0.12)
        assert xsec.area() == pytest.approx(np.pi * 0.06 ** 2, rel=1e-2)


class TestFuselageXSecSuperellipsePerimeter:
    def test_circle_perimeter(self):
        xsec = FuselageXSec(x=0.0, radius=0.1)
        assert xsec.perimeter() == pytest.approx(2 * np.pi * 0.1, rel=1e-2)

    def test_zero_width(self):
        xsec = FuselageXSec(x=0.0, width=0.0, height=0.2, shape=2.0)
        assert xsec.perimeter() == pytest.approx(0.4)

    def test_zero_height(self):
        xsec = FuselageXSec(x=0.0, width=0.2, height=0.0, shape=2.0)
        assert xsec.perimeter() == pytest.approx(0.4)


class TestFuselageXSecEquivalentRadius:
    def test_circle_area_preserve(self):
        xsec = FuselageXSec(x=0.0, radius=0.05)
        assert xsec.equivalent_radius(preserve="area") == pytest.approx(0.05, rel=1e-3)

    def test_circle_perimeter_preserve(self):
        xsec = FuselageXSec(x=0.0, radius=0.05)
        assert xsec.equivalent_radius(preserve="perimeter") == pytest.approx(0.05, rel=1e-2)

    def test_bad_preserve_raises(self):
        xsec = FuselageXSec(x=0.0, radius=0.05)
        with pytest.raises(ValueError):
            xsec.equivalent_radius(preserve="volume")


class TestFuselageXSecGet3DCoordinates:
    def test_output_shape(self):
        xsec = FuselageXSec(x=0.0, radius=0.1)
        theta = np.linspace(0, 2 * np.pi, 37)[:-1]  # 36 points
        center = np.array([0.5, 0.0, 0.0])
        pts = xsec.get_3D_coordinates(theta, center)
        assert pts.shape == (36, 3)

    def test_x_is_constant(self):
        xsec = FuselageXSec(x=0.0, radius=0.1)
        theta = np.linspace(0, 2 * np.pi, 37)[:-1]
        center = np.array([1.2, 0.0, 0.0])
        pts = xsec.get_3D_coordinates(theta, center)
        assert np.all(pts[:, 0] == pytest.approx(1.2))

    def test_circle_all_points_on_radius(self):
        xsec = FuselageXSec(x=0.0, radius=0.1)
        theta = np.linspace(0, 2 * np.pi, 361)[:-1]
        center = np.array([0.0, 0.0, 0.0])
        pts = xsec.get_3D_coordinates(theta, center)
        r = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
        assert np.allclose(r, 0.1, atol=1e-3)


class TestFuselageXSecTranslate:
    def test_translate_shifts_x(self):
        xsec = FuselageXSec(x=0.5, radius=0.1)
        moved = xsec.translate(0.3)
        assert moved.x == pytest.approx(0.8)
        assert xsec.x == pytest.approx(0.5)  # original unchanged


# ---------------------------------------------------------------------------
# Fuselage new methods (Task 3)
# ---------------------------------------------------------------------------

class TestFuselagePrismatoidVolume:
    def test_cylinder_volume(self):
        # Cylinder: constant radius 0.1m, length 1.0m → V = π * 0.1^2 * 1.0
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=1.0, radius=0.1),
            ]
        )
        expected = np.pi * 0.1 ** 2 * 1.0
        assert fuse.volume() == pytest.approx(expected, rel=1e-3)

    def test_cone_volume(self):
        # Cone: radius tapers 0.1 → 0, length 1.0m → V = (1/3) π * 0.1^2 * 1.0
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=1.0, width=0.0, height=0.0, shape=2.0),
            ]
        )
        expected = (1.0 / 3.0) * np.pi * 0.1 ** 2 * 1.0
        assert fuse.volume() == pytest.approx(expected, rel=1e-2)


class TestFuselageAreaProjected:
    def test_cylinder_xy_projection(self):
        # Constant diameter 0.2m, length 1.0m → top-down projected area = 0.2 * 1.0
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, width=0.2, height=0.1, shape=2.0),
                FuselageXSec(x=1.0, width=0.2, height=0.1, shape=2.0),
            ]
        )
        assert fuse.area_projected("XY") == pytest.approx(0.2 * 1.0, rel=1e-6)

    def test_cylinder_xz_projection(self):
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, width=0.2, height=0.1, shape=2.0),
                FuselageXSec(x=1.0, width=0.2, height=0.1, shape=2.0),
            ]
        )
        assert fuse.area_projected("XZ") == pytest.approx(0.1 * 1.0, rel=1e-6)

    def test_bad_type_raises(self):
        fuse = Fuselage(xsecs=[FuselageXSec(x=0.0, radius=0.1)])
        with pytest.raises(ValueError):
            fuse.area_projected("YZ")


class TestFuselageXCentroidProjected:
    def test_uniform_cylinder_centroid(self):
        # Uniform cylinder: centroid at mid-length = 0.5m
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=1.0, radius=0.1),
            ]
        )
        assert fuse.x_centroid_projected("XY") == pytest.approx(0.5, rel=1e-3)


class TestFuselageMeshBody:
    def test_returns_correct_shapes(self):
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=1.0, radius=0.1),
            ]
        )
        T = 12  # tangential_resolution
        pts, faces = fuse.mesh_body(tangential_resolution=T)
        n_stations = 2
        assert pts.shape == (n_stations * T, 3)
        # Faces: (n_stations - 1) * T quads
        assert faces.shape == ((n_stations - 1) * T, 4)

    def test_circle_points_on_correct_radius(self):
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=0.5, radius=0.1),
            ]
        )
        pts, _ = fuse.mesh_body(tangential_resolution=360)
        r = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
        assert np.allclose(r, 0.1, atol=1e-3)


class TestFuselageMeshLine:
    def test_centerline_x_coordinates(self):
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=1.0, radius=0.2),
            ]
        )
        pts = fuse.mesh_line(y_nondim=0.0, z_nondim=0.0)
        assert len(pts) == 2
        assert pts[0][0] == pytest.approx(0.0)
        assert pts[1][0] == pytest.approx(1.0)


class TestFuselageSubdivideSections:
    def test_ratio_2_doubles_xsecs(self):
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=1.0, radius=0.2),
            ]
        )
        sub = fuse.subdivide_sections(ratio=2)
        # 1 section → 2 sub-sections → 3 xsecs
        assert len(sub.xsecs) == 3

    def test_subdivided_length_preserved(self):
        fuse = Fuselage(
            xsecs=[
                FuselageXSec(x=0.0, radius=0.1),
                FuselageXSec(x=1.0, radius=0.2),
            ]
        )
        sub = fuse.subdivide_sections(ratio=4)
        assert sub.length() == pytest.approx(fuse.length(), rel=1e-9)

    def test_bad_ratio_raises(self):
        fuse = Fuselage(xsecs=[FuselageXSec(x=0.0, radius=0.1)])
        with pytest.raises(ValueError):
            fuse.subdivide_sections(ratio=1)


class TestFuselageTranslate:
    def test_translate_shifts_x_le(self):
        fuse = Fuselage(
            xsecs=[FuselageXSec(x=0.0, radius=0.1)],
            x_le=0.5,
        )
        moved = fuse.translate(np.array([1.0, 0.0, 0.0]))
        assert moved.x_le == pytest.approx(1.5)
        assert fuse.x_le == pytest.approx(0.5)  # original unchanged


# ---------------------------------------------------------------------------
# Fuselage (uses simple_fuselage fixture from conftest)
# ---------------------------------------------------------------------------

class TestFuselageLength:
    """Length = last x - first x."""

    def test_length(self, simple_fuselage):
        # xsecs at x = 0.0, 0.15, 0.70, 0.95
        assert simple_fuselage.length() == pytest.approx(0.95)


class TestFuselageVolume:
    """Volume by Prismatoid formula per section."""

    def test_volume(self, simple_fuselage):
        # Manual Prismatoid: V = sep/3 * (A_a + A_b + sqrt(A_a * A_b))
        # stations:  x = [0.0, 0.15, 0.70, 0.95]
        # radii:     r = [0.02, 0.06, 0.06, 0.02]
        sections = [
            (0.0, 0.15, 0.02, 0.06),
            (0.15, 0.70, 0.06, 0.06),
            (0.70, 0.95, 0.06, 0.02),
        ]
        expected_volume = sum(
            (xb - xa) / 3.0 * (
                np.pi * ra**2 + np.pi * rb**2 + np.sqrt(np.pi * ra**2 * np.pi * rb**2)
            )
            for xa, xb, ra, rb in sections
        )
        assert simple_fuselage.volume() == pytest.approx(expected_volume, rel=1e-6)


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
        assert simple_fuselage.wetted_area() == pytest.approx(expected_wetted, rel=5e-3)


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
