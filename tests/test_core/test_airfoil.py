"""Tests for Airfoil class — NACA generation, custom coordinates, and geometry queries."""

import numpy as np
import pytest

import aerisplane as ap


class TestFromNaca2412:
    """Tests for Airfoil.from_naca('2412')."""

    @pytest.fixture()
    def airfoil(self):
        return ap.Airfoil.from_naca("2412")

    def test_name(self, airfoil):
        assert airfoil.name == "naca2412"

    def test_coordinates_shape(self, airfoil):
        assert airfoil.coordinates.ndim == 2
        assert airfoil.coordinates.shape[1] == 2

    def test_coordinates_start_at_trailing_edge(self, airfoil):
        assert airfoil.coordinates[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_coordinates_end_at_trailing_edge(self, airfoil):
        assert airfoil.coordinates[-1, 0] == pytest.approx(1.0, abs=0.01)


class TestFromNaca0012:
    """Tests for symmetric NACA 0012 airfoil."""

    @pytest.fixture()
    def airfoil(self):
        return ap.Airfoil.from_naca("0012")

    def test_max_camber_approximately_zero(self, airfoil):
        assert airfoil.max_camber() == pytest.approx(0.0, abs=0.01)

    def test_thickness_approximately_012(self, airfoil):
        assert airfoil.thickness() == pytest.approx(0.12, abs=0.01)


class TestFromNaca2412Geometry:
    """Geometry properties for cambered NACA 2412."""

    @pytest.fixture()
    def airfoil(self):
        return ap.Airfoil.from_naca("2412")

    def test_thickness_approximately_012(self, airfoil):
        assert airfoil.thickness() == pytest.approx(0.12, abs=0.01)

    def test_max_camber_positive(self, airfoil):
        assert airfoil.max_camber() > 0


class TestAutomaticNacaGeneration:
    """Airfoil('naca0012') should auto-generate coordinates."""

    def test_auto_generates_coordinates(self):
        airfoil = ap.Airfoil("naca0012")
        assert airfoil.coordinates is not None
        assert airfoil.coordinates.ndim == 2
        assert airfoil.coordinates.shape[1] == 2


class TestCustomCoordinates:
    """Airfoil with user-supplied coordinates."""

    def test_custom_coordinates_accepted(self):
        coords = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        airfoil = ap.Airfoil("custom", coordinates=coords)
        assert airfoil.name == "custom"
        np.testing.assert_array_equal(airfoil.coordinates, coords)


class TestNoCoordinates:
    """Airfoil with unknown name and no coordinates."""

    def test_coordinates_are_none(self):
        airfoil = ap.Airfoil("nonexistent_foil_xyz")
        assert airfoil.coordinates is None

    def test_thickness_returns_zero(self):
        airfoil = ap.Airfoil("nonexistent_foil_xyz")
        assert airfoil.thickness() == 0.0


class TestAirfoilNondimPerimeter:
    def test_flat_plate_perimeter(self):
        # A flat plate (zero thickness) should have perimeter ≈ 2.0
        # (upper surface 1.0 + lower surface 1.0)
        af = ap.Airfoil(
            name="flat_plate",
            coordinates=np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.0],
                                   [0.5, 0.0], [1.0, 0.0]]),
        )
        assert af.nondim_perimeter() == pytest.approx(2.0, rel=1e-3)

    def test_naca0012_perimeter_above_two(self):
        # NACA 0012 is thicker than a flat plate, so perimeter > 2.0
        af = ap.Airfoil.from_naca("0012")
        assert af.nondim_perimeter() > 2.0
        assert af.nondim_perimeter() < 2.2  # sanity upper bound

    def test_no_coordinates_returns_two(self):
        af = ap.Airfoil(name="unknown_xyz_abc")
        assert af.nondim_perimeter() == pytest.approx(2.0)


class TestAirfoilNondimArea:
    def test_symmetric_airfoil_has_positive_area(self):
        af = ap.Airfoil.from_naca("0012")
        assert af.nondim_area() > 0.0

    def test_naca0012_area_approximate(self):
        # NACA 0012: thickness 12%, nondim area roughly 0.06-0.12
        af = ap.Airfoil.from_naca("0012")
        assert 0.06 < af.nondim_area() < 0.12

    def test_no_coordinates_returns_zero(self):
        af = ap.Airfoil(name="unknown_xyz_abc")
        assert af.nondim_area() == pytest.approx(0.0)
