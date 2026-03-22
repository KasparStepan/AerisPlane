"""Shared test fixtures for AerisPlane tests."""

import numpy as np
import pytest

import aerisplane as ap


@pytest.fixture
def naca2412():
    """A NACA 2412 airfoil."""
    return ap.Airfoil.from_naca("2412")


@pytest.fixture
def naca0012():
    """A symmetric NACA 0012 airfoil."""
    return ap.Airfoil.from_naca("0012")


@pytest.fixture
def carbon_fiber():
    """Carbon fiber material."""
    return ap.Material(
        name="Carbon Fiber",
        density=1600.0,
        E=70e9,
        yield_strength=600e6,
        poisson_ratio=0.3,
    )


@pytest.fixture
def cf_spar(carbon_fiber):
    """A 20mm OD carbon fiber tube spar at quarter-chord."""
    return ap.Spar(
        position=0.25,
        material=carbon_fiber,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


@pytest.fixture
def simple_wing(naca2412):
    """A simple tapered wing: 0.3m root chord, 0.15m tip chord, 0.8m semispan."""
    return ap.Wing(
        name="test_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=naca2412),
            ap.WingXSec(xyz_le=[0.02, 0.8, 0.05], chord=0.15, airfoil=naca2412),
        ],
        symmetric=True,
    )


@pytest.fixture
def rectangular_wing(naca0012):
    """A rectangular wing: 0.2m chord, 0.75m semispan, no taper, no sweep."""
    return ap.Wing(
        name="rect_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2, airfoil=naca0012),
            ap.WingXSec(xyz_le=[0, 0.75, 0], chord=0.2, airfoil=naca0012),
        ],
        symmetric=True,
    )


@pytest.fixture
def simple_fuselage():
    """A simple fuselage with 4 cross-sections."""
    return ap.Fuselage(
        name="test_fus",
        xsecs=[
            ap.FuselageXSec(x=0.0, radius=0.02),
            ap.FuselageXSec(x=0.15, radius=0.06),
            ap.FuselageXSec(x=0.70, radius=0.06),
            ap.FuselageXSec(x=0.95, radius=0.02),
        ],
    )


@pytest.fixture
def simple_motor():
    return ap.Motor(
        name="Test Motor",
        kv=1100,
        resistance=0.028,
        no_load_current=1.2,
        max_current=40.0,
        mass=0.152,
    )


@pytest.fixture
def simple_battery():
    return ap.Battery(
        name="Test 4S",
        capacity_ah=5.0,
        nominal_voltage=14.8,
        cell_count=4,
        c_rating=30.0,
        mass=0.42,
        internal_resistance=0.02,
    )
