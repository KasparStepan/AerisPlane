# tests/test_structures/conftest.py
"""Shared fixtures for structures module tests."""
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg


@pytest.fixture
def cf_spar():
    return ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


@pytest.fixture
def petg_skin():
    return ap.Skin(material=petg, thickness=0.8e-3)


@pytest.fixture
def naca2412():
    return ap.Airfoil.from_naca("2412")


@pytest.fixture
def naca0012():
    return ap.Airfoil.from_naca("0012")


@pytest.fixture
def rect_wing(cf_spar, petg_skin, naca2412):
    """Rectangular wing: 0.2 m chord, 0.75 m semispan, uniform NACA 2412."""
    return ap.Wing(
        name="rect_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.1, 0.0, 0.0], chord=0.2,
                        airfoil=naca2412, spar=cf_spar, skin=petg_skin),
            ap.WingXSec(xyz_le=[0.1, 0.75, 0.0], chord=0.2,
                        airfoil=naca2412, spar=cf_spar, skin=petg_skin),
        ],
        symmetric=True,
    )


@pytest.fixture
def simple_aircraft(rect_wing):
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.75, 0.0, 0.0], chord=0.12),
            ap.WingXSec(xyz_le=[0.75, 0.30, 0.0], chord=0.08),
        ],
        symmetric=True,
    )
    fuse = ap.Fuselage(
        name="fuse",
        xsecs=[
            ap.FuselageXSec(x=0.0, radius=0.02),
            ap.FuselageXSec(x=0.8, radius=0.02),
        ],
    )
    motor = ap.Motor(name="m", kv=1100, resistance=0.028,
                     no_load_current=1.2, max_current=40.0, mass=0.12)
    prop = ap.Propeller(diameter=0.254, pitch=0.127, mass=0.03)
    bat = ap.Battery(name="b", capacity_ah=2.2, nominal_voltage=14.8,
                     cell_count=4, c_rating=30.0, mass=0.2)
    esc = ap.ESC(name="e", max_current=40.0, mass=0.03)
    propulsion = ap.PropulsionSystem(
        motor=motor, propeller=prop, battery=bat, esc=esc,
        position=np.array([0., 0., 0.]),
    )
    return ap.Aircraft(
        name="TestPlane",
        wings=[rect_wing, htail],
        fuselages=[fuse],
        propulsion=propulsion,
        payload=ap.Payload(mass=0.1, cg=np.array([0.25, 0., 0.]), name="p"),
    )


@pytest.fixture
def cruise_condition():
    return ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=3.0, beta=0.0)
