"""Fixtures for weights module tests."""

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg


@pytest.fixture
def cf_material():
    """Carbon fiber tube material."""
    return carbon_fiber_tube


@pytest.fixture
def petg_material():
    """PETG 3D-print material."""
    return petg


@pytest.fixture
def cf_spar(cf_material):
    """20mm OD, 2mm wall carbon fiber tube spar at quarter-chord."""
    return ap.Spar(
        position=0.25,
        material=cf_material,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


@pytest.fixture
def petg_skin(petg_material):
    """0.8mm PETG skin."""
    return ap.Skin(material=petg_material, thickness=0.8e-3)


@pytest.fixture
def wing_with_structure(cf_spar, petg_skin):
    """Rectangular wing with spar and skin: 0.2m chord, 0.75m semispan."""
    return ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(
                xyz_le=[0.1, 0.0, 0.0], chord=0.2,
                spar=cf_spar, skin=petg_skin,
            ),
            ap.WingXSec(
                xyz_le=[0.1, 0.75, 0.0], chord=0.2,
                spar=cf_spar, skin=petg_skin,
            ),
        ],
        symmetric=True,
    )


@pytest.fixture
def htail_with_structure(cf_spar, petg_skin):
    """Small horizontal tail with spar and skin."""
    small_spar = ap.Spar(
        position=0.25,
        material=cf_spar.material,
        section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
    )
    thin_skin = ap.Skin(material=petg_skin.material, thickness=0.5e-3)
    return ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(
                xyz_le=[0.75, 0.0, 0.0], chord=0.12,
                spar=small_spar, skin=thin_skin,
            ),
            ap.WingXSec(
                xyz_le=[0.75, 0.30, 0.0], chord=0.08,
                spar=small_spar, skin=thin_skin,
            ),
        ],
        symmetric=True,
    )


@pytest.fixture
def simple_fuselage(petg_material):
    """Fuselage with PETG shell, 1mm wall, 4 cross-sections."""
    return ap.Fuselage(
        name="fuselage",
        xsecs=[
            ap.FuselageXSec(x=0.0, radius=0.02),
            ap.FuselageXSec(x=0.15, radius=0.06),
            ap.FuselageXSec(x=0.70, radius=0.06),
            ap.FuselageXSec(x=0.95, radius=0.02),
        ],
        material=petg_material,
        wall_thickness=0.001,
    )


@pytest.fixture
def test_propulsion():
    """Basic propulsion system at nose position."""
    motor = ap.Motor(
        name="Test Motor", kv=1100, resistance=0.028,
        no_load_current=1.2, max_current=40.0, mass=0.152,
    )
    prop = ap.Propeller(diameter=0.254, pitch=0.127, mass=0.030)
    battery = ap.Battery(
        name="Test 4S", capacity_ah=2.2, nominal_voltage=14.8,
        cell_count=4, c_rating=30.0, mass=0.200,
    )
    esc = ap.ESC(name="Test ESC", max_current=40.0, mass=0.035)
    return ap.PropulsionSystem(
        motor=motor, propeller=prop, battery=battery, esc=esc,
        position=np.array([0.0, 0.0, 0.0]),
    )


@pytest.fixture
def test_servo():
    """Small digital servo."""
    return ap.Servo(
        name="Test Servo", torque=3.0, speed=300.0,
        voltage=6.0, mass=0.024,
    )


@pytest.fixture
def aircraft_with_structure(
    wing_with_structure, htail_with_structure, simple_fuselage,
    test_propulsion, test_servo,
):
    """Complete aircraft with structure, propulsion, servos, and payload."""
    # Add aileron with servo to main wing
    wing_with_structure.control_surfaces = [
        ap.ControlSurface(
            name="aileron", span_start=0.6, span_end=0.95,
            chord_fraction=0.25, servo=test_servo,
        ),
    ]
    # Add elevator with servo to htail
    htail_with_structure.control_surfaces = [
        ap.ControlSurface(
            name="elevator", span_start=0.1, span_end=0.9,
            chord_fraction=0.30, servo=test_servo,
        ),
    ]

    payload = ap.Payload(mass=0.100, cg=np.array([0.25, 0.0, 0.0]), name="payload")

    return ap.Aircraft(
        name="TestPlane",
        wings=[wing_with_structure, htail_with_structure],
        fuselages=[simple_fuselage],
        propulsion=test_propulsion,
        payload=payload,
    )
