"""Fixtures for stability module tests."""

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg


@pytest.fixture
def cf_spar():
    """Carbon fiber tube spar."""
    return ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


@pytest.fixture
def petg_skin():
    """PETG 3D-print skin."""
    return ap.Skin(material=petg, thickness=0.8e-3)


@pytest.fixture
def main_wing(cf_spar, petg_skin):
    """Rectangular main wing: 0.2m chord, 0.75m semispan at x=0.1."""
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
def htail(cf_spar, petg_skin):
    """Horizontal tail at x=0.75, tapered, symmetric."""
    small_spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
    )
    thin_skin = ap.Skin(material=petg, thickness=0.5e-3)
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
        control_surfaces=[
            ap.ControlSurface(
                name="elevator", span_start=0.1, span_end=0.9,
                chord_fraction=0.30,
                servo=ap.Servo(
                    name="elevator_servo", torque=3.0, speed=300.0,
                    voltage=6.0, mass=0.024,
                ),
            ),
        ],
    )


@pytest.fixture
def vtail():
    """Vertical tail at x=0.75, non-symmetric.

    ASB convention: vtail span is defined along y-axis (the solver
    handles the physical vertical orientation).
    """
    return ap.Wing(
        name="vtail",
        xsecs=[
            ap.WingXSec(
                xyz_le=[0.75, 0.0, 0.0], chord=0.10,
            ),
            ap.WingXSec(
                xyz_le=[0.75, 0.20, 0.0], chord=0.06,
            ),
        ],
        symmetric=False,
    )


@pytest.fixture
def test_propulsion():
    """Basic propulsion system at nose."""
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
def stability_aircraft(main_wing, htail, vtail, test_propulsion):
    """Complete aircraft with main wing, htail, vtail, and propulsion."""
    payload = ap.Payload(mass=0.100, cg=np.array([0.25, 0.0, 0.0]), name="payload")

    return ap.Aircraft(
        name="StabilityTestPlane",
        wings=[main_wing, htail, vtail],
        fuselages=[
            ap.Fuselage(
                name="fuselage",
                xsecs=[
                    ap.FuselageXSec(x=0.0, radius=0.02),
                    ap.FuselageXSec(x=0.15, radius=0.06),
                    ap.FuselageXSec(x=0.70, radius=0.06),
                    ap.FuselageXSec(x=0.95, radius=0.02),
                ],
                material=petg,
                wall_thickness=0.001,
            ),
        ],
        propulsion=test_propulsion,
        payload=payload,
    )


@pytest.fixture
def flight_condition():
    """Cruise condition: 15 m/s at sea level."""
    return ap.FlightCondition(velocity=15.0, altitude=0.0, alpha=2.0, beta=0.0)


@pytest.fixture
def weight_result(stability_aircraft):
    """Weight analysis result for the stability test aircraft."""
    from aerisplane.weights import analyze as weight_analyze
    return weight_analyze(stability_aircraft)
