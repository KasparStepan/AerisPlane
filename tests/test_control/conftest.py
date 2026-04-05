"""Fixtures for control authority module tests."""

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg


@pytest.fixture(scope="module")
def cf_spar():
    return ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


@pytest.fixture(scope="module")
def petg_skin():
    return ap.Skin(material=petg, thickness=0.8e-3)


@pytest.fixture(scope="module")
def aileron_servo():
    return ap.Servo(
        name="Aileron Servo", torque=3.0, speed=300.0,
        voltage=6.0, mass=0.024,
    )


@pytest.fixture(scope="module")
def elevator_servo():
    return ap.Servo(
        name="Elevator Servo", torque=3.0, speed=300.0,
        voltage=6.0, mass=0.024,
    )


@pytest.fixture(scope="module")
def rudder_servo():
    return ap.Servo(
        name="Rudder Servo", torque=2.0, speed=400.0,
        voltage=6.0, mass=0.018,
    )


@pytest.fixture(scope="module")
def control_aircraft(cf_spar, petg_skin, aileron_servo, elevator_servo, rudder_servo):
    """Aircraft with aileron, elevator, and rudder — all with servos."""
    small_spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
    )
    thin_skin = ap.Skin(material=petg, thickness=0.5e-3)

    main_wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.1, 0.0, 0.0], chord=0.2, spar=cf_spar, skin=petg_skin),
            ap.WingXSec(xyz_le=[0.1, 0.75, 0.0], chord=0.2, spar=cf_spar, skin=petg_skin),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="aileron", span_start=0.6, span_end=0.95,
                chord_fraction=0.25, max_deflection=25.0, min_deflection=-25.0,
                symmetric=False, servo=aileron_servo,
            ),
        ],
    )

    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.75, 0.0, 0.0], chord=0.12, spar=small_spar, skin=thin_skin),
            ap.WingXSec(xyz_le=[0.75, 0.30, 0.0], chord=0.08, spar=small_spar, skin=thin_skin),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="elevator", span_start=0.1, span_end=0.9,
                chord_fraction=0.30, max_deflection=25.0, min_deflection=-25.0,
                symmetric=True, servo=elevator_servo,
            ),
        ],
    )

    vtail = ap.Wing(
        name="vtail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.75, 0.0, 0.00], chord=0.10),
            ap.WingXSec(xyz_le=[0.75, 0.0, 0.20], chord=0.06),
        ],
        symmetric=False,
        control_surfaces=[
            ap.ControlSurface(
                name="rudder", span_start=0.2, span_end=0.9,
                chord_fraction=0.35, max_deflection=25.0, min_deflection=-25.0,
                symmetric=True, servo=rudder_servo,
            ),
        ],
    )

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
    propulsion = ap.PropulsionSystem(
        motor=motor, propeller=prop, battery=battery, esc=esc,
        position=np.array([0.0, 0.0, 0.0]),
    )

    payload = ap.Payload(mass=0.100, cg=np.array([0.25, 0.0, 0.0]), name="payload")

    return ap.Aircraft(
        name="ControlTestPlane",
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
        propulsion=propulsion,
        payload=payload,
    )


@pytest.fixture(scope="module")
def flight_condition():
    return ap.FlightCondition(velocity=15.0, altitude=0.0, alpha=2.0, beta=0.0)


@pytest.fixture(scope="module")
def weight_result(control_aircraft):
    from aerisplane.weights import analyze as weight_analyze
    return weight_analyze(control_aircraft)


@pytest.fixture(scope="module")
def stability_result(control_aircraft, flight_condition, weight_result):
    from aerisplane.stability import analyze as stab_analyze
    return stab_analyze(control_aircraft, flight_condition, weight_result, aero_method="vlm")
