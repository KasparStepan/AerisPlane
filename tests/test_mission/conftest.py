"""Fixtures for mission module tests."""
import numpy as np
import pytest
import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg


@pytest.fixture(scope="module")
def perf_aircraft():
    """Simple aircraft for performance tests."""
    cf_spar = ap.Spar(
        position=0.25, material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )
    petg_skin = ap.Skin(material=petg, thickness=0.8e-3)

    main_wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.1, 0.0, 0.0], chord=0.2, spar=cf_spar, skin=petg_skin),
            ap.WingXSec(xyz_le=[0.1, 0.75, 0.0], chord=0.2, spar=cf_spar, skin=petg_skin),
        ],
        symmetric=True,
    )
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.75, 0.0, 0.0], chord=0.12),
            ap.WingXSec(xyz_le=[0.75, 0.30, 0.0], chord=0.08),
        ],
        symmetric=True,
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

    return ap.Aircraft(
        name="PerfTestPlane",
        wings=[main_wing, htail],
        fuselages=[ap.Fuselage(
            name="fuselage",
            xsecs=[
                ap.FuselageXSec(x=0.0, radius=0.02),
                ap.FuselageXSec(x=0.15, radius=0.06),
                ap.FuselageXSec(x=0.70, radius=0.06),
                ap.FuselageXSec(x=0.95, radius=0.02),
            ],
            material=petg, wall_thickness=0.001,
        )],
        propulsion=propulsion,
        payload=ap.Payload(mass=0.100, cg=np.array([0.25, 0.0, 0.0]), name="payload"),
    )


@pytest.fixture(scope="module")
def perf_weight_result(perf_aircraft):
    from aerisplane.weights import analyze as weight_analyze
    return weight_analyze(perf_aircraft)
