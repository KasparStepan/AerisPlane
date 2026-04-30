"""Test fixtures for io round-trip tests."""

import numpy as np
import pytest

import aerisplane as ap


@pytest.fixture
def aircraft_with_controls():
    """A symmetric wing + tail aircraft with elevator and ailerons."""
    af = ap.Airfoil.from_naca("2412")
    af_tail = ap.Airfoil.from_naca("0012")

    main_wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.30, airfoil=af),
            ap.WingXSec(xyz_le=[0.02, 0.50, 0.02], chord=0.22, airfoil=af),
            ap.WingXSec(xyz_le=[0.05, 1.00, 0.05], chord=0.15, airfoil=af, twist=-1.5),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="aileron", span_start=0.5, span_end=1.0,
                chord_fraction=0.25, symmetric=False, max_deflection=20.0,
            ),
        ],
    )
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.90, 0.00, 0.05], chord=0.12, airfoil=af_tail),
            ap.WingXSec(xyz_le=[0.92, 0.25, 0.05], chord=0.08, airfoil=af_tail),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="elevator", span_start=0.0, span_end=1.0,
                chord_fraction=0.30, symmetric=True,
            ),
        ],
    )
    fuse = ap.Fuselage(
        name="fuselage",
        xsecs=[
            ap.FuselageXSec(x=0.0, width=0.0, height=0.0, shape=2.0),
            ap.FuselageXSec(x=0.15, width=0.08, height=0.10, shape=4.0),
            ap.FuselageXSec(x=0.80, width=0.04, height=0.04, shape=2.0),
        ],
        x_le=-0.10,
    )
    return ap.Aircraft(
        name="test_plane",
        wings=[main_wing, htail],
        fuselages=[fuse],
        xyz_ref=[0.10, 0.0, 0.0],
    )


@pytest.fixture
def aircraft_full(aircraft_with_controls):
    """Same as aircraft_with_controls plus structures, propulsion, and payload."""
    from aerisplane.catalog.servos import kst_x08h

    cf = ap.Material(name="cf_uni", density=1600, E=70e9, yield_strength=600e6)
    foam = ap.Material(name="foamboard", density=80, E=20e6, yield_strength=1e6)
    spar = ap.Spar(
        position=0.25, material=cf,
        section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
    )
    aircraft_with_controls.wings[0].xsecs[0].spar = spar
    aircraft_with_controls.wings[0].xsecs[-1].spar = spar
    aircraft_with_controls.wings[0].control_surfaces[0].servo = kst_x08h
    aircraft_with_controls.fuselages[0].material = foam

    aircraft_with_controls.propulsion = ap.PropulsionSystem(
        motor=ap.Motor(name="m1", kv=1200, resistance=0.03, no_load_current=1.0,
                       max_current=30.0, mass=0.12),
        propeller=ap.Propeller(diameter=0.254, pitch=0.119, mass=0.018),
        battery=ap.Battery(name="b1", capacity_ah=4.0, nominal_voltage=14.8,
                           cell_count=4, c_rating=30.0, mass=0.40),
        esc=ap.ESC(name="e1", max_current=40.0, mass=0.030),
        position=np.array([-0.05, 0.0, 0.0]),
    )
    aircraft_with_controls.payload = ap.Payload(mass=0.5, cg=np.array([0.10, 0.0, 0.0]))
    return aircraft_with_controls
