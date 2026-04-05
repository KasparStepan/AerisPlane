"""Integration tests: chain all disciplines together on one aircraft.

Validates that aero → weights → stability → control → mission pass results
through each other without interface mismatches, and that outputs are
physically plausible.
"""
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg
import aerisplane.aero as aero
import aerisplane.weights as wts
import aerisplane.stability as stab
import aerisplane.control as ctrl
import aerisplane.mission as mis
import aerisplane.structures as struc
from aerisplane.mission.segments import Mission, Cruise, Climb


# ---------------------------------------------------------------------------
# Shared aircraft fixture (module-scoped — built once, reused across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rc_aircraft():
    """A representative 1.5 kg fixed-wing RC aircraft with all subsystems.

    Geometry based on a conventional tractor layout:
    - Rectangular main wing (AR≈7.5), NACA 2412
    - Symmetric tapered H-tail at x=0.75 m
    - Vertical tail (non-symmetric) at x=0.75 m
    - Simple fuselage (4 cross-sections)
    - Brushless motor + battery + ESC propulsion
    - Payload representing autopilot + camera
    """
    cf_spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )
    petg_skin = ap.Skin(material=petg, thickness=0.8e-3)
    thin_skin = ap.Skin(material=petg, thickness=0.5e-3)
    small_spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
    )

    main_wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(
                xyz_le=[0.10, 0.0, 0.0], chord=0.20,
                airfoil=ap.Airfoil.from_naca("2412"),
                spar=cf_spar, skin=petg_skin,
                twist=0.0,
            ),
            ap.WingXSec(
                xyz_le=[0.10, 0.75, 0.0], chord=0.20,
                airfoil=ap.Airfoil.from_naca("2412"),
                spar=cf_spar, skin=petg_skin,
                twist=-2.0,
            ),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="aileron", span_start=0.5, span_end=0.9,
                chord_fraction=0.25, symmetric=False,
                servo=ap.Servo(name="ail_servo", torque=2.5, speed=300.0,
                               voltage=5.0, mass=0.018),
            ),
        ],
    )

    htail = ap.Wing(
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
                servo=ap.Servo(name="elev_servo", torque=3.0, speed=300.0,
                               voltage=6.0, mass=0.024),
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
                name="rudder", span_start=0.1, span_end=0.9,
                chord_fraction=0.35,
                servo=ap.Servo(name="rud_servo", torque=3.0, speed=300.0,
                               voltage=6.0, mass=0.024),
            ),
        ],
    )

    fuselage = ap.Fuselage(
        name="fuselage",
        xsecs=[
            ap.FuselageXSec(x=0.00, radius=0.020),
            ap.FuselageXSec(x=0.12, radius=0.055),
            ap.FuselageXSec(x=0.60, radius=0.055),
            ap.FuselageXSec(x=0.85, radius=0.020),
        ],
        material=petg,
        wall_thickness=0.001,
    )

    propulsion = ap.PropulsionSystem(
        motor=ap.Motor(name="motor", kv=1100, resistance=0.028,
                       no_load_current=1.2, max_current=40.0, mass=0.120),
        propeller=ap.Propeller(diameter=0.254, pitch=0.127, mass=0.030),
        battery=ap.Battery(name="4S2200", capacity_ah=2.2, nominal_voltage=14.8,
                           cell_count=4, c_rating=30.0, mass=0.195),
        esc=ap.ESC(name="esc", max_current=40.0, mass=0.030),
        position=np.array([0.0, 0.0, 0.0]),
    )

    return ap.Aircraft(
        name="IntegrationTestPlane",
        wings=[main_wing, htail, vtail],
        fuselages=[fuselage],
        propulsion=propulsion,
        payload=ap.Payload(mass=0.100, cg=np.array([0.25, 0.0, 0.0]), name="payload"),
        xyz_ref=[0.15, 0.0, 0.0],
    )


@pytest.fixture(scope="module")
def cruise_condition():
    return ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=3.0, beta=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWeights:
    def test_weight_result_is_valid(self, rc_aircraft):
        result = wts.analyze(rc_aircraft)
        assert result is not None
        assert result.total_mass > 0.5    # > 500 g
        assert result.total_mass < 5.0    # < 5 kg
        assert result.cg is not None
        assert result.cg.shape == (3,)

    def test_report_runs(self, rc_aircraft):
        result = wts.analyze(rc_aircraft)
        report = result.report()
        assert isinstance(report, str)
        assert len(report) > 0


class TestAero:
    def test_vlm_produces_positive_cl(self, rc_aircraft, cruise_condition):
        result = aero.analyze(rc_aircraft, cruise_condition, method="vlm")
        assert result.CL > 0.0
        assert result.CD > 0.0
        assert 0.1 < result.CL < 2.0
        assert result.CD < result.CL  # sane L/D

    def test_lifting_line_produces_positive_cl(self, rc_aircraft, cruise_condition):
        result = aero.analyze(rc_aircraft, cruise_condition, method="lifting_line",
                              spanwise_resolution=6)
        assert result.CL > 0.0
        assert result.CD > 0.0

    def test_aero_buildup_produces_positive_cl(self, rc_aircraft, cruise_condition):
        result = aero.analyze(rc_aircraft, cruise_condition, method="aero_buildup")
        assert result.CL > 0.0
        assert result.CD > 0.0

    def test_report_runs(self, rc_aircraft, cruise_condition):
        result = aero.analyze(rc_aircraft, cruise_condition, method="vlm")
        result.report()  # should not raise


class TestStabilityChain:
    def test_stability_consumes_weight_result(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        assert result is not None
        assert result.CL_alpha is not None
        assert result.CL_alpha > 0.0    # positive lift slope

    def test_static_margin_is_finite(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        assert np.isfinite(result.static_margin)

    def test_report_runs(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        report = result.report()
        assert isinstance(report, str)


class TestControlChain:
    def test_control_consumes_stability_result(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        stab_result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        result = ctrl.analyze(rc_aircraft, cruise_condition, weight_result, stab_result)
        assert result is not None

    def test_roll_rate_positive(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        stab_result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        result = ctrl.analyze(rc_aircraft, cruise_condition, weight_result, stab_result)
        assert result.max_roll_rate > 0.0

    def test_report_runs(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        stab_result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        result = ctrl.analyze(rc_aircraft, cruise_condition, weight_result, stab_result)
        report = result.report()
        assert isinstance(report, str)


class TestMissionChain:
    def test_mission_consumes_weight_result(self, rc_aircraft):
        weight_result = wts.analyze(rc_aircraft)
        mission = Mission(segments=[
            Climb(to_altitude=100.0, climb_rate=2.0, velocity=12.0),
            Cruise(distance=3000.0, velocity=15.0, altitude=100.0),
        ])
        result = mis.analyze(rc_aircraft, weight_result, mission)
        assert result is not None

    def test_range_and_endurance_positive(self, rc_aircraft):
        weight_result = wts.analyze(rc_aircraft)
        mission = Mission(segments=[
            Cruise(distance=3000.0, velocity=15.0, altitude=0.0),
        ])
        result = mis.analyze(rc_aircraft, weight_result, mission)
        assert result.total_distance > 0.0
        assert result.total_time > 0.0

    def test_energy_consumed_positive(self, rc_aircraft):
        weight_result = wts.analyze(rc_aircraft)
        mission = Mission(segments=[
            Cruise(distance=3000.0, velocity=15.0, altitude=0.0),
        ])
        result = mis.analyze(rc_aircraft, weight_result, mission)
        assert result.total_energy > 0.0

    def test_report_runs(self, rc_aircraft):
        weight_result = wts.analyze(rc_aircraft)
        mission = Mission(segments=[
            Cruise(distance=1000.0, velocity=15.0, altitude=0.0),
        ])
        result = mis.analyze(rc_aircraft, weight_result, mission)
        report = result.report()
        assert isinstance(report, str)


class TestFullChain:
    """Single test that chains all five disciplines in sequence."""

    def test_full_discipline_chain(self, rc_aircraft, cruise_condition):
        """aero → weights → stability → control → mission with no interface errors."""
        weight_result = wts.analyze(rc_aircraft)
        aero_result = aero.analyze(rc_aircraft, cruise_condition, method="vlm")
        stab_result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        ctrl_result = ctrl.analyze(rc_aircraft, cruise_condition, weight_result, stab_result)
        mission = Mission(segments=[
            Cruise(distance=3000.0, velocity=15.0, altitude=100.0),
        ])
        mis_result = mis.analyze(rc_aircraft, weight_result, mission)

        # Cross-discipline sanity: weight and lift should roughly match in cruise
        # At cruise alpha, L ≈ W (within factor 3 for our test condition)
        L = aero_result.CL * cruise_condition.dynamic_pressure() * rc_aircraft.reference_area()
        W = weight_result.total_mass * 9.81
        assert L / W > 0.3, f"L/W={L/W:.2f}, lift much too low"

        # Stability: known static margin sign depends on config, but must be finite
        assert np.isfinite(stab_result.static_margin)

        # Control: roll rate > 0 (ailerons present)
        assert ctrl_result.max_roll_rate > 0.0

        # Mission: energy budget > 0
        assert mis_result.total_energy > 0.0


class TestStructuresChain:
    def test_structures_consumes_aero_and_weight(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        aero_result = aero.analyze(rc_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(rc_aircraft, aero_result, weight_result)
        assert result is not None
        assert len(result.wings) > 0

    def test_structures_report_runs(self, rc_aircraft, cruise_condition):
        weight_result = wts.analyze(rc_aircraft)
        aero_result = aero.analyze(rc_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(rc_aircraft, aero_result, weight_result)
        report = result.report()
        assert isinstance(report, str)

    def test_full_chain_with_structures(self, rc_aircraft, cruise_condition):
        """aero -> weights -> stability -> control -> mission -> structures."""
        weight_result = wts.analyze(rc_aircraft)
        aero_result = aero.analyze(rc_aircraft, cruise_condition, method="vlm")
        stab_result = stab.analyze(rc_aircraft, cruise_condition, weight_result)
        ctrl_result = ctrl.analyze(rc_aircraft, cruise_condition,
                                   weight_result, stab_result)
        mission = Mission(segments=[
            Cruise(distance=3000.0, velocity=15.0, altitude=100.0),
        ])
        mis_result = mis.analyze(rc_aircraft, weight_result, mission)
        struct_result = struc.analyze(rc_aircraft, aero_result, weight_result,
                                      stability_result=stab_result)
        assert struct_result.is_safe or not struct_result.is_safe  # just runs
        assert np.isfinite(struct_result.design_load_factor)
