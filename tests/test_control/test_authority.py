"""Tests for control authority computation."""

import numpy as np
import pytest

from aerisplane.control.authority import (
    compute_control_derivatives,
    compute_max_crosswind,
    compute_pitch_acceleration,
    compute_roll_rate,
    estimate_hinge_moment,
    estimate_roll_damping,
    find_control_surfaces,
)


class TestFindSurfaces:
    """Test control surface identification."""

    def test_finds_aileron(self, control_aircraft):
        surfaces = find_control_surfaces(control_aircraft)
        assert "aileron" in surfaces

    def test_finds_elevator(self, control_aircraft):
        surfaces = find_control_surfaces(control_aircraft)
        assert "elevator" in surfaces

    def test_finds_rudder(self, control_aircraft):
        surfaces = find_control_surfaces(control_aircraft)
        assert "rudder" in surfaces


class TestControlDerivatives:
    """Test control derivative computation."""

    def test_cl_delta_a_nonzero(self, control_aircraft, flight_condition, weight_result):
        deriv = compute_control_derivatives(
            control_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert abs(deriv.Cl_delta_a) > 1e-6, (
            f"Cl_delta_a = {deriv.Cl_delta_a}, expected non-zero"
        )

    def test_cm_delta_e_nonzero(self, control_aircraft, flight_condition, weight_result):
        deriv = compute_control_derivatives(
            control_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert abs(deriv.Cm_delta_e) > 1e-6, (
            f"Cm_delta_e = {deriv.Cm_delta_e}, expected non-zero"
        )

    def test_cn_delta_r_nonzero(self, control_aircraft, flight_condition, weight_result):
        deriv = compute_control_derivatives(
            control_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert abs(deriv.Cn_delta_r) > 1e-6, (
            f"Cn_delta_r = {deriv.Cn_delta_r}, expected non-zero"
        )

    def test_does_not_mutate_aircraft(self, control_aircraft, flight_condition, weight_result):
        original_ref = list(control_aircraft.xyz_ref)
        compute_control_derivatives(
            control_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert control_aircraft.xyz_ref == original_ref


class TestRollDamping:
    """Test roll damping estimation."""

    def test_cl_p_negative(self, control_aircraft, flight_condition):
        Cl_p = estimate_roll_damping(control_aircraft, flight_condition)
        assert Cl_p < 0, f"Cl_p = {Cl_p}, expected negative (damping)"

    def test_cl_p_reasonable_magnitude(self, control_aircraft, flight_condition):
        Cl_p = estimate_roll_damping(control_aircraft, flight_condition)
        # Typical Cl_p for RC aircraft: -0.2 to -0.8
        assert -2.0 < Cl_p < -0.05, (
            f"Cl_p = {Cl_p}, outside reasonable range"
        )


class TestRollRate:
    """Test steady-state roll rate computation."""

    def test_positive_roll_rate(self):
        p = compute_roll_rate(
            Cl_delta_a=-0.001, delta_a_max=25.0, Cl_p=-0.4,
            velocity=15.0, span=1.5,
        )
        assert p > 0

    def test_zero_aileron_gives_zero(self):
        p = compute_roll_rate(
            Cl_delta_a=0.0, delta_a_max=25.0, Cl_p=-0.4,
            velocity=15.0, span=1.5,
        )
        assert p == 0.0


class TestPitchAcceleration:
    """Test pitch acceleration computation."""

    def test_positive_acceleration(self):
        a = compute_pitch_acceleration(
            Cm_delta_e=-0.002, delta_e_max=25.0,
            dynamic_pressure=138.0, S_ref=0.3, c_ref=0.2, I_yy=0.01,
        )
        assert a > 0

    def test_zero_elevator_gives_zero(self):
        a = compute_pitch_acceleration(
            Cm_delta_e=0.0, delta_e_max=25.0,
            dynamic_pressure=138.0, S_ref=0.3, c_ref=0.2, I_yy=0.01,
        )
        assert a == 0.0


class TestMaxCrosswind:
    """Test crosswind computation."""

    def test_positive_crosswind(self):
        cw = compute_max_crosswind(
            Cn_delta_r=0.001, delta_r_max=25.0,
            Cn_beta=0.002, velocity=15.0,
        )
        assert cw > 0

    def test_zero_rudder_gives_zero(self):
        cw = compute_max_crosswind(
            Cn_delta_r=0.0, delta_r_max=25.0,
            Cn_beta=0.002, velocity=15.0,
        )
        assert cw == 0.0

    def test_zero_cn_beta_gives_inf(self):
        cw = compute_max_crosswind(
            Cn_delta_r=0.001, delta_r_max=25.0,
            Cn_beta=0.0, velocity=15.0,
        )
        assert np.isinf(cw)


class TestHingeMoment:
    """Test hinge moment estimation."""

    def test_hinge_moment_finite(self, control_aircraft, flight_condition):
        surfaces = find_control_surfaces(control_aircraft)
        q = flight_condition.dynamic_pressure()

        for stype, info in surfaces.items():
            hm = estimate_hinge_moment(info.surface, info.wing, q)
            if hm is not None:
                assert np.isfinite(hm), f"{stype} hinge moment not finite: {hm}"
                assert hm >= 0, f"{stype} hinge moment negative: {hm}"

    def test_no_servo_returns_none(self):
        """Surface without servo should return None for hinge moment."""
        import aerisplane as ap

        cs = ap.ControlSurface(
            name="test", span_start=0.5, span_end=0.9,
            chord_fraction=0.25, servo=None,
        )
        wing = ap.Wing(
            name="test_wing",
            xsecs=[
                ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2),
                ap.WingXSec(xyz_le=[0, 0.5, 0], chord=0.2),
            ],
            symmetric=True,
        )
        hm = estimate_hinge_moment(cs, wing, 138.0)
        assert hm is None
