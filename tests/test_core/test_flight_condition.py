"""Tests for FlightCondition from aerisplane.core.flight_condition."""

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.core.flight_condition import FlightCondition


# ---------------------------------------------------------------------------
# Sea-level constants for reference
# ---------------------------------------------------------------------------
SEA_LEVEL_DENSITY = 1.225          # kg/m^3
SEA_LEVEL_SPEED_OF_SOUND = 340.3   # m/s (approx at 288.15 K)


class TestDynamicPressure:
    """q = 0.5 * rho * V^2 at sea level."""

    def test_at_sea_level_20ms(self):
        fc = FlightCondition(velocity=20.0, altitude=0.0)
        expected_q = 0.5 * SEA_LEVEL_DENSITY * 20.0**2
        assert fc.dynamic_pressure() == pytest.approx(expected_q, rel=1e-4)

    def test_at_sea_level_zero_velocity(self):
        fc = FlightCondition(velocity=0.0, altitude=0.0)
        assert fc.dynamic_pressure() == pytest.approx(0.0)

    def test_altitude_reduces_q(self):
        """Same airspeed at altitude should give lower dynamic pressure
        because density decreases with altitude."""
        fc_low = FlightCondition(velocity=30.0, altitude=0.0)
        fc_high = FlightCondition(velocity=30.0, altitude=2000.0)
        assert fc_high.dynamic_pressure() < fc_low.dynamic_pressure()


class TestReynoldsNumber:
    """Re = rho * V * L / mu."""

    def test_known_reference_length(self):
        fc = FlightCondition(velocity=20.0, altitude=0.0)
        ref_length = 0.25  # 25 cm MAC
        _, _, rho, mu = fc.atmosphere()
        expected_re = rho * 20.0 * ref_length / mu
        assert fc.reynolds_number(ref_length) == pytest.approx(expected_re, rel=1e-6)

    def test_re_scales_linearly_with_velocity(self):
        ref_length = 0.2
        fc1 = FlightCondition(velocity=10.0, altitude=0.0)
        fc2 = FlightCondition(velocity=20.0, altitude=0.0)
        assert fc2.reynolds_number(ref_length) == pytest.approx(
            2.0 * fc1.reynolds_number(ref_length), rel=1e-6
        )


class TestMachNumber:
    """Mach = V / a, where a ~ 340.3 m/s at sea level."""

    def test_mach_at_sea_level(self):
        fc = FlightCondition(velocity=34.03, altitude=0.0)
        assert fc.mach() == pytest.approx(0.1, rel=1e-2)

    def test_subsonic(self):
        fc = FlightCondition(velocity=20.0, altitude=0.0)
        assert 0.0 < fc.mach() < 1.0


class TestAngleConversions:
    """alpha_rad and beta_rad convert degrees to radians."""

    def test_alpha_rad(self):
        fc = FlightCondition(velocity=20.0, alpha=5.0)
        assert fc.alpha_rad() == pytest.approx(np.radians(5.0))

    def test_beta_rad(self):
        fc = FlightCondition(velocity=20.0, beta=-3.0)
        assert fc.beta_rad() == pytest.approx(np.radians(-3.0))

    def test_zero_angles(self):
        fc = FlightCondition(velocity=20.0)
        assert fc.alpha_rad() == pytest.approx(0.0)
        assert fc.beta_rad() == pytest.approx(0.0)


class TestDeflections:
    """deflections dict defaults to empty."""

    def test_default_deflections_empty(self):
        fc = FlightCondition(velocity=20.0)
        assert fc.deflections == {}

    def test_deflections_stored(self):
        defl = {"elevator": -3.5, "aileron": 10.0}
        fc = FlightCondition(velocity=20.0, deflections=defl)
        assert fc.deflections == defl

    def test_default_not_shared_between_instances(self):
        """Each FlightCondition should get its own empty dict."""
        fc1 = FlightCondition(velocity=20.0)
        fc2 = FlightCondition(velocity=25.0)
        fc1.deflections["rudder"] = 5.0
        assert "rudder" not in fc2.deflections
