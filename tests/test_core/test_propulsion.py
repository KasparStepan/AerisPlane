"""Tests for Motor, Propeller, Battery, PropulsionSystem from aerisplane.core.propulsion."""

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.core.propulsion import Battery, ESC, Motor, Propeller, PropulsionSystem


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

class TestBatteryEnergy:
    """energy = capacity_ah * voltage * 3600."""

    def test_energy(self, simple_battery):
        # 5.0 Ah * 14.8 V * 3600 s/h = 266400 J
        expected = 5.0 * 14.8 * 3600.0
        assert simple_battery.energy() == pytest.approx(expected)


class TestBatteryMaxCurrent:
    """max_current = capacity_ah * c_rating."""

    def test_max_current(self, simple_battery):
        # 5.0 Ah * 30 C = 150 A
        expected = 5.0 * 30.0
        assert simple_battery.max_current() == pytest.approx(expected)


class TestBatteryVoltageUnderLoad:
    """voltage_under_load = nominal - I * R."""

    def test_no_load(self, simple_battery):
        assert simple_battery.voltage_under_load(0.0) == pytest.approx(14.8)

    def test_under_load(self, simple_battery):
        current = 20.0
        # 14.8 - 20.0 * 0.02 = 14.4
        expected = 14.8 - 20.0 * 0.02
        assert simple_battery.voltage_under_load(current) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

class TestMotorRPM:
    """rpm = kv * (V - I * R)."""

    def test_rpm(self, simple_motor):
        voltage = 14.8
        current = 20.0
        # 1100 * (14.8 - 20.0 * 0.028) = 1100 * (14.8 - 0.56) = 1100 * 14.24
        expected = 1100.0 * (14.8 - 20.0 * 0.028)
        assert simple_motor.rpm(voltage, current) == pytest.approx(expected)

    def test_no_load_rpm(self, simple_motor):
        """At zero current, RPM = kv * V."""
        voltage = 14.8
        expected = 1100.0 * 14.8
        assert simple_motor.rpm(voltage, 0.0) == pytest.approx(expected)


class TestMotorEfficiency:
    """Efficiency should be between 0 and 1 for reasonable inputs."""

    def test_efficiency_range(self, simple_motor):
        voltage = 14.8
        current = 20.0
        eff = simple_motor.efficiency(voltage, current)
        assert 0.0 < eff < 1.0

    def test_efficiency_at_no_load_current(self, simple_motor):
        """At no-load current, motor does no useful work -> efficiency = 0."""
        voltage = 14.8
        eff = simple_motor.efficiency(voltage, simple_motor.no_load_current)
        assert eff == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Propeller
# ---------------------------------------------------------------------------

class TestPropellerAdvanceRatio:
    """J = V / (n * D) where n = rpm / 60."""

    def test_advance_ratio(self):
        prop = Propeller(diameter=0.254, pitch=0.127)
        velocity = 15.0
        rpm = 8000.0
        n = rpm / 60.0
        expected_J = velocity / (n * 0.254)
        assert prop.advance_ratio(velocity, rpm) == pytest.approx(expected_J)

    def test_static_advance_ratio(self):
        """At zero RPM, advance ratio is 0."""
        prop = Propeller(diameter=0.254, pitch=0.127)
        assert prop.advance_ratio(15.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PropulsionSystem
# ---------------------------------------------------------------------------

class TestPropulsionSystemTotalMass:
    """total_mass = motor + propeller + battery + esc masses."""

    def test_total_mass(self, simple_motor, simple_battery):
        prop = Propeller(diameter=0.254, pitch=0.127, mass=0.03)
        esc = ESC(name="Test ESC", max_current=50.0, mass=0.035)
        system = PropulsionSystem(
            motor=simple_motor,
            propeller=prop,
            battery=simple_battery,
            esc=esc,
        )
        expected = simple_motor.mass + prop.mass + simple_battery.mass + esc.mass
        assert system.total_mass() == pytest.approx(expected)
