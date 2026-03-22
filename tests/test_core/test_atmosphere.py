"""Tests for the ISA atmosphere model (troposphere, 0-11000 m)."""

import pytest
from aerisplane.utils.atmosphere import isa, speed_of_sound


# ---------------------------------------------------------------------------
# Sea-level conditions (altitude = 0 m)
# ---------------------------------------------------------------------------

def test_isa_sea_level_temperature():
    temperature, _, _, _ = isa(0)
    assert temperature == pytest.approx(288.15, abs=0.01)


def test_isa_sea_level_pressure():
    _, pressure, _, _ = isa(0)
    assert pressure == pytest.approx(101325.0, abs=1.0)


def test_isa_sea_level_density():
    _, _, density, _ = isa(0)
    assert density == pytest.approx(1.225, abs=0.001)


# ---------------------------------------------------------------------------
# Tropopause conditions (altitude = 11000 m)
# ---------------------------------------------------------------------------

def test_isa_tropopause_temperature():
    temperature, _, _, _ = isa(11000)
    assert temperature == pytest.approx(216.65, abs=0.01)


def test_isa_tropopause_pressure():
    """Pressure at 11 km should be roughly 22632 Pa (ISA standard value)."""
    _, pressure, _, _ = isa(11000)
    assert pressure == pytest.approx(22632.0, rel=0.01)


def test_isa_tropopause_density():
    """Density at 11 km should be roughly 0.3639 kg/m^3."""
    _, _, density, _ = isa(11000)
    assert density == pytest.approx(0.3639, rel=0.01)


# ---------------------------------------------------------------------------
# Intermediate altitude (5000 m)
# ---------------------------------------------------------------------------

def test_isa_5000m_temperature():
    temperature, _, _, _ = isa(5000)
    assert temperature == pytest.approx(255.65, abs=0.1)


def test_isa_5000m_pressure():
    """Pressure at 5 km should be roughly 54019 Pa."""
    _, pressure, _, _ = isa(5000)
    assert pressure == pytest.approx(54019.0, rel=0.01)


def test_isa_5000m_density():
    """Density at 5 km should be roughly 0.7361 kg/m^3."""
    _, _, density, _ = isa(5000)
    assert density == pytest.approx(0.7361, rel=0.01)


# ---------------------------------------------------------------------------
# Speed of sound
# ---------------------------------------------------------------------------

def test_speed_of_sound_sea_level():
    a = speed_of_sound(0)
    assert a == pytest.approx(340.3, abs=0.5)


# ---------------------------------------------------------------------------
# Altitude clamping (negative and above 11000 m)
# ---------------------------------------------------------------------------

def test_negative_altitude_clipped_to_sea_level():
    """Negative altitudes should be clipped to 0, giving sea-level values."""
    temp_neg, pres_neg, rho_neg, mu_neg = isa(-500)
    temp_zero, pres_zero, rho_zero, mu_zero = isa(0)
    assert temp_neg == pytest.approx(temp_zero)
    assert pres_neg == pytest.approx(pres_zero)
    assert rho_neg == pytest.approx(rho_zero)
    assert mu_neg == pytest.approx(mu_zero)


def test_altitude_above_11000_clipped_to_tropopause():
    """Altitudes above 11000 m should be clipped to 11000 m."""
    temp_high, pres_high, rho_high, mu_high = isa(20000)
    temp_tropo, pres_tropo, rho_tropo, mu_tropo = isa(11000)
    assert temp_high == pytest.approx(temp_tropo)
    assert pres_high == pytest.approx(pres_tropo)
    assert rho_high == pytest.approx(rho_tropo)
    assert mu_high == pytest.approx(mu_tropo)
