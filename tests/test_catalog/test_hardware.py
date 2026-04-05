"""Tests for the hardware catalog: motors, batteries, propellers, servos."""
from aerisplane.catalog.motors import (
    sunnysky_x2216_1250, sunnysky_x2216_2400, tiger_mn3110_700,
    tiger_mn3110_780, tiger_mn2213_950, emax_mt2213_935, emax_mt2216_810,
    t_motor_f80_1900, t_motor_f60_2550, rctimer_5010_360,
    scorpion_m2205_2350, sunnysky_x2212_980, emax_rs2205_2600,
    tiger_mn4014_330, tiger_mn5212_340, axi_2217_20, turnigy_d3530_1400,
    hacker_a20_26, scorpion_hkii_2221_900, dualsky_eco_2315c_1100,
)
from aerisplane.core.propulsion import Motor
import pytest


def test_motor_type():
    assert isinstance(sunnysky_x2216_1250, Motor)


def test_motor_kv_positive():
    for m in [sunnysky_x2216_1250, tiger_mn3110_700, t_motor_f80_1900]:
        assert m.kv > 0


def test_motor_fields_plausible():
    m = sunnysky_x2216_1250
    assert 0 < m.kv < 5000
    assert 0 < m.resistance < 10
    assert 0 < m.no_load_current < 5
    assert 0 < m.max_current < 200
    assert 0 < m.mass < 1.0


# --- Battery tests ---

from aerisplane.catalog.batteries import (
    tattu_3s_2300, tattu_4s_1800, tattu_4s_3300, tattu_4s_5200,
    tattu_6s_10000, tattu_6s_16000, gens_ace_3s_2200, gens_ace_4s_4000,
    gens_ace_6s_6000, turnigy_nano_tech_3s_2200, turnigy_nano_tech_4s_5000,
    turnigy_nano_tech_6s_3300, multistar_4s_10000, ovonic_4s_2200, ovonic_6s_3300,
)
from aerisplane.core.propulsion import Battery


def test_battery_type():
    assert isinstance(tattu_3s_2300, Battery)


def test_battery_energy_positive():
    for b in [tattu_4s_1800, gens_ace_4s_4000, turnigy_nano_tech_6s_3300]:
        assert b.energy() > 0


def test_battery_max_current_consistent():
    b = tattu_4s_1800
    assert abs(b.max_current() - b.capacity_ah * b.c_rating) < 0.01


# --- Propeller tests ---

from aerisplane.catalog.propellers import (
    apc_10x4_7sf, apc_10x7e, apc_11x4_7sf, apc_12x6e, apc_13x4_7sf,
    apc_14x8_3mf, master_airscrew_10x5, master_airscrew_11x7,
    master_airscrew_14x7, tjd_14x8_5,
)
from aerisplane.core.propulsion import Propeller


def test_propeller_type():
    assert isinstance(apc_10x4_7sf, Propeller)


def test_propeller_diameter_inches():
    assert abs(apc_10x4_7sf.diameter - 0.254) < 0.001


def test_propeller_fields():
    p = apc_12x6e
    assert p.diameter > 0
    assert p.pitch > 0
    assert p.mass > 0


# --- Servo tests ---

from aerisplane.catalog.servos import (
    hitec_hs65mg, hitec_hs5086wb, hitec_hs7950th, savox_sh0255mg,
    savox_sc1256tg, futaba_s3003, futaba_s3305, towerpro_mg996r,
    kst_x08h, kst_ds215mg,
)
from aerisplane.core.control_surface import Servo


def test_servo_type():
    assert isinstance(hitec_hs65mg, Servo)


def test_servo_torque_positive():
    for s in [hitec_hs65mg, savox_sc1256tg, towerpro_mg996r]:
        assert s.torque > 0


def test_servo_fields():
    s = hitec_hs65mg
    assert s.torque > 0
    assert s.speed > 0
    assert s.voltage > 0
    assert s.mass > 0


# --- Discovery function tests ---

import aerisplane.catalog as catalog
from aerisplane.core.propulsion import Motor, Battery, Propeller
from aerisplane.core.control_surface import Servo


def test_list_motors_returns_motors():
    motors = catalog.list_motors()
    assert len(motors) >= 20
    assert all(isinstance(m, Motor) for m in motors)


def test_list_batteries_returns_batteries():
    batteries = catalog.list_batteries()
    assert len(batteries) >= 15
    assert all(isinstance(b, Battery) for b in batteries)


def test_list_propellers_returns_propellers():
    propellers = catalog.list_propellers()
    assert len(propellers) >= 10
    assert all(isinstance(p, Propeller) for p in propellers)


def test_list_servos_returns_servos():
    servos = catalog.list_servos()
    assert len(servos) >= 10
    assert all(isinstance(s, Servo) for s in servos)


def test_list_motors_unique_names():
    motors = catalog.list_motors()
    names = [m.name for m in motors]
    assert len(names) == len(set(names)), "Motor names must be unique"
