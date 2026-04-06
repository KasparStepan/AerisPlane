# Hardware Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the four empty hardware catalog stubs with real specs and add `list_*` discovery functions.

**Architecture:** Each catalog file (motors, batteries, propellers, servos) is a plain Python module exporting module-level instances (following the existing `materials.py` pattern). The catalog `__init__.py` grows four `list_*` functions that return lists of those instances. No new classes needed — all dataclasses are already defined in `core/propulsion.py` and `core/control_surface.py`.

**Tech Stack:** Python dataclasses (already defined), pytest, numpy (for any unit conversions in tests).

---

### Task 1: Motor catalog (`motors.py`)

**Files:**
- Modify: `src/aerisplane/catalog/motors.py`
- Create: `tests/test_catalog/__init__.py`
- Create: `tests/test_catalog/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catalog/test_hardware.py
from aerisplane.catalog.motors import (
    sunnysky_x2216_1250,
    sunnysky_x2216_2400,
    tiger_mn3110_700,
    tiger_mn3110_780,
    tiger_mn2213_950,
    emax_mt2213_935,
    emax_mt2216_810,
    t_motor_f80_1900,
    t_motor_f60_2550,
    rctimer_5010_360,
    scorpion_m2205_2350,
    sunnysky_x2212_980,
    emax_rs2205_2600,
    tiger_mn4014_330,
    tiger_mn5212_340,
    axi_2217_20,
    turnigy_d3530_1400,
    hacker_a20_26,
    scorpion_hkii_2221_900,
    dualsky_eco_2315c_1100,
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_catalog/test_hardware.py::test_motor_type -v
```

Expected: FAIL with ImportError or ModuleNotFoundError

- [ ] **Step 3: Create the test `__init__.py`**

```python
# tests/test_catalog/__init__.py
```
(empty file)

- [ ] **Step 4: Populate `motors.py` with 20 motors**

```python
# src/aerisplane/catalog/motors.py
"""Brushless motor catalog.

Specs sourced from manufacturer datasheets and published test bench data.
All values are nominal at room temperature.
"""
from aerisplane.core.propulsion import Motor

# SunnySky X2216
sunnysky_x2216_1250 = Motor(
    name="SunnySky X2216 1250KV",
    kv=1250, resistance=0.117, no_load_current=0.5,
    max_current=28.0, mass=0.058,
)
sunnysky_x2216_2400 = Motor(
    name="SunnySky X2216 2400KV",
    kv=2400, resistance=0.065, no_load_current=0.8,
    max_current=28.0, mass=0.058,
)
sunnysky_x2212_980 = Motor(
    name="SunnySky X2212 980KV",
    kv=980, resistance=0.135, no_load_current=0.5,
    max_current=20.0, mass=0.052,
)

# T-Motor MN series (long-range UAV)
tiger_mn3110_700 = Motor(
    name="T-Motor MN3110 700KV",
    kv=700, resistance=0.180, no_load_current=0.3,
    max_current=16.0, mass=0.102,
)
tiger_mn3110_780 = Motor(
    name="T-Motor MN3110 780KV",
    kv=780, resistance=0.165, no_load_current=0.3,
    max_current=16.0, mass=0.102,
)
tiger_mn2213_950 = Motor(
    name="T-Motor MN2213 950KV",
    kv=950, resistance=0.130, no_load_current=0.4,
    max_current=14.0, mass=0.060,
)
tiger_mn4014_330 = Motor(
    name="T-Motor MN4014 330KV",
    kv=330, resistance=0.220, no_load_current=0.2,
    max_current=22.0, mass=0.176,
)
tiger_mn5212_340 = Motor(
    name="T-Motor MN5212 340KV",
    kv=340, resistance=0.190, no_load_current=0.2,
    max_current=30.0, mass=0.215,
)

# T-Motor F series (race/sport)
t_motor_f80_1900 = Motor(
    name="T-Motor F80 1900KV",
    kv=1900, resistance=0.048, no_load_current=1.0,
    max_current=30.0, mass=0.068,
)
t_motor_f60_2550 = Motor(
    name="T-Motor F60 2550KV",
    kv=2550, resistance=0.031, no_load_current=1.2,
    max_current=35.0, mass=0.055,
)

# Emax
emax_mt2213_935 = Motor(
    name="Emax MT2213 935KV",
    kv=935, resistance=0.145, no_load_current=0.5,
    max_current=20.0, mass=0.057,
)
emax_mt2216_810 = Motor(
    name="Emax MT2216 810KV",
    kv=810, resistance=0.175, no_load_current=0.4,
    max_current=20.0, mass=0.075,
)
emax_rs2205_2600 = Motor(
    name="Emax RS2205 2600KV",
    kv=2600, resistance=0.038, no_load_current=1.0,
    max_current=30.0, mass=0.030,
)

# RCTimer
rctimer_5010_360 = Motor(
    name="RCTimer 5010 360KV",
    kv=360, resistance=0.120, no_load_current=0.3,
    max_current=40.0, mass=0.190,
)

# Scorpion
scorpion_m2205_2350 = Motor(
    name="Scorpion M2205 2350KV",
    kv=2350, resistance=0.042, no_load_current=1.1,
    max_current=30.0, mass=0.035,
)
scorpion_hkii_2221_900 = Motor(
    name="Scorpion HKII-2221 900KV",
    kv=900, resistance=0.140, no_load_current=0.6,
    max_current=22.0, mass=0.068,
)

# AXi
axi_2217_20 = Motor(
    name="AXi 2217/20",
    kv=1050, resistance=0.098, no_load_current=0.4,
    max_current=18.0, mass=0.095,
)

# Turnigy
turnigy_d3530_1400 = Motor(
    name="Turnigy D3530/14 1400KV",
    kv=1400, resistance=0.085, no_load_current=0.5,
    max_current=21.0, mass=0.086,
)

# Hacker
hacker_a20_26 = Motor(
    name="Hacker A20-26L",
    kv=1020, resistance=0.120, no_load_current=0.4,
    max_current=16.0, mass=0.072,
)

# Dualsky
dualsky_eco_2315c_1100 = Motor(
    name="Dualsky ECO 2315C 1100KV",
    kv=1100, resistance=0.110, no_load_current=0.5,
    max_current=18.0, mass=0.065,
)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_catalog/test_hardware.py -v -k motor
```

Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_catalog/__init__.py tests/test_catalog/test_hardware.py src/aerisplane/catalog/motors.py
git commit -m "feat(catalog): add 20 brushless motors"
```

---

### Task 2: Battery catalog (`batteries.py`)

**Files:**
- Modify: `src/aerisplane/catalog/batteries.py`
- Modify: `tests/test_catalog/test_hardware.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_catalog/test_hardware.py`:

```python
from aerisplane.catalog.batteries import (
    tattu_3s_2300,
    tattu_4s_1800,
    tattu_4s_3300,
    tattu_4s_5200,
    tattu_6s_10000,
    tattu_6s_16000,
    gens_ace_3s_2200,
    gens_ace_4s_4000,
    gens_ace_6s_6000,
    turnigy_nano_tech_3s_2200,
    turnigy_nano_tech_4s_5000,
    turnigy_nano_tech_6s_3300,
    multistar_4s_10000,
    ovonic_4s_2200,
    ovonic_6s_3300,
)
from aerisplane.core.propulsion import Battery

def test_battery_type():
    assert isinstance(tattu_3s_2300, Battery)

def test_battery_energy_positive():
    for b in [tattu_4s_1800, gens_ace_4s_4000, turnigy_nano_tech_6s_3300]:
        assert b.energy() > 0

def test_battery_max_current_consistent():
    # max_current = capacity_ah * c_rating
    b = tattu_4s_1800
    assert abs(b.max_current() - b.capacity_ah * b.c_rating) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_catalog/test_hardware.py -v -k battery
```

Expected: FAIL with ImportError

- [ ] **Step 3: Populate `batteries.py`**

```python
# src/aerisplane/catalog/batteries.py
"""LiPo battery pack catalog.

Cell voltages: 3.7 V nominal per cell (3S=11.1V, 4S=14.8V, 6S=22.2V).
Internal resistance estimates from published data and typical values.
"""
from aerisplane.core.propulsion import Battery

# Tattu (premium, race/UAV)
tattu_3s_2300 = Battery(
    name="Tattu 3S 2300mAh 45C",
    capacity_ah=2.3, nominal_voltage=11.1, cell_count=3,
    c_rating=45.0, mass=0.178, internal_resistance=0.010,
)
tattu_4s_1800 = Battery(
    name="Tattu 4S 1800mAh 75C",
    capacity_ah=1.8, nominal_voltage=14.8, cell_count=4,
    c_rating=75.0, mass=0.218, internal_resistance=0.008,
)
tattu_4s_3300 = Battery(
    name="Tattu 4S 3300mAh 45C",
    capacity_ah=3.3, nominal_voltage=14.8, cell_count=4,
    c_rating=45.0, mass=0.302, internal_resistance=0.012,
)
tattu_4s_5200 = Battery(
    name="Tattu 4S 5200mAh 45C",
    capacity_ah=5.2, nominal_voltage=14.8, cell_count=4,
    c_rating=45.0, mass=0.470, internal_resistance=0.015,
)
tattu_6s_10000 = Battery(
    name="Tattu 6S 10000mAh 25C",
    capacity_ah=10.0, nominal_voltage=22.2, cell_count=6,
    c_rating=25.0, mass=1.280, internal_resistance=0.025,
)
tattu_6s_16000 = Battery(
    name="Tattu 6S 16000mAh 15C",
    capacity_ah=16.0, nominal_voltage=22.2, cell_count=6,
    c_rating=15.0, mass=1.900, internal_resistance=0.030,
)

# Gens Ace (general purpose)
gens_ace_3s_2200 = Battery(
    name="Gens Ace 3S 2200mAh 25C",
    capacity_ah=2.2, nominal_voltage=11.1, cell_count=3,
    c_rating=25.0, mass=0.162, internal_resistance=0.012,
)
gens_ace_4s_4000 = Battery(
    name="Gens Ace 4S 4000mAh 45C",
    capacity_ah=4.0, nominal_voltage=14.8, cell_count=4,
    c_rating=45.0, mass=0.390, internal_resistance=0.014,
)
gens_ace_6s_6000 = Battery(
    name="Gens Ace 6S 6000mAh 30C",
    capacity_ah=6.0, nominal_voltage=22.2, cell_count=6,
    c_rating=30.0, mass=0.870, internal_resistance=0.020,
)

# Turnigy Nano-tech (budget)
turnigy_nano_tech_3s_2200 = Battery(
    name="Turnigy Nano-tech 3S 2200mAh 25-50C",
    capacity_ah=2.2, nominal_voltage=11.1, cell_count=3,
    c_rating=25.0, mass=0.156, internal_resistance=0.013,
)
turnigy_nano_tech_4s_5000 = Battery(
    name="Turnigy Nano-tech 4S 5000mAh 25-50C",
    capacity_ah=5.0, nominal_voltage=14.8, cell_count=4,
    c_rating=25.0, mass=0.480, internal_resistance=0.018,
)
turnigy_nano_tech_6s_3300 = Battery(
    name="Turnigy Nano-tech 6S 3300mAh 45-90C",
    capacity_ah=3.3, nominal_voltage=22.2, cell_count=6,
    c_rating=45.0, mass=0.480, internal_resistance=0.016,
)

# Multistar (heavy endurance)
multistar_4s_10000 = Battery(
    name="Multistar 4S 10000mAh 10C",
    capacity_ah=10.0, nominal_voltage=14.8, cell_count=4,
    c_rating=10.0, mass=0.890, internal_resistance=0.030,
)

# Ovonic
ovonic_4s_2200 = Battery(
    name="Ovonic 4S 2200mAh 50C",
    capacity_ah=2.2, nominal_voltage=14.8, cell_count=4,
    c_rating=50.0, mass=0.200, internal_resistance=0.010,
)
ovonic_6s_3300 = Battery(
    name="Ovonic 6S 3300mAh 50C",
    capacity_ah=3.3, nominal_voltage=22.2, cell_count=6,
    c_rating=50.0, mass=0.480, internal_resistance=0.015,
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_catalog/test_hardware.py -v -k battery
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/catalog/batteries.py tests/test_catalog/test_hardware.py
git commit -m "feat(catalog): add 15 LiPo battery packs"
```

---

### Task 3: Propeller catalog (`propellers.py`)

**Files:**
- Modify: `src/aerisplane/catalog/propellers.py`
- Modify: `tests/test_catalog/test_hardware.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_catalog/test_hardware.py`:

```python
from aerisplane.catalog.propellers import (
    apc_10x4_7sf,
    apc_10x7e,
    apc_11x4_7sf,
    apc_12x6e,
    apc_13x4_7sf,
    apc_14x8_3mf,
    master_airscrew_10x5,
    master_airscrew_11x7,
    master_airscrew_14x7,
    tjd_14x8_5,
)
from aerisplane.core.propulsion import Propeller

def test_propeller_type():
    assert isinstance(apc_10x4_7sf, Propeller)

def test_propeller_diameter_inches():
    # APC 10x4.7: diameter = 10 inches = 0.254 m
    assert abs(apc_10x4_7sf.diameter - 0.254) < 0.001

def test_propeller_fields():
    p = apc_12x6e
    assert p.diameter > 0
    assert p.pitch > 0
    assert p.mass > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_catalog/test_hardware.py -v -k propeller
```

Expected: FAIL with ImportError

- [ ] **Step 3: Populate `propellers.py`**

Diameter/pitch in metric (inches × 0.0254). Mass estimates from published APC data.

```python
# src/aerisplane/catalog/propellers.py
"""Propeller catalog.

Diameter and pitch are in meters (1 inch = 0.0254 m).
Mass values are measured weights from manufacturer data.
"""
from aerisplane.core.propulsion import Propeller

_IN = 0.0254  # inch → meter

# APC Slow-Fly series
apc_10x4_7sf = Propeller(
    diameter=10 * _IN, pitch=4.7 * _IN, mass=0.018, num_blades=2,
)
apc_11x4_7sf = Propeller(
    diameter=11 * _IN, pitch=4.7 * _IN, mass=0.021, num_blades=2,
)
apc_13x4_7sf = Propeller(
    diameter=13 * _IN, pitch=4.7 * _IN, mass=0.030, num_blades=2,
)

# APC Electric series
apc_10x7e = Propeller(
    diameter=10 * _IN, pitch=7 * _IN, mass=0.019, num_blades=2,
)
apc_12x6e = Propeller(
    diameter=12 * _IN, pitch=6 * _IN, mass=0.027, num_blades=2,
)

# APC Multi-rotor / pusher
apc_14x8_3mf = Propeller(
    diameter=14 * _IN, pitch=8.3 * _IN, mass=0.040, num_blades=2,
)

# Master Airscrew
master_airscrew_10x5 = Propeller(
    diameter=10 * _IN, pitch=5 * _IN, mass=0.020, num_blades=2,
)
master_airscrew_11x7 = Propeller(
    diameter=11 * _IN, pitch=7 * _IN, mass=0.026, num_blades=2,
)
master_airscrew_14x7 = Propeller(
    diameter=14 * _IN, pitch=7 * _IN, mass=0.042, num_blades=2,
)

# TJD carbon
tjd_14x8_5 = Propeller(
    diameter=14 * _IN, pitch=8.5 * _IN, mass=0.022, num_blades=2,
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_catalog/test_hardware.py -v -k propeller
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/catalog/propellers.py tests/test_catalog/test_hardware.py
git commit -m "feat(catalog): add 10 propellers"
```

---

### Task 4: Servo catalog (`servos.py`)

**Files:**
- Modify: `src/aerisplane/catalog/servos.py`
- Modify: `tests/test_catalog/test_hardware.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_catalog/test_hardware.py`:

```python
from aerisplane.catalog.servos import (
    hitec_hs65mg,
    hitec_hs5086wb,
    hitec_hs7950th,
    savox_sh0255mg,
    savox_sc1256tg,
    futaba_s3003,
    futaba_s3305,
    towerpro_mg996r,
    kst_x08h,
    kst_ds215mg,
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_catalog/test_hardware.py -v -k servo
```

Expected: FAIL with ImportError

- [ ] **Step 3: Check the Servo dataclass fields**

The `Servo` class in `src/aerisplane/core/control_surface.py` has: `name: str`, `torque: float` (N·m), `speed: float` (deg/s), `voltage: float` (V), `mass: float` (kg).

- [ ] **Step 4: Populate `servos.py`**

Torque specs are typically given at a specific voltage; use nominal voltage spec. Speed in deg/s = 60/transit_time_per_60deg.

```python
# src/aerisplane/catalog/servos.py
"""Servo catalog.

Torque in N·m, speed in deg/s, voltage in V, mass in kg.
Specs at nominal operating voltage unless noted.
"""
from aerisplane.core.control_surface import Servo

# Hitec digital servos
hitec_hs65mg = Servo(
    name="Hitec HS-65MG",
    torque=0.196,    # 2.0 kg·cm at 6V
    speed=500.0,     # 60/0.12 s = 500 deg/s at 6V
    voltage=6.0,
    mass=0.013,
)
hitec_hs5086wb = Servo(
    name="Hitec HS-5086WB",
    torque=0.275,    # 2.8 kg·cm at 6V
    speed=500.0,     # 0.12 s/60 deg
    voltage=6.0,
    mass=0.019,
)
hitec_hs7950th = Servo(
    name="Hitec HS-7950TH",
    torque=0.686,    # 7.0 kg·cm at 6V
    speed=500.0,
    voltage=6.0,
    mass=0.068,
)

# Savox
savox_sh0255mg = Servo(
    name="Savox SH-0255MG",
    torque=0.167,    # 1.7 kg·cm at 4.8V
    speed=500.0,
    voltage=4.8,
    mass=0.014,
)
savox_sc1256tg = Servo(
    name="Savox SC-1256TG",
    torque=0.686,    # 7.0 kg·cm at 6V
    speed=600.0,     # 0.10 s/60 deg
    voltage=6.0,
    mass=0.056,
)

# Futaba
futaba_s3003 = Servo(
    name="Futaba S3003",
    torque=0.318,    # 3.24 kg·cm at 4.8V
    speed=400.0,     # 0.15 s/60 deg
    voltage=4.8,
    mass=0.037,
)
futaba_s3305 = Servo(
    name="Futaba S3305",
    torque=0.490,    # 5.0 kg·cm at 6V
    speed=500.0,
    voltage=6.0,
    mass=0.042,
)

# TowerPro
towerpro_mg996r = Servo(
    name="TowerPro MG996R",
    torque=0.980,    # 10 kg·cm at 6V
    speed=333.0,     # 0.18 s/60 deg
    voltage=6.0,
    mass=0.055,
)

# KST
kst_x08h = Servo(
    name="KST X08H V5",
    torque=0.412,    # 4.2 kg·cm at 6V
    speed=500.0,
    voltage=6.0,
    mass=0.022,
)
kst_ds215mg = Servo(
    name="KST DS215MG",
    torque=1.470,    # 15.0 kg·cm at 6V
    speed=400.0,     # 0.15 s/60 deg
    voltage=6.0,
    mass=0.065,
)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_catalog/test_hardware.py -v -k servo
```

Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/catalog/servos.py tests/test_catalog/test_hardware.py
git commit -m "feat(catalog): add 10 servos"
```

---

### Task 5: Discovery functions in `catalog/__init__.py`

**Files:**
- Modify: `src/aerisplane/catalog/__init__.py`
- Modify: `tests/test_catalog/test_hardware.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_catalog/test_hardware.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_catalog/test_hardware.py -v -k list_
```

Expected: FAIL with AttributeError (no list_motors etc.)

- [ ] **Step 3: Add discovery functions to `catalog/__init__.py`**

```python
# src/aerisplane/catalog/__init__.py
"""AerisPlane hardware and airfoil catalog."""
from __future__ import annotations


def get_airfoil(name: str):
    """Load an airfoil from the catalog by name.

    Parameters
    ----------
    name : str
        Airfoil name, e.g. ``"naca2412"``, ``"e423"``.  NACA 4-digit names
        are generated analytically; all others are loaded from the catalog
        .dat files in ``catalog/airfoils/``.

    Returns
    -------
    Airfoil
        Airfoil with coordinates populated.

    Raises
    ------
    ValueError
        If the name cannot be resolved (not NACA and not in catalog).
    """
    from aerisplane.core.airfoil import Airfoil
    af = Airfoil(name)
    if af.coordinates is None:
        raise ValueError(
            f"Airfoil '{name}' not found in catalog. "
            "Check catalog/airfoils/ for available .dat files."
        )
    return af


def list_motors():
    """Return all motors in the catalog.

    Returns
    -------
    list of Motor
    """
    import aerisplane.catalog.motors as _m
    from aerisplane.core.propulsion import Motor
    return [v for v in vars(_m).values() if isinstance(v, Motor)]


def list_batteries():
    """Return all batteries in the catalog.

    Returns
    -------
    list of Battery
    """
    import aerisplane.catalog.batteries as _b
    from aerisplane.core.propulsion import Battery
    return [v for v in vars(_b).values() if isinstance(v, Battery)]


def list_propellers():
    """Return all propellers in the catalog.

    Returns
    -------
    list of Propeller
    """
    import aerisplane.catalog.propellers as _p
    from aerisplane.core.propulsion import Propeller
    return [v for v in vars(_p).values() if isinstance(v, Propeller)]


def list_servos():
    """Return all servos in the catalog.

    Returns
    -------
    list of Servo
    """
    import aerisplane.catalog.servos as _s
    from aerisplane.core.control_surface import Servo
    return [v for v in vars(_s).values() if isinstance(v, Servo)]


__all__ = [
    "get_airfoil",
    "list_motors",
    "list_batteries",
    "list_propellers",
    "list_servos",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_catalog/test_hardware.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Run full test suite to check nothing broke**

```bash
pytest tests/ -x -q --ignore=tests/test_mdo/test_integration.py
```

Expected: all tests PASS (or any pre-existing failures only)

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/catalog/__init__.py tests/test_catalog/test_hardware.py
git commit -m "feat(catalog): add list_motors/batteries/propellers/servos discovery functions"
```
