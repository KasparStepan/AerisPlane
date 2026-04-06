# Flight Performance Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `mission/` module with flight performance envelope analysis — power curves, characteristic speeds, climb rate, glide performance, endurance/range — for small electric RC/UAV aircraft.

**Architecture:** The module has two layers. `performance.py` computes point performance at a single (altitude, speed) condition: power required, power available, characteristic speeds, climb rate, glide ratio. `envelope.py` sweeps across altitudes and speeds to build the full flight envelope. `segments.py` + `__init__.py` compose segments into a mission energy budget. Result dataclasses in `result.py` provide `plot()` and `report()`.

**Tech Stack:** numpy, scipy.optimize (for speed intersections), matplotlib/seaborn (plotting), existing `aero.analyze()` and `core.propulsion` modules.

**Reference document:** `docs/plans/uav-flight-envelope-methods.md` — comprehensive equations and methods.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/aerisplane/mission/performance.py` | Point performance: power required/available at one condition, motor-prop operating point, drag polar fitting |
| `src/aerisplane/mission/envelope.py` | Flight envelope: sweep over altitudes/speeds, compute characteristic speeds, climb rate profile, ceilings |
| `src/aerisplane/mission/segments.py` | Mission segment dataclasses (Climb, Cruise, Loiter, Descent, Return, Mission) |
| `src/aerisplane/mission/result.py` | `PerformanceResult` (single altitude) + `EnvelopeResult` (multi-altitude) + `MissionResult` with `plot()` and `report()` |
| `src/aerisplane/mission/__init__.py` | Entry points: `performance()`, `envelope()`, `analyze()` |
| `tests/test_mission/conftest.py` | Test fixtures: aircraft, flight condition, weight result |
| `tests/test_mission/test_performance.py` | Tests for point performance functions |
| `tests/test_mission/test_envelope.py` | Tests for envelope computation |
| `tests/test_mission/test_mission.py` | Integration tests for mission `analyze()` |

---

## Task 1: Mission segment dataclasses

**Files:**
- Create: `src/aerisplane/mission/segments.py`
- Create: `tests/test_mission/__init__.py`
- Create: `tests/test_mission/test_segments.py`

These are pure data containers with no logic — quick to implement and they define
the vocabulary for the rest of the module.

- [ ] **Step 1: Write segment dataclass tests**

```python
# tests/test_mission/test_segments.py
"""Tests for mission segment dataclasses."""
import pytest
from aerisplane.mission.segments import Climb, Cruise, Loiter, Descent, Return, Mission


class TestSegments:
    def test_climb_defaults(self):
        s = Climb(to_altitude=100.0, climb_rate=2.0, velocity=12.0)
        assert s.name == "climb"
        assert s.to_altitude == 100.0

    def test_cruise_defaults(self):
        s = Cruise(distance=5000.0, velocity=15.0)
        assert s.altitude == 100.0

    def test_loiter_defaults(self):
        s = Loiter(duration=600.0, velocity=12.0)
        assert s.altitude == 100.0

    def test_descent_defaults(self):
        s = Descent(to_altitude=0.0)
        assert s.descent_rate == 2.0
        assert s.velocity == 15.0

    def test_return_segment(self):
        s = Return(distance=5000.0, velocity=15.0)
        assert s.name == "return"

    def test_mission_holds_segments(self):
        m = Mission(segments=[
            Climb(to_altitude=100, climb_rate=2.0, velocity=12.0),
            Cruise(distance=5000, velocity=15.0),
        ])
        assert len(m.segments) == 2
        assert m.start_altitude == 0.0
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `.venv/bin/python -m pytest tests/test_mission/test_segments.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement segment dataclasses**

```python
# src/aerisplane/mission/segments.py
"""Mission segment definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


@dataclass
class Climb:
    """Climbing segment."""
    to_altitude: float      # target altitude [m MSL]
    climb_rate: float       # rate of climb [m/s]
    velocity: float         # true airspeed during climb [m/s]
    name: str = "climb"


@dataclass
class Cruise:
    """Steady level cruise segment."""
    distance: float         # horizontal distance [m]
    velocity: float         # cruise airspeed [m/s]
    altitude: float = 100.0 # cruise altitude [m MSL]
    name: str = "cruise"


@dataclass
class Loiter:
    """Loitering (circling/holding) segment at constant altitude."""
    duration: float         # loiter time [s]
    velocity: float         # loiter airspeed [m/s]
    altitude: float = 100.0 # loiter altitude [m MSL]
    name: str = "loiter"


@dataclass
class Return:
    """Return leg — functionally identical to cruise, semantically distinct."""
    distance: float
    velocity: float
    altitude: float = 100.0
    name: str = "return"


@dataclass
class Descent:
    """Descending segment (partial power or glide)."""
    to_altitude: float          # target altitude [m MSL]
    descent_rate: float = 2.0   # descent rate [m/s] (positive = descending)
    velocity: float = 15.0      # airspeed [m/s]
    name: str = "descent"


@dataclass
class Mission:
    """Ordered sequence of mission segments.

    Segments execute in order. Each segment's start altitude is inherited
    from the previous segment's end altitude (or start_altitude for the first).
    """
    segments: list[Union[Climb, Cruise, Loiter, Return, Descent]]
    start_altitude: float = 0.0
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_mission/test_segments.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/mission/segments.py tests/test_mission/
git commit -m "feat(mission): add mission segment dataclasses"
```

---

## Task 2: Point performance — power required

**Files:**
- Create: `src/aerisplane/mission/performance.py`
- Create: `tests/test_mission/conftest.py`
- Create: `tests/test_mission/test_performance.py`

This task implements the core `power_required` and `drag_at_speed` functions.
These compute P_R(V) = D(V) * V using either the aero solver directly or a
fitted parabolic drag polar.

The **fitted drag polar** approach is important: rather than calling VLM at every
speed point (expensive), we run VLM at 2-3 speeds, fit CD0 and k, then evaluate
the analytic drag polar across the full speed range. This makes envelope sweeps
fast (~3 aero calls total instead of ~50).

**Key equations from the research doc (Section 1.1):**

```
CL = 2W / (rho * V^2 * S)
D(V) = 0.5 * rho * V^2 * S * CD0 + 2 * k * W^2 / (rho * V^2 * S)
P_R(V) = D(V) * V
```

- [ ] **Step 1: Write conftest with aircraft and weight fixtures**

```python
# tests/test_mission/conftest.py
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
```

- [ ] **Step 2: Write failing tests for performance functions**

```python
# tests/test_mission/test_performance.py
"""Tests for point performance functions."""
import numpy as np
import pytest
from aerisplane.mission.performance import (
    fit_drag_polar,
    power_required,
    power_available,
    DragPolar,
)


class TestDragPolar:
    def test_fit_produces_positive_cd0(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        assert polar.CD0 > 0

    def test_fit_produces_positive_k(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        assert polar.k > 0

    def test_ld_max_reasonable(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        ld_max = polar.ld_max()
        assert 5 < ld_max < 40, f"L/D max = {ld_max}, outside 5-40"


class TestPowerRequired:
    def test_positive_at_cruise_speed(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        pr = power_required(15.0, polar, perf_weight_result.total_mass, altitude=0.0)
        assert pr > 0

    def test_u_shaped_curve(self, perf_aircraft, perf_weight_result):
        """Power required should be higher at very low and very high speeds."""
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        mass = perf_weight_result.total_mass
        pr_low = power_required(8.0, polar, mass, altitude=0.0)
        pr_mid = power_required(14.0, polar, mass, altitude=0.0)
        pr_high = power_required(25.0, polar, mass, altitude=0.0)
        assert pr_mid < pr_low, "P_R should decrease from low speed to mid speed"
        assert pr_mid < pr_high, "P_R should increase from mid speed to high speed"


class TestPowerAvailable:
    def test_positive_at_cruise(self, perf_aircraft):
        pa = power_available(perf_aircraft.propulsion, 15.0, altitude=0.0)
        assert pa > 0

    def test_decreases_with_altitude(self, perf_aircraft):
        pa_sl = power_available(perf_aircraft.propulsion, 15.0, altitude=0.0)
        pa_hi = power_available(perf_aircraft.propulsion, 15.0, altitude=2000.0)
        assert pa_hi < pa_sl, "Power available should decrease with altitude"
```

- [ ] **Step 3: Run tests — expect ImportError**

Run: `.venv/bin/python -m pytest tests/test_mission/test_performance.py -v`

- [ ] **Step 4: Implement performance.py**

```python
# src/aerisplane/mission/performance.py
"""Point performance equations for level flight, climb, and glide.

Implements the methods from the UAV flight envelope research document.
All equations assume steady, unaccelerated flight conditions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aerisplane.aero import analyze as aero_analyze
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.propulsion import PropulsionSystem
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult


G = 9.81  # gravitational acceleration [m/s^2]


@dataclass
class DragPolar:
    """Parabolic drag polar: CD = CD0 + k * CL^2.

    Fitted from a small number of aero evaluations at different speeds.

    Parameters
    ----------
    CD0 : float
        Zero-lift drag coefficient.
    k : float
        Induced drag factor: k = 1 / (pi * AR * e).
    S_ref : float
        Reference wing area [m^2].
    """

    CD0: float
    k: float
    S_ref: float

    def cd(self, cl: float) -> float:
        """Total drag coefficient at given CL."""
        return self.CD0 + self.k * cl**2

    def ld_max(self) -> float:
        """Maximum lift-to-drag ratio."""
        return 1.0 / (2.0 * math.sqrt(self.CD0 * self.k))

    def cl_for_ld_max(self) -> float:
        """CL at maximum L/D (minimum drag speed)."""
        return math.sqrt(self.CD0 / self.k)

    def cl_for_min_power(self) -> float:
        """CL at minimum power required (best endurance)."""
        return math.sqrt(3.0 * self.CD0 / self.k)


def fit_drag_polar(
    aircraft: Aircraft,
    weight_result: WeightResult,
    altitude: float = 0.0,
    aero_method: str = "vlm",
    speeds: tuple[float, ...] = (10.0, 15.0, 20.0),
    **aero_kwargs,
) -> DragPolar:
    """Fit a parabolic drag polar from aero evaluations at several speeds.

    Runs the aero solver at each speed, computes CL and CD, then fits
    CD = CD0 + k * CL^2 via least squares.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft definition.
    weight_result : WeightResult
        Weight result for CG (moment reference).
    altitude : float
        Altitude for the analysis [m MSL].
    aero_method : str
        Aero solver method.
    speeds : tuple of float
        Airspeeds to evaluate [m/s]. At least 2 required.
    **aero_kwargs
        Extra kwargs for aero.analyze().

    Returns
    -------
    DragPolar
        Fitted drag polar.
    """
    import copy

    ac = copy.deepcopy(aircraft)
    cg = weight_result.cg
    ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]

    S = aircraft.reference_area()
    W = weight_result.total_mass * G
    _, _, rho, _ = isa(altitude)

    cls = []
    cds = []

    for V in speeds:
        # Compute alpha for level flight: CL_required = 2W / (rho * V^2 * S)
        cl_req = 2.0 * W / (rho * V**2 * S) if (rho * V**2 * S) > 0 else 0.0
        # Estimate alpha from thin-airfoil CL_alpha ~ 2*pi*AR/(AR+2) per radian
        AR = aircraft.main_wing().aspect_ratio() if aircraft.main_wing() else 6.0
        cl_alpha_rad = 2.0 * np.pi * AR / (AR + 2.0)
        alpha_est = np.degrees(cl_req / cl_alpha_rad) if cl_alpha_rad > 0 else 2.0

        cond = FlightCondition(velocity=V, altitude=altitude, alpha=alpha_est)
        result = aero_analyze(ac, cond, method=aero_method, **aero_kwargs)

        cls.append(result.CL)
        cds.append(result.CD)

    cls = np.array(cls)
    cds = np.array(cds)

    # Fit CD = CD0 + k * CL^2 via least squares
    # A @ [CD0, k]^T = cds  where A = [[1, CL1^2], [1, CL2^2], ...]
    A = np.column_stack([np.ones_like(cls), cls**2])
    coeffs, _, _, _ = np.linalg.lstsq(A, cds, rcond=None)
    CD0 = max(coeffs[0], 1e-4)  # enforce positive
    k = max(coeffs[1], 0.01)     # enforce positive

    return DragPolar(CD0=CD0, k=k, S_ref=S)


def power_required(
    velocity: float,
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> float:
    """Power required for steady level flight [W].

    P_R = 0.5 * rho * V^3 * S * CD0 + 2 * k * W^2 / (rho * V * S)

    Parameters
    ----------
    velocity : float
        True airspeed [m/s].
    polar : DragPolar
        Fitted drag polar.
    mass : float
        Aircraft mass [kg].
    altitude : float
        Altitude [m MSL].

    Returns
    -------
    float
        Power required [W].
    """
    _, _, rho, _ = isa(altitude)
    W = mass * G
    S = polar.S_ref

    if velocity <= 0:
        return float("inf")

    # Parasitic + induced power
    P_parasitic = 0.5 * rho * S * polar.CD0 * velocity**3
    P_induced = 2.0 * polar.k * W**2 / (rho * S * velocity)

    return P_parasitic + P_induced


def power_available(
    propulsion: PropulsionSystem,
    velocity: float,
    altitude: float = 0.0,
    throttle: float = 1.0,
) -> float:
    """Power available from the propulsion system [W].

    Computes thrust * velocity at the motor-prop operating point.

    Parameters
    ----------
    propulsion : PropulsionSystem
        Propulsion system.
    velocity : float
        True airspeed [m/s].
    altitude : float
        Altitude [m MSL].
    throttle : float
        Throttle setting 0-1.

    Returns
    -------
    float
        Power available (thrust power) [W].
    """
    _, _, rho, _ = isa(altitude)

    # ESC voltage: throttle * battery voltage
    V_bat = propulsion.battery.voltage_under_load(0.0)  # approximate
    V_esc = throttle * V_bat

    # Motor no-load RPM at this voltage
    rpm_max = propulsion.motor.kv * V_esc
    if rpm_max <= 0:
        return 0.0

    # Find operating RPM: motor torque = prop torque
    # Use max current as upper bound
    max_current = min(
        propulsion.motor.max_current,
        propulsion.esc.max_current,
        propulsion.battery.max_current(),
    )
    V_bat_loaded = propulsion.battery.voltage_under_load(max_current)
    V_esc_loaded = throttle * V_bat_loaded
    rpm_at_max_I = propulsion.motor.rpm(V_esc_loaded, max_current)

    # Thrust at this operating point
    rpm = max(rpm_at_max_I, 0.0)
    thrust = propulsion.propeller.thrust(rpm, velocity, rho)

    return max(thrust * velocity, 0.0)
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_mission/test_performance.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/mission/performance.py tests/test_mission/
git commit -m "feat(mission): add point performance functions (P_R, P_A, drag polar)"
```

---

## Task 3: Characteristic speeds

**Files:**
- Modify: `src/aerisplane/mission/performance.py`
- Modify: `tests/test_mission/test_performance.py`

Add functions to compute the key characteristic speeds from the drag polar:
stall speed, best endurance speed (V_mp), best range speed (V*), and max speed.

**Key equations (research doc Sections 2.1-2.4):**

```
V_stall = sqrt(2W / (rho * S * CL_max))
V* = sqrt(2W / (rho * S)) * (k / CD0)^(1/4)          # best range / max L/D
V_mp = 3^(-1/4) * V* ≈ 0.76 * V*                      # best endurance
V_max = largest V where P_A(V) >= P_R(V)                # numerical scan
```

- [ ] **Step 1: Write failing tests for characteristic speeds**

```python
# Append to tests/test_mission/test_performance.py
from aerisplane.mission.performance import (
    stall_speed,
    best_range_speed,
    best_endurance_speed,
    max_level_speed,
)


class TestCharacteristicSpeeds:
    def test_stall_speed_positive(self, perf_weight_result):
        vs = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=0.0)
        assert vs > 0

    def test_stall_speed_reasonable(self, perf_weight_result):
        """Stall speed for a ~2kg aircraft should be 5-15 m/s."""
        vs = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=0.0)
        assert 3 < vs < 20, f"V_stall = {vs:.1f} m/s"

    def test_stall_increases_with_altitude(self, perf_weight_result):
        vs_sl = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=0.0)
        vs_hi = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=2000.0)
        assert vs_hi > vs_sl

    def test_best_range_speed(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        vr = best_range_speed(polar, perf_weight_result.total_mass, altitude=0.0)
        assert vr > 0

    def test_best_endurance_slower_than_range(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        mass = perf_weight_result.total_mass
        ve = best_endurance_speed(polar, mass, altitude=0.0)
        vr = best_range_speed(polar, mass, altitude=0.0)
        assert ve < vr, "V_endurance should be slower than V_range"

    def test_max_speed_above_cruise(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        vmax = max_level_speed(polar, perf_weight_result.total_mass,
                               perf_aircraft.propulsion, altitude=0.0)
        if vmax is not None:
            assert vmax > 10.0
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement characteristic speed functions**

Add to `performance.py`:

```python
def stall_speed(
    mass: float,
    S: float,
    CL_max: float = 1.4,
    altitude: float = 0.0,
) -> float:
    """1-g stall speed [m/s].

    V_stall = sqrt(2W / (rho * S * CL_max))
    """
    _, _, rho, _ = isa(altitude)
    W = mass * G
    return math.sqrt(2.0 * W / (rho * S * CL_max))


def best_range_speed(
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> float:
    """Speed for maximum L/D (best range) [m/s].

    V* = sqrt(2W / (rho * S * CL*))  where CL* = sqrt(CD0 / k)
    """
    _, _, rho, _ = isa(altitude)
    W = mass * G
    cl_star = polar.cl_for_ld_max()
    return math.sqrt(2.0 * W / (rho * polar.S_ref * cl_star))


def best_endurance_speed(
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> float:
    """Speed for minimum power required (best endurance) [m/s].

    V_mp = 3^(-1/4) * V*  ≈ 0.76 * V*
    """
    v_star = best_range_speed(polar, mass, altitude)
    return v_star * 3.0**(-0.25)


def max_level_speed(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    v_max_search: float = 60.0,
    dv: float = 0.5,
) -> float | None:
    """Maximum level flight speed [m/s].

    Scans speed range to find largest V where P_A >= P_R.
    Returns None if P_A < P_R everywhere (cannot fly).
    """
    v_max_found = None

    for v_int in range(int(5 / dv), int(v_max_search / dv)):
        v = v_int * dv
        pr = power_required(v, polar, mass, altitude)
        pa = power_available(propulsion, v, altitude)
        if pa >= pr:
            v_max_found = v

    return v_max_found
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/mission/performance.py tests/test_mission/test_performance.py
git commit -m "feat(mission): add characteristic speed functions"
```

---

## Task 4: Climb rate, ceilings, and glide performance

**Files:**
- Modify: `src/aerisplane/mission/performance.py`
- Modify: `tests/test_mission/test_performance.py`

**Key equations (research doc Sections 3 and 4):**

```
ROC(V) = (P_A(V) - P_R(V)) / W                         # excess power method
V_best_glide = V* = V_LD_max                             # same as best range speed
Glide ratio = L/D_max
Glide range = (L/D)_max * altitude
V_min_sink = V_mp ≈ 0.76 * V*                            # same as best endurance speed
```

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_mission/test_performance.py
from aerisplane.mission.performance import (
    rate_of_climb,
    max_rate_of_climb,
    glide_range,
    GlidePerformance,
    glide_performance,
)


class TestClimb:
    def test_roc_positive_at_cruise(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        roc = rate_of_climb(15.0, polar, perf_weight_result.total_mass,
                            perf_aircraft.propulsion, altitude=0.0)
        assert roc > 0, f"ROC = {roc:.2f} m/s, expected positive"

    def test_max_roc_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        roc_max, v_y = max_rate_of_climb(
            polar, perf_weight_result.total_mass,
            perf_aircraft.propulsion, altitude=0.0
        )
        assert roc_max > 0
        assert v_y > 0

    def test_roc_decreases_with_altitude(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        mass = perf_weight_result.total_mass
        prop = perf_aircraft.propulsion
        roc_sl, _ = max_rate_of_climb(polar, mass, prop, altitude=0.0)
        roc_hi, _ = max_rate_of_climb(polar, mass, prop, altitude=2000.0)
        assert roc_hi < roc_sl


class TestGlide:
    def test_glide_range_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        r = glide_range(polar, from_altitude=100.0)
        assert r > 0

    def test_glide_performance(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        gp = glide_performance(polar, perf_weight_result.total_mass, altitude=0.0)
        assert gp.best_glide_ratio > 0
        assert gp.best_glide_speed > 0
        assert gp.min_sink_speed > 0
        assert gp.min_sink_rate > 0
        assert gp.min_sink_speed < gp.best_glide_speed
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement climb and glide functions**

Add to `performance.py`:

```python
@dataclass
class GlidePerformance:
    """Glide performance summary."""
    best_glide_ratio: float    # L/D max
    best_glide_speed: float    # m/s
    min_sink_speed: float      # m/s (speed for minimum sink rate)
    min_sink_rate: float       # m/s (vertical speed at min sink speed)


def rate_of_climb(
    velocity: float,
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
) -> float:
    """Rate of climb at given speed [m/s].

    ROC = (P_A - P_R) / W
    """
    W = mass * G
    if W <= 0:
        return 0.0
    pr = power_required(velocity, polar, mass, altitude)
    pa = power_available(propulsion, velocity, altitude)
    return (pa - pr) / W


def max_rate_of_climb(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    v_min: float = 5.0,
    v_max: float = 40.0,
    dv: float = 0.5,
) -> tuple[float, float]:
    """Maximum rate of climb and speed for best climb [m/s].

    Returns (ROC_max, V_y).
    """
    best_roc = -float("inf")
    best_v = v_min

    for v_int in range(int(v_min / dv), int(v_max / dv)):
        v = v_int * dv
        roc = rate_of_climb(v, polar, mass, propulsion, altitude)
        if roc > best_roc:
            best_roc = roc
            best_v = v

    return best_roc, best_v


def glide_range(
    polar: DragPolar,
    from_altitude: float,
) -> float:
    """Still-air glide range from given altitude [m].

    R = (L/D)_max * altitude
    """
    return polar.ld_max() * from_altitude


def glide_performance(
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> GlidePerformance:
    """Compute glide performance summary."""
    ld_max = polar.ld_max()
    v_bg = best_range_speed(polar, mass, altitude)
    v_ms = best_endurance_speed(polar, mass, altitude)

    # Min sink rate: P_R_min / W (sink rate = D*V / W at V_mp)
    pr_min = power_required(v_ms, polar, mass, altitude)
    W = mass * G
    min_sink = pr_min / W if W > 0 else 0.0

    return GlidePerformance(
        best_glide_ratio=ld_max,
        best_glide_speed=v_bg,
        min_sink_speed=v_ms,
        min_sink_rate=min_sink,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/mission/performance.py tests/test_mission/test_performance.py
git commit -m "feat(mission): add climb rate, ceilings, and glide performance"
```

---

## Task 5: Endurance and range

**Files:**
- Modify: `src/aerisplane/mission/performance.py`
- Modify: `tests/test_mission/test_performance.py`

**Key equations (research doc Section 5):**

```
Endurance = E_battery * eta / P_R_min                    # at best endurance speed
Range = E_battery * eta * (L/D)_max / W                  # at best range speed
```

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_mission/test_performance.py
from aerisplane.mission.performance import max_endurance, max_range


class TestEnduranceRange:
    def test_endurance_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        e = max_endurance(polar, perf_weight_result.total_mass,
                          perf_aircraft.propulsion, altitude=0.0)
        assert e > 0

    def test_endurance_reasonable(self, perf_aircraft, perf_weight_result):
        """Endurance for a 2.2Ah 4S battery at ~2kg should be 5-60 minutes."""
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        e = max_endurance(polar, perf_weight_result.total_mass,
                          perf_aircraft.propulsion, altitude=0.0)
        assert 60 < e < 7200, f"Endurance = {e:.0f}s ({e/60:.1f} min)"

    def test_range_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        r = max_range(polar, perf_weight_result.total_mass,
                      perf_aircraft.propulsion, altitude=0.0)
        assert r > 0

    def test_range_reasonable(self, perf_aircraft, perf_weight_result):
        """Range should be 1-50 km for a typical RC aircraft."""
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        r = max_range(polar, perf_weight_result.total_mass,
                      perf_aircraft.propulsion, altitude=0.0)
        assert 500 < r < 100_000, f"Range = {r:.0f}m ({r/1000:.1f} km)"
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement endurance and range**

Add to `performance.py`:

```python
def max_endurance(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    eta_total: float | None = None,
) -> float:
    """Maximum endurance at best endurance speed [seconds].

    E = E_battery * eta / P_R_min

    Parameters
    ----------
    eta_total : float or None
        Total propulsive efficiency. If None, estimated from motor and prop.
    """
    v_mp = best_endurance_speed(polar, mass, altitude)
    pr_min = power_required(v_mp, polar, mass, altitude)

    if eta_total is None:
        eta_total = _estimate_efficiency(propulsion, v_mp, altitude)

    E_bat = propulsion.battery.energy()

    if pr_min <= 0:
        return float("inf")

    return E_bat * eta_total / pr_min


def max_range(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    eta_total: float | None = None,
) -> float:
    """Maximum range at best range speed [meters].

    R = E_battery * eta * (L/D)_max / W
    """
    W = mass * G

    if eta_total is None:
        v_star = best_range_speed(polar, mass, altitude)
        eta_total = _estimate_efficiency(propulsion, v_star, altitude)

    E_bat = propulsion.battery.energy()
    ld_max = polar.ld_max()

    if W <= 0:
        return 0.0

    return E_bat * eta_total * ld_max / W


def _estimate_efficiency(
    propulsion: PropulsionSystem,
    velocity: float,
    altitude: float,
) -> float:
    """Estimate total propulsive efficiency (motor * prop).

    Uses the propulsion model at the given operating point.
    Falls back to 0.5 if computation fails.
    """
    _, _, rho, _ = isa(altitude)

    # Get operating RPM at max throttle
    V_bat = propulsion.battery.voltage_under_load(0.0)
    rpm = propulsion.motor.kv * V_bat

    if rpm <= 0 or velocity <= 0:
        return 0.5

    # Prop efficiency
    eta_prop = propulsion.propeller.efficiency(rpm, velocity, rho)

    # Motor efficiency (approximate at mid current)
    mid_current = 0.5 * propulsion.motor.max_current
    eta_motor = propulsion.motor.efficiency(V_bat, mid_current)

    eta_total = eta_prop * eta_motor

    # Clamp to reasonable range
    return max(min(eta_total, 0.85), 0.1) if eta_total > 0 else 0.5
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/mission/performance.py tests/test_mission/test_performance.py
git commit -m "feat(mission): add endurance and range calculations"
```

---

## Task 6: Flight envelope (multi-altitude sweep)

**Files:**
- Create: `src/aerisplane/mission/envelope.py`
- Create: `tests/test_mission/test_envelope.py`

Sweeps performance across altitudes to build the full flight envelope:
speed limits vs altitude, climb rate vs altitude, service/absolute ceiling.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mission/test_envelope.py
"""Tests for flight envelope computation."""
import numpy as np
import pytest
from aerisplane.mission.envelope import compute_envelope, EnvelopeResult


class TestEnvelope:
    def test_returns_envelope_result(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert isinstance(env, EnvelopeResult)

    def test_altitudes_ascending(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert all(env.altitudes[i] <= env.altitudes[i+1]
                   for i in range(len(env.altitudes) - 1))

    def test_stall_speed_increases(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert env.stall_speeds[-1] > env.stall_speeds[0]

    def test_max_roc_decreases(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert env.max_rocs[0] > env.max_rocs[-1]

    def test_report_non_empty(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert len(env.report()) > 100

    def test_sea_level_summary(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert env.ld_max > 0
        assert env.endurance_s > 0
        assert env.range_m > 0
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement envelope.py**

```python
# src/aerisplane/mission/envelope.py
"""Flight envelope computation across altitudes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.mission.performance import (
    DragPolar,
    best_endurance_speed,
    best_range_speed,
    fit_drag_polar,
    glide_performance,
    max_endurance,
    max_level_speed,
    max_range,
    max_rate_of_climb,
    power_available,
    power_required,
    stall_speed,
)
from aerisplane.weights.result import WeightResult


@dataclass
class EnvelopeResult:
    """Flight envelope across altitudes.

    All arrays are indexed by altitude.
    """

    altitudes: np.ndarray           # [m]
    stall_speeds: np.ndarray        # [m/s]
    best_endurance_speeds: np.ndarray
    best_range_speeds: np.ndarray
    max_speeds: np.ndarray          # [m/s], NaN where flight impossible
    max_rocs: np.ndarray            # [m/s]
    best_climb_speeds: np.ndarray   # V_y [m/s]

    # Scalar summaries (at sea level / reference altitude)
    ld_max: float
    endurance_s: float              # max endurance [s]
    range_m: float                  # max range [m]
    service_ceiling: float          # altitude [m] where ROC = 0.5 m/s
    absolute_ceiling: float         # altitude [m] where ROC = 0

    # Glide
    best_glide_ratio: float
    best_glide_speed: float         # at reference altitude [m/s]
    min_sink_rate: float            # [m/s]

    # Drag polar at reference altitude
    polar: DragPolar

    # Power curve data at reference altitude (for plotting)
    _v_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    _pr_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    _pa_grid: np.ndarray = field(default_factory=lambda: np.array([]))

    def report(self) -> str:
        lines = []
        lines.append("AerisPlane Flight Performance Envelope")
        lines.append("=" * 60)

        lines.append("")
        lines.append("Drag Polar")
        lines.append("-" * 40)
        lines.append(f"  CD0             {self.polar.CD0:>10.5f}")
        lines.append(f"  k               {self.polar.k:>10.5f}")
        lines.append(f"  L/D max         {self.ld_max:>10.1f}")

        lines.append("")
        lines.append("Characteristic Speeds (sea level)")
        lines.append("-" * 40)
        lines.append(f"  Stall           {self.stall_speeds[0]:>10.1f}  m/s")
        lines.append(f"  Best endurance  {self.best_endurance_speeds[0]:>10.1f}  m/s")
        lines.append(f"  Best range      {self.best_range_speeds[0]:>10.1f}  m/s")
        v_max_sl = self.max_speeds[0]
        if np.isnan(v_max_sl):
            lines.append(f"  Max speed            N/A")
        else:
            lines.append(f"  Max speed       {v_max_sl:>10.1f}  m/s")

        lines.append("")
        lines.append("Climb Performance")
        lines.append("-" * 40)
        lines.append(f"  Max ROC (SL)    {self.max_rocs[0]:>10.1f}  m/s")
        lines.append(f"  V_y (SL)        {self.best_climb_speeds[0]:>10.1f}  m/s")
        lines.append(f"  Service ceiling {self.service_ceiling:>10.0f}  m")
        lines.append(f"  Absolute ceiling{self.absolute_ceiling:>10.0f}  m")

        lines.append("")
        lines.append("Glide Performance")
        lines.append("-" * 40)
        lines.append(f"  Best glide L/D  {self.best_glide_ratio:>10.1f}")
        lines.append(f"  Best glide V    {self.best_glide_speed:>10.1f}  m/s")
        lines.append(f"  Min sink rate   {self.min_sink_rate:>10.2f}  m/s")

        lines.append("")
        lines.append("Endurance & Range")
        lines.append("-" * 40)
        lines.append(f"  Max endurance   {self.endurance_s / 60:>10.1f}  min")
        lines.append(f"  Max range       {self.range_m / 1000:>10.1f}  km")

        return "\n".join(lines)

    def plot(self):
        """Flight envelope summary: 4 subplots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- Top-left: Power curves at sea level ---
        ax = axes[0, 0]
        if len(self._v_grid) > 0:
            ax.plot(self._v_grid, self._pr_grid, label="P required", color=PALETTE[0], lw=2)
            ax.plot(self._v_grid, self._pa_grid, label="P available", color=PALETTE[1], lw=2)
            ax.set_xlabel("Airspeed [m/s]")
            ax.set_ylabel("Power [W]")
            ax.set_title("Power Curves (Sea Level)", fontweight="bold")
            ax.legend()
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        # --- Top-right: Speed envelope vs altitude ---
        ax = axes[0, 1]
        ax.plot(self.stall_speeds, self.altitudes, label="Stall", color=PALETTE[3], lw=2)
        valid_max = ~np.isnan(self.max_speeds)
        if valid_max.any():
            ax.plot(self.max_speeds[valid_max], self.altitudes[valid_max],
                    label="Max speed", color=PALETTE[1], lw=2)
        ax.fill_betweenx(self.altitudes, self.stall_speeds,
                         np.where(valid_max, self.max_speeds, self.stall_speeds),
                         alpha=0.15, color=PALETTE[0], label="Flyable region")
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title("Speed Envelope", fontweight="bold")
        ax.legend(fontsize=8)

        # --- Bottom-left: ROC vs altitude ---
        ax = axes[1, 0]
        ax.plot(self.max_rocs, self.altitudes, color=PALETTE[2], lw=2)
        ax.axvline(0.5, color=PALETTE[3], ls="--", lw=1, label="Service ceiling (0.5 m/s)")
        ax.axvline(0, color="gray", ls="-", lw=0.8)
        ax.set_xlabel("Max Rate of Climb [m/s]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title("Climb Performance", fontweight="bold")
        ax.legend(fontsize=8)

        # --- Bottom-right: Summary text ---
        ax = axes[1, 1]
        ax.axis("off")
        summary = (
            f"L/D max: {self.ld_max:.1f}\n"
            f"Stall (SL): {self.stall_speeds[0]:.1f} m/s\n"
            f"Best range V: {self.best_range_speeds[0]:.1f} m/s\n"
            f"Best endurance V: {self.best_endurance_speeds[0]:.1f} m/s\n"
            f"Max ROC (SL): {self.max_rocs[0]:.1f} m/s\n"
            f"Service ceiling: {self.service_ceiling:.0f} m\n"
            f"Best glide ratio: {self.best_glide_ratio:.1f}\n"
            f"Min sink: {self.min_sink_rate:.2f} m/s\n"
            f"Endurance: {self.endurance_s / 60:.1f} min\n"
            f"Range: {self.range_m / 1000:.1f} km"
        )
        ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=12,
                verticalalignment="center", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
        ax.set_title("Summary", fontweight="bold")

        fig.suptitle("Flight Performance Envelope", fontsize=14, fontweight="bold")
        fig.tight_layout(pad=1.0)
        return fig


def compute_envelope(
    aircraft: Aircraft,
    weight_result: WeightResult,
    CL_max: float = 1.4,
    altitudes: np.ndarray | None = None,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> EnvelopeResult:
    """Compute flight performance envelope across altitudes.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft definition (must have propulsion system).
    weight_result : WeightResult
        Weight analysis result.
    CL_max : float
        Maximum lift coefficient for stall speed estimation.
    altitudes : array or None
        Altitude array [m]. Default: 0 to 3000 m in 200 m steps.
    aero_method : str
        Aero solver for drag polar fitting.
    """
    if altitudes is None:
        altitudes = np.arange(0, 3200, 200, dtype=float)

    mass = weight_result.total_mass
    S = aircraft.reference_area()
    prop = aircraft.propulsion

    # Fit drag polar at sea level (assume weak altitude dependence for CD0, k)
    polar = fit_drag_polar(
        aircraft, weight_result, altitude=0.0,
        aero_method=aero_method, **aero_kwargs,
    )

    n_alt = len(altitudes)
    stall_speeds_arr = np.zeros(n_alt)
    ve_arr = np.zeros(n_alt)
    vr_arr = np.zeros(n_alt)
    vmax_arr = np.full(n_alt, np.nan)
    roc_arr = np.zeros(n_alt)
    vy_arr = np.zeros(n_alt)

    for i, alt in enumerate(altitudes):
        stall_speeds_arr[i] = stall_speed(mass, S, CL_max, alt)
        ve_arr[i] = best_endurance_speed(polar, mass, alt)
        vr_arr[i] = best_range_speed(polar, mass, alt)

        if prop is not None:
            vm = max_level_speed(polar, mass, prop, alt)
            vmax_arr[i] = vm if vm is not None else np.nan
            roc_max, v_y = max_rate_of_climb(polar, mass, prop, alt)
            roc_arr[i] = roc_max
            vy_arr[i] = v_y

    # Ceilings
    service_ceiling = float(altitudes[-1])
    absolute_ceiling = float(altitudes[-1])
    for i, alt in enumerate(altitudes):
        if roc_arr[i] <= 0.5 and i > 0:
            # Interpolate
            if roc_arr[i - 1] > 0.5:
                frac = (roc_arr[i - 1] - 0.5) / (roc_arr[i - 1] - roc_arr[i])
                service_ceiling = altitudes[i - 1] + frac * (alt - altitudes[i - 1])
            break
    else:
        service_ceiling = float(altitudes[-1])

    for i, alt in enumerate(altitudes):
        if roc_arr[i] <= 0 and i > 0:
            frac = roc_arr[i - 1] / (roc_arr[i - 1] - roc_arr[i])
            absolute_ceiling = altitudes[i - 1] + frac * (alt - altitudes[i - 1])
            break
    else:
        absolute_ceiling = float(altitudes[-1])

    # Glide and endurance/range at sea level
    gp = glide_performance(polar, mass, altitude=0.0)

    endurance = 0.0
    range_m = 0.0
    if prop is not None:
        endurance = max_endurance(polar, mass, prop, altitude=0.0)
        range_m = max_range(polar, mass, prop, altitude=0.0)

    # Power curve data for plotting
    v_grid = np.linspace(max(stall_speeds_arr[0] * 0.8, 3.0), 40.0, 80)
    pr_grid = np.array([power_required(v, polar, mass, 0.0) for v in v_grid])
    pa_grid = np.zeros_like(v_grid)
    if prop is not None:
        pa_grid = np.array([power_available(prop, v, 0.0) for v in v_grid])

    return EnvelopeResult(
        altitudes=altitudes,
        stall_speeds=stall_speeds_arr,
        best_endurance_speeds=ve_arr,
        best_range_speeds=vr_arr,
        max_speeds=vmax_arr,
        max_rocs=roc_arr,
        best_climb_speeds=vy_arr,
        ld_max=polar.ld_max(),
        endurance_s=endurance,
        range_m=range_m,
        service_ceiling=service_ceiling,
        absolute_ceiling=absolute_ceiling,
        best_glide_ratio=gp.best_glide_ratio,
        best_glide_speed=gp.best_glide_speed,
        min_sink_rate=gp.min_sink_rate,
        polar=polar,
        _v_grid=v_grid,
        _pr_grid=pr_grid,
        _pa_grid=pa_grid,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/mission/envelope.py tests/test_mission/test_envelope.py
git commit -m "feat(mission): add flight envelope computation across altitudes"
```

---

## Task 7: Mission segment analysis and result

**Files:**
- Create: `src/aerisplane/mission/result.py`
- Modify: `src/aerisplane/mission/__init__.py`
- Create: `tests/test_mission/test_mission.py`

This wires segments into an energy budget. For each segment, compute power
required, duration, distance, energy consumed. Sum up and compare to battery.

- [ ] **Step 1: Write failing tests for mission analysis**

```python
# tests/test_mission/test_mission.py
"""Integration tests for mission analyze()."""
import pytest
from aerisplane.mission import analyze, performance as perf_fn, envelope as env_fn
from aerisplane.mission.segments import Mission, Climb, Cruise, Loiter, Descent
from aerisplane.mission.result import MissionResult, SegmentResult


class TestMissionAnalyze:
    def test_returns_mission_result(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Climb(to_altitude=100, climb_rate=2.0, velocity=12.0),
            Cruise(distance=3000, velocity=15.0, altitude=100.0),
            Descent(to_altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert isinstance(result, MissionResult)

    def test_total_energy_positive(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=3000, velocity=15.0, altitude=100.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert result.total_energy > 0

    def test_feasible_short_mission(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=1000, velocity=15.0, altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert result.feasible

    def test_segment_count_matches(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Climb(to_altitude=100, climb_rate=2.0, velocity=12.0),
            Cruise(distance=3000, velocity=15.0, altitude=100.0),
            Descent(to_altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert len(result.segments) == 3

    def test_report_non_empty(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=3000, velocity=15.0, altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert len(result.report()) > 100

    def test_energy_margin_between_0_and_1(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=1000, velocity=15.0, altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert 0 <= result.energy_margin <= 1.0
```

- [ ] **Step 2: Run tests — expect ImportError**

- [ ] **Step 3: Implement result.py**

```python
# src/aerisplane/mission/result.py
"""Mission analysis result dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SegmentResult:
    """Energy budget for one mission segment."""
    name: str
    duration: float         # seconds
    distance: float         # meters (horizontal)
    energy: float           # Joules consumed
    avg_power: float        # Watts (average)
    avg_speed: float        # m/s
    altitude_start: float   # m
    altitude_end: float     # m


@dataclass
class MissionResult:
    """Complete mission energy budget result."""
    total_energy: float             # Joules consumed
    total_time: float               # seconds
    total_distance: float           # meters
    battery_energy_available: float # Joules
    energy_margin: float            # fraction remaining (0=empty, 1=full)
    feasible: bool                  # enough battery for the mission
    segments: list[SegmentResult]

    def report(self) -> str:
        lines = []
        lines.append("AerisPlane Mission Analysis")
        lines.append("=" * 75)
        lines.append("")

        header = (
            f"{'Segment':<16} {'Duration':>8} {'Distance':>9} "
            f"{'Energy':>9} {'Avg Power':>10} {'Alt':>10}"
        )
        lines.append(header)
        lines.append("-" * 75)

        for seg in self.segments:
            alt_str = f"{seg.altitude_start:.0f}->{seg.altitude_end:.0f}m"
            lines.append(
                f"{seg.name:<16} {seg.duration:>7.0f}s {seg.distance:>8.0f}m "
                f"{seg.energy / 3600:>8.1f}Wh {seg.avg_power:>9.1f}W {alt_str:>10}"
            )

        lines.append("-" * 75)
        lines.append(
            f"{'TOTAL':<16} {self.total_time:>7.0f}s {self.total_distance:>8.0f}m "
            f"{self.total_energy / 3600:>8.1f}Wh"
        )

        lines.append("")
        lines.append(f"Battery energy:   {self.battery_energy_available / 3600:.1f} Wh")
        lines.append(f"Energy used:      {self.total_energy / 3600:.1f} Wh")
        lines.append(f"Energy margin:    {self.energy_margin * 100:.1f}%")
        lines.append(f"Mission time:     {self.total_time / 60:.1f} min")
        status = "FEASIBLE" if self.feasible else "NOT FEASIBLE"
        lines.append(f"Status:           {status}")

        return "\n".join(lines)

    def plot(self):
        """Mission profile: energy budget bars + altitude profile."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, (ax_energy, ax_alt) = plt.subplots(1, 2, figsize=(14, 6))

        # Energy budget bars
        names = [s.name for s in self.segments]
        energies_wh = [s.energy / 3600 for s in self.segments]
        colors = sns.color_palette("husl", n_colors=max(len(self.segments), 1))

        bars = ax_energy.barh(names, energies_wh, color=colors, edgecolor="white")
        for bar, wh in zip(bars, energies_wh):
            ax_energy.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                          f"{wh:.1f} Wh", va="center", fontsize=9)
        ax_energy.set_xlabel("Energy [Wh]")
        ax_energy.set_title("Energy Budget", fontweight="bold")

        # Altitude profile
        t_cumul = 0.0
        for seg in self.segments:
            t_start = t_cumul
            t_end = t_cumul + seg.duration
            ax_alt.plot([t_start / 60, t_end / 60],
                       [seg.altitude_start, seg.altitude_end],
                       "o-", color=PALETTE[0], lw=2, markersize=4)
            ax_alt.text((t_start + t_end) / 2 / 60, (seg.altitude_start + seg.altitude_end) / 2,
                       seg.name, fontsize=8, ha="center", va="bottom")
            t_cumul = t_end

        ax_alt.set_xlabel("Time [min]")
        ax_alt.set_ylabel("Altitude [m]")
        ax_alt.set_title("Altitude Profile", fontweight="bold")
        ax_alt.set_ylim(bottom=-10)

        fig.suptitle(
            f"Mission: {self.total_time/60:.1f} min, "
            f"{self.total_energy/3600:.1f}/{self.battery_energy_available/3600:.1f} Wh "
            f"({'FEASIBLE' if self.feasible else 'INFEASIBLE'})",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(pad=1.0)
        return fig
```

- [ ] **Step 4: Implement mission __init__.py**

```python
# src/aerisplane/mission/__init__.py
"""Mission analysis module.

Public API
----------
performance(aircraft, weight_result, altitude=0.0) -> DragPolar
    Fit a drag polar at the given altitude.

envelope(aircraft, weight_result, CL_max=1.4) -> EnvelopeResult
    Compute full flight performance envelope.

analyze(aircraft, weight_result, mission) -> MissionResult
    Run mission energy budget analysis.
"""
from __future__ import annotations

from aerisplane.core.aircraft import Aircraft
from aerisplane.mission.envelope import EnvelopeResult, compute_envelope
from aerisplane.mission.performance import (
    DragPolar,
    fit_drag_polar,
    power_required,
    _estimate_efficiency,
)
from aerisplane.mission.result import MissionResult, SegmentResult
from aerisplane.mission.segments import (
    Climb, Cruise, Descent, Loiter, Mission, Return,
)
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult

G = 9.81


def performance(
    aircraft: Aircraft,
    weight_result: WeightResult,
    altitude: float = 0.0,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> DragPolar:
    """Fit a drag polar at the given altitude."""
    return fit_drag_polar(aircraft, weight_result, altitude, aero_method, **aero_kwargs)


def envelope(
    aircraft: Aircraft,
    weight_result: WeightResult,
    CL_max: float = 1.4,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> EnvelopeResult:
    """Compute full flight performance envelope."""
    return compute_envelope(
        aircraft, weight_result, CL_max=CL_max,
        aero_method=aero_method, **aero_kwargs,
    )


def analyze(
    aircraft: Aircraft,
    weight_result: WeightResult,
    mission: Mission,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> MissionResult:
    """Run mission energy budget analysis.

    For each segment, computes duration, distance, and energy required.
    Compares total energy to battery capacity.
    """
    mass = weight_result.total_mass
    W = mass * G
    prop = aircraft.propulsion

    # Fit drag polar once at representative altitude
    polar = fit_drag_polar(aircraft, weight_result, altitude=0.0,
                           aero_method=aero_method, **aero_kwargs)

    E_battery = prop.battery.energy() if prop else 0.0

    seg_results = []
    alt_current = mission.start_altitude
    total_energy = 0.0
    total_time = 0.0
    total_distance = 0.0

    for seg in mission.segments:
        sr = _analyze_segment(seg, alt_current, polar, mass, prop)
        seg_results.append(sr)
        total_energy += sr.energy
        total_time += sr.duration
        total_distance += sr.distance
        alt_current = sr.altitude_end

    energy_margin = max(0.0, 1.0 - total_energy / E_battery) if E_battery > 0 else 0.0
    feasible = total_energy <= E_battery

    return MissionResult(
        total_energy=total_energy,
        total_time=total_time,
        total_distance=total_distance,
        battery_energy_available=E_battery,
        energy_margin=energy_margin,
        feasible=feasible,
        segments=seg_results,
    )


def _analyze_segment(seg, alt_start, polar, mass, prop):
    """Compute energy for a single mission segment."""
    W = mass * G

    if isinstance(seg, Climb):
        alt_end = seg.to_altitude
        dh = alt_end - alt_start
        duration = abs(dh) / seg.climb_rate if seg.climb_rate > 0 else 0.0
        distance = seg.velocity * duration

        # Power = D*V + W*climb_rate
        pr_level = power_required(seg.velocity, polar, mass, alt_start)
        pr_climb = pr_level + W * seg.climb_rate

        eta = _estimate_efficiency(prop, seg.velocity, alt_start) if prop else 0.5
        p_elec = pr_climb / eta if eta > 0 else pr_climb
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    elif isinstance(seg, (Cruise, Return)):
        alt_end = seg.altitude
        duration = seg.distance / seg.velocity if seg.velocity > 0 else 0.0
        distance = seg.distance

        pr = power_required(seg.velocity, polar, mass, seg.altitude)
        eta = _estimate_efficiency(prop, seg.velocity, seg.altitude) if prop else 0.5
        p_elec = pr / eta if eta > 0 else pr
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    elif isinstance(seg, Loiter):
        alt_end = seg.altitude
        duration = seg.duration
        distance = seg.velocity * duration

        pr = power_required(seg.velocity, polar, mass, seg.altitude)
        eta = _estimate_efficiency(prop, seg.velocity, seg.altitude) if prop else 0.5
        p_elec = pr / eta if eta > 0 else pr
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    elif isinstance(seg, Descent):
        alt_end = seg.to_altitude
        dh = alt_start - alt_end
        duration = abs(dh) / seg.descent_rate if seg.descent_rate > 0 else 0.0
        distance = seg.velocity * duration

        # Partial power descent: assume 50% of level power
        pr = power_required(seg.velocity, polar, mass, alt_start) * 0.5
        eta = _estimate_efficiency(prop, seg.velocity, alt_start) if prop else 0.5
        p_elec = pr / eta if eta > 0 else pr
        energy = p_elec * duration

        return SegmentResult(
            name=seg.name, duration=duration, distance=distance,
            energy=energy, avg_power=p_elec, avg_speed=seg.velocity,
            altitude_start=alt_start, altitude_end=alt_end,
        )

    else:
        raise ValueError(f"Unknown segment type: {type(seg)}")


__all__ = [
    "analyze", "performance", "envelope",
    "MissionResult", "EnvelopeResult", "DragPolar",
]
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_mission/ -v`

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/mission/ tests/test_mission/
git commit -m "feat(mission): add mission segment analysis with energy budget"
```

---

## Task 8: Run full test suite and verify

- [ ] **Step 1: Run all mission tests**

Run: `.venv/bin/python -m pytest tests/test_mission/ -v`
Expected: All pass

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: No new failures (only pre-existing airfoil failures if not yet fixed)

- [ ] **Step 3: Verify physical reasonableness**

Run a quick sanity check script:

```python
import aerisplane as ap
from aerisplane.weights import analyze as weight_analyze
from aerisplane.mission import envelope

# Use a test aircraft fixture
# ... (same as conftest)
# wr = weight_analyze(aircraft)
# env = envelope(aircraft, wr)
# print(env.report())
```

Check that:
- L/D max is 5-30 for an RC aircraft
- Stall speed is 5-15 m/s for ~2 kg
- Endurance is 5-60 min for a 2.2Ah 4S battery
- Range is 1-50 km
- Service ceiling is 500-5000 m
- Min sink rate is 0.3-2.0 m/s

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(mission): complete flight performance envelope module"
```
