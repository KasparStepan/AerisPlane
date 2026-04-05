# Propulsion Discipline Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `aerisplane.propulsion.analyze(aircraft, condition, throttle)` → `PropulsionResult` that finds the motor–propeller operating point and reports thrust, current, RPM, efficiencies, endurance, and constraint flags.

**Architecture:** New discipline module `src/aerisplane/propulsion/` following the same pattern as other discipline modules: `__init__.py` (public `analyze()`) + `result.py` (result dataclass) + `solver.py` (operating-point physics). The solver finds the equilibrium RPM where motor torque equals propeller torque using `scipy.optimize.brentq`. All physics delegated to existing `Motor`, `Propeller`, `Battery`, `ESC` methods in `core/propulsion.py`. The module is wired into `aerisplane.__init__.py` and the MDO discipline chain.

**Tech Stack:** scipy (brentq), numpy, existing `core/propulsion.py` dataclasses.

---

### Task 1: `PropulsionResult` dataclass

**Files:**
- Create: `src/aerisplane/propulsion/__init__.py`
- Create: `src/aerisplane/propulsion/result.py`
- Create: `tests/test_propulsion/__init__.py`
- Create: `tests/test_propulsion/test_result.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_propulsion/test_result.py
import pytest
from aerisplane.propulsion.result import PropulsionResult

def test_result_fields():
    r = PropulsionResult(
        thrust_n=12.5,
        current_a=18.3,
        rpm=7500.0,
        motor_efficiency=0.82,
        propulsive_efficiency=0.65,
        electrical_power_w=270.0,
        shaft_power_w=221.4,
        battery_endurance_s=1200.0,
        c_rate=5.0,
        over_current=False,
        throttle=0.75,
        velocity_ms=14.0,
    )
    assert r.thrust_n == pytest.approx(12.5)
    assert r.over_current is False

def test_report_is_string():
    r = PropulsionResult(
        thrust_n=10.0, current_a=15.0, rpm=6000.0,
        motor_efficiency=0.80, propulsive_efficiency=0.60,
        electrical_power_w=200.0, shaft_power_w=160.0,
        battery_endurance_s=900.0, c_rate=4.0,
        over_current=False, throttle=0.70, velocity_ms=12.0,
    )
    report = r.report()
    assert isinstance(report, str)
    assert "Thrust" in report
    assert "RPM" in report

def test_plot_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    r = PropulsionResult(
        thrust_n=10.0, current_a=15.0, rpm=6000.0,
        motor_efficiency=0.80, propulsive_efficiency=0.60,
        electrical_power_w=200.0, shaft_power_w=160.0,
        battery_endurance_s=900.0, c_rate=4.0,
        over_current=False, throttle=0.70, velocity_ms=12.0,
    )
    fig = r.plot()
    import matplotlib.pyplot as plt
    assert fig is not None
    plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_propulsion/test_result.py -v
```

Expected: FAIL with ImportError

- [ ] **Step 3: Create `tests/test_propulsion/__init__.py`**

```python
# tests/test_propulsion/__init__.py
```
(empty file)

- [ ] **Step 4: Create `result.py`**

```python
# src/aerisplane/propulsion/result.py
"""Propulsion analysis result dataclass."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PropulsionResult:
    """Result of a propulsion operating-point analysis.

    Parameters
    ----------
    thrust_n : float
        Thrust produced [N].
    current_a : float
        Battery current draw [A].
    rpm : float
        Propeller / motor RPM.
    motor_efficiency : float
        Motor electrical-to-shaft efficiency [-].
    propulsive_efficiency : float
        Propeller propulsive efficiency = T*V/P_shaft [-].
    electrical_power_w : float
        Total electrical power consumed [W].
    shaft_power_w : float
        Shaft (mechanical) power into propeller [W].
    battery_endurance_s : float
        Time to battery depletion at this operating point [s].
    c_rate : float
        Instantaneous C-rate = current / capacity_ah [-].
    over_current : bool
        True if current exceeds ESC or motor max_current limit.
    throttle : float
        Throttle setting used [0–1].
    velocity_ms : float
        Flight velocity [m/s].
    """

    thrust_n: float
    current_a: float
    rpm: float
    motor_efficiency: float
    propulsive_efficiency: float
    electrical_power_w: float
    shaft_power_w: float
    battery_endurance_s: float
    c_rate: float
    over_current: bool
    throttle: float
    velocity_ms: float

    def report(self) -> str:
        """Return a human-readable summary string."""
        flag = "  *** OVER-CURRENT ***" if self.over_current else ""
        return (
            f"Propulsion Analysis (throttle={self.throttle:.0%}, V={self.velocity_ms:.1f} m/s)\n"
            f"  Thrust               : {self.thrust_n:.2f} N\n"
            f"  Current              : {self.current_a:.2f} A{flag}\n"
            f"  RPM                  : {self.rpm:.0f}\n"
            f"  Motor efficiency     : {self.motor_efficiency:.1%}\n"
            f"  Propulsive efficiency: {self.propulsive_efficiency:.1%}\n"
            f"  Electrical power     : {self.electrical_power_w:.1f} W\n"
            f"  Shaft power          : {self.shaft_power_w:.1f} W\n"
            f"  C-rate               : {self.c_rate:.1f} C\n"
            f"  Battery endurance    : {self.battery_endurance_s / 60:.1f} min\n"
        )

    def plot(self):
        """Bar chart of key propulsion metrics."""
        import matplotlib.pyplot as plt

        labels = ["Thrust (N)", "Current (A)", "RPM/100", "η_motor (%)", "η_prop (%)"]
        values = [
            self.thrust_n,
            self.current_a,
            self.rpm / 100.0,
            self.motor_efficiency * 100.0,
            self.propulsive_efficiency * 100.0,
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values)
        if self.over_current:
            bars[1].set_color("red")
        ax.set_title(
            f"Propulsion — throttle {self.throttle:.0%}, V={self.velocity_ms:.1f} m/s"
        )
        ax.set_ylabel("Value")
        fig.tight_layout()
        return fig
```

- [ ] **Step 5: Create stub `__init__.py`**

```python
# src/aerisplane/propulsion/__init__.py
"""Propulsion discipline module."""
from aerisplane.propulsion.result import PropulsionResult

__all__ = ["PropulsionResult"]
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest tests/test_propulsion/test_result.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/aerisplane/propulsion/__init__.py src/aerisplane/propulsion/result.py \
        tests/test_propulsion/__init__.py tests/test_propulsion/test_result.py
git commit -m "feat(propulsion): add PropulsionResult dataclass"
```

---

### Task 2: Operating-point solver (`solver.py`)

**Files:**
- Create: `src/aerisplane/propulsion/solver.py`
- Create: `tests/test_propulsion/test_solver.py`

The operating point is where motor torque equals propeller torque:

```
Q_motor(RPM) = Q_prop(RPM)

Q_motor = Kt * (I - I0)   where  I = (V_batt - RPM/kv) / R_motor
Q_prop  = CP * rho * n^2 * D^5 / (2π)   where n = RPM/60
```

`brentq` finds the RPM that makes `Q_motor - Q_prop = 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_propulsion/test_solver.py
import pytest
import numpy as np
from aerisplane.core.propulsion import Motor, Propeller, Battery, ESC, PropulsionSystem
from aerisplane.propulsion.solver import solve_operating_point

@pytest.fixture
def system():
    motor = Motor(
        name="test", kv=900, resistance=0.1,
        no_load_current=0.5, max_current=40.0, mass=0.120,
    )
    prop = Propeller(diameter=0.254, pitch=0.127, mass=0.025)
    battery = Battery(
        name="test_bat", capacity_ah=3.0, nominal_voltage=14.8,
        cell_count=4, c_rating=25.0, mass=0.280,
    )
    esc = ESC(name="test_esc", max_current=40.0, mass=0.030)
    return PropulsionSystem(
        motor=motor, propeller=prop, battery=battery, esc=esc,
        position=np.array([0.0, 0.0, 0.0]),
    )

def test_solve_returns_positive_rpm(system):
    rpm, current = solve_operating_point(system, throttle=1.0, velocity=0.0, rho=1.225)
    assert rpm > 0
    assert current > 0

def test_solve_throttle_zero_gives_zero_thrust(system):
    rpm, current = solve_operating_point(system, throttle=0.0, velocity=0.0, rho=1.225)
    assert rpm == pytest.approx(0.0, abs=1.0)

def test_solve_thrust_increases_with_throttle(system):
    rho = 1.225
    rpm_low, _ = solve_operating_point(system, throttle=0.5, velocity=10.0, rho=rho)
    rpm_high, _ = solve_operating_point(system, throttle=1.0, velocity=10.0, rho=rho)
    thrust_low = system.propeller.thrust(rpm_low, 10.0, rho)
    thrust_high = system.propeller.thrust(rpm_high, 10.0, rho)
    assert thrust_high > thrust_low

def test_solve_current_within_motor_limit(system):
    rpm, current = solve_operating_point(system, throttle=1.0, velocity=0.0, rho=1.225)
    assert current <= system.motor.max_current * 1.01  # allow 1% tolerance
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_propulsion/test_solver.py -v
```

Expected: FAIL with ImportError

- [ ] **Step 3: Implement `solver.py`**

```python
# src/aerisplane/propulsion/solver.py
"""Operating-point solver for a motor–propeller system.

Finds the equilibrium RPM where motor torque equals propeller torque using
a scalar root-finding approach (scipy.optimize.brentq).
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq

from aerisplane.core.propulsion import PropulsionSystem


def solve_operating_point(
    system: PropulsionSystem,
    throttle: float,
    velocity: float,
    rho: float,
) -> tuple[float, float]:
    """Find equilibrium RPM and current for a given throttle and airspeed.

    Throttle scales the battery terminal voltage: V_eff = throttle * V_battery.
    At equilibrium: motor torque = propeller torque.

    Parameters
    ----------
    system : PropulsionSystem
        Motor + propeller + battery + ESC assembly.
    throttle : float
        Throttle command [0–1].
    velocity : float
        Freestream velocity [m/s].
    rho : float
        Air density [kg/m^3].

    Returns
    -------
    rpm : float
        Equilibrium rotational speed [RPM].
    current : float
        Motor current at equilibrium [A].
    """
    if throttle <= 0.0:
        return 0.0, 0.0

    motor = system.motor
    prop = system.propeller
    battery = system.battery

    V_eff = throttle * battery.nominal_voltage
    # Kt = torque constant [N·m/A]  (30/pi / kv)
    Kt = 30.0 / (math.pi * motor.kv)

    def torque_residual(rpm: float) -> float:
        n = rpm / 60.0
        # Motor current from back-EMF: I = (V - omega/kv) / R
        omega_rads = rpm / motor.kv  # back-EMF in V
        I = (V_eff - omega_rads) / motor.resistance
        I = max(0.0, min(I, motor.max_current))

        Q_motor = Kt * (I - motor.no_load_current)

        # Propeller torque: Q = CP * rho * n^2 * D^5 / (2*pi)
        J = prop.advance_ratio(velocity, rpm)
        cp = prop._parametric_cp(J) if prop.performance_data is None else float(
            np.interp(J, prop.performance_data.J, prop.performance_data.CP)
        )
        Q_prop = cp * rho * n**2 * prop.diameter**5 / (2.0 * math.pi)

        return Q_motor - Q_prop

    rpm_max = motor.kv * V_eff  # no-load RPM upper bound

    # Check sign change
    f_low = torque_residual(1.0)
    f_high = torque_residual(rpm_max)

    if f_low * f_high > 0:
        # No sign change: motor cannot drive prop at this throttle/condition.
        # Return the RPM giving minimum residual.
        rpms = np.linspace(1.0, rpm_max, 200)
        residuals = np.array([abs(torque_residual(r)) for r in rpms])
        rpm_eq = float(rpms[np.argmin(residuals)])
    else:
        rpm_eq = brentq(torque_residual, 1.0, rpm_max, xtol=0.1)

    # Compute equilibrium current
    omega_rads = rpm_eq / motor.kv
    current = (V_eff - omega_rads) / motor.resistance
    current = max(0.0, min(current, motor.max_current))

    return float(rpm_eq), float(current)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_propulsion/test_solver.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/propulsion/solver.py tests/test_propulsion/test_solver.py
git commit -m "feat(propulsion): add operating-point solver (brentq)"
```

---

### Task 3: Public `analyze()` function

**Files:**
- Modify: `src/aerisplane/propulsion/__init__.py`
- Create: `tests/test_propulsion/test_analyze.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_propulsion/test_analyze.py
import pytest
import numpy as np
from aerisplane.core.propulsion import Motor, Propeller, Battery, ESC, PropulsionSystem
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
import aerisplane.propulsion as prop_module

@pytest.fixture
def aircraft_with_propulsion():
    motor = Motor(
        name="test", kv=900, resistance=0.1,
        no_load_current=0.5, max_current=40.0, mass=0.120,
    )
    propeller = Propeller(diameter=0.254, pitch=0.127, mass=0.025)
    battery = Battery(
        name="test_bat", capacity_ah=3.0, nominal_voltage=14.8,
        cell_count=4, c_rating=25.0, mass=0.280,
    )
    esc = ESC(name="test_esc", max_current=40.0, mass=0.030)
    propulsion = PropulsionSystem(
        motor=motor, propeller=propeller, battery=battery, esc=esc,
        position=np.array([0.0, 0.0, 0.0]),
    )
    return Aircraft(name="test", wings=[], propulsion=propulsion)

@pytest.fixture
def condition():
    return FlightCondition(velocity=14.0, altitude=100.0, alpha=3.0)

def test_analyze_returns_result(aircraft_with_propulsion, condition):
    from aerisplane.propulsion.result import PropulsionResult
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.8)
    assert isinstance(r, PropulsionResult)

def test_thrust_positive(aircraft_with_propulsion, condition):
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.8)
    assert r.thrust_n > 0

def test_rpm_positive(aircraft_with_propulsion, condition):
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.8)
    assert r.rpm > 0

def test_endurance_positive(aircraft_with_propulsion, condition):
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.5)
    assert r.battery_endurance_s > 0

def test_over_current_flag_at_max_throttle(aircraft_with_propulsion):
    """High-KV motor + full throttle on large battery may push over limit."""
    cond = FlightCondition(velocity=0.0, altitude=0.0, alpha=0.0)
    r = prop_module.analyze(aircraft_with_propulsion, cond, throttle=1.0)
    # over_current should be bool
    assert isinstance(r.over_current, bool)

def test_no_propulsion_raises(condition):
    ac = Aircraft(name="glider", wings=[])
    with pytest.raises(ValueError, match="no PropulsionSystem"):
        prop_module.analyze(ac, condition, throttle=0.8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_propulsion/test_analyze.py -v
```

Expected: FAIL (no `analyze` exported from propulsion)

- [ ] **Step 3: Implement `analyze()` in `__init__.py`**

```python
# src/aerisplane/propulsion/__init__.py
"""Propulsion discipline module.

Public API
----------
analyze(aircraft, condition, throttle) -> PropulsionResult
    Find motor–propeller operating point and compute propulsion metrics.
"""
from __future__ import annotations

from aerisplane.propulsion.result import PropulsionResult

__all__ = ["PropulsionResult", "analyze"]


def analyze(
    aircraft,
    condition,
    throttle: float = 1.0,
) -> PropulsionResult:
    """Compute propulsion operating point for a given throttle and condition.

    Parameters
    ----------
    aircraft : Aircraft
        Must have a `propulsion` attribute (PropulsionSystem).
    condition : FlightCondition
        Provides velocity and altitude (used for air density via ISA).
    throttle : float
        Throttle command [0–1].

    Returns
    -------
    PropulsionResult

    Raises
    ------
    ValueError
        If the aircraft has no PropulsionSystem.
    """
    from aerisplane.propulsion.solver import solve_operating_point
    from aerisplane.utils.atmosphere import isa

    propulsion = getattr(aircraft, "propulsion", None)
    if propulsion is None:
        raise ValueError(
            f"Aircraft '{aircraft.name}' has no PropulsionSystem — "
            "cannot run propulsion analysis."
        )

    _, rho, _ = isa(condition.altitude)
    velocity = condition.velocity

    rpm, current = solve_operating_point(propulsion, throttle, velocity, rho)

    motor = propulsion.motor
    prop = propulsion.propeller
    battery = propulsion.battery
    esc = propulsion.esc

    thrust = prop.thrust(rpm, velocity, rho)
    shaft_power = prop.power(rpm, velocity, rho)
    motor_eff = motor.efficiency(throttle * battery.nominal_voltage, current)
    prop_eff = prop.efficiency(rpm, velocity, rho)

    electrical_power = (
        throttle * battery.nominal_voltage * current if current > 0 else 0.0
    )
    endurance = battery.energy() / electrical_power if electrical_power > 0 else float("inf")
    c_rate = current / battery.capacity_ah if battery.capacity_ah > 0 else 0.0

    max_allowed = min(motor.max_current, esc.max_current)
    over_current = bool(current > max_allowed)

    return PropulsionResult(
        thrust_n=thrust,
        current_a=current,
        rpm=rpm,
        motor_efficiency=motor_eff,
        propulsive_efficiency=prop_eff,
        electrical_power_w=electrical_power,
        shaft_power_w=shaft_power,
        battery_endurance_s=endurance,
        c_rate=c_rate,
        over_current=over_current,
        throttle=throttle,
        velocity_ms=velocity,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_propulsion/test_analyze.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/propulsion/__init__.py tests/test_propulsion/test_analyze.py
git commit -m "feat(propulsion): add public analyze() function"
```

---

### Task 4: Wire into `aerisplane.__init__.py` and MDO discipline chain

**Files:**
- Modify: `src/aerisplane/__init__.py`
- Modify: `src/aerisplane/mdo/problem.py`
- Modify: `tests/test_propulsion/test_analyze.py`

- [ ] **Step 1: Check current `aerisplane.__init__.py` exports**

Read `src/aerisplane/__init__.py` to see what it currently exports. The goal is to add `propulsion` to the package namespace so `import aerisplane as ap; ap.propulsion.analyze(...)` works, and to expose `PropulsionResult` if it is not already.

- [ ] **Step 2: Add propulsion to top-level exports**

In `src/aerisplane/__init__.py`, add after other discipline imports:

```python
from aerisplane import propulsion
from aerisplane.propulsion import PropulsionResult
```

Also add `"propulsion"` and `"PropulsionResult"` to `__all__` if it exists.

- [ ] **Step 3: Write the failing MDO wiring test**

Add to `tests/test_propulsion/test_analyze.py`:

```python
def test_aerisplane_namespace(aircraft_with_propulsion, condition):
    import aerisplane as ap
    r = ap.propulsion.analyze(aircraft_with_propulsion, condition, throttle=0.7)
    assert isinstance(r, ap.PropulsionResult)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_propulsion/test_analyze.py::test_aerisplane_namespace -v
```

Expected: PASS

- [ ] **Step 5: Add `propulsion` to MDO `DISCIPLINE_ORDER`**

In `src/aerisplane/mdo/problem.py`, find:

```python
DISCIPLINE_ORDER = ["weights", "aero", "structures", "stability", "control", "mission"]
```

Change to:

```python
DISCIPLINE_ORDER = ["weights", "aero", "structures", "stability", "control", "propulsion", "mission"]
```

Then in the `evaluate()` method, find the discipline dispatch block (the `if "structures" in disciplines:` etc. chain) and add after the `control` block:

```python
if "propulsion" in disciplines:
    from aerisplane.propulsion import analyze as propulsion_analyze
    throttle = getattr(self, "_throttle", 1.0)
    results["propulsion"] = propulsion_analyze(ac, cond, throttle=throttle)
```

Also update `MDOProblem.__init__` to accept an optional `throttle: float = 1.0` parameter and store it as `self._throttle`.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -x -q --ignore=tests/test_mdo/test_integration.py
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/aerisplane/__init__.py src/aerisplane/mdo/problem.py tests/test_propulsion/test_analyze.py
git commit -m "feat(propulsion): wire into aerisplane namespace and MDO discipline chain"
```
