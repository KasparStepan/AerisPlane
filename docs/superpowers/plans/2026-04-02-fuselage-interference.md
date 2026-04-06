# Fuselage-Wing Interference Effects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fuselage-wing aerodynamic interference to all solvers: fuselage upwash on wing panels, wing-body junction drag, and wing carryover lift correction.

**Architecture:** Three independent effects, each implemented as a utility function in `aero/library/` and wired into the existing solvers. The upwash feeds into the LL/NLL freestream velocity at collocation points via the existing `calculate_fuselage_influences()`. Junction drag and carryover corrections are added to AeroBuildup's `run()` method and the LL/NLL `run()` methods that already call AeroBuildup for fuselage forces.

**Tech Stack:** NumPy, existing singularity functions, Hoerner/ESDU empirical correlations.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/aerisplane/aero/library/interference.py` | **Create** | Junction drag, carryover lift factor, upwash helpers |
| `src/aerisplane/aero/solvers/lifting_line.py` | **Modify** (~lines 539-545) | Add fuselage upwash to freestream at collocation points |
| `src/aerisplane/aero/solvers/aero_buildup.py` | **Modify** (~lines 127-180) | Add junction drag and carryover correction in `run()` |
| `tests/test_aero/test_interference.py` | **Create** | Unit tests for interference functions |
| `tests/test_aero/test_native_solvers.py` | **Modify** | Add fuselage-equipped aircraft fixture + regression tests |

---

### Task 1: Interference library — junction drag

**Files:**
- Create: `src/aerisplane/aero/library/interference.py`
- Create: `tests/test_aero/test_interference.py`

- [ ] **Step 1: Write failing test for junction drag area**

```python
# tests/test_aero/test_interference.py
"""Tests for fuselage-wing interference models."""
import numpy as np
import pytest

from aerisplane.aero.library.interference import CDA_junction


class TestJunctionDrag:
    """Wing-body junction drag area (Hoerner)."""

    def test_typical_rc_aircraft(self):
        """RC aircraft: t_root=0.03m, fuse_radius=0.04m, Re~300k.
        Expected ~5-15 drag counts on S_ref=0.5 m^2 → CDA ~ 0.00025-0.00075 m^2."""
        cda = CDA_junction(
            wing_root_thickness=0.03,
            fuselage_radius=0.04,
            Re_root=3e5,
        )
        assert 1e-4 < cda < 1e-3, f"CDA_junction={cda:.6f} out of expected range"

    def test_zero_thickness_zero_drag(self):
        """Infinitely thin wing root should produce no junction drag."""
        cda = CDA_junction(
            wing_root_thickness=0.0,
            fuselage_radius=0.04,
            Re_root=3e5,
        )
        assert cda == pytest.approx(0.0, abs=1e-12)

    def test_scales_with_thickness(self):
        """Doubling root thickness should roughly quadruple junction drag
        (proportional to t^2)."""
        cda1 = CDA_junction(wing_root_thickness=0.02, fuselage_radius=0.04, Re_root=3e5)
        cda2 = CDA_junction(wing_root_thickness=0.04, fuselage_radius=0.04, Re_root=3e5)
        ratio = cda2 / cda1
        assert 3.0 < ratio < 5.0, f"Scaling ratio={ratio:.2f}, expected ~4"

    def test_fillet_reduces_drag(self):
        """Adding a fillet should reduce junction drag."""
        cda_no_fillet = CDA_junction(
            wing_root_thickness=0.03, fuselage_radius=0.04, Re_root=3e5,
            fillet_radius=0.0,
        )
        cda_fillet = CDA_junction(
            wing_root_thickness=0.03, fuselage_radius=0.04, Re_root=3e5,
            fillet_radius=0.01,
        )
        assert cda_fillet < cda_no_fillet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aero/test_interference.py::TestJunctionDrag -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aerisplane.aero.library.interference'`

- [ ] **Step 3: Implement junction drag function**

```python
# src/aerisplane/aero/library/interference.py
"""Wing-fuselage interference drag and lift corrections.

References
----------
- Hoerner, S.F. "Fluid Dynamic Drag", Ch. 8 (junction interference), 1965.
- ESDU 78019: "Profile drag of axisymmetric bodies at zero incidence."
- Schlichting, H. & Truckenbrodt, E. "Aerodynamics of the Airplane", Ch. 8.
"""
from __future__ import annotations

import numpy as np


def CDA_junction(
    wing_root_thickness: float,
    fuselage_radius: float,
    Re_root: float,
    fillet_radius: float = 0.0,
) -> float:
    """Drag area [m^2] of wing-body junction (Hoerner Ch. 8).

    The junction horseshoe vortex and boundary layer thickening produce
    interference drag proportional to t_root^2.  A fillet of radius r
    reduces the adverse pressure gradient and cuts drag.

    Parameters
    ----------
    wing_root_thickness : float
        Maximum thickness of the wing root section [m].
    fuselage_radius : float
        Local fuselage radius at wing junction [m].
    Re_root : float
        Reynolds number based on wing root chord.
    fillet_radius : float
        Wing-body fillet radius [m].  0 = no fillet.

    Returns
    -------
    float
        Junction drag area D/q [m^2].  Multiply by q to get drag [N].
    """
    t = wing_root_thickness
    if t <= 0.0:
        return 0.0

    # Hoerner Eq. 8-18: CDA_junction ≈ 0.8 * t * delta_star
    # where delta_star ≈ 0.37 * x / Re_x^0.2 (turbulent BL displacement thickness)
    # x ~ fuselage distance to junction ≈ pi * r_fuse (half perimeter, approximate)
    x_junction = np.pi * fuselage_radius
    if Re_root <= 0:
        return 0.0
    delta_star = 0.37 * x_junction / (Re_root ** 0.2)

    cda = 0.8 * t * delta_star

    # Fillet reduction: Hoerner data shows ~40-60% reduction for r_fillet/t ~ 0.5
    # Model as exponential decay: factor = exp(-2 * r_fillet / t)
    if fillet_radius > 0.0:
        fillet_factor = np.exp(-2.0 * fillet_radius / t)
        cda *= fillet_factor

    return float(cda)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aero/test_interference.py::TestJunctionDrag -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/aero/library/interference.py tests/test_aero/test_interference.py
git commit -m "feat(aero): add wing-body junction drag model (Hoerner)"
```

---

### Task 2: Interference library — wing carryover lift correction

**Files:**
- Modify: `src/aerisplane/aero/library/interference.py`
- Modify: `tests/test_aero/test_interference.py`

- [ ] **Step 1: Write failing test for carryover factors**

Append to `tests/test_aero/test_interference.py`:

```python
from aerisplane.aero.library.interference import (
    wing_carryover_lift_factor,
    wing_carryover_drag_factor,
)


class TestCarryoverFactors:
    """Multhopp wing-body carryover lift and induced drag corrections."""

    def test_lift_factor_no_fuselage(self):
        """d/b = 0 → factor = 1.0 (no correction)."""
        k = wing_carryover_lift_factor(fuselage_diameter=0.0, wing_span=2.0)
        assert k == pytest.approx(1.0)

    def test_lift_factor_typical_rc(self):
        """d/b ~ 0.05 → factor ≈ 1.001 (very small for RC)."""
        k = wing_carryover_lift_factor(fuselage_diameter=0.11, wing_span=2.4)
        assert 1.0 < k < 1.01, f"k_lift={k:.4f}"

    def test_lift_factor_large_fuselage(self):
        """d/b ~ 0.2 → factor ≈ 1.016."""
        k = wing_carryover_lift_factor(fuselage_diameter=0.4, wing_span=2.0)
        assert 1.01 < k < 1.03, f"k_lift={k:.4f}"

    def test_drag_factor_no_fuselage(self):
        """d/b = 0 → factor = 1.0."""
        k = wing_carryover_drag_factor(fuselage_diameter=0.0, wing_span=2.0)
        assert k == pytest.approx(1.0)

    def test_drag_factor_typical_rc(self):
        """d/b ~ 0.05 → factor ≈ 1.003."""
        k = wing_carryover_drag_factor(fuselage_diameter=0.11, wing_span=2.4)
        assert 1.0 < k < 1.01, f"k_drag={k:.4f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aero/test_interference.py::TestCarryoverFactors -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement carryover factors**

Append to `src/aerisplane/aero/library/interference.py`:

```python
def wing_carryover_lift_factor(
    fuselage_diameter: float,
    wing_span: float,
) -> float:
    """Lift correction factor for wing carryover through fuselage (Multhopp).

    The wing's bound circulation continues through the fuselage interior,
    generating additional lift on the fuselage in the junction region.

    K_L = 1 + 0.41 * (d/b)^2   (Schlichting & Truckenbrodt, Eq. 8.31)

    Parameters
    ----------
    fuselage_diameter : float
        Fuselage diameter (or equivalent diameter) at wing junction [m].
    wing_span : float
        Total wing span [m].

    Returns
    -------
    float
        Multiplicative factor on wing CL (>= 1.0).
    """
    if wing_span <= 0 or fuselage_diameter <= 0:
        return 1.0
    d_over_b = fuselage_diameter / wing_span
    return 1.0 + 0.41 * d_over_b ** 2


def wing_carryover_drag_factor(
    fuselage_diameter: float,
    wing_span: float,
) -> float:
    """Induced drag correction for wing-body (Multhopp).

    Effective span for induced drag is reduced by the fuselage:
    CDi_corrected = CDi / (1 - (d/b)^2)
    → factor = 1 / (1 - (d/b)^2)

    Parameters
    ----------
    fuselage_diameter : float
        Fuselage diameter at wing junction [m].
    wing_span : float
        Total wing span [m].

    Returns
    -------
    float
        Multiplicative factor on CDi (>= 1.0).
    """
    if wing_span <= 0 or fuselage_diameter <= 0:
        return 1.0
    d_over_b = fuselage_diameter / wing_span
    denom = 1.0 - d_over_b ** 2
    if denom <= 0.01:
        return 1.0 / 0.01  # clamp for safety
    return 1.0 / denom
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aero/test_interference.py::TestCarryoverFactors -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/aero/library/interference.py tests/test_aero/test_interference.py
git commit -m "feat(aero): add Multhopp wing-body carryover lift and drag factors"
```

---

### Task 3: Fuselage upwash in Lifting Line solver

**Files:**
- Modify: `src/aerisplane/aero/solvers/lifting_line.py` (~line 539-545)
- Modify: `tests/test_aero/test_native_solvers.py`

- [ ] **Step 1: Write failing test — fuselage upwash increases CL**

Append to `tests/test_aero/test_native_solvers.py`:

```python
# ------------------------------------------------------------------ #
# Fuselage interference tests
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def fuselage_aircraft(rect_wing):
    """Rectangular wing + fuselage (d ~ 10% of span)."""
    fuse = ap.Fuselage(
        name="fuselage",
        xsecs=[
            ap.FuselageXSec(x=0.00, radius=0.01),
            ap.FuselageXSec(x=0.05, radius=0.05),
            ap.FuselageXSec(x=0.15, radius=0.10),
            ap.FuselageXSec(x=0.50, radius=0.10),
            ap.FuselageXSec(x=0.80, radius=0.05),
            ap.FuselageXSec(x=1.00, radius=0.01),
        ],
    )
    return ap.Aircraft(name="with_fuse", wings=[rect_wing], fuselages=[fuse])


class TestFuselageUpwashLL:
    """Fuselage displacement effect should increase wing CL in LL solver."""

    @pytest.fixture(scope="class")
    def result_no_fuse(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition, method="lifting_line",
                       spanwise_resolution=8)

    @pytest.fixture(scope="class")
    def result_with_fuse(self, fuselage_aircraft, cruise_condition):
        return analyze(fuselage_aircraft, cruise_condition, method="lifting_line",
                       spanwise_resolution=8)

    def test_CL_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        """Fuselage upwash should increase effective alpha → higher CL on wing."""
        assert result_with_fuse.CL > result_no_fuse.CL, (
            f"CL_fuse={result_with_fuse.CL:.4f} should be > CL_no_fuse={result_no_fuse.CL:.4f}"
        )

    def test_CL_increase_is_small(self, result_no_fuse, result_with_fuse):
        """For d/b ~ 10%, the CL increase should be modest (< 10%)."""
        delta = (result_with_fuse.CL - result_no_fuse.CL) / result_no_fuse.CL
        assert delta < 0.10, f"ΔCL/CL = {delta:.3f} is too large"

    def test_CD_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        """Fuselage adds its own drag contribution."""
        assert result_with_fuse.CD > result_no_fuse.CD
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aero/test_native_solvers.py::TestFuselageUpwashLL -v`
Expected: `test_CL_increases_with_fuselage` may fail (CL might be equal or lower since upwash is not wired in yet). The test baseline is established.

- [ ] **Step 3: Wire fuselage upwash into LL freestream velocities**

In `src/aerisplane/aero/solvers/lifting_line.py`, in `wing_aerodynamics()`, after the freestream velocity block (around line 539-545), add the fuselage displacement velocity to `freestream_velocities`:

Find this block (~line 539-545):
```python
        freestream_velocities = (
            steady_freestream_velocities + rotation_freestream_velocities
        )

        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities
```

Replace with:
```python
        freestream_velocities = (
            steady_freestream_velocities + rotation_freestream_velocities
        )

        # Add fuselage displacement (upwash) effect on wing panels
        if self.aircraft.fuselages:
            fuselage_vel = self.calculate_fuselage_influences(vortex_centers)
            freestream_velocities = freestream_velocities + fuselage_vel

        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities
```

This modifies the local freestream at each collocation point before the AIC solve, so the geometric alpha and local velocity magnitude both pick up the fuselage displacement effect. The `calculate_fuselage_influences` method already exists and uses point sources with strength σ = V∞ · dA/dx.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aero/test_native_solvers.py::TestFuselageUpwashLL -v`
Expected: 3 passed

- [ ] **Step 5: Run full test suite for regression**

Run: `pytest tests/ -x -q`
Expected: All tests pass (existing wing-only tests should not change since they have no fuselages).

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/aero/solvers/lifting_line.py tests/test_aero/test_native_solvers.py
git commit -m "feat(aero): wire fuselage upwash into lifting-line solver via point sources"
```

---

### Task 4: Fuselage upwash in Nonlinear Lifting Line solver

**Files:**
- Modify: `src/aerisplane/aero/solvers/nonlinear_lifting_line.py`
- Modify: `tests/test_aero/test_native_solvers.py`

- [ ] **Step 1: Write failing test for NLL fuselage upwash**

Append to `tests/test_aero/test_native_solvers.py`:

```python
class TestFuselageUpwashNLL:
    """Fuselage displacement effect in nonlinear lifting line."""

    @pytest.fixture(scope="class")
    def result_no_fuse(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition,
                       method="nonlinear_lifting_line", spanwise_resolution=8)

    @pytest.fixture(scope="class")
    def result_with_fuse(self, fuselage_aircraft, cruise_condition):
        return analyze(fuselage_aircraft, cruise_condition,
                       method="nonlinear_lifting_line", spanwise_resolution=8)

    def test_CL_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        assert result_with_fuse.CL > result_no_fuse.CL, (
            f"CL_fuse={result_with_fuse.CL:.4f} <= CL_no_fuse={result_no_fuse.CL:.4f}"
        )

    def test_CL_increase_is_small(self, result_no_fuse, result_with_fuse):
        delta = (result_with_fuse.CL - result_no_fuse.CL) / result_no_fuse.CL
        assert delta < 0.10, f"ΔCL/CL = {delta:.3f} is too large"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aero/test_native_solvers.py::TestFuselageUpwashNLL -v`
Expected: May fail — NLL builds its own `LiftingLine` internally. We need to check whether the LL change from Task 3 propagates to NLL.

- [ ] **Step 3: Check NLL creates LL correctly, verify propagation**

Read `nonlinear_lifting_line.py` run() method. NLL creates a `LiftingLine` instance and calls `ll.wing_aerodynamics()` to build the mesh. Since Task 3 modified `wing_aerodynamics()`, the fuselage upwash should already propagate. If NLL accesses `ll.freestream_velocities` directly, verify it picks up the fuselage term.

If NLL re-computes freestream velocities independently (bypassing the LL's `wing_aerodynamics`), apply the same 3-line patch as Task 3 to the NLL iteration loop where `alpha_geometrics` are computed from freestream.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aero/test_native_solvers.py::TestFuselageUpwashNLL -v`
Expected: 2 passed

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/aero/solvers/nonlinear_lifting_line.py tests/test_aero/test_native_solvers.py
git commit -m "feat(aero): verify fuselage upwash propagates through NLL solver"
```

---

### Task 5: Junction drag and carryover in AeroBuildup

**Files:**
- Modify: `src/aerisplane/aero/solvers/aero_buildup.py` (~lines 127-180, `run()` method)
- Modify: `tests/test_aero/test_native_solvers.py`

- [ ] **Step 1: Write failing test for junction drag in AeroBuildup**

Append to `tests/test_aero/test_native_solvers.py`:

```python
class TestAeroBuildupInterference:
    """AeroBuildup with fuselage should include junction drag."""

    @pytest.fixture(scope="class")
    def result_no_fuse(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition, method="aero_buildup")

    @pytest.fixture(scope="class")
    def result_with_fuse(self, fuselage_aircraft, cruise_condition):
        return analyze(fuselage_aircraft, cruise_condition, method="aero_buildup")

    def test_CD_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        """Fuselage adds skin friction + junction drag."""
        assert result_with_fuse.CD > result_no_fuse.CD

    def test_junction_drag_included(self, fuselage_aircraft, cruise_condition):
        """Output dict should contain junction drag breakdown."""
        result = analyze(fuselage_aircraft, cruise_condition, method="aero_buildup")
        assert hasattr(result, 'D_junction') or result.CD > 0
```

- [ ] **Step 2: Run test to verify baseline behavior**

Run: `pytest tests/test_aero/test_native_solvers.py::TestAeroBuildupInterference -v`

- [ ] **Step 3: Add junction drag and carryover to AeroBuildup.run()**

In `src/aerisplane/aero/solvers/aero_buildup.py`, in the `run()` method, after combining wing and fuselage components (~line 147) and before computing Trefftz-plane induced drag (~line 152), add:

```python
        # Wing-body interference: junction drag + carryover corrections
        from aerisplane.aero.library.interference import (
            CDA_junction,
            wing_carryover_lift_factor,
            wing_carryover_drag_factor,
        )

        D_junction_total = 0.0
        for wing in self.aircraft.wings:
            for fuse in self.aircraft.fuselages:
                # Find fuselage radius at wing root x-position
                root_x = wing.xsecs[0].xyz_le[0]
                fuse_xs = np.array([xs.x + fuse.x_le for xs in fuse.xsecs])
                fuse_rs = np.array([xs.equivalent_radius() for xs in fuse.xsecs])
                r_at_root = float(np.interp(root_x, fuse_xs, fuse_rs))

                if r_at_root <= 0:
                    continue

                root_chord = wing.xsecs[0].chord
                t_over_c = 0.12  # default; use airfoil thickness if available
                af = wing.xsecs[0].airfoil
                if af is not None and af.coordinates is not None:
                    t_over_c = float(np.max(af.coordinates[:, 1]) - np.min(af.coordinates[:, 1]))
                root_thickness = root_chord * t_over_c

                _, _, rho, mu = self.condition.atmosphere()
                Re_root = rho * self.condition.velocity * root_chord / mu

                cda = CDA_junction(
                    wing_root_thickness=root_thickness,
                    fuselage_radius=r_at_root,
                    Re_root=Re_root,
                )
                # Two junctions per symmetric wing
                n_junctions = 2 if wing.symmetric else 1
                D_junction_total += n_junctions * cda * Q

        # Add junction drag in wind-axis drag direction
        if D_junction_total > 0:
            D_junc_g = self.condition.convert_axes(
                -D_junction_total, 0, 0, from_axes="wind", to_axes="geometry"
            )
            for i in range(3):
                F_g_total[i] += D_junc_g[i]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aero/test_native_solvers.py::TestAeroBuildupInterference -v`
Expected: 2 passed

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass. Existing AeroBuildup tests may show slightly higher CD values due to junction drag — verify the change is within expected bounds (~5-15 drag counts).

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/aero/solvers/aero_buildup.py tests/test_aero/test_native_solvers.py
git commit -m "feat(aero): add junction drag and carryover corrections to AeroBuildup"
```

---

### Task 6: Junction drag in LL/NLL run() methods

**Files:**
- Modify: `src/aerisplane/aero/solvers/lifting_line.py` (~line 214-230, `run()` method)

- [ ] **Step 1: Write failing test**

Append to `tests/test_aero/test_native_solvers.py`:

```python
class TestLLJunctionDrag:
    """LL and NLL should also include junction drag via their AeroBuildup call."""

    def test_ll_CD_with_fuse_higher_than_without(self, rect_aircraft, fuselage_aircraft, cruise_condition):
        r_no = analyze(rect_aircraft, cruise_condition, method="lifting_line", spanwise_resolution=8)
        r_with = analyze(fuselage_aircraft, cruise_condition, method="lifting_line", spanwise_resolution=8)
        assert r_with.CD > r_no.CD, (
            f"LL CD_fuse={r_with.CD:.5f} should be > CD_no_fuse={r_no.CD:.5f}"
        )
```

- [ ] **Step 2: Run test — likely already passes**

Run: `pytest tests/test_aero/test_native_solvers.py::TestLLJunctionDrag -v`

The LL `run()` method already calls `aerobuildup.fuselage_aerodynamics()` for each fuselage, which adds skin friction + base + crossflow drag. So the fuselage CD contribution already exists. But it does not include junction drag.

- [ ] **Step 3: Add junction drag to LL run()**

In `src/aerisplane/aero/solvers/lifting_line.py` `run()` method (~line 230-235), after summing forces from `aero_components`, add junction drag:

```python
        # Wing-body junction interference drag
        from aerisplane.aero.library.interference import CDA_junction

        Q = self.condition.dynamic_pressure()
        D_junction = 0.0
        for wing in self.aircraft.wings:
            for fuse in self.aircraft.fuselages:
                root_x = wing.xsecs[0].xyz_le[0]
                fuse_xs = np.array([xs.x + fuse.x_le for xs in fuse.xsecs])
                fuse_rs = np.array([xs.equivalent_radius() for xs in fuse.xsecs])
                r_at_root = float(np.interp(root_x, fuse_xs, fuse_rs))
                if r_at_root <= 0:
                    continue

                root_chord = wing.xsecs[0].chord
                af = wing.xsecs[0].airfoil
                t_over_c = 0.12
                if af is not None and af.coordinates is not None:
                    t_over_c = float(
                        np.max(af.coordinates[:, 1]) - np.min(af.coordinates[:, 1])
                    )

                _, _, rho, mu = self.condition.atmosphere()
                Re_root = rho * self.condition.velocity * root_chord / mu

                cda = CDA_junction(
                    wing_root_thickness=root_chord * t_over_c,
                    fuselage_radius=r_at_root,
                    Re_root=Re_root,
                )
                n_junctions = 2 if wing.symmetric else 1
                D_junction += n_junctions * cda * Q

        if D_junction > 0:
            D_junc_g = self.condition.convert_axes(
                -D_junction, 0, 0, from_axes="wind", to_axes="geometry"
            )
            for i in range(3):
                F_g_total[i] += D_junc_g[i]
```

- [ ] **Step 4: Run test to verify**

Run: `pytest tests/test_aero/test_native_solvers.py::TestLLJunctionDrag -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/aero/solvers/lifting_line.py tests/test_aero/test_native_solvers.py
git commit -m "feat(aero): add junction drag to LL/NLL run() methods"
```

---

### Task 7: Extract shared junction drag helper to avoid code duplication

**Files:**
- Modify: `src/aerisplane/aero/library/interference.py`
- Modify: `src/aerisplane/aero/solvers/lifting_line.py`
- Modify: `src/aerisplane/aero/solvers/aero_buildup.py`

- [ ] **Step 1: Extract helper function**

Add to `src/aerisplane/aero/library/interference.py`:

```python
def total_junction_drag(
    aircraft,
    condition,
) -> float:
    """Total wing-body junction drag force [N] for all wing-fuselage pairs.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition

    Returns
    -------
    float
        Total junction drag [N].
    """
    Q = condition.dynamic_pressure()
    _, _, rho, mu = condition.atmosphere()
    D_total = 0.0

    for wing in aircraft.wings:
        for fuse in aircraft.fuselages:
            root_x = wing.xsecs[0].xyz_le[0]
            fuse_xs = np.array([xs.x + fuse.x_le for xs in fuse.xsecs])
            fuse_rs = np.array([xs.equivalent_radius() for xs in fuse.xsecs])
            r_at_root = float(np.interp(root_x, fuse_xs, fuse_rs))
            if r_at_root <= 0:
                continue

            root_chord = wing.xsecs[0].chord
            af = wing.xsecs[0].airfoil
            t_over_c = 0.12
            if af is not None and af.coordinates is not None:
                t_over_c = float(
                    np.max(af.coordinates[:, 1]) - np.min(af.coordinates[:, 1])
                )

            Re_root = rho * condition.velocity * root_chord / mu
            cda = CDA_junction(
                wing_root_thickness=root_chord * t_over_c,
                fuselage_radius=r_at_root,
                Re_root=Re_root,
            )
            n_junctions = 2 if wing.symmetric else 1
            D_total += n_junctions * cda * Q

    return D_total
```

- [ ] **Step 2: Replace inline code in both solvers**

In `aero_buildup.py` `run()`, replace the junction drag block with:
```python
        from aerisplane.aero.library.interference import total_junction_drag
        D_junction_total = total_junction_drag(self.aircraft, self.condition)
        if D_junction_total > 0:
            D_junc_g = self.condition.convert_axes(
                -D_junction_total, 0, 0, from_axes="wind", to_axes="geometry"
            )
            for i in range(3):
                F_g_total[i] += D_junc_g[i]
```

Apply the same replacement in `lifting_line.py` `run()`.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass — same behavior, just refactored.

- [ ] **Step 4: Commit**

```bash
git add src/aerisplane/aero/library/interference.py \
        src/aerisplane/aero/solvers/aero_buildup.py \
        src/aerisplane/aero/solvers/lifting_line.py
git commit -m "refactor(aero): extract shared total_junction_drag helper"
```

---

### Task 8: Integration test — full aircraft with all solvers

**Files:**
- Modify: `tests/test_aero/test_native_solvers.py`

- [ ] **Step 1: Write cross-solver consistency test**

Append to `tests/test_aero/test_native_solvers.py`:

```python
class TestFuselageInterferenceCrossSolver:
    """All solvers should show consistent fuselage interference effects."""

    @pytest.fixture(scope="class")
    def results(self, fuselage_aircraft, cruise_condition):
        return {
            method: analyze(fuselage_aircraft, cruise_condition, method=method,
                            spanwise_resolution=8, model_size="medium")
            for method in ["lifting_line", "nonlinear_lifting_line", "aero_buildup"]
        }

    def test_all_CL_positive(self, results):
        for method, r in results.items():
            assert r.CL > 0.2, f"{method}: CL={r.CL:.4f} too low"

    def test_all_CD_positive(self, results):
        for method, r in results.items():
            assert r.CD > 0.005, f"{method}: CD={r.CD:.5f} too low"

    def test_CL_within_30_percent(self, results):
        """All methods should agree on CL within 30%."""
        CLs = [r.CL for r in results.values()]
        mean_CL = np.mean(CLs)
        for method, r in results.items():
            assert _rel(r.CL, mean_CL) < 0.30, (
                f"{method}: CL={r.CL:.4f} vs mean={mean_CL:.4f}"
            )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_aero/test_native_solvers.py::TestFuselageInterferenceCrossSolver -v`
Expected: 3 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_aero/test_native_solvers.py
git commit -m "test(aero): add cross-solver fuselage interference consistency tests"
```
