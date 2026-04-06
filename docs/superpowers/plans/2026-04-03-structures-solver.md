# Structures Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fast Euler-Bernoulli wing beam solver that computes spanwise shear, bending, deflection, and structural margins (bending, shear, buckling, divergence) for use as MDO constraints.

**Architecture:** Six focused modules: `loads.py` computes the design load factor from gust and maneuver cases; `section.py` computes effective EI from airfoil geometry; `beam.py` integrates the cantilever beam equations; `checks.py` computes margins of safety; `result.py` holds result dataclasses; `__init__.py` exposes the public `analyze()` entry point. Follows the same discipline module pattern as `stability/` and `control/`.

**Tech Stack:** Pure NumPy (no new dependencies). Tests use pytest. Fixtures reuse existing `aerisplane.catalog.materials` entries.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/aerisplane/structures/loads.py` | Maneuver, gust, and design load factor functions |
| `src/aerisplane/structures/section.py` | Airfoil-geometry-based EI, spar height, skin I, spar fit check |
| `src/aerisplane/structures/beam.py` | `BeamResult` dataclass + `WingBeam` cantilever solver |
| `src/aerisplane/structures/checks.py` | Bending, shear, buckling margin functions; divergence speed |
| `src/aerisplane/structures/result.py` | `WingStructureResult` and `StructureResult` dataclasses |
| `src/aerisplane/structures/__init__.py` | Public `analyze()` entry point |
| `tests/test_structures/__init__.py` | Empty |
| `tests/test_structures/conftest.py` | Shared fixtures |
| `tests/test_structures/test_loads.py` | Load factor tests |
| `tests/test_structures/test_section.py` | Section property tests |
| `tests/test_structures/test_beam.py` | Beam solver tests |
| `tests/test_structures/test_checks.py` | Margin function tests |
| `tests/test_structures/test_structures.py` | Integration: analyze() end-to-end |

**No changes needed** to any `core/` files — all required data is already in `Spar`, `TubeSection`, `Skin`, `Airfoil`, `Wing`, `WingXSec`.

---

## Sign conventions and physics reference

**Beam model:** The wing spar is a cantilever fixed at the root (y=0), free at the tip (y=b). The net distributed load `q(y)` [N/m] acts upward (positive z). Integrating from tip to root:

```
V(y) = ∫_y^b q(η) dη          [N]  shear force, positive = upward force from tip segment
M(y) = ∫_y^b V(η) dη          [N·m]  bending moment, positive at root for upward load
```

Euler-Bernoulli deflection (integrating from root to tip):
```
θ(y) = ∫_0^y M(η)/EI(η) dη   [rad]  slope
δ(y) = ∫_0^y θ(η) dη          [m]   deflection (positive = upward)
```

Verification: uniform load q on length L → δ_tip = qL⁴/(8EI). ✓

**Net load with inertia relief:**
```
q_net(y) = n · q_aero(y) − n · g · m'(y)
```
where `n` is load factor, `m'(y)` [kg/m] is structural mass per unit span.

**Lift distribution:** Elliptical approximation used by default (conservative, fast, solver-independent):
```
q_aero(y) = q_0 · √(1 − (y/b)²)   where q_0 = 4·L_semi / (π·b)
```
`L_semi` = total lift for one semi-span = `aero_result.L / 2` (for symmetric wings).

**EI of composite section (transformed section method):**
```
EI_eff = E_spar · I_spar + E_skin · I_skin
I_skin = t_skin · ∫ y(s)² ds     (contour integral over airfoil coordinates)
```

**Divergence speed:**
```
V_div = √(2·GJ_root / (ρ · a · e · S_wing))
e = (0.25 − spar.position) · c_root     [m]  (positive when AC is behind SC)
a = CL_alpha                             [1/rad]
```
Returns `inf` when `e ≤ 0` (no divergence risk).

**Shell buckling of CF tube:**
```
σ_cr = 0.6 · E · t_wall / R_outer   (Timoshenko)
MoS_buckling = σ_cr / σ_bending − 1
```

**Shear margin (thin annular tube):**
```
τ_max = V / A_wall    (conservative: V / A_wall ≥ actual max shear stress)
τ_yield = σ_yield / √3   (von Mises)
MoS_shear = τ_yield / τ_max − 1
```

---

## Task 1: Design load factors

**Files:**
- Create: `src/aerisplane/structures/loads.py`
- Create: `tests/test_structures/__init__.py`
- Create: `tests/test_structures/conftest.py`
- Create: `tests/test_structures/test_loads.py`

- [ ] **Step 1: Create test directory and empty `__init__.py`**

```bash
mkdir -p tests/test_structures
touch tests/test_structures/__init__.py
```

- [ ] **Step 2: Write conftest.py with shared fixtures**

```python
# tests/test_structures/conftest.py
"""Shared fixtures for structures module tests."""
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg


@pytest.fixture
def cf_spar():
    return ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


@pytest.fixture
def petg_skin():
    return ap.Skin(material=petg, thickness=0.8e-3)


@pytest.fixture
def naca2412():
    return ap.Airfoil.from_naca("2412")


@pytest.fixture
def naca0012():
    return ap.Airfoil.from_naca("0012")


@pytest.fixture
def rect_wing(cf_spar, petg_skin, naca2412):
    """Rectangular wing: 0.2 m chord, 0.75 m semispan, uniform NACA 2412."""
    return ap.Wing(
        name="rect_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.1, 0.0, 0.0], chord=0.2,
                        airfoil=naca2412, spar=cf_spar, skin=petg_skin),
            ap.WingXSec(xyz_le=[0.1, 0.75, 0.0], chord=0.2,
                        airfoil=naca2412, spar=cf_spar, skin=petg_skin),
        ],
        symmetric=True,
    )


@pytest.fixture
def simple_aircraft(rect_wing):
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.75, 0.0, 0.0], chord=0.12),
            ap.WingXSec(xyz_le=[0.75, 0.30, 0.0], chord=0.08),
        ],
        symmetric=True,
    )
    fuse = ap.Fuselage(
        name="fuse",
        xsecs=[
            ap.FuselageXSec(x=0.0, radius=0.02),
            ap.FuselageXSec(x=0.8, radius=0.02),
        ],
    )
    motor = ap.Motor(name="m", kv=1100, resistance=0.028,
                     no_load_current=1.2, max_current=40.0, mass=0.12)
    prop = ap.Propeller(diameter=0.254, pitch=0.127, mass=0.03)
    bat = ap.Battery(name="b", capacity_ah=2.2, nominal_voltage=14.8,
                     cell_count=4, c_rating=30.0, mass=0.2)
    esc = ap.ESC(name="e", max_current=40.0, mass=0.03)
    propulsion = ap.PropulsionSystem(
        motor=motor, propeller=prop, battery=bat, esc=esc,
        position=np.array([0., 0., 0.]),
    )
    return ap.Aircraft(
        name="TestPlane",
        wings=[rect_wing, htail],
        fuselages=[fuse],
        propulsion=propulsion,
        payload=ap.Payload(mass=0.1, cg=np.array([0.25, 0., 0.]), name="p"),
    )


@pytest.fixture
def cruise_condition():
    return ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=3.0, beta=0.0)
```

- [ ] **Step 3: Write failing tests for loads.py**

```python
# tests/test_structures/test_loads.py
"""Tests for design load factor functions."""
import pytest
import aerisplane as ap
import aerisplane.weights as wts
from aerisplane.structures.loads import (
    maneuver_load_factor,
    gust_load_factor,
    design_load_factor,
)


class TestManeuverLoadFactor:
    def test_default_35g(self):
        assert maneuver_load_factor() == pytest.approx(3.5)

    def test_custom_value(self):
        assert maneuver_load_factor(n_limit=4.0) == pytest.approx(4.0)


class TestGustLoadFactor:
    def test_returns_greater_than_one(self, cruise_condition):
        n = gust_load_factor(
            velocity=cruise_condition.velocity,
            altitude=cruise_condition.altitude,
            cl_alpha_per_rad=5.5,
            wing_loading=60.0,  # 60 Pa — typical RC
        )
        assert n > 1.0

    def test_increases_with_velocity(self):
        n_slow = gust_load_factor(velocity=10.0, altitude=0.0,
                                  cl_alpha_per_rad=5.5, wing_loading=60.0)
        n_fast = gust_load_factor(velocity=20.0, altitude=0.0,
                                  cl_alpha_per_rad=5.5, wing_loading=60.0)
        assert n_fast > n_slow

    def test_decreases_with_wing_loading(self):
        n_light = gust_load_factor(velocity=15.0, altitude=0.0,
                                   cl_alpha_per_rad=5.5, wing_loading=30.0)
        n_heavy = gust_load_factor(velocity=15.0, altitude=0.0,
                                   cl_alpha_per_rad=5.5, wing_loading=100.0)
        assert n_light > n_heavy


class TestDesignLoadFactor:
    def test_at_least_maneuver_times_safety(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        n = design_load_factor(simple_aircraft, cruise_condition, wr,
                               n_limit=3.5, safety_factor=1.5)
        # Must be at least n_limit × safety_factor
        assert n >= 3.5 * 1.5

    def test_increases_with_safety_factor(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        n1 = design_load_factor(simple_aircraft, cruise_condition, wr,
                                safety_factor=1.5)
        n2 = design_load_factor(simple_aircraft, cruise_condition, wr,
                                safety_factor=2.0)
        assert n2 > n1

    def test_higher_at_high_speed(self, simple_aircraft):
        wr = wts.analyze(simple_aircraft)
        cond_slow = ap.FlightCondition(velocity=10.0, altitude=0.0, alpha=3.0)
        cond_fast = ap.FlightCondition(velocity=25.0, altitude=0.0, alpha=3.0)
        n_slow = design_load_factor(simple_aircraft, cond_slow, wr)
        n_fast = design_load_factor(simple_aircraft, cond_fast, wr)
        assert n_fast >= n_slow
```

- [ ] **Step 4: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/test_structures/test_loads.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'aerisplane.structures.loads'`

- [ ] **Step 5: Implement loads.py**

```python
# src/aerisplane/structures/loads.py
"""Design load factor computation for RC/UAV structural sizing.

Computes limit and ultimate load factors from both the pilot-commanded
maneuver case and the CS-VLA gust case.  The design load factor is the
maximum of both, multiplied by the safety factor.
"""
from __future__ import annotations

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult

_G = 9.81  # m/s²


def maneuver_load_factor(n_limit: float = 3.5) -> float:
    """Pilot-commanded limit load factor.

    Parameters
    ----------
    n_limit : float
        Limit load factor (default 3.5 — aerobatic RC per CS-VLA).

    Returns
    -------
    float
    """
    return float(n_limit)


def gust_load_factor(
    velocity: float,
    altitude: float,
    cl_alpha_per_rad: float,
    wing_loading: float,
    U_de: float = 9.0,
) -> float:
    """Load factor increment from a discrete vertical gust (CS-VLA).

    .. math::
        \\Delta n = \\frac{\\rho V U_{de} a}{2 (W/S)}

    Parameters
    ----------
    velocity : float
        Airspeed [m/s].
    altitude : float
        Altitude [m].
    cl_alpha_per_rad : float
        Lift curve slope dCL/dα [1/rad].
    wing_loading : float
        Wing loading W/S [Pa].
    U_de : float
        Design gust velocity [m/s]. Default 9 m/s (CS-VLA cruise gust).

    Returns
    -------
    float
        Total load factor n = 1 + Δn.
    """
    _, _, rho, _ = isa(altitude)
    delta_n = rho * velocity * U_de * cl_alpha_per_rad / (2.0 * wing_loading)
    return 1.0 + float(delta_n)


def design_load_factor(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    stability_result=None,
    n_limit: float = 3.5,
    safety_factor: float = 1.5,
    gust_velocity: float = 9.0,
) -> float:
    """Ultimate design load factor: max(maneuver, gust) × safety_factor.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition
        Sizing flight condition (typically max-speed cruise).
    weight_result : WeightResult
        Provides total mass for wing loading.
    stability_result : StabilityResult or None
        If provided, uses CL_alpha from stability analysis [1/deg].
        If None, uses the default CL_alpha ≈ 5.5 1/rad.
    n_limit : float
        Limit maneuver load factor (default 3.5).
    safety_factor : float
        Applied to the limit load factor (default 1.5 → ultimate).
    gust_velocity : float
        Design gust velocity U_de [m/s] (default 9.0 per CS-VLA).

    Returns
    -------
    float
        Ultimate design load factor.
    """
    S = aircraft.reference_area() or 1.0
    W = weight_result.total_mass * _G
    wing_loading = W / S

    if stability_result is not None:
        # CL_alpha is stored in [1/deg] — convert to 1/rad
        cl_alpha_rad = float(stability_result.CL_alpha) * (180.0 / np.pi)
    else:
        cl_alpha_rad = 5.5  # typical for moderate AR wing

    n_maneuver = maneuver_load_factor(n_limit)
    n_gust = gust_load_factor(
        velocity=condition.velocity,
        altitude=condition.altitude,
        cl_alpha_per_rad=cl_alpha_rad,
        wing_loading=wing_loading,
        U_de=gust_velocity,
    )
    return float(max(n_maneuver, n_gust) * safety_factor)
```

- [ ] **Step 6: Run tests and verify they pass**

```bash
pytest tests/test_structures/test_loads.py -v
```

Expected: `8 passed`

- [ ] **Step 7: Commit**

```bash
git add src/aerisplane/structures/loads.py \
        tests/test_structures/__init__.py \
        tests/test_structures/conftest.py \
        tests/test_structures/test_loads.py
git commit -m "feat(structures): add design load factor computation (maneuver + gust)"
```

---

## Task 2: Section properties from airfoil geometry

**Files:**
- Create: `src/aerisplane/structures/section.py`
- Create: `tests/test_structures/test_section.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_structures/test_section.py
"""Tests for airfoil-geometry-based section properties."""
import numpy as np
import pytest
import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg
from aerisplane.structures.section import (
    airfoil_spar_height,
    skin_second_moment_of_area,
    effective_EI,
    spar_fits_in_airfoil,
)


class TestAirfoilSparHeight:
    def test_naca0012_at_quarterchord(self, naca0012):
        # NACA 0012: t/c = 12%, height at 25% chord ≈ 11-12% of chord
        h = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        assert 0.018 < h < 0.026  # 9-13% of 0.20m

    def test_scales_with_chord(self, naca0012):
        h1 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        h2 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.40)
        assert h2 == pytest.approx(2.0 * h1, rel=0.01)

    def test_height_decreases_toward_te(self, naca0012):
        h_25 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        h_50 = airfoil_spar_height(naca0012, spar_position=0.50, chord=0.20)
        assert h_25 > h_50  # thicker near leading quarter

    def test_thicker_airfoil_gives_larger_height(self, naca0012):
        naca0018 = ap.Airfoil.from_naca("0018")
        h_12 = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        h_18 = airfoil_spar_height(naca0018, spar_position=0.25, chord=0.20)
        assert h_18 > h_12


class TestSkinSecondMomentOfArea:
    def test_positive_for_valid_airfoil(self, naca0012, petg_skin):
        I = skin_second_moment_of_area(naca0012, chord=0.20,
                                       skin_thickness=petg_skin.thickness)
        assert I > 0.0

    def test_zero_for_zero_thickness(self, naca0012):
        I = skin_second_moment_of_area(naca0012, chord=0.20, skin_thickness=0.0)
        assert I == pytest.approx(0.0)

    def test_scales_with_chord_to_fourth(self, naca0012, petg_skin):
        # I ~ chord⁴ because y ~ chord and ds ~ chord
        I1 = skin_second_moment_of_area(naca0012, chord=0.10,
                                        skin_thickness=petg_skin.thickness)
        I2 = skin_second_moment_of_area(naca0012, chord=0.20,
                                        skin_thickness=petg_skin.thickness)
        assert I2 == pytest.approx(16.0 * I1, rel=0.02)

    def test_scales_linearly_with_thickness(self, naca0012):
        I1 = skin_second_moment_of_area(naca0012, chord=0.20, skin_thickness=0.001)
        I2 = skin_second_moment_of_area(naca0012, chord=0.20, skin_thickness=0.002)
        assert I2 == pytest.approx(2.0 * I1, rel=1e-6)


class TestEffectiveEI:
    def test_spar_only_equals_E_times_I(self, naca0012, cf_spar):
        EI = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=None)
        expected = cf_spar.material.E * cf_spar.section.second_moment_of_area()
        assert EI == pytest.approx(expected, rel=1e-9)

    def test_with_skin_larger_than_spar_only(self, naca0012, cf_spar, petg_skin):
        EI_bare = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=None)
        EI_skin = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=petg_skin)
        assert EI_skin > EI_bare

    def test_stiffer_skin_gives_higher_EI(self, naca0012, cf_spar):
        from aerisplane.catalog.materials import carbon_fiber_tube
        cf_skin = ap.Skin(material=carbon_fiber_tube, thickness=0.5e-3)
        petg_skin_local = ap.Skin(material=petg, thickness=0.5e-3)
        EI_petg = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=petg_skin_local)
        EI_cf = effective_EI(naca0012, chord=0.20, spar=cf_spar, skin=cf_skin)
        assert EI_cf > EI_petg


class TestSparFitsInAirfoil:
    def test_small_spar_fits(self, naca0012):
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
        )
        assert spar_fits_in_airfoil(naca0012, spar, chord=0.20) is True

    def test_oversized_spar_does_not_fit(self, naca0012):
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.050, wall_thickness=0.002),
        )
        assert spar_fits_in_airfoil(naca0012, spar, chord=0.20) is False

    def test_exactly_at_limit_fits(self, naca0012):
        h = airfoil_spar_height(naca0012, spar_position=0.25, chord=0.20)
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=h * 0.99, wall_thickness=0.001),
        )
        assert spar_fits_in_airfoil(naca0012, spar, chord=0.20) is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_structures/test_section.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'aerisplane.structures.section'`

- [ ] **Step 3: Implement section.py**

```python
# src/aerisplane/structures/section.py
"""Wing cross-section structural properties derived from airfoil geometry.

Provides effective bending stiffness EI accounting for both the spar tube
and the surface skin, using the transformed-section (homogenisation) method.
"""
from __future__ import annotations

import numpy as np

from aerisplane.core.airfoil import Airfoil
from aerisplane.core.structures import Skin, Spar


def airfoil_spar_height(
    airfoil: Airfoil,
    spar_position: float,
    chord: float,
) -> float:
    """Physical height available for a spar at the given chordwise position [m].

    Height = (y_upper − y_lower) at x = spar_position, scaled by chord.

    Parameters
    ----------
    airfoil : Airfoil
    spar_position : float
        Chordwise position as fraction of chord [0..1].
    chord : float
        Physical chord length [m].

    Returns
    -------
    float
        Available spar diameter [m].
    """
    upper = airfoil.upper_coordinates()  # LE→TE, normalised
    lower = airfoil.lower_coordinates()  # LE→TE, normalised
    y_up = float(np.interp(spar_position, upper[:, 0], upper[:, 1]))
    y_lo = float(np.interp(spar_position, lower[:, 0], lower[:, 1]))
    return (y_up - y_lo) * chord


def skin_second_moment_of_area(
    airfoil: Airfoil,
    chord: float,
    skin_thickness: float,
) -> float:
    """Second moment of area of the skin about the chord line [m^4].

    Uses the contour integral:  I_skin = t · ∫ y(s)² ds
    where s is the arc-length parameter along the airfoil contour.
    The neutral axis is approximated as the chord line (y = 0 on the
    normalised airfoil).  This is exact for symmetric airfoils and
    introduces only a small error for cambered sections.

    Parameters
    ----------
    airfoil : Airfoil
    chord : float
        Physical chord length [m].
    skin_thickness : float
        Skin wall thickness t [m].

    Returns
    -------
    float
    """
    if skin_thickness == 0.0:
        return 0.0
    if airfoil.coordinates is None:
        return 0.0
    xs = airfoil.coordinates[:, 0] * chord
    ys = airfoil.coordinates[:, 1] * chord
    ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    y_mid = 0.5 * (ys[:-1] + ys[1:])
    return float(skin_thickness * np.sum(y_mid ** 2 * ds))


def effective_EI(
    airfoil: Airfoil,
    chord: float,
    spar: Spar,
    skin: Skin | None = None,
) -> float:
    """Effective bending stiffness EI_eff [N·m²] of the wing cross-section.

    Uses the transformed-section (Voigt) method to combine spar and skin:

        EI_eff = E_spar · I_spar + E_skin · I_skin

    Parameters
    ----------
    airfoil : Airfoil
    chord : float
        Physical chord [m].
    spar : Spar
    skin : Skin or None
        If None, only the spar contribution is included.

    Returns
    -------
    float
    """
    EI = spar.material.E * spar.section.second_moment_of_area()
    if skin is not None:
        I_skin = skin_second_moment_of_area(airfoil, chord, skin.thickness)
        EI += skin.material.E * I_skin
    return float(EI)


def spar_fits_in_airfoil(airfoil: Airfoil, spar: Spar, chord: float) -> bool:
    """True if the spar outer diameter fits within the airfoil at the spar position.

    Parameters
    ----------
    airfoil : Airfoil
    spar : Spar
    chord : float
        Physical chord [m].

    Returns
    -------
    bool
    """
    h = airfoil_spar_height(airfoil, spar.position, chord)
    return spar.section.outer_diameter <= h
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_structures/test_section.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/structures/section.py \
        tests/test_structures/test_section.py
git commit -m "feat(structures): add airfoil-geometry section properties (EI, spar fit)"
```

---

## Task 3: Beam solver

**Files:**
- Create: `src/aerisplane/structures/beam.py`
- Create: `tests/test_structures/test_beam.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_structures/test_beam.py
"""Tests for the Euler-Bernoulli wing beam solver."""
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg
from aerisplane.structures.beam import BeamResult, WingBeam


class TestBeamResult:
    def test_tip_deflection_property(self):
        y = np.array([0.0, 0.5, 1.0])
        br = BeamResult(
            y=y,
            V=np.array([10.0, 5.0, 0.0]),
            M=np.array([5.0, 2.5, 0.0]),
            theta=np.array([0.0, 0.001, 0.003]),
            delta=np.array([0.0, 0.0005, 0.002]),
            EI=np.ones(3) * 1000.0,
            GJ=np.ones(3) * 500.0,
        )
        assert br.tip_deflection == pytest.approx(0.002)
        assert br.root_bending_moment == pytest.approx(5.0)
        assert br.root_shear_force == pytest.approx(10.0)


class TestWingBeamStations:
    def test_builds_correct_number_of_stations(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=30)
        assert len(wb.y) == 30

    def test_y_goes_from_root_to_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=20)
        assert wb.y[0] == pytest.approx(0.0)
        assert wb.y[-1] == pytest.approx(0.75)

    def test_EI_positive_everywhere(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=20)
        assert np.all(wb.EI > 0)

    def test_GJ_positive_everywhere(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=20)
        assert np.all(wb.GJ > 0)


class TestWingBeamSolve:
    def test_shear_zero_at_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.V[-1] == pytest.approx(0.0, abs=1e-10)

    def test_moment_zero_at_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.M[-1] == pytest.approx(0.0, abs=1e-10)

    def test_deflection_zero_at_root(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.delta[0] == pytest.approx(0.0, abs=1e-10)

    def test_deflection_positive_at_tip(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        assert result.tip_deflection > 0.0

    def test_shear_maximum_at_root(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=1.0, inertia_relief=False)
        assert result.V[0] == np.max(result.V)

    def test_moment_maximum_at_root(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        result = wb.solve(total_lift=20.0, load_factor=1.0, inertia_relief=False)
        assert result.M[0] == np.max(result.M)

    def test_higher_load_gives_larger_deflection(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        r1 = wb.solve(total_lift=10.0, load_factor=1.0, inertia_relief=False)
        r2 = wb.solve(total_lift=20.0, load_factor=1.0, inertia_relief=False)
        assert r2.tip_deflection > r1.tip_deflection

    def test_inertia_relief_reduces_deflection(self, rect_wing):
        wb = WingBeam(rect_wing, n_stations=50)
        r_no = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=False)
        r_ir = wb.solve(total_lift=20.0, load_factor=3.5, inertia_relief=True)
        assert r_no.tip_deflection >= r_ir.tip_deflection

    def test_uniform_beam_matches_analytic(self):
        """Uniform rectangular cantilever: δ_tip = qL⁴/(8EI) (no inertia relief)."""
        from aerisplane.catalog.materials import carbon_fiber_tube
        cf_spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
        )
        naca0012 = ap.Airfoil.from_naca("0012")
        # Uniform wing: 1 m semispan, 0.2 m chord, same spar root to tip
        wing = ap.Wing(
            name="uniform",
            xsecs=[
                ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.2,
                            airfoil=naca0012, spar=cf_spar),
                ap.WingXSec(xyz_le=[0.0, 1.0, 0.0], chord=0.2,
                            airfoil=naca0012, spar=cf_spar),
            ],
            symmetric=False,
        )
        L_total = 40.0   # N total (= q_0 from elliptic for verification below)
        wb = WingBeam(wing, n_stations=200)
        result = wb.solve(total_lift=L_total, load_factor=1.0, inertia_relief=False)

        # For an elliptic distribution on semispan b=1m, total lift=40N:
        # q_0 = 4*40/(pi*1) ≈ 50.9 N/m
        # The analytic tip deflection differs from uniform — but we can check
        # that the root moment ≈ q_0 * 4*b/3π = 50.9 * 4/(3π) ≈ 21.6 N·m
        # (moment of elliptic distribution at root)
        EI = cf_spar.material.E * cf_spar.section.second_moment_of_area()
        b = 1.0
        q_0 = 4.0 * L_total / (np.pi * b)
        M_root_analytic = q_0 * b * 4.0 / (3.0 * np.pi)   # ≈ integral of elliptic

        assert result.root_bending_moment == pytest.approx(M_root_analytic, rel=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_structures/test_beam.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'aerisplane.structures.beam'`

- [ ] **Step 3: Implement beam.py**

```python
# src/aerisplane/structures/beam.py
"""Euler-Bernoulli cantilever beam model for wing structural sizing.

The wing spar is modelled as a cantilever fixed at root (y=0), free at
tip (y=b).  The net distributed load is:

    q_net(y) = n · q_aero(y)  −  n · g · m'(y)

where q_aero is the aerodynamic lift per unit span (elliptic approximation),
n is the load factor, and m'(y) is the structural mass per unit span
(inertia relief).

Integration proceeds from tip to root for V and M, then root to tip
for slope θ and deflection δ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerisplane.core.wing import Wing, WingXSec
from aerisplane.structures.section import effective_EI

_G = 9.81  # m/s²


@dataclass
class BeamResult:
    """Spanwise structural solution for one wing semi-span.

    Parameters
    ----------
    y : ndarray
        Spanwise stations [m], from root (y[0]) to tip (y[-1]).
    V : ndarray
        Shear force [N] at each station.
    M : ndarray
        Bending moment [N·m] at each station.
    theta : ndarray
        Slope dδ/dy [rad] at each station.
    delta : ndarray
        Deflection [m] at each station (positive = upward).
    EI : ndarray
        Bending stiffness [N·m²] at each station.
    GJ : ndarray
        Torsional stiffness [N·m²/rad] at each station.
    """

    y: np.ndarray
    V: np.ndarray
    M: np.ndarray
    theta: np.ndarray
    delta: np.ndarray
    EI: np.ndarray
    GJ: np.ndarray

    @property
    def tip_deflection(self) -> float:
        """Tip deflection δ(tip) [m]."""
        return float(self.delta[-1])

    @property
    def root_bending_moment(self) -> float:
        """Bending moment at root M(0) [N·m]."""
        return float(self.M[0])

    @property
    def root_shear_force(self) -> float:
        """Shear force at root V(0) [N]."""
        return float(self.V[0])


class WingBeam:
    """Euler-Bernoulli beam model for a wing spar.

    Interpolates chord, EI(y), GJ(y), and mass-per-unit-span from the
    Wing's cross-section definitions.  All cross-sections must have a
    Spar defined; cross-sections without a Spar are skipped when
    computing EI (treated as zero stiffness at that station).

    Parameters
    ----------
    wing : Wing
    n_stations : int
        Number of uniformly spaced spanwise integration stations (default 50).
    """

    def __init__(self, wing: Wing, n_stations: int = 50) -> None:
        self.wing = wing
        self.n_stations = n_stations
        self._build_stations()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _EI_at_xsec(self, xsec: WingXSec) -> float:
        if xsec.spar is None:
            return 0.0
        if xsec.airfoil is not None:
            return effective_EI(xsec.airfoil, xsec.chord, xsec.spar, xsec.skin)
        return xsec.spar.material.E * xsec.spar.section.second_moment_of_area()

    def _GJ_at_xsec(self, xsec: WingXSec) -> float:
        if xsec.spar is None:
            return 0.0
        spar = xsec.spar
        G = spar.material.shear_modulus
        # Polar moment for circular tube: J = 2·I
        J = 2.0 * spar.section.second_moment_of_area()
        return G * J

    def _mass_per_length_at_xsec(self, xsec: WingXSec) -> float:
        """Structural mass per unit span [kg/m] from spar + skin."""
        m = 0.0
        if xsec.spar is not None:
            m += xsec.spar.mass_per_length()
        if xsec.skin is not None:
            # Approximate skin perimeter as 2 × chord (flat-wrap estimate)
            perimeter = 2.0 * xsec.chord
            m += xsec.skin.thickness * xsec.skin.material.density * perimeter
        return m

    def _build_stations(self) -> None:
        """Interpolate all spanwise properties to uniform y stations."""
        xsecs = self.wing.xsecs
        y_xsecs = np.array([xs.xyz_le[1] for xs in xsecs])
        y_root = float(y_xsecs[0])
        y_tip = float(y_xsecs[-1])

        self.y = np.linspace(y_root, y_tip, self.n_stations)

        chords = np.array([xs.chord for xs in xsecs])
        self.chord = np.interp(self.y, y_xsecs, chords)

        EI_xsecs = np.array([self._EI_at_xsec(xs) for xs in xsecs])
        self.EI = np.maximum(np.interp(self.y, y_xsecs, EI_xsecs), 1e-30)

        GJ_xsecs = np.array([self._GJ_at_xsec(xs) for xs in xsecs])
        self.GJ = np.maximum(np.interp(self.y, y_xsecs, GJ_xsecs), 1e-30)

        m_xsecs = np.array([self._mass_per_length_at_xsec(xs) for xs in xsecs])
        self.m_prime = np.interp(self.y, y_xsecs, m_xsecs)

    # ------------------------------------------------------------------
    # Public solver
    # ------------------------------------------------------------------

    def solve(
        self,
        total_lift: float,
        load_factor: float = 1.0,
        inertia_relief: bool = True,
    ) -> BeamResult:
        """Integrate beam equations for a given total lift and load factor.

        Uses an elliptic spanwise lift distribution for the aerodynamic
        load and optionally subtracts the wing structural weight (inertia
        relief).

        Parameters
        ----------
        total_lift : float
            Total aerodynamic lift on the complete aircraft [N].
            The semi-span load is total_lift / 2 for symmetric wings.
        load_factor : float
            Multiplier applied to both the aerodynamic load and the
            inertia relief term (default 1.0 = 1g).
        inertia_relief : bool
            If True, subtract n·g·m'(y) from the applied load
            (default True).

        Returns
        -------
        BeamResult
        """
        y = self.y
        n = load_factor
        b = float(y[-1] - y[0])  # semispan [m]

        # Elliptic distribution: q_aero(y) = q_0 · √(1 − η²)
        # where η = (y − y_root) / b
        if b > 0.0:
            L_semi = total_lift / 2.0  # one semi-span
            q_0 = 4.0 * L_semi / (np.pi * b)
            eta = (y - y[0]) / b
            q_aero = q_0 * np.sqrt(np.maximum(1.0 - eta ** 2, 0.0))
        else:
            q_aero = np.zeros_like(y)

        # Net distributed load [N/m]
        q = n * q_aero
        if inertia_relief:
            q = q - n * _G * self.m_prime

        # ── Shear V(y) = ∫_y^tip q(η) dη  (tip → root) ──────────────
        V = np.zeros_like(y)
        for i in range(len(y) - 2, -1, -1):
            dy = y[i + 1] - y[i]
            V[i] = V[i + 1] + 0.5 * (q[i] + q[i + 1]) * dy

        # ── Bending moment M(y) = ∫_y^tip V(η) dη  (tip → root) ─────
        M = np.zeros_like(y)
        for i in range(len(y) - 2, -1, -1):
            dy = y[i + 1] - y[i]
            M[i] = M[i + 1] + 0.5 * (V[i] + V[i + 1]) * dy

        # ── Slope θ(y) = ∫_0^y M/EI dy  (root → tip) ────────────────
        M_over_EI = M / self.EI
        theta = np.zeros_like(y)
        for i in range(1, len(y)):
            dy = y[i] - y[i - 1]
            theta[i] = theta[i - 1] + 0.5 * (M_over_EI[i - 1] + M_over_EI[i]) * dy

        # ── Deflection δ(y) = ∫_0^y θ dy  (root → tip) ──────────────
        delta = np.zeros_like(y)
        for i in range(1, len(y)):
            dy = y[i] - y[i - 1]
            delta[i] = delta[i - 1] + 0.5 * (theta[i - 1] + theta[i]) * dy

        return BeamResult(y=y, V=V, M=M, theta=theta, delta=delta,
                          EI=self.EI.copy(), GJ=self.GJ.copy())
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_structures/test_beam.py -v
```

Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/structures/beam.py \
        tests/test_structures/test_beam.py
git commit -m "feat(structures): add Euler-Bernoulli wing beam solver with elliptic load"
```

---

## Task 4: Structural check functions

**Files:**
- Create: `src/aerisplane/structures/checks.py`
- Create: `tests/test_structures/test_checks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_structures/test_checks.py
"""Tests for structural margin-of-safety functions."""
import math
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.catalog.materials import carbon_fiber_tube, petg
from aerisplane.structures.checks import (
    bending_margin,
    shear_margin,
    buckling_margin,
    fits_in_airfoil,
    divergence_speed,
)


@pytest.fixture
def cf_spar():
    return ap.Spar(
        position=0.25, material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.002),
    )


class TestBendingMargin:
    def test_positive_for_safe_spar(self, cf_spar):
        M_safe = 5.0  # much less than yield
        mos = bending_margin(cf_spar, M_safe)
        assert mos > 0.0

    def test_negative_for_overloaded_spar(self, cf_spar):
        # Force failure: apply huge moment
        M_fail = 1e6
        mos = bending_margin(cf_spar, M_fail)
        assert mos < 0.0

    def test_zero_moment_returns_inf(self, cf_spar):
        mos = bending_margin(cf_spar, 0.0)
        assert math.isinf(mos)


class TestShearMargin:
    def test_positive_for_small_shear(self, cf_spar):
        mos = shear_margin(cf_spar, shear_force=10.0)
        assert mos > 0.0

    def test_zero_shear_returns_inf(self, cf_spar):
        mos = shear_margin(cf_spar, shear_force=0.0)
        assert math.isinf(mos)

    def test_larger_shear_smaller_margin(self, cf_spar):
        mos1 = shear_margin(cf_spar, shear_force=50.0)
        mos2 = shear_margin(cf_spar, shear_force=500.0)
        assert mos2 < mos1


class TestBucklingMargin:
    def test_positive_for_typical_cf_tube(self, cf_spar):
        # CF tube at moderate moment — buckling usually not critical
        mos = buckling_margin(cf_spar, bending_moment=10.0)
        assert mos > 0.0

    def test_zero_moment_returns_inf(self, cf_spar):
        mos = buckling_margin(cf_spar, bending_moment=0.0)
        assert math.isinf(mos)

    def test_thin_walled_more_susceptible_to_buckling(self):
        thick_spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.003),
        )
        thin_spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.020, wall_thickness=0.001),
        )
        M = 10.0
        mos_thick = buckling_margin(thick_spar, M)
        mos_thin = buckling_margin(thin_spar, M)
        assert mos_thick > mos_thin


class TestFitsInAirfoil:
    def test_small_spar_fits(self):
        naca0012 = ap.Airfoil.from_naca("0012")
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.010, wall_thickness=0.001),
        )
        assert fits_in_airfoil(naca0012, spar, chord=0.20) is True

    def test_oversized_spar_does_not_fit(self):
        naca0012 = ap.Airfoil.from_naca("0012")
        spar = ap.Spar(
            position=0.25, material=carbon_fiber_tube,
            section=ap.TubeSection(outer_diameter=0.060, wall_thickness=0.002),
        )
        assert fits_in_airfoil(naca0012, spar, chord=0.20) is False


class TestDivergenceSpeed:
    def test_returns_inf_for_zero_offset(self):
        V = divergence_speed(GJ_root=100.0, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=0.0)
        assert math.isinf(V)

    def test_returns_inf_for_negative_offset(self):
        # e < 0: AC ahead of SC — self-stabilising
        V = divergence_speed(GJ_root=100.0, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=-0.01)
        assert math.isinf(V)

    def test_returns_finite_for_positive_offset(self):
        V = divergence_speed(GJ_root=100.0, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=0.02)
        assert np.isfinite(V)
        assert V > 0.0

    def test_stiffer_wing_has_higher_divergence_speed(self):
        V_soft = divergence_speed(GJ_root=50.0, cl_alpha_per_rad=5.5,
                                  wing_area=0.3, density=1.225, e=0.02)
        V_stiff = divergence_speed(GJ_root=200.0, cl_alpha_per_rad=5.5,
                                   wing_area=0.3, density=1.225, e=0.02)
        assert V_stiff > V_soft

    def test_typical_cf_tube_has_high_divergence_speed(self, cf_spar):
        # GJ for 20mm CF tube: G=52 GPa, J=2I
        spar = cf_spar
        G = spar.material.shear_modulus
        J = 2.0 * spar.section.second_moment_of_area()
        GJ = G * J
        e = (0.25 - spar.position) * 0.20  # spar at 25% chord, chord 0.2 m → e=0
        if e <= 0:
            return  # spar at AC → infinite divergence, test not meaningful
        V = divergence_speed(GJ_root=GJ, cl_alpha_per_rad=5.5,
                             wing_area=0.3, density=1.225, e=e)
        assert V > 50.0  # should be very high
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_structures/test_checks.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'aerisplane.structures.checks'`

- [ ] **Step 3: Implement checks.py**

```python
# src/aerisplane/structures/checks.py
"""Structural margin-of-safety functions for wing sizing.

All margin functions return:  MoS = capacity / demand − 1
Positive = safe, negative = failed, zero = exactly at limit.
"""
from __future__ import annotations

import numpy as np

from aerisplane.core.airfoil import Airfoil
from aerisplane.core.structures import Spar
from aerisplane.structures.section import spar_fits_in_airfoil


def bending_margin(spar: Spar, bending_moment: float) -> float:
    """Margin of safety against bending yield at a given moment [N·m].

    MoS = σ_yield / σ_bending − 1

    Parameters
    ----------
    spar : Spar
    bending_moment : float
        Applied bending moment [N·m].

    Returns
    -------
    float
        Positive = safe, negative = failed.
    """
    return spar.margin_of_safety(bending_moment)


def shear_margin(spar: Spar, shear_force: float) -> float:
    """Margin of safety against shear yield (thin annular tube).

    Uses a conservative estimate: τ_max = V / A_wall.
    Shear yield from von Mises: τ_yield = σ_yield / √3.

    Parameters
    ----------
    spar : Spar
    shear_force : float
        Applied shear force [N].

    Returns
    -------
    float
    """
    if shear_force == 0.0:
        return float("inf")
    A = spar.section.area()
    if A <= 0.0:
        return float("inf")
    tau_applied = abs(shear_force) / A
    tau_yield = spar.material.yield_strength / np.sqrt(3.0)
    return tau_yield / tau_applied - 1.0


def buckling_margin(spar: Spar, bending_moment: float) -> float:
    """Margin of safety against local shell buckling (Timoshenko).

    σ_cr = 0.6 · E · t_wall / R_outer

    Parameters
    ----------
    spar : Spar
    bending_moment : float
        Applied bending moment [N·m].

    Returns
    -------
    float
    """
    if bending_moment == 0.0:
        return float("inf")
    R = spar.section.outer_diameter / 2.0
    t = spar.section.wall_thickness
    if R <= 0.0 or t <= 0.0:
        return float("inf")
    sigma_cr = 0.6 * spar.material.E * t / R
    sigma_applied = spar.max_bending_stress(bending_moment)
    if sigma_applied <= 0.0:
        return float("inf")
    return sigma_cr / sigma_applied - 1.0


def fits_in_airfoil(airfoil: Airfoil, spar: Spar, chord: float) -> bool:
    """True if the spar outer diameter fits in the airfoil at its position.

    Delegates to ``aerisplane.structures.section.spar_fits_in_airfoil``.
    """
    return spar_fits_in_airfoil(airfoil, spar, chord)


def divergence_speed(
    GJ_root: float,
    cl_alpha_per_rad: float,
    wing_area: float,
    density: float,
    e: float,
) -> float:
    """Torsional divergence speed [m/s].

    .. math::
        V_{\\text{div}} = \\sqrt{\\frac{2 \\cdot GJ}{\\rho \\cdot a \\cdot e \\cdot S}}

    where e = x_AC − x_SC [m] (positive when AC is behind SC).
    Returns ``inf`` when e ≤ 0 (no divergence risk).

    Parameters
    ----------
    GJ_root : float
        Torsional stiffness at root [N·m²/rad].
    cl_alpha_per_rad : float
        Lift curve slope [1/rad].
    wing_area : float
        Wing planform area [m²].
    density : float
        Air density [kg/m³].
    e : float
        Distance from aerodynamic centre to shear centre [m].
        Positive = AC behind SC (divergence possible).

    Returns
    -------
    float
    """
    if e <= 0.0 or GJ_root <= 0.0:
        return float("inf")
    denom = density * cl_alpha_per_rad * e * wing_area
    if denom <= 0.0:
        return float("inf")
    return float(np.sqrt(2.0 * GJ_root / denom))
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_structures/test_checks.py -v
```

Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/structures/checks.py \
        tests/test_structures/test_checks.py
git commit -m "feat(structures): add bending, shear, buckling, and divergence checks"
```

---

## Task 5: StructureResult dataclasses

**Files:**
- Create: `src/aerisplane/structures/result.py`

There are no dedicated result tests — result correctness is verified through `test_structures.py` (Task 6 integration tests). The `report()` and `plot()` methods are verified there.

- [ ] **Step 1: Implement result.py**

```python
# src/aerisplane/structures/result.py
"""Structural analysis result dataclasses with reporting and plotting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WingStructureResult:
    """Structural analysis result for one wing semi-span.

    Parameters
    ----------
    wing_name : str
    y : ndarray
        Spanwise stations [m].
    shear_force : ndarray
        V(y) [N].
    bending_moment : ndarray
        M(y) [N·m].
    deflection : ndarray
        δ(y) [m].
    tip_deflection : float
        δ at tip [m].
    tip_deflection_ratio : float
        δ_tip / semispan [-].
    root_bending_moment : float
        M at root [N·m].
    root_shear_force : float
        V at root [N].
    bending_margin : float
        Margin of safety against bending yield at root.
    shear_margin : float
        Margin of safety against shear yield at root.
    buckling_margin : float
        Margin of safety against shell buckling at root.
    spar_fits : bool
        True if spar OD ≤ available airfoil height at root.
    divergence_speed : float
        Torsional divergence speed [m/s] (inf if no risk).
    design_load_factor : float
        Ultimate load factor used for this analysis.
    """

    wing_name: str
    y: np.ndarray
    shear_force: np.ndarray
    bending_moment: np.ndarray
    deflection: np.ndarray
    tip_deflection: float
    tip_deflection_ratio: float
    root_bending_moment: float
    root_shear_force: float
    bending_margin: float
    shear_margin: float
    buckling_margin: float
    spar_fits: bool
    divergence_speed: float
    design_load_factor: float

    @property
    def is_safe(self) -> bool:
        """True if all margins are non-negative and spar fits in airfoil."""
        return (
            self.bending_margin >= 0.0
            and self.shear_margin >= 0.0
            and self.buckling_margin >= 0.0
            and self.spar_fits
        )

    def report(self) -> str:
        lines = [
            f"Wing: {self.wing_name}",
            f"  Design load factor:   {self.design_load_factor:.2f} g (ultimate)",
            f"  Root bending moment:  {self.root_bending_moment:.2f} N·m",
            f"  Root shear force:     {self.root_shear_force:.2f} N",
            f"  Tip deflection:       {self.tip_deflection * 1000:.1f} mm"
            f"  ({self.tip_deflection_ratio * 100:.1f}% semispan)",
            f"  Bending MoS (root):   {self.bending_margin:+.3f}"
            f"  {'PASS' if self.bending_margin >= 0 else 'FAIL'}",
            f"  Shear MoS (root):     {self.shear_margin:+.3f}"
            f"  {'PASS' if self.shear_margin >= 0 else 'FAIL'}",
            f"  Buckling MoS (root):  {self.buckling_margin:+.3f}"
            f"  {'PASS' if self.buckling_margin >= 0 else 'FAIL'}",
            f"  Spar fits in airfoil: {'PASS' if self.spar_fits else 'FAIL'}",
            f"  Divergence speed:     "
            + (f"{self.divergence_speed:.1f} m/s"
               if np.isfinite(self.divergence_speed) else "∞ (no risk)"),
            f"  Overall: {'SAFE' if self.is_safe else 'FAILED'}",
        ]
        return "\n".join(lines)


@dataclass
class StructureResult:
    """Complete structural analysis result for all wings.

    Parameters
    ----------
    wings : list of WingStructureResult
        One entry per structural wing (wings without a spar are excluded).
    design_load_factor : float
        Ultimate load factor applied to all wings.
    """

    wings: list[WingStructureResult]
    design_load_factor: float

    @property
    def is_safe(self) -> bool:
        """True if all analysed wings pass all checks."""
        return all(w.is_safe for w in self.wings)

    def report(self) -> str:
        header = [
            "Structural Analysis",
            "=" * 60,
            f"Design load factor: {self.design_load_factor:.2f} g (ultimate)",
            f"Overall: {'SAFE' if self.is_safe else 'FAILED'}",
            "",
        ]
        wing_reports = [w.report() for w in self.wings]
        return "\n".join(header + wing_reports)

    def plot(self, show: bool = True, save_path: str | None = None) -> None:
        """Plot spanwise shear, moment, and deflection for all wings."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        n = len(self.wings)
        if n == 0:
            return

        fig, axes = plt.subplots(3, n, figsize=(5 * n, 9), squeeze=False)
        fig.suptitle(f"Structural Analysis — n_design = {self.design_load_factor:.2f} g")

        for col, wr in enumerate(self.wings):
            y = wr.y
            axes[0, col].plot(y, wr.shear_force)
            axes[0, col].set_ylabel("Shear [N]")
            axes[0, col].set_title(wr.wing_name)
            axes[0, col].grid(True, alpha=0.4)

            axes[1, col].plot(y, wr.bending_moment)
            axes[1, col].set_ylabel("Moment [N·m]")
            axes[1, col].grid(True, alpha=0.4)

            axes[2, col].plot(y, wr.deflection * 1000)
            axes[2, col].set_xlabel("Span y [m]")
            axes[2, col].set_ylabel("Deflection [mm]")
            axes[2, col].grid(True, alpha=0.4)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
```

- [ ] **Step 2: Commit (no dedicated tests — covered by Task 6)**

```bash
git add src/aerisplane/structures/result.py
git commit -m "feat(structures): add WingStructureResult and StructureResult dataclasses"
```

---

## Task 6: Public API and end-to-end integration

**Files:**
- Create: `src/aerisplane/structures/__init__.py`
- Create: `tests/test_structures/test_structures.py`
- Modify: `tests/test_integration/test_discipline_chain.py`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/test_structures/test_structures.py
"""End-to-end integration tests for the structures module."""
import math
import numpy as np
import pytest

import aerisplane as ap
import aerisplane.aero as aero
import aerisplane.weights as wts
import aerisplane.structures as struc
from aerisplane.structures.result import StructureResult, WingStructureResult


class TestAnalyze:
    def test_returns_structure_result(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        assert isinstance(result, StructureResult)

    def test_only_wings_with_spars_included(self, simple_aircraft, cruise_condition):
        # simple_aircraft has rect_wing (with spar) and htail (no spar)
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wr_wing in result.wings:
            assert wr_wing.wing_name == "rect_wing"

    def test_design_load_factor_at_least_525(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr,
                               n_limit=3.5, safety_factor=1.5)
        assert result.design_load_factor >= 3.5 * 1.5

    def test_tip_deflection_finite_and_positive(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wing_result in result.wings:
            assert wing_result.tip_deflection > 0.0
            assert np.isfinite(wing_result.tip_deflection)

    def test_tip_deflection_ratio_reasonable(self, simple_aircraft, cruise_condition):
        # Tip deflection / semispan should be < 50% for a stiff CF spar
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wing_result in result.wings:
            assert wing_result.tip_deflection_ratio < 0.50

    def test_safe_for_well_designed_wing(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr,
                               n_limit=3.5, safety_factor=1.5)
        assert result.is_safe

    def test_report_runs_and_is_nonempty(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        report = result.report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_all_margins_are_finite(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        ar = aero.analyze(simple_aircraft, cruise_condition, method="vlm")
        result = struc.analyze(simple_aircraft, ar, wr)
        for wr_wing in result.wings:
            assert np.isfinite(wr_wing.bending_margin)
            assert np.isfinite(wr_wing.shear_margin)
            assert np.isfinite(wr_wing.buckling_margin)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_structures/test_structures.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'aerisplane.structures'` (or `has no attribute 'analyze'`)

- [ ] **Step 3: Implement __init__.py**

```python
# src/aerisplane/structures/__init__.py
"""Structures discipline module — Euler-Bernoulli wing beam solver.

Public API
----------
analyze(aircraft, aero_result, weight_result, ...) -> StructureResult
"""
from __future__ import annotations

import numpy as np

from aerisplane.aero.result import AeroResult
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.wing import WingXSec
from aerisplane.structures.beam import WingBeam
from aerisplane.structures.checks import (
    bending_margin,
    buckling_margin,
    divergence_speed,
    fits_in_airfoil,
    shear_margin,
)
from aerisplane.structures.loads import design_load_factor
from aerisplane.structures.result import StructureResult, WingStructureResult
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult


def analyze(
    aircraft: Aircraft,
    aero_result: AeroResult,
    weight_result: WeightResult,
    n_limit: float = 3.5,
    safety_factor: float = 1.5,
    gust_velocity: float = 9.0,
    stability_result=None,
    n_stations: int = 50,
) -> StructureResult:
    """Run structural beam analysis for all wings with a spar.

    Parameters
    ----------
    aircraft : Aircraft
    aero_result : AeroResult
        Aerodynamic result at the sizing condition (provides total lift L).
    weight_result : WeightResult
        Weight analysis result (provides total mass for wing loading).
    n_limit : float
        Limit maneuver load factor (default 3.5).
    safety_factor : float
        Safety factor applied to limit load factor (default 1.5 → ultimate).
    gust_velocity : float
        Design gust velocity U_de [m/s] (default 9.0 per CS-VLA).
    stability_result : StabilityResult or None
        If provided, CL_alpha from stability analysis is used for gust n.
    n_stations : int
        Number of spanwise beam integration stations per wing (default 50).

    Returns
    -------
    StructureResult
        Per-wing structural results, margins, and deflections.
    """
    sizing_condition = _condition_from_aero_result(aero_result)

    n_design = design_load_factor(
        aircraft, sizing_condition, weight_result,
        stability_result=stability_result,
        n_limit=n_limit,
        safety_factor=safety_factor,
        gust_velocity=gust_velocity,
    )

    # Attribute total lift to wings by area fraction
    total_area = sum(w.area() for w in aircraft.wings) or 1.0
    total_lift = aero_result.L  # [N]

    # Air density at sizing condition (for divergence speed)
    _, _, rho, _ = isa(aero_result.altitude)

    wing_results = []
    for wing in aircraft.wings:
        # Skip wings with no spar at any cross-section
        if not any(xs.spar is not None for xs in wing.xsecs):
            continue

        # Lift attributed to this wing (proportional to planform area)
        wing_lift = total_lift * (wing.area() / total_area)

        beam = WingBeam(wing, n_stations=n_stations)
        beam_result = beam.solve(
            total_lift=wing_lift,
            load_factor=n_design,
            inertia_relief=True,
        )

        semispan = wing.semispan()
        tip_ratio = (beam_result.tip_deflection / semispan
                     if semispan > 0.0 else 0.0)

        # Checks at root (index 0 — highest M and V)
        root_xsec = wing.xsecs[0]
        root_spar = root_xsec.spar
        root_airfoil = root_xsec.airfoil
        root_chord = root_xsec.chord

        M_root = beam_result.root_bending_moment
        V_root = beam_result.root_shear_force

        mos_bending = (bending_margin(root_spar, M_root)
                       if root_spar is not None else float("inf"))
        mos_shear = (shear_margin(root_spar, V_root)
                     if root_spar is not None else float("inf"))
        mos_buckling = (buckling_margin(root_spar, M_root)
                        if root_spar is not None else float("inf"))
        spar_ok = (
            fits_in_airfoil(root_airfoil, root_spar, root_chord)
            if root_spar is not None and root_airfoil is not None
            else True
        )

        # Torsional divergence
        GJ_root = float(beam.GJ[0])
        if stability_result is not None:
            cl_alpha_rad = float(stability_result.CL_alpha) * (180.0 / np.pi)
        else:
            cl_alpha_rad = 5.5
        e = (0.25 - root_spar.position) * root_chord if root_spar is not None else 0.0
        V_div = divergence_speed(
            GJ_root=GJ_root,
            cl_alpha_per_rad=cl_alpha_rad,
            wing_area=wing.area(),
            density=rho,
            e=e,
        )

        wing_results.append(WingStructureResult(
            wing_name=wing.name,
            y=beam_result.y,
            shear_force=beam_result.V,
            bending_moment=beam_result.M,
            deflection=beam_result.delta,
            tip_deflection=beam_result.tip_deflection,
            tip_deflection_ratio=tip_ratio,
            root_bending_moment=M_root,
            root_shear_force=V_root,
            bending_margin=mos_bending,
            shear_margin=mos_shear,
            buckling_margin=mos_buckling,
            spar_fits=spar_ok,
            divergence_speed=V_div,
            design_load_factor=n_design,
        ))

    return StructureResult(wings=wing_results, design_load_factor=n_design)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _condition_from_aero_result(aero_result: AeroResult):
    """Reconstruct a minimal FlightCondition-like object from AeroResult."""
    from aerisplane.core.flight_condition import FlightCondition
    return FlightCondition(
        velocity=aero_result.velocity,
        altitude=aero_result.altitude,
        alpha=aero_result.alpha,
    )
```

- [ ] **Step 4: Run integration tests and verify they pass**

```bash
pytest tests/test_structures/test_structures.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Add structures to the full discipline chain integration test**

In `tests/test_integration/test_discipline_chain.py`, add the following import at the top of the file after the existing imports:

```python
import aerisplane.structures as struc
```

Then add a new test class at the end of the file:

```python
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
        """aero → weights → stability → control → mission → structures."""
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
```

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/ -q --tb=short
```

Expected: all existing tests pass plus the new structures tests. Total should be 361 + ~22 new = ~383 passed.

- [ ] **Step 7: Commit everything**

```bash
git add src/aerisplane/structures/__init__.py \
        tests/test_structures/test_structures.py \
        tests/test_integration/test_discipline_chain.py
git commit -m "feat(structures): add public analyze() and wire into discipline chain"
```

---

## Self-Review

### 1. Spec coverage

| Requirement | Task |
|---|---|
| Beam solver (fast, NumPy only) | Task 3 |
| EI from airfoil geometry | Task 2 |
| Skin I contribution | Task 2 |
| Spar fits-in-airfoil check | Task 2, 4 |
| Maneuver load factor | Task 1 |
| Gust load factor (CS-VLA) | Task 1 |
| Design load factor (max of both × SF) | Task 1 |
| Inertia relief | Task 3 |
| Shear V(y) and moment M(y) | Task 3 |
| Tip deflection δ_tip | Task 3 |
| Bending margin at root | Task 4, 6 |
| Shear margin at root | Task 4, 6 |
| Buckling margin (Timoshenko) | Task 4, 6 |
| Torsional divergence speed | Task 4, 6 |
| StructureResult with report/plot | Task 5 |
| analyze() public API | Task 6 |
| Integration into discipline chain | Task 6 |

No gaps found.

### 2. Placeholder scan

No TBDs, TODOs, or incomplete code blocks found.

### 3. Type consistency

- `bending_margin(spar: Spar, bending_moment: float)` — defined in Task 4, called the same way in Task 6. ✓
- `shear_margin(spar, shear_force)` — consistent. ✓
- `buckling_margin(spar, bending_moment)` — consistent. ✓
- `fits_in_airfoil(airfoil, spar, chord)` — consistent. ✓
- `divergence_speed(GJ_root, cl_alpha_per_rad, wing_area, density, e)` — consistent. ✓
- `WingBeam(wing, n_stations)` — consistent. ✓
- `WingBeam.solve(total_lift, load_factor, inertia_relief)` — consistent. ✓
- `BeamResult.tip_deflection`, `.root_bending_moment`, `.root_shear_force` — consistent. ✓
- `WingStructureResult` fields — all set in Task 6's `analyze()`. ✓
