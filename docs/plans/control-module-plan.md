# Control Authority Module Implementation Plan

**Date:** 2026-03-28
**Branch:** `feature/vendor-asb-aero`
**Status:** COMPLETE

## Overview

Implement `src/aerisplane/control/` — the control authority analysis module. This module
answers the question: "Can the aircraft's control surfaces actually control it?"

It computes control derivatives (how much moment does each surface produce per degree of
deflection), then translates those into physical performance metrics: maximum roll rate,
pitch acceleration, crosswind capability, and servo load margins.

### What the module does, step by step

1. **Identifies all control surfaces** on the aircraft (ailerons, elevator, rudder, flaps)
   by scanning `aircraft.wings[*].control_surfaces`. Each surface has a name, span extent,
   chord fraction, max/min deflection, and optionally an assigned servo.

2. **Computes control derivatives** by running aero at baseline and with each surface
   deflected to its max deflection. The difference in moment coefficients gives:
   - `Cl_delta_a` — roll moment per degree of aileron
   - `Cm_delta_e` — pitch moment per degree of elevator
   - `Cn_delta_r` — yaw moment per degree of rudder

3. **Estimates roll damping** (`Cl_p`) from strip theory so we can compute steady-state
   roll rate. Roll damping is the aerodynamic moment that opposes roll — it increases
   with roll rate until equilibrium.

4. **Computes physical authority metrics:**
   - Max roll rate from aileron
   - Max pitch acceleration from elevator
   - Max crosswind for coordinated flight from rudder

5. **Estimates hinge moments** to check if the assigned servos are strong enough.

6. **Normalizes authority** to a 0–1 scale against common RC/UAV requirements so the
   user can see at a glance if control is adequate.

### Inputs required

| Input | Source | What it provides |
|-------|--------|-----------------|
| `Aircraft` | user | Control surface geometry, wing geometry |
| `FlightCondition` | user | Velocity, altitude, alpha → dynamic pressure |
| `WeightResult` | `weights.analyze()` | Mass, CG, inertia tensor (I_xx, I_yy, I_zz) |
| `StabilityResult` | `stability.analyze()` | Cn_beta (for crosswind calc), trim state |
| `aero_method` | user | Which aero solver to use for the deflection runs |

### Outputs

A `ControlResult` dataclass with:
- Roll metrics: max_roll_rate, aileron_authority, Cl_delta_a
- Pitch metrics: elevator_authority, Cm_delta_e, max_pitch_acceleration
- Yaw metrics: rudder_authority, Cn_delta_r, max_crosswind
- Servo loads: hinge moments for each surface (None if no servo assigned)
- `report()` — formatted text summary
- `plot()` — authority bar chart + servo load margin diagram

---

## Files to Create / Edit

| File | Action | Purpose |
|------|--------|---------|
| `src/aerisplane/control/result.py` | Write | `ControlResult` dataclass with `plot()` and `report()` |
| `src/aerisplane/control/authority.py` | Write | Core computation: derivatives, roll rate, hinge moments |
| `src/aerisplane/control/__init__.py` | Write | `analyze()` entry point |
| `tests/test_control/__init__.py` | Write | Package init |
| `tests/test_control/conftest.py` | Write | Test fixtures (aircraft with aileron, elevator, rudder) |
| `tests/test_control/test_authority.py` | Write | Unit tests for authority computation |
| `tests/test_control/test_control.py` | Write | Integration tests for `analyze()` |

---

## Implementation Steps

### Step 1: `result.py` — ControlResult dataclass

Define the result dataclass following the spec and the pattern established in
`stability/result.py` and `weights/result.py`.

**Fields:**

```python
@dataclass
class ControlResult:
    # --- Roll ---
    max_roll_rate: float          # deg/s — steady-state roll rate at max aileron
    aileron_authority: float      # 0-1 normalized (1.0 = meets 180 deg/s requirement)
    Cl_delta_a: float             # dCl/d(aileron) [1/deg]

    # --- Pitch ---
    elevator_authority: float     # 0-1 normalized (1.0 = meets requirement)
    Cm_delta_e: float             # dCm/d(elevator) [1/deg]
    max_pitch_acceleration: float # deg/s^2 — max angular accel from full elevator

    # --- Yaw ---
    rudder_authority: float       # 0-1 normalized (1.0 = meets requirement)
    Cn_delta_r: float             # dCn/d(rudder) [1/deg]
    max_crosswind: float          # m/s — max crosswind speed for coordinated flight

    # --- Servo loads ---
    aileron_hinge_moment: float | None   # N*m — estimated hinge moment at max deflection
    elevator_hinge_moment: float | None  # N*m
    rudder_hinge_moment: float | None    # N*m

    # --- Servo adequacy ---
    aileron_servo_margin: float | None   # ratio: servo_torque / hinge_moment (>1 = OK)
    elevator_servo_margin: float | None
    rudder_servo_margin: float | None
```

**Methods:**
- `report()` — Formatted text table with:
  - Control derivatives section (Cl_da, Cm_de, Cn_dr)
  - Authority section (roll rate, pitch accel, crosswind)
  - Servo section (hinge moment vs servo torque, margin)
  - Pass/fail indicators for each axis

- `plot()` — matplotlib Figure with 2 subplots:
  1. **Authority bar chart**: horizontal bars for roll/pitch/yaw authority (0–1 scale)
     with a vertical line at 1.0 marking the requirement threshold. Color-coded
     green (>1.0) / yellow (0.5–1.0) / red (<0.5).
  2. **Servo load margin**: bar chart of servo torque margin for each surface.
     Shows required torque vs available torque. Surfaces without servos are omitted.

**NaN/None conventions:**
- If no aileron exists → `Cl_delta_a = 0.0`, `max_roll_rate = 0.0`, `aileron_authority = 0.0`
- If no elevator exists → `Cm_delta_e = 0.0`, etc.
- If no rudder exists → `Cn_delta_r = 0.0`, `max_crosswind = 0.0`
- If no servo assigned → `*_hinge_moment = None`, `*_servo_margin = None`

---

### Step 2: `authority.py` — Control derivative computation

This is the core computation file. It contains several functions:

#### 2a. `compute_control_derivatives(aircraft, condition, aero_method, **aero_kwargs)`

For each control surface type (aileron, elevator, rudder), run two aero evaluations:

1. **Baseline** — all control surfaces at zero deflection
2. **Deflected** — the target surface at its `max_deflection`, others at zero

Compute the derivative:
```
Cl_delta_a = (Cl_deflected - Cl_baseline) / max_deflection_deg
Cm_delta_e = (Cm_deflected - Cm_baseline) / max_deflection_deg
Cn_delta_r = (Cn_deflected - Cn_baseline) / max_deflection_deg
```

**Important detail:** The moment reference point must be set to the CG (same as
stability module). Use `copy.deepcopy(aircraft)` and set `xyz_ref` to the CG.

**Surface identification:** Scan all `aircraft.wings[*].control_surfaces` and
classify by name:
- Name contains "aileron" or "ail" → aileron
- Name contains "elevator" or "elev" → elevator
- Name contains "rudder" or "rud" → rudder
- Name contains "flap" → skip (not a control authority surface)

If multiple surfaces of the same type exist (e.g., left + right aileron), they
should be deflected together (the VLM already handles symmetric/antisymmetric
via the `ControlSurface.symmetric` flag).

**Number of aero evaluations:** 1 baseline + 1 per surface type found. Typically
3–4 total (baseline + aileron + elevator + rudder).

#### 2b. `estimate_roll_damping(aircraft, condition)`

Roll damping `Cl_p` is the derivative dCl/d(p̂) where p̂ = p*b/(2V) is the
non-dimensional roll rate. This tells us how quickly roll is damped.

**Strip theory estimate:**
```
Cl_p = -(CL_alpha_2d / 4) * integral from 0 to 1 of:
    (c(η) / c_ref) * η² dη

where:
    η = 2y/b (non-dimensional spanwise position)
    c(η) = local chord
    c_ref = reference chord (MAC)
    CL_alpha_2d ≈ 2*pi (thin airfoil theory) or from Wing CL_alpha
```

This is a purely geometric calculation — no aero solver calls needed. It uses
the main wing's chord distribution from `wing._chords()` and `wing._y_stations()`.

**For our simple rectangular wing:** Cl_p ≈ -(CL_alpha * b²) / (12 * S) simplified.

#### 2c. `compute_roll_rate(Cl_delta_a, delta_a_max, Cl_p, V, b)`

Steady-state roll rate from the aileron:
```
p_ss = -(Cl_delta_a * delta_a_max) / Cl_p * (2 * V / b)
```

Convert from rad/s to deg/s for the result.

**Physical meaning:** At steady-state roll, the aileron rolling moment exactly
balances the roll damping moment. More aileron authority or less damping → faster roll.

#### 2d. `compute_pitch_acceleration(Cm_delta_e, delta_e_max, q, S, c, I_yy)`

Maximum pitch angular acceleration from full elevator deflection:
```
alpha_dot_dot = (Cm_delta_e * delta_e_max * q * S * c) / I_yy
```

Convert from rad/s² to deg/s².

**Physical meaning:** How quickly can the elevator pitch the aircraft? Important
for maneuverability and recovery from stall.

#### 2e. `compute_max_crosswind(Cn_delta_r, delta_r_max, Cn_beta)`

Maximum sideslip (and hence crosswind) the rudder can counteract:
```
beta_max = (Cn_delta_r * delta_r_max) / Cn_beta  [degrees]
crosswind = V * sin(beta_max)                     [m/s]
```

**Physical meaning:** In a crosswind landing, the rudder must generate enough
yaw moment to fly at a sideslip angle. More rudder authority or less weathercock
stability → more crosswind capability.

Requires `Cn_beta` from `StabilityResult`. If `Cn_beta ≤ 0` (directionally
unstable), set `max_crosswind = inf` (rudder is not the limiting factor).

#### 2f. `estimate_hinge_moment(cs, wing, condition)`

Simplified hinge moment estimate using thin-airfoil theory:
```
H = Ch * q * S_cs * c_cs

where:
    Ch ≈ -0.6 * deflection_rad  (thin airfoil hinge moment coefficient)
    S_cs = surface area of the control surface [m²]
    c_cs = mean chord of the control surface [m]
    q = dynamic pressure [Pa]
```

Surface area is computed from:
```
S_cs = chord_fraction * integral(chord, dy) over [span_start, span_end] of the wing
```

For symmetric wings, this is the single-side area (the servo only drives one side).

The hinge moment is compared to the servo torque to get the servo margin:
```
servo_margin = servo.torque / abs(hinge_moment)
```

A margin > 1.0 means the servo has enough torque. Typical design target is > 1.5.

---

### Step 3: Authority normalization

Normalize each axis to a 0–1 scale against typical RC/UAV requirements:

| Axis | Requirement for authority = 1.0 | Source |
|------|--------------------------------|--------|
| Roll | 180 deg/s roll rate | Common RC aerobatic baseline |
| Pitch | Cm_delta_e sufficient to trim ±10° alpha range | Trim authority |
| Yaw | Rudder can hold 5 m/s crosswind | Common RC landing crosswind |

These are hardcoded defaults. The user can override them later via kwargs.

```python
aileron_authority = min(max_roll_rate / 180.0, 1.0)
elevator_authority = min(abs(Cm_delta_e * 25.0) / required_Cm_range, 1.0)
rudder_authority = min(max_crosswind / 5.0, 1.0)
```

---

### Step 4: `__init__.py` — analyze() entry point

```python
def analyze(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    stability_result: StabilityResult,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> ControlResult:
```

**Orchestration:**
1. Deep copy aircraft, set `xyz_ref` to CG from `weight_result`
2. Call `compute_control_derivatives()` → get Cl_da, Cm_de, Cn_dr + baseline aero
3. Call `estimate_roll_damping()` → get Cl_p
4. Call `compute_roll_rate()` → get max_roll_rate
5. Call `compute_pitch_acceleration()` using I_yy from `weight_result.inertia_tensor[1,1]`
6. Call `compute_max_crosswind()` using Cn_beta from `stability_result`
7. For each surface with a servo, call `estimate_hinge_moment()` → get hinge moments
8. Compute servo margins
9. Normalize authority metrics
10. Return `ControlResult`

**Total aero evaluations:** 1 baseline + 1 per surface type (typically 3–4 total).
Much cheaper than stability (which runs 5 + trim search).

---

### Step 5: Tests

#### `conftest.py`

Test fixtures need an aircraft with all three control surface types:
- **Main wing** with aileron (antisymmetric deflection, `symmetric=False`)
- **Horizontal tail** with elevator (symmetric deflection, `symmetric=True`)
- **Vertical tail** with rudder (non-symmetric wing, `symmetric=True` on surface)

Reuse the stability test aircraft pattern but ensure:
- Aileron has servo assigned
- Elevator has servo assigned
- Rudder has servo assigned (add one to the vtail)
- Reasonable max_deflection values (±25° default)

Also provide `flight_condition`, `weight_result`, and `stability_result` fixtures
(the stability result can be pre-computed once and cached with `@pytest.fixture(scope="module")`
to avoid repeated 5-point FD computation).

#### `test_authority.py` — Unit tests

- `test_cl_delta_a_nonzero` — aileron produces roll moment
- `test_cl_delta_a_sign` — positive aileron deflection → negative Cl (right roll)
  or positive depending on convention — just check it's non-zero
- `test_cm_delta_e_nonzero` — elevator produces pitch moment
- `test_cm_delta_e_negative` — trailing-edge-down elevator → nose-down Cm (negative)
- `test_cn_delta_r_nonzero` — rudder produces yaw moment
- `test_roll_damping_negative` — Cl_p should be negative (damping)
- `test_roll_rate_positive` — max roll rate > 0
- `test_pitch_acceleration_positive` — max pitch accel > 0
- `test_hinge_moment_finite` — hinge moments are finite numbers
- `test_no_surface_returns_zero` — aircraft without aileron → Cl_da = 0

#### `test_control.py` — Integration tests

- `test_returns_control_result` — `analyze()` returns ControlResult
- `test_report_non_empty` — `report()` has content
- `test_authority_range` — all authority values are 0 ≤ value ≤ some reasonable max
- `test_servo_margins_positive` — if servo exists, margin > 0
- `test_does_not_mutate_aircraft` — input aircraft unchanged
- `test_crosswind_positive` — max_crosswind ≥ 0

---

### Step 6: Verification

- [ ] All tests pass
- [ ] Run on the test aircraft and print `report()` — verify numbers are physically sensible
- [ ] Check that roll rate, pitch acceleration, and crosswind are in realistic ranges
  for a ~2 kg RC aircraft at 15 m/s
- [ ] Verify servo margins make sense (small servos on small surfaces → margin > 1)

---

## Key Design Decisions

1. **Forward finite differences for control derivatives** — not central differences.
   Only 1 deflected case per surface (baseline + max deflection). Central differences
   would need ± deflection (2 extra runs per surface). Since control derivatives are
   roughly linear in deflection, forward FD is adequate and saves aero evaluations.

2. **StabilityResult as input** — `Cn_beta` comes from stability analysis, not
   recomputed. This avoids 5 redundant aero calls and enforces the dependency:
   stability must be run before control.

3. **Strip theory for Cl_p** — avoids extra aero calls. Roll damping from strip
   theory is well-established and accurate enough for conceptual design.

4. **Thin-airfoil hinge moments** — simplified but reasonable for initial sizing.
   Real hinge moments depend on airfoil shape, gap seal, etc. This gives a
   conservative estimate for servo selection.

5. **Authority normalization** — hardcoded baselines (180 deg/s roll, 5 m/s crosswind).
   These are reasonable for RC/UAV but could be made configurable later.

## Dependencies

```
weights.analyze() → WeightResult (mass, CG, inertia)
       ↓
stability.analyze() → StabilityResult (Cn_beta)
       ↓
control.analyze() → ControlResult
```

The user must run `weights → stability → control` in order. The `analyze()` function
signature enforces this by requiring both `WeightResult` and `StabilityResult` as inputs.

## Estimated Aero Evaluations

| Operation | Calls |
|-----------|-------|
| Baseline (zero deflection) | 1 |
| Aileron deflected | 1 |
| Elevator deflected | 1 |
| Rudder deflected | 1 |
| **Total** | **4** |

Compare: stability module uses 5 + trim search (~7–9 total).
Combined stability + control: ~11–13 aero evaluations.
