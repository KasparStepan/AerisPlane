# Stability Module Implementation Plan

**Date:** 2026-03-28
**Branch:** `feature/vendor-asb-aero`
**Status:** COMPLETE

## Overview

Implement `src/aerisplane/stability/` module with solver-agnostic central finite-difference
stability derivatives. The module uses 5 aero evaluations to compute longitudinal and
lateral-directional derivatives, neutral point, static margin, trim, and tail volume coefficients.

**Inputs required:**
- `Aircraft` — geometry
- `FlightCondition` — operating point
- `WeightResult` — CG position (from `weights.analyze()`)
- `aero_method` — which aero solver to use (default `"vlm"`)

## Files to Create / Edit

| File | Action | Purpose |
|------|--------|---------|
| `src/aerisplane/stability/result.py` | Write | `StabilityResult` dataclass with `plot()` and `report()` |
| `src/aerisplane/stability/derivatives.py` | Write | `compute_derivatives()` — 5-point central FD |
| `src/aerisplane/stability/__init__.py` | Write | `analyze()` entry point |
| `tests/test_stability/__init__.py` | Write | Package init |
| `tests/test_stability/conftest.py` | Write | Test fixtures (aircraft with tail surfaces) |
| `tests/test_stability/test_derivatives.py` | Write | Tests for derivative computation |
| `tests/test_stability/test_stability.py` | Write | Integration tests for `analyze()` |

## Implementation Steps

### Step 1: `result.py` — StabilityResult dataclass
- [x] Define `StabilityResult` with all fields from spec:
  - Longitudinal: `static_margin`, `neutral_point`, `Cm_alpha`, `CL_alpha`
  - Lateral: `Cl_beta`, `Cn_beta`
  - Trim: `trim_alpha`, `trim_elevator`
  - Tail volumes: `Vh`, `Vv`
  - CG envelope: `cg_forward_limit`, `cg_aft_limit`
  - Reference: `cg_x` (input CG), `mac` (reference chord)
- [x] `report()` — formatted text output
- [x] `plot()` — Cm vs alpha curve + static margin diagram

### Step 2: `derivatives.py` — Central finite differences
- [x] `compute_derivatives(aircraft, condition, weight_result, aero_method, **aero_kwargs)`
  - Set `aircraft.xyz_ref` to CG from `weight_result` before aero calls
  - Run 5 aero evaluations:
    1. baseline (alpha, beta)
    2. (alpha + 0.5, beta)
    3. (alpha - 0.5, beta)
    4. (alpha, beta + 1.0)
    5. (alpha, beta - 1.0)
  - Compute: `CL_alpha`, `Cm_alpha`, `Cl_beta`, `Cn_beta`
  - Compute neutral point: `x_np = x_cg - (Cm_alpha / CL_alpha) * c_ref`
  - Compute static margin: `SM = (x_np - x_cg) / MAC`

### Step 3: Trim computation
- [x] `compute_trim(aircraft, condition, weight_result, aero_method, **aero_kwargs)`
  - Find `trim_alpha` where Cm = 0 (scipy.optimize.brentq)
  - Find `trim_elevator` for level flight (CL matches weight) at trim_alpha
  - Handle case where no elevator exists → return NaN

### Step 4: Tail volume coefficients
- [x] `compute_tail_volumes(aircraft)`
  - Identify horizontal tail and vertical tail from `aircraft.wings`
    - Heuristic: wing with smallest area that is symmetric → htail
    - Wing that is not symmetric and has ~90° dihedral → vtail
    - Or: match by name containing "htail"/"vtail"/"horizontal"/"vertical"
  - `Vh = (S_h * l_h) / (S_w * c_w)` where l_h = distance from wing AC to tail AC
  - `Vv = (S_v * l_v) / (S_w * b_w)` where l_v = distance from wing AC to vtail AC

### Step 5: `__init__.py` — analyze() entry point
- [x] Wire everything together:
  ```python
  def analyze(aircraft, condition, weight_result, aero_method="vlm", **aero_kwargs) -> StabilityResult
  ```
  - Call `compute_derivatives()`
  - Call `compute_trim()`
  - Call `compute_tail_volumes()`
  - Compute CG envelope limits (forward/aft from NP ± margin)
  - Return `StabilityResult`

### Step 6: Tests
- [x] `conftest.py` — aircraft fixture with main wing + htail + vtail + propulsion
- [x] `test_derivatives.py`:
  - Test that CL_alpha > 0
  - Test that Cm_alpha < 0 for stable aircraft
  - Test that static margin is positive and reasonable (5-30% MAC)
  - Test neutral point is aft of CG
- [x] `test_stability.py`:
  - Integration test: `analyze()` returns valid `StabilityResult`
  - Test `report()` produces non-empty string
  - Test trim_alpha is within reasonable range
  - Test tail volume coefficients are positive and reasonable

### Step 7: Verification
- [x] All tests pass
- [x] Run on a known aircraft config and verify results are physically reasonable
- [x] Verify `report()` output is clear and complete

## Key Design Decisions

1. **Solver-agnostic approach** — use `aero.analyze()` as black box, works with all 4 solvers
2. **Central finite differences** (not forward FD) — more accurate, spec-compliant
3. **Set xyz_ref to CG** — moments must be computed about the CG for stability derivatives
4. **Step sizes from spec** — d_alpha = 0.5 deg, d_beta = 1.0 deg
5. **Trim via root-finding** — scipy.optimize.brentq for robustness
6. **Tail identification by name** — user names wings, we match on common patterns

## Notes / Issues

- The `aircraft.xyz_ref` sets the moment reference point for aero solvers.
  We need to temporarily set it to the CG for stability analysis, then restore it.
  Alternative: copy the aircraft object. → Use `copy.deepcopy` to avoid side effects.
- VLM is inviscid — trim elevator may not be perfectly accurate but is fine for conceptual design.
- No vtail in test fixtures yet — need to create one for tail volume coefficient tests.
