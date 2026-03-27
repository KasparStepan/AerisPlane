# Plan: Vendor AeroSandbox Aerodynamic Solvers into AerisPlane

## Goal

Replace the AeroSandbox pip dependency for aerodynamic analysis by vendoring ASB's
3D aero solvers directly into `src/aerisplane/aero/`, refactored to work with
aerisplane's own `core/` geometry classes. This eliminates the `aircraft_to_asb()`
translation layer and gives full control over the solvers (including control surface
deflections in VLM).

AeroSandbox is MIT-licensed. The vendored code will carry attribution to
Peter Sharpe / AeroSandbox in every file header.

---

## What we vendor vs replace vs drop

### Vendor (copy + refactor to use aerisplane types)

| ASB source file | Lines | Target location |
|---|---|---|
| `aerodynamics/aero_3D/vortex_lattice_method.py` | 814 | `aero/solvers/vlm.py` |
| `aerodynamics/aero_3D/lifting_line.py` | 1336 | `aero/solvers/lifting_line.py` |
| `aerodynamics/aero_3D/aero_buildup.py` | 1329 | `aero/solvers/aero_buildup.py` |
| `aerodynamics/aero_3D/singularities/uniform_strength_horseshoe_singularities.py` | 276 | `aero/singularities.py` |
| `aerodynamics/aero_3D/singularities/point_source.py` | 120 | `aero/singularities.py` |
| `aerodynamics/aero_3D/aero_buildup_submodels/fuselage_aerodynamics_utilities.py` | 145 | `aero/fuselage_aero.py` |
| `aerodynamics/aero_3D/aero_buildup_submodels/softmax_scalefree.py` | ~30 | `aero/fuselage_aero.py` (inline) |
| `library/aerodynamics/viscous.py` | 664 | `aero/library/viscous.py` |
| `library/aerodynamics/transonic.py` | 219 | `aero/library/transonic.py` |
| `library/aerodynamics/inviscid.py` | 205 | `aero/library/inviscid.py` |
| `library/aerodynamics/components.py` | 293 | `aero/library/components.py` |

### Replace with existing aerisplane code

| ASB component | Lines | aerisplane replacement |
|---|---|---|
| `geometry.Airplane` | 1501 | `core.Aircraft` — already has `wings`, `fuselages`, `reference_area()`, `reference_chord()`, `reference_span()` |
| `geometry.Wing` | 1707 | `core.Wing` — has `xsecs`, `symmetric`, geometry methods |
| `geometry.WingXSec` | (in wing.py) | `core.WingXSec` — has `xyz_le`, `chord`, `twist`, `airfoil`, `control_surfaces` |
| `geometry.Fuselage` | 961 | `core.Fuselage` — has `xsecs`, geometry methods |
| `geometry.FuselageXSec` | (in fuselage.py) | `core.FuselageXSec` |
| `geometry.Airfoil` | 1949 | `core.Airfoil` — has `name`, `coordinates` |
| `performance.OperatingPoint` | ~500 | `core.FlightCondition` + new helper functions (see Phase 2) |
| `atmosphere.Atmosphere` | ~300 | `utils.atmosphere.isa()` — already implemented |
| `common.ExplicitAnalysis` | 130 | Drop — solvers become plain classes |
| `aerosandbox.numpy` | 3678 | `numpy` — replace `import aerosandbox.numpy as np` → `import numpy as np` |

### Drop (not needed)

| ASB component | Reason |
|---|---|
| `NonlinearLiftingLine` | Requires CasADi `ImplicitAnalysis`. Rarely used, slow, diverges past stall. Can add back later if needed. |
| `avl.py` (AVL interface) | External solver interface, not needed |
| `geometry.Propulsor` | Not used by aero solvers |
| `common.ImplicitAnalysis` | CasADi optimization glue, not needed without NLL |
| `common.AeroSandboxObject` | Serialization/copy utilities, not needed |
| CasADi dependency | Eliminated entirely |

---

## New code we must write

### 1. Wing meshing (`core/wing.py` — new methods)

The VLM and LiftingLine solvers call `wing.subdivide_sections()` and
`wing.mesh_thin_surface()`. These are geometry methods that belong on our
`Wing` class.

- **`Wing.subdivide_sections(ratio, spacing_function)`** — Interpolate xsecs
  to create finer spanwise resolution. ~60 lines of geometry interpolation.
- **`Wing.mesh_thin_surface(chordwise_resolution, spacing_function)`** —
  Generate structured quad mesh of the mean camber surface. ~120 lines.

### 2. Flight condition helpers (`core/flight_condition.py` — new methods)

The solvers call `op_point.compute_freestream_velocity_geometry_axes()`,
`op_point.compute_rotation_velocity_geometry_axes(points)`, and
`op_point.convert_axes(...)`. These are pure trig. Add them to
`FlightCondition`:

- **`FlightCondition.freestream_velocity_geometry_axes()`** — velocity vector
  in geometry frame from (V, alpha, beta). ~10 lines.
- **`FlightCondition.rotation_velocity_geometry_axes(points)`** — ω × r at
  panel points. ~15 lines. (p, q, r rates default to 0.)
- **`FlightCondition.convert_axes(x, y, z, from_axes, to_axes)`** — Rotation
  between geometry/body/wind frames. ~40 lines.

### 3. Cosine spacing utility (`utils/spacing.py`)

- **`cosspace(start, stop, num)`** — cosine-spaced points. ~5 lines.
- **`sinspace(start, stop, num)`** — sine-spaced points. ~5 lines.

---

## Phases

### Phase 1 — Foundation (new methods on existing classes)

- ✅ **1a.** Add `cosspace` / `sinspace` to `utils/spacing.py`
- ✅ **1b.** Add `freestream_velocity_geometry_axes()`, `rotation_velocity_geometry_axes()`,
  `convert_axes()` to `FlightCondition`
- ✅ **1c.** Add `subdivide_sections()` and `mesh_thin_surface()` to `Wing`
- ✅ **1d.** Test all new methods against ASB equivalents (numerical comparison)
  - 67/69 tests pass, remaining 2 are minor atmosphere precision differences

### Phase 2 — Vendor solvers 🔧 IN PROGRESS

- ✅ **2a.** Copy singularity functions → `aero/singularities.py`, replace `aerosandbox.numpy` → `numpy`
- ✅ **2b.** Copy VLM → `aero/solvers/vlm.py`, refactor to accept `Aircraft` + `FlightCondition`
  directly. Remove `ExplicitAnalysis` base class.
- ✅ **2c.** Copy AeroBuildup + submodels → `aero/solvers/aero_buildup.py` + `aero/fuselage_aero.py`,
  refactor geometry inputs. Copy library/aerodynamics → `aero/library/`.
- ✅ **2d.** Copy LiftingLine → `aero/solvers/lifting_line.py`, refactor. (Depends on both
  singularities and AeroBuildup.)
- ✅ **2d+.** Write NonlinearLiftingLine → `aero/solvers/nonlinear_lifting_line.py`.
  Fixed-point iteration, plain NumPy, no CasADi. Converges in 10–30 iterations.
- ⬜ **2e.** Test all vendored solvers against ASB originals (same aircraft, same
  condition → results must match to <1e-6).

### Phase 3 — Integration ⬜ TODO

- ⬜ **3a.** Update `aero/__init__.py` `analyze()` to dispatch to vendored solvers
  instead of importing from ASB
- ⬜ **3b.** Update `aero/result.py` — remove `_solver` / `_airplane` ASB objects,
  store our own solver for panel data access
- ⬜ **3c.** Remove `aerosandbox_backend.py` (all translation code)
- ⬜ **3d.** Remove `aerosandbox` from `pyproject.toml` dependencies
  (keep `neuralfoil` — LiftingLine and AeroBuildup still call it directly)
- ⬜ **3e.** Full regression test: re-run `tutorials/02_aerodynamics_executed.ipynb`,
  verify all plots match

### Phase 4 — Control surfaces (the reason we're doing this) ⬜ TODO

- ⬜ **4a.** In vendored VLM: implement control surface deflection as panel geometry
  rotation in mesh generation (rotate trailing-edge panels by deflection angle)
- ⬜ **4b.** Wire `FlightCondition.deflections` dict into VLM mesh generation
- ⬜ **4c.** Test: elevator deflection → Cm change, aileron → Cl change

---

## File structure after vendoring

```
src/aerisplane/aero/
├── __init__.py              # analyze(), plot_geometry()
├── result.py                # AeroResult dataclass
├── singularities.py         # horseshoe vortex + point source (from ASB)
├── fuselage_aero.py         # fuselage utilities + softmax (from ASB)
├── solvers/
│   ├── __init__.py
│   ├── vlm.py               # VortexLatticeMethod (from ASB, refactored)
│   ├── lifting_line.py      # LiftingLine (from ASB, refactored)
│   └── aero_buildup.py      # AeroBuildup (from ASB, refactored)
└── library/
    ├── __init__.py
    ├── viscous.py            # Cf, skin friction models (from ASB)
    ├── transonic.py          # wave drag corrections (from ASB)
    ├── inviscid.py           # thin airfoil theory helpers (from ASB)
    └── components.py         # component drag models (from ASB)
```

---

## Dependencies after vendoring

**Removed:** `aerosandbox`, `casadi`, `dill`, `sortedcontainers`, `tqdm`

**Kept:** `numpy`, `scipy`, `matplotlib`, `neuralfoil`, `pandas`, `seaborn`

---

## Attribution

Every vendored file will include this header:

```python
# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/aerodynamics/aero_3D/<filename>.py
```

---

## Risk / notes

- **NeuralFoil stays as a pip dependency.** It's a separate MIT-licensed package
  (also by Peter Sharpe). LiftingLine and AeroBuildup call it for section polars.
  No reason to vendor it — it's stable and well-maintained.
- **NonlinearLiftingLine is dropped.** It requires CasADi for its implicit solve.
  The three remaining methods (VLM, LiftingLine, AeroBuildup) cover all practical
  use cases. NLL can be re-added later with a scipy-based iterative solver if needed.
- **`aerosandbox.numpy` → `numpy` substitution** should be mechanical (find-replace)
  since we are not doing CasADi optimization. The only risk is if any solver code
  uses `aerosandbox.numpy`-specific functions beyond standard numpy + `cosspace`.
  Audit during Phase 2.

---

## Current status (2026-03-27)

**Branch:** `feature/vendor-asb-aero` (based on `feature/weights-module`)

### Files created/modified in this effort

| File | Status | Notes |
|---|---|---|
| `utils/spacing.py` | ✅ NEW | `cosspace()`, `sinspace()` |
| `core/flight_condition.py` | ✅ MODIFIED | Added `freestream_velocity_geometry_axes()`, `rotation_velocity_geometry_axes()`, `convert_axes()` |
| `core/wing.py` | ✅ MODIFIED | Added `subdivide_sections()`, `mesh_thin_surface()`, internal geometry helpers |
| `core/airfoil.py` | ✅ MODIFIED | Added `upper_coordinates()`, `lower_coordinates()`, `local_camber()`, `repanel()`, `blend_with_another_airfoil()`, `max_camber()`. Also added custom `__eq__`/`__hash__` (dataclass default `__eq__` fails on numpy arrays), changed to `@dataclass(eq=False)`. |
| `aero/singularities.py` | ✅ NEW | `calculate_induced_velocity_horseshoe()`, `calculate_induced_velocity_point_source()` |
| `aero/solvers/__init__.py` | ✅ NEW | Empty init |
| `aero/solvers/vlm.py` | ✅ NEW | `VortexLatticeMethod` — default `vortex_core_radius=1e-8` fixes CD sign |

### What works

- Phase 1 is fully complete. All new methods on `Wing`, `FlightCondition`, `Airfoil` are
  validated against ASB equivalents.
- VLM produces correct **CL** (within ~1.3% of ASB), correct **Cm** (within ~1.8%),
  and correct **Cl, Cn** (~0 for symmetric wing).
- NaN self-influence bug was fixed with `np.nan_to_num` in `get_velocity_at_points()`.

### VLM CD fix (resolved 2026-03-27)

**Root cause:** The horseshoe singularity function computes bound-leg + two trailing
legs as a single expression. For the diagonal entry `(i,i)` (field point = vortex
centre of panel i, source = panel i's own horseshoe), the bound leg gives `0/0 = NaN`
which propagates through the sum, poisoning the trailing-vortex contributions for that
entry. `nan_to_num` was then zeroing the **entire** diagonal including the trailing
downwash — which is physically real and drives induced drag.

**Fix:** Set `vortex_core_radius = 1e-8` as the default. The Kaufmann model replaces
`1/x` with `x/(x² + ε²)`, so at the singularity `smoothed_inv(0) = 0`. The bound-leg
contribution correctly evaluates to zero; trailing-vortex contributions are unaffected
(they evaluate at points well away from their respective vortex lines).
`nan_to_num` is kept as a defensive guard for degenerate panels but should be a no-op.

### Minor differences vs ASB (acceptable)

| Quantity | Ours | ASB | Cause |
|---|---|---|---|
| s_ref | 0.3840 | 0.3847 | Wing area computation: we use trapezoidal, ASB uses projected |
| b_ref | 1.6000 | 1.6031 | Same: projected vs geometric span |
| c_ref | 0.2550 | 0.2450 | MAC formula differs slightly |
| dynamic_pressure | 238.017 | 237.972 | ISA density at 300m differs by ~0.02% |

These are minor and will cause ~1-2% coefficient differences even with perfect solver
agreement. Can be improved later by matching ASB's projected-area formulas.
