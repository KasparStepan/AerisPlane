# AerisPlane Public Release: API Docs, Documentation Site, and Tutorials

**Date:** 2026-04-05
**Status:** Approved

---

## 1. Goal

Make AerisPlane usable by people other than its author. This means: complete API
documentation for every public module, two missing tutorial notebooks, a hosted
documentation site on GitHub Pages, and the supporting files that any public Python
package needs (contributing guide, changelog, CI).

---

## 2. Audience

Two groups, served by different layers of the same site:

- **Engineering students and hobbyists** — need step-by-step tutorials, conceptual
  explanations, worked examples with plots
- **Aerospace engineers and researchers** — need accurate API reference, physics context,
  sign conventions, and enough depth to extend or integrate the library

---

## 3. Documentation Site

### 3.1 Stack

| Tool | Purpose |
|---|---|
| MkDocs | Static site generator |
| Material theme | Navigation, search, dark/light mode, mobile |
| mkdocstrings | Auto-render Python docstrings into API reference pages |
| mkdocs-jupyter | Render `.ipynb` tutorials with stored cell outputs (no re-execution) |
| GitHub Pages | Hosting via `gh-pages` branch |

### 3.2 Navigation Structure

```
index.md                        Landing page (from README)
installation.md                 pip install, optional extras
getting-started.md              5-minute prose walkthrough ("hello wing")

Tutorials/
  01_getting_started            Aircraft definition, geometry plotting
  02_aerodynamics               Alpha/velocity/altitude sweeps, polar plots
  02_weight_buildup             Mass breakdown, CG analysis
  03_control_surfaces           Defining surfaces, elevator/aileron sweeps
  04_flow_visualisation         Surface Cp, streamlines, interactive Plotly
  05_flight_performance         Flight envelope, performance maps
  06_full_discipline_chain      All modules in one loop
  07_mdo_optimization           NEW — MDO problem setup, optimize, sensitivity
  08_propulsion_endurance       NEW — catalog selection, propulsion analysis, endurance

API Reference/
  core.md                       Aircraft, Wing, WingXSec, Fuselage, FlightCondition, ...
  aero.md                       analyze(), AeroResult, plot_geometry()
  weights.md                    analyze(), WeightResult
  structures.md                 NEW — analyze(), StructuresResult, Spar, TubeSection
  stability.md                  analyze(), StabilityResult
  control.md                    NEW — analyze(), ControlResult
  mission.md                    analyze(), MissionResult, segment types
  propulsion.md                 NEW — analyze(), PropulsionResult
  mdo.md                        NEW — MDOProblem, DesignVar, Constraint, Objective, sensitivity()
  catalog.md                    NEW — list_*() functions, hardware tables

Architecture/
  overview.md                   What AerisPlane is and is not (fix stale "UAVDesign" text)
  architecture.md               Layered design, dependency rules, backend pattern

contributing.md
changelog.md
```

---

## 4. Docstrings

### 4.1 Style

NumPy docstring format throughout. Matches the conventions of NumPy, SciPy, and
AeroSandbox — the libraries this audience already knows.

### 4.2 Coverage

**Full docstrings** (Parameters, Returns, Raises, one Example):
- All `analyze()` entry points in every discipline module
- All result dataclasses — every field with unit and description
- All `core/` geometry classes: constructor parameters + key computed properties
  (`span()`, `area()`, `mean_aerodynamic_chord()`)
- `MDOProblem`, `DesignVar`, `Constraint`, `Objective`, `AirfoilPool`
- `FlightCondition` — all parameters with units and sign conventions
- `catalog` list functions — signature + return type

**One-liner docstrings only:**
- Solver internals (`VortexLatticeMethod`, `BeamSolver`)
- Backend adapter functions (`aircraft_to_asb()`)
- Utility functions in `utils/`

**No docstrings:**
- Private/internal files: `_paths.py`, `_np_compat.py`, `_`-prefixed methods

### 4.3 mkdocstrings Integration

Each API Markdown page uses `::: aerisplane.module.ClassName` blocks to auto-render
docstrings below the hand-written narrative sections. The two sources appear together
on the same page: prose context at the top, auto-generated reference below.

---

## 5. New API Documentation Pages

Six modules currently have no API docs. Each page follows the same structure:
Quick start → `analyze()` signature table → result dataclass fields table →
methods → examples. mkdocstrings blocks pull in docstrings.

### 5.1 `core.md`

Covers: `Aircraft`, `Wing`, `WingXSec`, `Fuselage`, `FuselageXSec`, `Airfoil`,
`ControlSurface`, `FlightCondition`, `Servo`.

Special sections:
- Coordinate system diagram: `xyz_le` convention, geometry axes vs body axes
- Control surface sign convention (consolidated from `api-aero.md`)
- `FlightCondition.deflections` dict usage

### 5.2 `structures.md`

Covers: `structures.analyze()` → `StructuresResult`, `Spar`, `TubeSection`, `Material`.

Special sections:
- Failure modes: yield stress, buckling, tip deflection limit
- Load factor input and how it maps to distributed loads
- How to interpret the safety factor output
- Spar geometry parameters (outer diameter, wall thickness, material)

### 5.3 `control.md`

Covers: `control.analyze()` → `ControlResult`.

Special sections:
- Roll rate, pitch authority, rudder authority, servo load outputs
- How `ControlSurface` deflection limits feed in
- Interpreting control authority margins

### 5.4 `propulsion.md`

Covers: `propulsion.analyze()` → `PropulsionResult`, `PropulsionSystem`,
`Motor`, `Battery`, `Propeller`, `ESC`.

Special sections:
- Throttle input range (0.0–1.0)
- Outputs: thrust [N], RPM, current [A], motor efficiency, propulsive efficiency,
  C-rate utilization, estimated endurance [min]
- Over-current flag
- Pattern: pick hardware from catalog → build `PropulsionSystem` → analyze

### 5.5 `mdo.md`

Covers: `MDOProblem`, `DesignVar`, `Constraint`, `Objective`, `AirfoilPool`,
`OptimizationResult`, `SensitivityResult`, `optimize()`, `sensitivity()`.

Special sections:
- String-path design variable syntax: `"wings[0].xsecs[1].chord"`
- Path resolution at construction time (`validate()`)
- Evaluation caching (keyed on design vector)
- Driver selection: `"scipy_de"`, `"scipy_slsqp"`, `"pygmo_sade"`, etc.
- Checkpoint and resume
- Sensitivity analysis: normalized gradients, ranked table
- `AirfoilPool`: discrete airfoil selection as a design variable

### 5.6 `catalog.md`

Covers: `list_motors()`, `list_batteries()`, `list_propellers()`, `list_servos()`,
`list_materials()`, and the hardware constants (e.g. `MOTORS.sunnysky_x2216`).

Special sections:
- Full tables of included items with key specs (KV, capacity, diameter, etc.)
- How to select and use catalog items in a design
- How to define custom hardware using the same `core/` dataclasses

---

## 6. New Tutorial Notebooks

### 6.1 `07_mdo_optimization.ipynb`

**Audience:** Engineers who want to run an optimization, not just analyze.

**Content:**
1. Reuse geometry from Tutorial 01 as the baseline design
2. Define an `MDOProblem`:
   - 3 design variables: tip chord (`wings[0].xsecs[1].chord`), semi-span
     (`wings[0].xsecs[1].xyz_le[1]`), cruise speed (`FlightCondition.velocity`)
   - Bounds on each variable
   - Objective: maximize endurance from `mission.analyze()`
   - Constraints: stall speed < 12 m/s, structural safety factor > 1.5,
     tip deflection < 10% semi-span
3. Run optimizer with `driver="scipy_de"` (differential evolution — robust, no gradients)
4. Plot convergence history (objective vs iteration)
5. Run `sensitivity()` — ranked table of which variable matters most
6. Print side-by-side `report()` of baseline vs optimal design

### 6.2 `08_propulsion_and_endurance.ipynb`

**Audience:** Engineers sizing a propulsion system and estimating flight time.

**Content:**
1. Browse the catalog: `list_motors()`, `list_batteries()`, `list_propellers()`
   — display as formatted tables
2. Build a `PropulsionSystem` from catalog items (motor + prop + battery + ESC)
3. Run `propulsion.analyze()` at cruise throttle (0.6) — show thrust, RPM,
   current, C-rate, motor efficiency, propulsive efficiency
4. Throttle sweep (0.2 → 1.0): plot thrust and efficiency curves
5. Compute endurance estimate from battery capacity and average current
6. Swap motor and battery combos — compare endurance side-by-side in a bar chart
7. Integrate into the full discipline chain: pass `PropulsionResult` into
   `mission.analyze()` to get range and endurance with a real propulsion model

---

## 7. Supporting Files

### 7.1 `CONTRIBUTING.md`

- Editable install: `pip install -e ".[dev]"`
- Tests: `pytest tests/`
- Lint: `ruff check src/`
- Preview docs locally: `mkdocs serve`
- Branch naming: `feature/`, `fix/`, `docs/`
- Note that the project is in early development — issues and PRs are welcome

### 7.2 `CHANGELOG.md`

Follows [Keep a Changelog](https://keepachangelog.com) format.

- `[Unreleased]` section at top
- `v0.1.0` entry covering: aero (VLM, LLT, NLL, AeroBuildup, flow viz), weights,
  structures (beam solver), stability (finite-difference derivatives), control authority,
  mission (point performance), propulsion, MDO (ScipyDriver, PygmoDriver, sensitivity),
  hardware catalog (motors, batteries, propellers, servos, 2175 airfoils)

### 7.3 Stale Docs Fix

`docs/overview.md` and `docs/architecture.md` refer to "UAVDesign" and `AircraftConfig`.
Update to use `aerisplane` package name and `Aircraft` class throughout. These become the
Architecture section in the MkDocs site.

### 7.4 `pyproject.toml` Additions

Add `[project.urls]` block:
```toml
[project.urls]
Homepage = "https://github.com/KasparStepan/AerisPlane"
Documentation = "https://kasparstepan.github.io/AerisPlane"
Repository = "https://github.com/KasparStepan/AerisPlane"
"Bug Tracker" = "https://github.com/KasparStepan/AerisPlane/issues"
```

Add MkDocs dependencies to `[project.optional-dependencies]`:
```toml
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
    "mkdocs-jupyter>=0.24",
]
```

---

## 8. GitHub Actions CI

### 8.1 `tests.yml`

Trigger: push and PR to `main`.
Matrix: Python 3.10 and 3.12.
Steps: install `.[dev]`, run `pytest tests/`.

### 8.2 `docs.yml`

Trigger: push to `main`.
Steps: install `.[docs]`, run `mkdocs gh-deploy --force`.
Deploys to `gh-pages` branch → served at `https://kasparstepan.github.io/AerisPlane`.

---

## 9. Out of Scope

- Type annotations / mypy (not requested)
- PyPI release / `twine upload` (not requested)
- Sphinx (replaced by MkDocs)
- Any new features or module implementations
