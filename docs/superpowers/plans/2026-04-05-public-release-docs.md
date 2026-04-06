# Public Release: API Docs, Site, and Tutorials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AerisPlane usable by others by completing API documentation, building a MkDocs site on GitHub Pages, writing docstrings for undocumented public API, adding two missing tutorial notebooks, and creating the supporting files every public Python package needs.

**Architecture:** MkDocs + Material theme deployed to GitHub Pages via GitHub Actions. All content lives under `docs/` (tutorials moved from repo root). Hand-written API Markdown pages include `:::` mkdocstrings blocks that auto-render Python docstrings below the narrative text.

**Tech Stack:** MkDocs ≥ 1.5, mkdocs-material ≥ 9, mkdocstrings[python] ≥ 0.24, mkdocs-jupyter ≥ 0.24, GitHub Actions.

---

## Task 1: Move tutorials under docs/ and install MkDocs

**Files:**
- `git mv tutorials/ docs/tutorials/`
- Create: `mkdocs.yml`
- Modify: `README.md` (update tutorial paths)

- [ ] **Step 1: Move tutorials directory**

```bash
git mv tutorials docs/tutorials
```

- [ ] **Step 2: Install MkDocs and plugins**

```bash
pip install "mkdocs>=1.5" "mkdocs-material>=9.0" "mkdocstrings[python]>=0.24" "mkdocs-jupyter>=0.24"
```

- [ ] **Step 3: Create `mkdocs.yml`**

```yaml
site_name: AerisPlane
site_url: https://kasparstepan.github.io/AerisPlane
repo_url: https://github.com/KasparStepan/AerisPlane
repo_name: KasparStepan/AerisPlane
docs_dir: docs

theme:
  name: material
  palette:
    - scheme: default
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.highlight
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: false
            show_root_heading: true
            members_order: source
            show_if_no_docstring: false
  - mkdocs-jupyter:
      execute: false
      include_source: true

nav:
  - Home: index.md
  - Installation: installation.md
  - Getting Started: getting-started.md
  - Tutorials:
    - "01 — Aircraft Definition": tutorials/01_getting_started.ipynb
    - "02 — Aerodynamics": tutorials/02_aerodynamics_executed.ipynb
    - "02 — Weight Buildup": tutorials/02_weight_buildup.ipynb
    - "03 — Control Surfaces": tutorials/03_control_surfaces.ipynb
    - "04 — Flow Visualisation": tutorials/04_flow_visualisation.ipynb
    - "05 — Flight Performance": tutorials/05_flight_performance.ipynb
    - "06 — Full Discipline Chain": tutorials/06_full_discipline_chain.ipynb
    - "07 — MDO Optimization": tutorials/07_mdo_optimization.ipynb
    - "08 — Propulsion & Endurance": tutorials/08_propulsion_and_endurance.ipynb
  - API Reference:
    - Core: api/core.md
    - Aerodynamics: api/aero.md
    - Weights: api/weights.md
    - Structures: api/structures.md
    - Stability: api/stability.md
    - Control: api/control.md
    - Mission: api/mission.md
    - Propulsion: api/propulsion.md
    - MDO: api/mdo.md
    - Catalog: api/catalog.md
  - Architecture:
    - Overview: architecture/overview.md
    - Design: architecture/design.md
  - Contributing: contributing.md
  - Changelog: changelog.md

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - tables
  - toc:
      permalink: true
```

- [ ] **Step 4: Update README.md tutorial table**

Replace all occurrences of `tutorials/` with `docs/tutorials/` in the Tutorials section of README.md.

- [ ] **Step 5: Create placeholder docs that the nav references**

Create empty placeholder files so `mkdocs build` doesn't error before we write them:

```bash
touch docs/index.md docs/installation.md docs/getting-started.md
mkdir -p docs/api docs/architecture
touch docs/api/core.md docs/api/aero.md docs/api/weights.md docs/api/structures.md
touch docs/api/stability.md docs/api/control.md docs/api/mission.md
touch docs/api/propulsion.md docs/api/mdo.md docs/api/catalog.md
touch docs/architecture/overview.md docs/architecture/design.md
touch docs/contributing.md docs/changelog.md
```

- [ ] **Step 6: Verify site builds without errors**

```bash
mkdocs build --strict 2>&1 | head -40
```

Expected: build completes. Some empty pages are fine at this stage.

- [ ] **Step 7: Commit**

```bash
git add mkdocs.yml docs/ README.md
git commit -m "feat(docs): add MkDocs site infrastructure, move tutorials to docs/"
```

---

## Task 2: Landing pages — index.md, installation.md, getting-started.md

**Files:**
- Write: `docs/index.md`
- Write: `docs/installation.md`
- Write: `docs/getting-started.md`

- [ ] **Step 1: Write `docs/index.md`**

```markdown
# AerisPlane

Conceptual MDO toolkit for fixed-wing RC/UAV aircraft design (1–20 kg class).

AerisPlane is a Python library for analysing and optimising small unmanned aircraft
at the conceptual design stage. It covers aerodynamics, weights, structures, stability,
control authority, and mission performance — all wired together through a lightweight
MDO layer.

## Module overview

| Module | Description |
|---|---|
| `aerisplane.core` | Geometry dataclasses: `Aircraft`, `Wing`, `Fuselage`, `FlightCondition`, ... |
| `aerisplane.aero` | Aerodynamic analysis (VLM, LiftingLine, AeroBuildup, flow viz) |
| `aerisplane.weights` | Component mass buildup, CG analysis |
| `aerisplane.structures` | Euler–Bernoulli wing beam solver |
| `aerisplane.stability` | Numerical stability derivatives |
| `aerisplane.control` | Control authority and servo loads |
| `aerisplane.mission` | Point-performance energy budget |
| `aerisplane.propulsion` | Motor/battery/propeller operating point |
| `aerisplane.mdo` | Optimisation: design variables, constraints, SciPy/pygmo drivers |
| `aerisplane.catalog` | Hardware database — motors, batteries, propellers, servos |

## Quick start

```python
import aerisplane as ap
from aerisplane.aero import analyze

wing = ap.Wing(
    name="main_wing",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28, airfoil=ap.Airfoil(name="ag35")),
        ap.WingXSec(xyz_le=[0.07, 1.20, 0.05], chord=0.14, twist=-2.0, airfoil=ap.Airfoil(name="ag35")),
    ],
)
aircraft = ap.Aircraft(name="MyPlane", wings=[wing])
cond = ap.FlightCondition(velocity=16.0, altitude=0.0, alpha=4.0)
result = analyze(aircraft, cond, method="vlm")
result.report()
```
```

- [ ] **Step 2: Write `docs/installation.md`**

```markdown
# Installation

## Requirements

- Python ≥ 3.10
- numpy, scipy, matplotlib, neuralfoil (installed automatically)

## Standard install

```bash
pip install aerisplane
```

## Editable install (development)

```bash
git clone https://github.com/KasparStepan/AerisPlane.git
cd AerisPlane
pip install -e ".[dev]"
```

## Optional extras

```bash
pip install aerisplane[oas]          # OpenAeroStruct for detailed structural analysis
pip install aerisplane[optimize]     # pygmo global optimizers
pip install aerisplane[interactive]  # Plotly interactive flow visualisation
pip install aerisplane[all]          # everything
```
```

- [ ] **Step 3: Write `docs/getting-started.md`**

```markdown
# Getting Started

A five-minute introduction to the AerisPlane workflow.

## 1. Define geometry

An aircraft is built from `Wing` and `Fuselage` objects composed of cross-sections.
The main wing, horizontal tail, and vertical tail are all `Wing` objects — there is
no separate tail class.

```python
import aerisplane as ap

# Main wing with two cross-sections (root and tip)
wing = ap.Wing(
    name="main_wing",
    symmetric=True,   # mirror across y=0 plane
    xsecs=[
        ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28, airfoil=ap.Airfoil(name="ag35")),
        ap.WingXSec(xyz_le=[0.07, 1.20, 0.05], chord=0.14, twist=-2.0, airfoil=ap.Airfoil(name="ag35")),
    ],
)

# Horizontal tail with elevator
htail = ap.Wing(
    name="htail",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.95, 0.00, 0.00], chord=0.15, airfoil=ap.Airfoil(name="naca0012")),
        ap.WingXSec(xyz_le=[0.98, 0.40, 0.00], chord=0.10, airfoil=ap.Airfoil(name="naca0012")),
    ],
    control_surfaces=[
        ap.ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
                          chord_fraction=0.35, symmetric=True),
    ],
)

aircraft = ap.Aircraft(name="MyPlane", wings=[wing, htail])
```

Coordinate convention: `xyz_le = [x, y, z]` is the leading-edge position in aircraft
frame. `x` points aft, `y` points right, `z` points up. For symmetric wings, define
the right side (y ≥ 0) only.

## 2. Run an aerodynamic analysis

```python
from aerisplane.aero import analyze, plot_geometry

# Plot the geometry to verify it looks right
plot_geometry(aircraft, style="three_view")

# Run VLM at a single operating point
cond = ap.FlightCondition(velocity=16.0, altitude=0.0, alpha=4.0)
result = analyze(aircraft, cond, method="vlm")
result.report()
```

Available methods: `"vlm"` (fast, inviscid), `"lifting_line"` (viscous with NeuralFoil),
`"nonlinear_lifting_line"` (captures stall), `"aero_buildup"` (semi-empirical).

## 3. Check weights

```python
from aerisplane.weights import analyze as weight_analysis

wr = weight_analysis(aircraft)
wr.report()
wr.plot_mass_breakdown()
```

## 4. Run the full discipline chain

```python
from aerisplane import weights, stability

wr = weights.analyze(aircraft)
aircraft.xyz_ref = wr.cg.tolist()   # set moment reference to CG

sr = stability.analyze(aircraft, cond, wr)
print(f"Static margin: {sr.static_margin:.1%} MAC")
print(f"Cm_alpha:      {sr.Cm_alpha:.4f} 1/deg  ({'stable' if sr.Cm_alpha < 0 else 'UNSTABLE'})")
```

See the [Tutorials](tutorials/01_getting_started.ipynb) for deeper worked examples
covering every discipline module.
```

- [ ] **Step 4: Verify site builds and preview**

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000` and check the landing pages render correctly.

- [ ] **Step 5: Commit**

```bash
git add docs/index.md docs/installation.md docs/getting-started.md
git commit -m "docs: add landing page, installation guide, getting-started walkthrough"
```

---

## Task 3: Supporting files — CONTRIBUTING.md, CHANGELOG.md, stale docs fix

**Files:**
- Write: `docs/contributing.md`
- Write: `docs/changelog.md`
- Write: `docs/architecture/overview.md` (fix stale "UAVDesign")
- Write: `docs/architecture/design.md` (fix stale "UAVDesign")

- [ ] **Step 1: Write `docs/contributing.md`**

```markdown
# Contributing

AerisPlane is in early development. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/KasparStepan/AerisPlane.git
cd AerisPlane
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/
pytest tests/test_aero/   # aerodynamics only
```

## Linting

```bash
ruff check src/
```

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview the site.

## Branch naming

- `feature/<name>` — new functionality
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation only

Submit a pull request against `main`. Please include a brief description of what
changed and why.
```

- [ ] **Step 2: Write `docs/changelog.md`**

```markdown
# Changelog

All notable changes to AerisPlane are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.1.0] — 2026-04-05

### Added

**Aerodynamics (`aerisplane.aero`)**
- Vortex Lattice Method (VLM) — inviscid 3-D, arbitrary geometry
- Prandtl Lifting Line with NeuralFoil section polars
- Nonlinear Lifting Line (fixed-point stall iteration)
- AeroBuildup semi-empirical method (NeuralFoil wings + Jorgensen fuselage)
- Surface pressure and streamline flow visualisation
- Interactive Plotly 3-D flow field (`plot_interactive()`)

**Weights (`aerisplane.weights`)**
- Component-based mass buildup from geometry and material properties
- CG computation and inertia tensor estimate
- Override mechanism for replacing estimates with measurements

**Structures (`aerisplane.structures`)**
- Euler–Bernoulli wing beam solver
- Failure checks: bending margin, shear margin, buckling margin
- Tip deflection ratio and torsional divergence speed
- OpenAeroStruct adapter (optional dependency)

**Stability (`aerisplane.stability`)**
- Numerical stability derivatives via central finite differences
- Static margin, neutral point, Cm_alpha, CL_alpha
- Lateral-directional: Cl_beta, Cn_beta
- Rate derivatives: CL_q, Cm_q, Cl_p, Cn_p, CY_p, Cn_r, Cl_r, CY_r
- Dynamic mode estimates: short-period frequency and damping

**Control (`aerisplane.control`)**
- Control authority: roll rate, pitch authority, rudder authority
- Servo hinge moment estimation and servo load margin
- Finite-difference control derivatives

**Mission (`aerisplane.mission`)**
- Segment-based energy budget: Climb, Cruise, Loiter, Descent, Return
- Flight envelope: Vs, Vmin_power, Vmax_range, Vy, ceiling
- Range–endurance–speed performance curves

**Propulsion (`aerisplane.propulsion`)**
- Motor/propeller/battery/ESC operating-point solver
- Outputs: thrust, RPM, current, motor efficiency, propulsive efficiency,
  battery endurance, C-rate, over-current flag

**MDO (`aerisplane.mdo`)**
- `MDOProblem`: design variables, constraints, objectives
- String-path design variable syntax (`"wings[0].xsecs[1].chord"`)
- Evaluation caching keyed on design vector
- SciPy drivers: `scipy_de`, `scipy_minimize`, `scipy_shgo`
- pygmo drivers: `pygmo_de`, `pygmo_sade`, `pygmo_nsga2`
- Checkpoint/resume
- Sensitivity analysis: finite-difference gradients, normalized ranking

**Catalog (`aerisplane.catalog`)**
- 20 brushless motors, 15 LiPo batteries, 10 propellers, 10 servos
- 2175 airfoil `.dat` files
- `list_motors()`, `list_batteries()`, `list_propellers()`, `list_servos()`
```

- [ ] **Step 3: Write `docs/architecture/overview.md`** (clean version of stale `docs/overview.md`)

```markdown
# Overview

AerisPlane is a Python MDO toolkit for fixed-wing RC/UAV conceptual design (1–20 kg class).

## What AerisPlane is

- A conceptual design MDO pipeline for RC/UAV aircraft
- A discipline integration layer: aerodynamics, weights, structures, stability,
  control, and mission analysis wired into a single optimization loop
- A Python library with a clean, readable API for engineers and students

## What AerisPlane is not

- Not a CFD/FEA solver — it delegates to backend solvers (AeroSandbox, OpenAeroStruct)
- Not a GUI application
- Not a replacement for AeroSandbox — it builds on top of it and adds disciplines
  AeroSandbox does not cover
- Not a flight simulator or 6-DOF dynamics tool

## Relationship to AeroSandbox

AerisPlane uses AeroSandbox as a backend aero solver via an adapter layer.
It does **not** inherit from or tightly couple to AeroSandbox's class hierarchy.
The `core/` module depends only on `numpy` and has no AeroSandbox imports.
Translation to AeroSandbox objects happens exclusively in `aero/aerosandbox_backend.py`.

## Intended workflow

1. **Define** — build an `Aircraft` from `Wing`, `Fuselage`, and hardware catalog items
2. **Analyse** — run discipline modules individually or in sequence
3. **Optimise** — define an `MDOProblem`, run it, inspect `OptimizationResult`
```

- [ ] **Step 4: Write `docs/architecture/design.md`** (clean version of stale `docs/architecture.md`)

```markdown
# Architecture

## Layered design

```
┌──────────────────────────────────────────────────┐
│                    mdo/                           │  Optimization orchestration
│  MDOProblem, DesignVar, Constraint, Objective     │
├──────────────────────────────────────────────────┤
│  aero/ │ weights/ │ structures/ │ stability/ │   │  Discipline modules
│  control/ │ mission/ │ propulsion/           │   │  Each: analyze() → Result
├──────────────────────────────────────────────────┤
│                    core/                          │  Data model
│  Aircraft, Wing, Fuselage, FlightCondition, ...  │  numpy only, no solvers
├──────────────────────────────────────────────────┤
│                   catalog/                        │  Hardware database
│  Motors, Batteries, Servos, Propellers            │
├──────────────────────────────────────────────────┤
│                    utils/                         │  Shared utilities
│  ISA atmosphere, units, plotting style            │
└──────────────────────────────────────────────────┘
```

## Dependency rules

- `core/` depends on **numpy only** — no AeroSandbox, no backend imports
- `catalog/` depends on `core/`
- Each discipline module depends on `core/`, `utils/`, and its backends (lazy-imported)
- `mdo/` depends on `core/` and all discipline modules
- Nothing depends on `mdo/` — it is the top of the tree

## Backend pattern

Solvers are lazy-imported behind string-based method selection:

```python
result = analyze(aircraft, condition, method="vlm")          # native VLM
result = analyze(aircraft, condition, method="aero_buildup") # AeroBuildup
```

Each adapter: translate core objects in → call solver → translate out to result dataclass.

## Every discipline follows the same pattern

```
discipline/__init__.py    # analyze(*args) → DisciplineResult
discipline/result.py      # DisciplineResult dataclass with .report() and .plot()
discipline/solver.py      # physics implementation
```
```

- [ ] **Step 5: Commit**

```bash
git add docs/contributing.md docs/changelog.md docs/architecture/
git commit -m "docs: add contributing guide, changelog, architecture pages"
```

---

## Task 4: pyproject.toml — add URLs and docs optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml** (already done above — current content known)

- [ ] **Step 2: Add `[project.urls]` and `docs` optional dependency**

In `pyproject.toml`, after the `[project]` block add:

```toml
[project.urls]
Homepage = "https://github.com/KasparStepan/AerisPlane"
Documentation = "https://kasparstepan.github.io/AerisPlane"
Repository = "https://github.com/KasparStepan/AerisPlane"
"Bug Tracker" = "https://github.com/KasparStepan/AerisPlane/issues"
```

In `[project.optional-dependencies]`, add:

```toml
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
    "mkdocs-jupyter>=0.24",
]
```

Also update the `all` extra to include docs:

```toml
all = ["aerisplane[oas,optimize,interactive,dev,docs]"]
```

- [ ] **Step 3: Verify package installs cleanly**

```bash
pip install -e ".[docs]" --quiet
python -c "import mkdocs; import mkdocstrings; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add project URLs and docs optional dependency"
```

---

## Task 5: GitHub Actions CI

**Files:**
- Create: `.github/workflows/tests.yml`
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Create `.github/workflows/` directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Write `.github/workflows/tests.yml`**

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v
```

- [ ] **Step 3: Write `.github/workflows/docs.yml`**

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install docs dependencies
        run: pip install -e ".[docs]"

      - name: Deploy to GitHub Pages
        run: mkdocs gh-deploy --force
```

- [ ] **Step 4: Commit**

```bash
git add .github/
git commit -m "ci: add tests and docs deployment GitHub Actions workflows"
```

---

## Task 6: Docstrings — core geometry classes

**Files:**
- Modify: `src/aerisplane/core/airfoil.py`
- Modify: `src/aerisplane/core/control_surface.py`
- Modify: `src/aerisplane/core/propulsion.py`
- Modify: `src/aerisplane/catalog/__init__.py`

- [ ] **Step 1: Add docstring to `Airfoil` class**

In `src/aerisplane/core/airfoil.py`, the `Airfoil` class currently has a partial docstring. Replace with:

```python
@dataclass(eq=False)
class Airfoil:
    """Airfoil defined by name and/or coordinate array.

    If ``name`` is given and ``coordinates`` is None, the catalog is searched
    for a matching ``.dat`` file. NACA 4-digit names (e.g. ``"naca2412"``) are
    generated analytically if not found in the catalog.

    Parameters
    ----------
    name : str
        Airfoil name, e.g. ``"ag35"``, ``"naca2412"``, ``"e387"``.
        Used for catalog lookup and plot labels.
    coordinates : ndarray of shape (N, 2) or None
        Explicit (x, y) coordinate array in Selig format (upper surface first,
        x from 0 to 1 and back to 0). If None, loaded from catalog by name.

    Examples
    --------
    >>> af = Airfoil(name="ag35")         # loads from catalog
    >>> af = Airfoil(name="naca2412")     # generated analytically
    >>> af = Airfoil(name="custom", coordinates=np.array([[1,0],[0.5,0.06],[0,0],...]))
    """
```

- [ ] **Step 2: Add docstring to `ControlSurface`**

In `src/aerisplane/core/control_surface.py`, add full docstring to the `ControlSurface` class:

```python
@dataclass
class ControlSurface:
    """A hinged control surface on a wing.

    Parameters
    ----------
    name : str
        Surface name used to key ``FlightCondition.deflections``.
        Conventional names: ``"elevator"``, ``"aileron"``, ``"rudder"``, ``"flap"``.
    span_start : float
        Start of the surface as a fraction of the wing semi-span [0, 1].
    span_end : float
        End of the surface as a fraction of the wing semi-span [0, 1].
    chord_fraction : float
        Hinge chord fraction [0, 1]. 0.25 means the surface occupies the aft 25%
        of the local chord.
    symmetric : bool
        If True, the surface deflects the same way on both sides of the wing
        (elevator, flap). If False, the sign of the deflection is mirrored on
        the left side (aileron, split rudder).
    max_deflection : float
        Maximum deflection magnitude [deg]. Used for authority calculations.

    Sign convention
    ---------------
    Positive deflection = trailing edge down.
    For ``symmetric=False`` (ailerons): positive = right side TE-down, left side TE-up.

    Examples
    --------
    >>> elevator = ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
    ...                            chord_fraction=0.35, symmetric=True, max_deflection=25.0)
    >>> aileron = ControlSurface(name="aileron", span_start=0.5, span_end=0.9,
    ...                           chord_fraction=0.25, symmetric=False, max_deflection=20.0)
    """
```

- [ ] **Step 3: Add docstrings to `Motor`, `Battery`, `Propeller`, `ESC`, `PropulsionSystem` in `core/propulsion.py`**

Read `src/aerisplane/core/propulsion.py` first, then add docstrings to each class constructor. Example for `Motor`:

```python
@dataclass
class Motor:
    """Brushless DC motor model.

    Parameters
    ----------
    name : str
        Motor name for display.
    kv : float
        Motor velocity constant [RPM/V].
    resistance : float
        Winding resistance [Ohm].
    no_load_current : float
        No-load current [A].
    max_current : float
        Maximum continuous current [A].
    mass : float
        Motor mass [kg].

    Examples
    --------
    >>> from aerisplane.catalog.motors import sunnysky_x2216_1250
    >>> motor = sunnysky_x2216_1250   # 1250 KV, 0.117 Ohm, 28 A max, 58 g
    """
```

- [ ] **Step 4: Add docstrings to catalog `list_*` functions**

In `src/aerisplane/catalog/__init__.py`:

```python
def list_motors() -> list:
    """Return all motors in the catalog as a list of :class:`~aerisplane.core.propulsion.Motor`.

    Returns
    -------
    list of Motor
        All motor instances defined in ``aerisplane.catalog.motors``.

    Examples
    --------
    >>> from aerisplane.catalog import list_motors
    >>> motors = list_motors()
    >>> for m in motors:
    ...     print(f"{m.name:40s}  KV={m.kv:5.0f}  max_I={m.max_current:.0f}A")
    """
```

Apply the same pattern for `list_batteries`, `list_propellers`, `list_servos`.

- [ ] **Step 5: Verify docstrings render**

```bash
python -c "
import aerisplane as ap
from aerisplane.catalog import list_motors
help(ap.Airfoil)
help(ap.ControlSurface)
help(list_motors)
"
```

Expected: formatted docstrings printed without errors.

- [ ] **Step 6: Commit**

```bash
git add src/aerisplane/core/airfoil.py src/aerisplane/core/control_surface.py \
        src/aerisplane/core/propulsion.py src/aerisplane/catalog/__init__.py
git commit -m "docs: add docstrings to Airfoil, ControlSurface, Motor/Battery/Propeller, catalog list_*"
```

---

## Task 7: Docstrings — mission and propulsion analyze()

**Files:**
- Modify: `src/aerisplane/mission/__init__.py`
- Modify: `src/aerisplane/propulsion/__init__.py`
- Modify: `src/aerisplane/mission/segments.py`

- [ ] **Step 1: Expand `mission.analyze()` docstring**

In `src/aerisplane/mission/__init__.py`, replace the `analyze()` docstring with:

```python
def analyze(
    aircraft: Aircraft,
    weight_result: WeightResult,
    mission: Mission,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> MissionResult:
    """Run segment-by-segment mission energy budget analysis.

    Fits a drag polar once at sea level, then evaluates each segment
    in order (Climb, Cruise, Loiter, Descent, Return) for duration,
    distance, and electrical energy required.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft with a ``propulsion`` attribute (used for battery energy and
        efficiency estimation).
    weight_result : WeightResult
        Weight analysis result (provides total mass for power-required calculation).
    mission : Mission
        Mission profile with segments and start altitude.
    aero_method : str, optional
        Aero solver used for drag polar fitting. Default ``"vlm"``.
    **aero_kwargs
        Additional keyword arguments passed to ``aero.analyze()``.

    Returns
    -------
    MissionResult
        Total energy, time, distance, battery margin, feasibility flag, and
        per-segment breakdown.

    Examples
    --------
    >>> from aerisplane.mission import analyze
    >>> from aerisplane.mission.segments import Mission, Climb, Cruise, Loiter
    >>> mission = Mission(
    ...     start_altitude=0.0,
    ...     segments=[
    ...         Climb(name="climb", velocity=14.0, climb_rate=2.0, to_altitude=100.0),
    ...         Cruise(name="cruise", velocity=18.0, altitude=100.0, distance=5000.0),
    ...         Loiter(name="loiter", velocity=16.0, altitude=100.0, duration=600.0),
    ...     ]
    ... )
    >>> result = analyze(aircraft, weight_result, mission)
    >>> print(f"Feasible: {result.feasible}  margin: {result.energy_margin:.1%}")
    """
```

- [ ] **Step 2: Expand `propulsion.analyze()` docstring**

In `src/aerisplane/propulsion/__init__.py`, replace the docstring with:

```python
def analyze(aircraft, condition, throttle: float = 1.0) -> PropulsionResult:
    """Compute propulsion system operating point at a given throttle.

    Solves motor RPM and current by matching torque supply and demand,
    then computes thrust, efficiency, battery C-rate, and estimated endurance.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft with a ``propulsion`` attribute
        (:class:`~aerisplane.core.propulsion.PropulsionSystem`).
    condition : FlightCondition
        Operating point. Uses ``condition.velocity`` and ``condition.altitude``.
    throttle : float, optional
        Throttle setting in [0, 1]. Default 1.0 (full throttle).

    Returns
    -------
    PropulsionResult
        Thrust [N], current [A], RPM, motor efficiency, propulsive efficiency,
        electrical power [W], shaft power [W], endurance [s], C-rate,
        and over-current flag.

    Raises
    ------
    ValueError
        If ``aircraft.propulsion`` is None.

    Examples
    --------
    >>> from aerisplane.propulsion import analyze
    >>> result = analyze(aircraft, condition, throttle=0.6)
    >>> print(result.report())
    """
```

- [ ] **Step 3: Add docstrings to `Mission`, `Climb`, `Cruise`, `Loiter`, `Descent`, `Return` in `segments.py`**

Read `src/aerisplane/mission/segments.py` first, then add minimal docstrings:

```python
@dataclass
class Mission:
    """Mission profile composed of sequential segments.

    Parameters
    ----------
    segments : list
        Ordered list of mission segments (Climb, Cruise, Loiter, Descent, Return).
    start_altitude : float
        Starting altitude [m]. Default 0.0.
    """

@dataclass
class Climb:
    """Climb segment at constant velocity and climb rate.

    Parameters
    ----------
    name : str
    velocity : float  — True airspeed during climb [m/s]
    climb_rate : float  — Rate of climb [m/s]
    to_altitude : float  — Target altitude at end of segment [m]
    """

@dataclass
class Cruise:
    """Cruise segment at constant velocity and altitude.

    Parameters
    ----------
    name : str
    velocity : float  — True airspeed [m/s]
    altitude : float  — Cruise altitude [m]
    distance : float  — Horizontal distance [m]
    """

@dataclass
class Loiter:
    """Loiter segment at constant velocity, altitude, and duration.

    Parameters
    ----------
    name : str
    velocity : float  — Loiter airspeed [m/s]
    altitude : float  — Loiter altitude [m]
    duration : float  — Loiter time [s]
    """
```

- [ ] **Step 4: Verify**

```bash
python -c "
from aerisplane.mission import analyze
from aerisplane.propulsion import analyze as p_analyze
help(analyze)
help(p_analyze)
"
```

Expected: full docstrings printed.

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/mission/ src/aerisplane/propulsion/
git commit -m "docs: expand docstrings for mission.analyze(), propulsion.analyze(), segment classes"
```

---

## Task 8: Existing API docs — rename and add mkdocstrings blocks

**Files:**
- Rename: `docs/api/api-aero.md` → `docs/api/aero.md`
- Rename: `docs/api/api-weights.md` → `docs/api/weights.md`
- Rename: `docs/api/api-stability.md` → `docs/api/stability.md`
- Rename: `docs/api/api-mission.md` → `docs/api/mission.md`

- [ ] **Step 1: Rename existing API docs**

```bash
git mv docs/api/api-aero.md docs/api/aero.md
git mv docs/api/api-weights.md docs/api/weights.md
git mv docs/api/api-stability.md docs/api/stability.md
git mv docs/api/api-mission.md docs/api/mission.md
```

- [ ] **Step 2: Append mkdocstrings block to `docs/api/aero.md`**

At the end of `docs/api/aero.md`, append:

```markdown
---

## Auto-generated reference

::: aerisplane.aero.analyze

::: aerisplane.aero.result.AeroResult
```

- [ ] **Step 3: Append mkdocstrings blocks to `docs/api/weights.md`**

At the end of `docs/api/weights.md`, append:

```markdown
---

## Auto-generated reference

::: aerisplane.weights.analyze

::: aerisplane.weights.result.WeightResult
```

- [ ] **Step 4: Append mkdocstrings blocks to `docs/api/stability.md`**

At the end of `docs/api/stability.md`, append:

```markdown
---

## Auto-generated reference

::: aerisplane.stability.analyze

::: aerisplane.stability.result.StabilityResult
```

- [ ] **Step 5: Append mkdocstrings blocks to `docs/api/mission.md`**

At the end of `docs/api/mission.md`, append:

```markdown
---

## Auto-generated reference

::: aerisplane.mission.analyze

::: aerisplane.mission.result.MissionResult
```

- [ ] **Step 6: Verify site builds with renamed files**

```bash
mkdocs build --strict 2>&1 | grep -E "(ERROR|WARNING)" | head -20
```

Expected: no errors about missing files.

- [ ] **Step 7: Commit**

```bash
git add docs/api/
git commit -m "docs: rename API doc files, add mkdocstrings blocks to existing pages"
```

---

## Task 9: New API doc — `docs/api/core.md`

**Files:**
- Write: `docs/api/core.md`

- [ ] **Step 1: Write `docs/api/core.md`**

```markdown
# Core — `aerisplane.core`

The `core` module contains the geometry and component dataclasses used by every
discipline module. It depends only on `numpy` — no solvers or backends.

---

## Coordinate system

All positions use the aircraft body frame:

- **x** — positive aft (toward the tail)
- **y** — positive right (starboard wing)
- **z** — positive up

`xyz_le` on `WingXSec` and `Wing` is the leading-edge position in this frame.
For symmetric wings, define the right side (y ≥ 0) only; the left side is mirrored
automatically.

---

## `Aircraft`

::: aerisplane.core.aircraft.Aircraft

---

## `Wing` and `WingXSec`

::: aerisplane.core.wing.Wing

::: aerisplane.core.wing.WingXSec

---

## `Fuselage` and `FuselageXSec`

::: aerisplane.core.fuselage.Fuselage

::: aerisplane.core.fuselage.FuselageXSec

---

## `Airfoil`

::: aerisplane.core.airfoil.Airfoil

---

## `ControlSurface` and `Servo`

::: aerisplane.core.control_surface.ControlSurface

::: aerisplane.core.control_surface.Servo

### Sign convention

| Surface | Positive deflection |
|---|---|
| elevator | trailing edge down (nose-up moment) |
| flap | trailing edge down (lift increase) |
| aileron (symmetric=False) | right side TE-down / left side TE-up |
| rudder | trailing edge left (nose-right moment) |

---

## `FlightCondition`

::: aerisplane.core.flight_condition.FlightCondition

### Control surface deflections

Deflections are passed through `FlightCondition.deflections` — a dict mapping surface
name to angle in degrees. The surface name must match `ControlSurface.name` on the wing.

```python
cond = ap.FlightCondition(
    velocity=18.0,
    altitude=100.0,
    alpha=3.5,
    deflections={"elevator": -5.0, "aileron": 10.0},
)
```

---

## Structural components

::: aerisplane.core.structures.Spar

::: aerisplane.core.structures.TubeSection

::: aerisplane.core.structures.Material

---

## Propulsion components

::: aerisplane.core.propulsion.Motor

::: aerisplane.core.propulsion.Propeller

::: aerisplane.core.propulsion.Battery

::: aerisplane.core.propulsion.ESC

::: aerisplane.core.propulsion.PropulsionSystem
```

- [ ] **Step 2: Verify page builds**

```bash
mkdocs build --strict 2>&1 | grep -i "core" | head -10
```

Expected: no errors related to core.md.

- [ ] **Step 3: Commit**

```bash
git add docs/api/core.md
git commit -m "docs: add core API reference page"
```

---

## Task 10: New API doc — `docs/api/structures.md`

**Files:**
- Write: `docs/api/structures.md`

- [ ] **Step 1: Write `docs/api/structures.md`**

```markdown
# Structures — `aerisplane.structures`

Wing spar structural analysis using an Euler–Bernoulli beam model.
Checks bending yield, shear yield, buckling, and tip deflection
against ultimate design loads. Requires `Spar` definitions on wing cross-sections.

---

## Quick start

```python
import aerisplane as ap
from aerisplane.aero import analyze as aero_analyze
from aerisplane.weights import analyze as weight_analyze
from aerisplane.structures import analyze
from aerisplane.core.structures import Spar, TubeSection
from aerisplane.catalog.materials import CFRP_UD

# Add a spar to the wing root and tip cross-sections
spar = Spar(position=0.25, tube=TubeSection(outer_diameter=0.018, wall_thickness=0.0015),
            material=CFRP_UD)
wing.xsecs[0].spar = spar
wing.xsecs[1].spar = spar

cond = ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=6.0)
aero_r = aero_analyze(aircraft, cond, method="vlm")
weight_r = weight_analyze(aircraft)
result = analyze(aircraft, aero_r, weight_r, n_limit=4.0)
result.report()
```

---

## `analyze()`

::: aerisplane.structures.analyze

---

## `StructureResult`

::: aerisplane.structures.result.StructureResult

::: aerisplane.structures.result.WingStructureResult

### Margin of safety interpretation

| Field | Safe if ... |
|---|---|
| `bending_margin` | > 0 |
| `shear_margin` | > 0 |
| `buckling_margin` | > 0 |
| `tip_deflection_ratio` | < 0.10 (10% of semi-span, typical) |
| `spar_fits` | True |

---

## Spar definition

::: aerisplane.core.structures.Spar

::: aerisplane.core.structures.TubeSection

::: aerisplane.core.structures.Material
```

- [ ] **Step 2: Commit**

```bash
git add docs/api/structures.md
git commit -m "docs: add structures API reference page"
```

---

## Task 11: New API docs — `control.md`, `propulsion.md`

**Files:**
- Write: `docs/api/control.md`
- Write: `docs/api/propulsion.md`

- [ ] **Step 1: Write `docs/api/control.md`**

```markdown
# Control — `aerisplane.control`

Control authority analysis: roll rate, pitch authority, rudder authority, and servo
loads at a given flight condition. Computes control derivatives by forward
finite-differencing the aerodynamic solver.

Requires `ControlSurface` objects on the wings and a completed `stability.analyze()`.

---

## Quick start

```python
from aerisplane.weights import analyze as weight_analyze
from aerisplane.stability import analyze as stab_analyze
from aerisplane.control import analyze

wr = weight_analyze(aircraft)
aircraft.xyz_ref = wr.cg.tolist()

cond = ap.FlightCondition(velocity=18.0, altitude=0.0, alpha=4.0)
sr = stab_analyze(aircraft, cond, wr)
result = analyze(aircraft, cond, wr, sr, aero_method="vlm")
print(result.report())
```

---

## `analyze()`

::: aerisplane.control.analyze

---

## `ControlResult`

::: aerisplane.control.result.ControlResult

### Interpreting authority values

`aileron_authority`, `elevator_authority`, `rudder_authority` are normalised [0, 1]:
- **1.0** — surface can meet or exceed the reference requirement
- **< 1.0** — surface is undersized relative to the reference requirement

Reference values:
- Roll: 60 deg/s minimum roll rate requirement
- Crosswind: 5 m/s crosswind capability requirement
- Elevator: ability to trim over a ±10° alpha range
```

- [ ] **Step 2: Write `docs/api/propulsion.md`**

```markdown
# Propulsion — `aerisplane.propulsion`

Computes the motor/propeller/battery operating point at a given throttle setting
and flight velocity. Returns thrust, current draw, RPM, efficiency, C-rate, and
estimated battery endurance.

---

## Quick start

```python
from aerisplane.catalog.motors import sunnysky_x2216_1250
from aerisplane.catalog.batteries import tattu_4s_5200
from aerisplane.catalog.propellers import apc_10x4_7sf
from aerisplane.core.propulsion import ESC, PropulsionSystem
from aerisplane.propulsion import analyze

propulsion = PropulsionSystem(
    motor=sunnysky_x2216_1250,
    propeller=apc_10x4_7sf,
    battery=tattu_4s_5200,
    esc=ESC(max_current=60.0, resistance=0.002),
)
aircraft = ap.Aircraft(name="MyPlane", wings=[wing], propulsion=propulsion)
cond = ap.FlightCondition(velocity=16.0, altitude=0.0)

result = analyze(aircraft, cond, throttle=0.75)
print(result.report())
```

---

## `analyze()`

::: aerisplane.propulsion.analyze

---

## `PropulsionResult`

::: aerisplane.propulsion.result.PropulsionResult

---

## Hardware catalog

See the [Catalog](catalog.md) page for available motors, batteries, and propellers.
To use catalog hardware:

```python
from aerisplane.catalog import list_motors, list_batteries, list_propellers

motors = list_motors()
# pick by name
motor = next(m for m in motors if "SunnySky" in m.name and m.kv == 1250)
```
```

- [ ] **Step 3: Commit**

```bash
git add docs/api/control.md docs/api/propulsion.md
git commit -m "docs: add control and propulsion API reference pages"
```

---

## Task 12: New API doc — `docs/api/mdo.md`

**Files:**
- Write: `docs/api/mdo.md`

- [ ] **Step 1: Write `docs/api/mdo.md`**

```markdown
# MDO — `aerisplane.mdo`

Multidisciplinary optimisation orchestration. Defines the problem (design variables,
constraints, objective), evaluates the full discipline chain for each candidate design,
and drives external optimisers.

---

## Minimal example

```python
import aerisplane as ap
from aerisplane.mdo import MDOProblem, DesignVar, Constraint, Objective
from aerisplane.mission.segments import Mission, Cruise, Loiter

# Baseline aircraft (must have propulsion for mission analysis)
# ... define aircraft as normal ...

mission = Mission(
    start_altitude=0.0,
    segments=[
        Cruise(name="cruise", velocity=18.0, altitude=100.0, distance=8000.0),
        Loiter(name="loiter", velocity=16.0, altitude=100.0, duration=900.0),
    ]
)

problem = MDOProblem(
    aircraft=aircraft,
    condition=ap.FlightCondition(velocity=18.0, altitude=100.0),
    design_variables=[
        DesignVar("wings[0].xsecs[1].chord",       lower=0.08, upper=0.22, scale=0.15),
        DesignVar("wings[0].xsecs[1].xyz_le[1]",   lower=0.8,  upper=1.8,  scale=1.2),
    ],
    constraints=[
        Constraint("structures.wings[0].bending_margin", lower=0.0),
        Constraint("stability.static_margin",             lower=0.05, upper=0.20),
    ],
    objective=Objective("mission.total_energy", maximize=False),
    mission=mission,
)

result = problem.optimize(method="scipy_de", options={"maxiter": 200, "seed": 42})
result.report()
```

---

## Design variable paths

Design variables use dot-bracket path syntax into the `Aircraft` object:

| Example path | What it controls |
|---|---|
| `"wings[0].xsecs[1].chord"` | Tip chord of first wing |
| `"wings[0].xsecs[1].xyz_le[1]"` | Semi-span (y of tip leading edge) |
| `"wings[0].xsecs[0].twist"` | Root twist angle |
| `"wings[1].xsecs[0].chord"` | Root chord of second wing (htail) |

Paths are validated at `MDOProblem` construction time — a `ValueError` is raised
immediately if a path does not resolve on the aircraft.

---

## Constraint and objective paths

Constraint and objective paths reference discipline result fields:

| Prefix | Example | Notes |
|---|---|---|
| `"aero."` | `"aero.CL"` | Fields of `AeroResult` |
| `"weights."` | `"weights.total_mass"` | Fields of `WeightResult` |
| `"structures."` | `"structures.wings[0].bending_margin"` | Fields of `WingStructureResult` |
| `"stability."` | `"stability.static_margin"` | Fields of `StabilityResult` |
| `"control."` | `"control.aileron_authority"` | Fields of `ControlResult` |
| `"mission."` | `"mission.total_energy"` | Fields of `MissionResult` |
| `"propulsion."` | `"propulsion.c_rate"` | Fields of `PropulsionResult` |

---

## Optimizer methods

| Method | Driver | Description |
|---|---|---|
| `"scipy_de"` | SciPy | Differential Evolution — global, no gradients, handles integers |
| `"scipy_minimize"` | SciPy | Local gradient-free (Nelder-Mead) or gradient-based (SLSQP) |
| `"scipy_shgo"` | SciPy | Simplicial Homology Global Optimisation |
| `"pygmo_de"` | pygmo | Differential Evolution (requires `pip install pygmo`) |
| `"pygmo_sade"` | pygmo | Self-Adaptive DE (more robust) |
| `"pygmo_nsga2"` | pygmo | Multi-objective NSGA-II |

Recommended default: `"scipy_de"` — robust, handles box constraints naturally,
no gradient needed, works with integer variables (airfoil pool).

---

## Checkpoint and resume

```python
# First run — saves checkpoint every 50 evaluations
result = problem.optimize(
    method="scipy_de",
    checkpoint_path="runs/opt_run1",
    checkpoint_interval=50,
)

# Resume from checkpoint after interruption
result = problem.optimize(
    method="scipy_de",
    checkpoint_path="runs/opt_run1",   # same path — automatically resumed
)
```

---

## Sensitivity analysis

After optimization, compute normalized finite-difference gradients to understand
which design variables matter most:

```python
sens = problem.sensitivity(result.x_optimal)
print(sens.report())
```

---

## Auto-generated reference

::: aerisplane.mdo.problem.MDOProblem

::: aerisplane.mdo.problem.DesignVar

::: aerisplane.mdo.problem.Constraint

::: aerisplane.mdo.problem.Objective

::: aerisplane.mdo.problem.AirfoilPool

::: aerisplane.mdo.result.OptimizationResult

::: aerisplane.mdo.sensitivity.SensitivityResult
```

- [ ] **Step 2: Commit**

```bash
git add docs/api/mdo.md
git commit -m "docs: add MDO API reference page with worked examples"
```

---

## Task 13: New API doc — `docs/api/catalog.md`

**Files:**
- Write: `docs/api/catalog.md`

- [ ] **Step 1: Write `docs/api/catalog.md`**

````markdown
# Catalog — `aerisplane.catalog`

Hardware database of real-world components for direct use in `PropulsionSystem`
and `Aircraft` definitions.

---

## Browsing the catalog

```python
from aerisplane.catalog import list_motors, list_batteries, list_propellers, list_servos

motors = list_motors()
for m in motors:
    print(f"{m.name:40s}  KV={m.kv:5.0f}  I_max={m.max_current:.0f}A  mass={m.mass*1000:.0f}g")
```

---

## Using catalog items

```python
from aerisplane.catalog.motors    import sunnysky_x2216_1250
from aerisplane.catalog.batteries import tattu_4s_5200
from aerisplane.catalog.propellers import apc_10x4_7sf
from aerisplane.core.propulsion import ESC, PropulsionSystem

propulsion = PropulsionSystem(
    motor=sunnysky_x2216_1250,
    propeller=apc_10x4_7sf,
    battery=tattu_4s_5200,
    esc=ESC(max_current=60.0, resistance=0.002),
)
```

---

## Motors

| Variable | Name | KV | Max I [A] | Mass [g] |
|---|---|---|---|---|
| `sunnysky_x2216_1250` | SunnySky X2216 1250KV | 1250 | 28 | 58 |
| `sunnysky_x2216_2400` | SunnySky X2216 2400KV | 2400 | 28 | 58 |
| `sunnysky_x2212_980` | SunnySky X2212 980KV | 980 | 20 | 52 |
| `tiger_mn3110_700` | T-Motor MN3110 700KV | 700 | 16 | 102 |
| `tiger_mn3110_780` | T-Motor MN3110 780KV | 780 | 16 | 102 |
| `tiger_mn2213_950` | T-Motor MN2213 950KV | 950 | 14 | 60 |
| `tiger_mn4014_330` | T-Motor MN4014 330KV | 330 | 22 | 176 |
| `tiger_mn5212_340` | T-Motor MN5212 340KV | 340 | 30 | 215 |
| `t_motor_f80_1900` | T-Motor F80 1900KV | 1900 | 30 | 68 |
| `t_motor_f60_2550` | T-Motor F60 2550KV | 2550 | 35 | 55 |
| `emax_mt2213_935` | Emax MT2213 935KV | 935 | 20 | 57 |
| `emax_mt2216_810` | Emax MT2216 810KV | 810 | 20 | 75 |
| `emax_rs2205_2600` | Emax RS2205 2600KV | 2600 | 30 | 30 |
| `rctimer_5010_360` | RCTimer 5010 360KV | 360 | 40 | 190 |
| `scorpion_m2205_2350` | Scorpion M2205 2350KV | 2350 | 30 | 35 |
| `scorpion_hkii_2221_900` | Scorpion HKII-2221 900KV | 900 | 22 | 68 |
| `axi_2217_20` | AXi 2217/20 | 1050 | 18 | 95 |
| `turnigy_d3530_1400` | Turnigy D3530/14 1400KV | 1400 | 21 | 86 |
| `hacker_a20_26` | Hacker A20-26L | 1020 | 16 | 72 |
| `dualsky_eco_2315c_1100` | Dualsky ECO 2315C 1100KV | 1100 | 18 | 65 |

Import from `aerisplane.catalog.motors`.

---

## Batteries

| Variable | Name | Capacity [Ah] | Voltage [V] | C-rating | Mass [g] |
|---|---|---|---|---|---|
| `tattu_3s_2300` | Tattu 3S 2300mAh 45C | 2.3 | 11.1 | 45 | 178 |
| `tattu_4s_1800` | Tattu 4S 1800mAh 75C | 1.8 | 14.8 | 75 | 218 |
| `tattu_4s_3300` | Tattu 4S 3300mAh 45C | 3.3 | 14.8 | 45 | 302 |
| `tattu_4s_5200` | Tattu 4S 5200mAh 45C | 5.2 | 14.8 | 45 | 470 |
| `tattu_6s_10000` | Tattu 6S 10000mAh 25C | 10.0 | 22.2 | 25 | 1280 |
| `tattu_6s_16000` | Tattu 6S 16000mAh 15C | 16.0 | 22.2 | 15 | 1900 |
| `gens_ace_3s_2200` | Gens Ace 3S 2200mAh 25C | 2.2 | 11.1 | 25 | 162 |
| `gens_ace_4s_4000` | Gens Ace 4S 4000mAh 45C | 4.0 | 14.8 | 45 | 390 |
| `gens_ace_6s_6000` | Gens Ace 6S 6000mAh 30C | 6.0 | 22.2 | 30 | 870 |
| `turnigy_nano_tech_3s_2200` | Turnigy Nano-tech 3S 2200mAh | 2.2 | 11.1 | 25 | 156 |
| `turnigy_nano_tech_4s_5000` | Turnigy Nano-tech 4S 5000mAh | 5.0 | 14.8 | 25 | 480 |
| `turnigy_nano_tech_6s_3300` | Turnigy Nano-tech 6S 3300mAh | 3.3 | 22.2 | 45 | 480 |
| `multistar_4s_10000` | Multistar 4S 10000mAh 10C | 10.0 | 14.8 | 10 | 890 |
| `ovonic_4s_2200` | Ovonic 4S 2200mAh 50C | 2.2 | 14.8 | 50 | 200 |
| `ovonic_6s_3300` | Ovonic 6S 3300mAh 50C | 3.3 | 22.2 | 50 | 480 |

Import from `aerisplane.catalog.batteries`.

---

## Propellers

| Variable | Diameter × Pitch | Mass [g] |
|---|---|---|
| `apc_10x4_7sf` | 10×4.7 in | 18 |
| `apc_11x4_7sf` | 11×4.7 in | 21 |
| `apc_13x4_7sf` | 13×4.7 in | 30 |
| `apc_10x7e` | 10×7 in | 19 |
| `apc_12x6e` | 12×6 in | 27 |
| `apc_14x8_3mf` | 14×8.3 in | 40 |
| `master_airscrew_10x5` | 10×5 in | 20 |
| `master_airscrew_11x7` | 11×7 in | 26 |
| `master_airscrew_14x7` | 14×7 in | 42 |
| `tjd_14x8_5` | 14×8.5 in | 22 |

Import from `aerisplane.catalog.propellers`.

---

## Airfoils

The airfoil catalog contains 2175 `.dat` files. Load by name:

```python
af = ap.Airfoil(name="ag35")       # load from catalog
af = ap.Airfoil(name="naca2412")   # generated analytically (NACA 4-digit)
```

---

## Auto-generated reference

::: aerisplane.catalog.list_motors

::: aerisplane.catalog.list_batteries

::: aerisplane.catalog.list_propellers

::: aerisplane.catalog.list_servos

::: aerisplane.catalog.get_airfoil
````

- [ ] **Step 2: Commit**

```bash
git add docs/api/catalog.md
git commit -m "docs: add catalog API reference page with hardware tables"
```

---

## Task 14: Tutorial 07 — MDO optimization notebook

**Files:**
- Create: `docs/tutorials/07_mdo_optimization.ipynb`

- [ ] **Step 1: Create the notebook**

Create `docs/tutorials/07_mdo_optimization.ipynb` with the following cells. Use
`nbformat` version 4, kernel `python3`.

**Cell 1 — markdown:**
```markdown
# Tutorial 07 — MDO Optimization

Define a multidisciplinary optimization problem, run it with differential evolution,
plot convergence, and inspect which design variables matter most using sensitivity analysis.

**What you'll learn:**
- Set up an `MDOProblem` with design variables, constraints, and an objective
- Run `scipy_de` differential evolution optimizer
- Plot convergence history
- Use `sensitivity()` to rank design variable influence
- Compare baseline vs optimal design side-by-side
```

**Cell 2 — code (baseline aircraft):**
```python
import numpy as np
import matplotlib.pyplot as plt
import aerisplane as ap
from aerisplane.core.structures import Spar, TubeSection
from aerisplane.catalog.materials import CFRP_UD
from aerisplane.catalog.motors import sunnysky_x2216_1250
from aerisplane.catalog.batteries import tattu_4s_5200
from aerisplane.catalog.propellers import apc_10x4_7sf
from aerisplane.core.propulsion import ESC, PropulsionSystem

# Wing with CFRP spar
spar = Spar(
    position=0.25,
    tube=TubeSection(outer_diameter=0.018, wall_thickness=0.0015),
    material=CFRP_UD,
)
wing = ap.Wing(
    name="main_wing",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28,
                    airfoil=ap.Airfoil(name="ag35"), spar=spar),
        ap.WingXSec(xyz_le=[0.07, 1.20, 0.05], chord=0.14, twist=-2.0,
                    airfoil=ap.Airfoil(name="ag35"), spar=spar),
    ],
    control_surfaces=[
        ap.ControlSurface(name="aileron", span_start=0.5, span_end=0.9,
                          chord_fraction=0.25, symmetric=False, max_deflection=20.0),
    ],
)

htail = ap.Wing(
    name="htail",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.95, 0.00, 0.00], chord=0.15,
                    airfoil=ap.Airfoil(name="naca0012")),
        ap.WingXSec(xyz_le=[0.98, 0.40, 0.00], chord=0.10,
                    airfoil=ap.Airfoil(name="naca0012")),
    ],
    control_surfaces=[
        ap.ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
                          chord_fraction=0.35, symmetric=True, max_deflection=25.0),
    ],
)

vtail = ap.Wing(
    name="vtail",
    symmetric=False,
    xsecs=[
        ap.WingXSec(xyz_le=[0.90, 0.00, 0.00], chord=0.18,
                    airfoil=ap.Airfoil(name="naca0012")),
        ap.WingXSec(xyz_le=[0.95, 0.00, 0.30], chord=0.12,
                    airfoil=ap.Airfoil(name="naca0012")),
    ],
    control_surfaces=[
        ap.ControlSurface(name="rudder", span_start=0.0, span_end=1.0,
                          chord_fraction=0.30, symmetric=True, max_deflection=25.0),
    ],
)

propulsion = PropulsionSystem(
    motor=sunnysky_x2216_1250,
    propeller=apc_10x4_7sf,
    battery=tattu_4s_5200,
    esc=ESC(max_current=60.0, resistance=0.002),
)

aircraft = ap.Aircraft(
    name="baseline",
    wings=[wing, htail, vtail],
    propulsion=propulsion,
)

print(f"Baseline span : {wing.span():.2f} m")
print(f"Baseline area : {wing.area():.4f} m²")
print(f"Baseline AR   : {wing.span()**2 / wing.area():.2f}")
```

**Cell 3 — markdown:**
```markdown
## Define the mission

We want to minimize total mission energy (equivalent to maximizing endurance).
The mission is a simple cruise + loiter profile.
```

**Cell 4 — code (mission and MDO problem):**
```python
from aerisplane.mission.segments import Mission, Cruise, Loiter
from aerisplane.mdo import MDOProblem, DesignVar, Constraint, Objective

mission = Mission(
    start_altitude=0.0,
    segments=[
        Cruise(name="cruise_out", velocity=18.0, altitude=100.0, distance=5000.0),
        Loiter(name="loiter",     velocity=16.0, altitude=100.0, duration=600.0),
        Cruise(name="cruise_back",velocity=18.0, altitude=100.0, distance=5000.0),
    ]
)

condition = ap.FlightCondition(velocity=18.0, altitude=100.0)

problem = MDOProblem(
    aircraft=aircraft,
    condition=condition,
    design_variables=[
        DesignVar("wings[0].xsecs[1].chord",     lower=0.08, upper=0.22, scale=0.15),
        DesignVar("wings[0].xsecs[1].xyz_le[1]", lower=0.80, upper=1.80, scale=1.20),
    ],
    constraints=[
        Constraint("structures.wings[0].bending_margin", lower=0.0),
        Constraint("stability.static_margin", lower=0.05, upper=0.25),
    ],
    objective=Objective("mission.total_energy", maximize=False),
    mission=mission,
    alpha=4.0,
)
print("Problem validated. Design variables:")
for dv in problem._dvars:
    print(f"  {dv.path}  [{dv.lower}, {dv.upper}]")
```

**Cell 5 — markdown:**
```markdown
## Run the optimizer

Differential evolution (`scipy_de`) is a population-based global optimizer —
no gradient information needed, robust to local minima.
`maxiter=80` is enough for a 2-variable problem to converge.
```

**Cell 6 — code (optimize):**
```python
import logging
logging.basicConfig(level=logging.WARNING)  # suppress INFO logs from the discipline chain

result = problem.optimize(
    method="scipy_de",
    options={"maxiter": 80, "seed": 42, "popsize": 10},
    verbose=False,
    report_interval=20,
)
print(result.report())
```

**Cell 7 — markdown:**
```markdown
## Convergence history

The optimizer tracks every evaluation. We can plot the best objective value
over evaluations to see how quickly it converged.
```

**Cell 8 — code (convergence plot):**
```python
history = result.convergence_history
# convergence_history stores raw (negated) objective values for maximize=False
# total_energy is minimized, so no sign flip needed
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, lw=1.5, color="steelblue")
ax.set_xlabel("Evaluation number")
ax.set_ylabel("Best total energy [J]")
ax.set_title("Convergence history — scipy_de")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Cell 9 — markdown:**
```markdown
## Sensitivity analysis

Which design variable actually matters more? Sensitivity analysis computes
normalized finite-difference gradients at the optimal point.
```

**Cell 10 — code (sensitivity):**
```python
sens = problem.sensitivity(result.x_optimal)
print(sens.report())
```

**Cell 11 — markdown:**
```markdown
## Baseline vs optimal comparison

Print a side-by-side comparison of the key design values and discipline results.
```

**Cell 12 — code (comparison):**
```python
print("Design variable comparison:")
print(f"{'Variable':<45}  {'Baseline':>12}  {'Optimal':>12}")
print("-" * 72)
for path, (v0, vopt) in result.variables.items():
    print(f"{path:<45}  {v0:>12.4f}  {vopt:>12.4f}")

print(f"\nObjective (total energy [J]): {result.objective_initial:.1f} → {result.objective_optimal:.1f}")
print(f"Improvement: {(result.objective_initial - result.objective_optimal)/result.objective_initial:.1%}")
print(f"Constraints satisfied: {result.constraints_satisfied}")
print(f"Total evaluations: {result.n_evaluations}")
```

- [ ] **Step 2: Save notebook as proper JSON**

The notebook must be valid JSON with `nbformat=4`. Write the cells as a Python
script that generates the `.ipynb` file:

```python
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

cells = [
    # Cell 1 - intro markdown
    nbformat.v4.new_markdown_cell("# Tutorial 07 — MDO Optimization\n\nDefine a multidisciplinary optimization problem, run it with differential evolution,\nplot convergence, and inspect which design variables matter most using sensitivity analysis.\n\n**What you'll learn:**\n- Set up an `MDOProblem` with design variables, constraints, and an objective\n- Run `scipy_de` differential evolution optimizer\n- Plot convergence history\n- Use `sensitivity()` to rank design variable influence\n- Compare baseline vs optimal design side-by-side"),
    # ... add all other cells following the content above
]
nb.cells = cells

with open("docs/tutorials/07_mdo_optimization.ipynb", "w") as f:
    nbformat.write(nb, f)
```

Run this script to generate the notebook, then delete the script:

```bash
python generate_tutorial_07.py
rm generate_tutorial_07.py
```

- [ ] **Step 3: Execute the notebook to store outputs**

```bash
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute --inplace docs/tutorials/07_mdo_optimization.ipynb
```

- [ ] **Step 4: Verify the notebook renders in the site**

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000/tutorials/07_mdo_optimization/` and verify cells and outputs render.

- [ ] **Step 5: Commit**

```bash
git add docs/tutorials/07_mdo_optimization.ipynb
git commit -m "docs: add Tutorial 07 — MDO optimization walkthrough"
```

---

## Task 15: Tutorial 08 — Propulsion and endurance notebook

**Files:**
- Create: `docs/tutorials/08_propulsion_and_endurance.ipynb`

- [ ] **Step 1: Create the notebook with the following cells**

**Cell 1 — markdown:**
```markdown
# Tutorial 08 — Propulsion and Endurance

Browse the hardware catalog, set up a propulsion system, analyze it at different
throttle settings, estimate endurance, and integrate with mission analysis.

**What you'll learn:**
- Browse motors, batteries, and propellers in the catalog
- Build a `PropulsionSystem` and run `propulsion.analyze()`
- Throttle sweep: thrust and efficiency curves
- Compute and compare endurance for different hardware combos
- Feed propulsion results into the mission discipline
```

**Cell 2 — code (browse catalog):**
```python
import aerisplane as ap
from aerisplane.catalog import list_motors, list_batteries, list_propellers

motors = list_motors()
print(f"{'Name':<42} {'KV':>6} {'I_max':>7} {'mass':>7}")
print("-" * 67)
for m in sorted(motors, key=lambda x: x.kv):
    print(f"{m.name:<42} {m.kv:>6.0f} {m.max_current:>6.0f}A {m.mass*1000:>6.0f}g")
```

**Cell 3 — code (browse batteries):**
```python
batteries = list_batteries()
print(f"{'Name':<42} {'Cap [Ah]':>9} {'V':>6} {'C':>5} {'mass':>7}")
print("-" * 74)
for b in sorted(batteries, key=lambda x: x.capacity_ah):
    print(f"{b.name:<42} {b.capacity_ah:>9.1f} {b.nominal_voltage:>6.1f} {b.c_rating:>5.0f}C {b.mass*1000:>6.0f}g")
```

**Cell 4 — code (build propulsion system):**
```python
from aerisplane.catalog.motors import sunnysky_x2216_1250
from aerisplane.catalog.batteries import tattu_4s_5200
from aerisplane.catalog.propellers import apc_10x4_7sf
from aerisplane.core.propulsion import ESC, PropulsionSystem
from aerisplane.propulsion import analyze as prop_analyze

propulsion = PropulsionSystem(
    motor=sunnysky_x2216_1250,
    propeller=apc_10x4_7sf,
    battery=tattu_4s_5200,
    esc=ESC(max_current=60.0, resistance=0.002),
)

# Build a simple aircraft for the analysis
wing = ap.Wing(
    name="main_wing",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28, airfoil=ap.Airfoil(name="ag35")),
        ap.WingXSec(xyz_le=[0.07, 1.20, 0.05], chord=0.14, twist=-2.0,
                    airfoil=ap.Airfoil(name="ag35")),
    ],
)
aircraft = ap.Aircraft(name="MyPlane", wings=[wing], propulsion=propulsion)

# Single operating point — cruise throttle
cond = ap.FlightCondition(velocity=16.0, altitude=0.0)
result = prop_analyze(aircraft, cond, throttle=0.6)
print(result.report())
```

**Cell 5 — markdown:**
```markdown
## Throttle sweep

How does thrust and efficiency change from idle to full throttle?
```

**Cell 6 — code (throttle sweep):**
```python
import numpy as np
import matplotlib.pyplot as plt

throttles = np.linspace(0.2, 1.0, 17)
thrusts, motor_effs, prop_effs, currents = [], [], [], []

for t in throttles:
    r = prop_analyze(aircraft, cond, throttle=t)
    thrusts.append(r.thrust_n)
    motor_effs.append(r.motor_efficiency * 100)
    prop_effs.append(r.propulsive_efficiency * 100)
    currents.append(r.current_a)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(throttles * 100, thrusts, "b-o", ms=4)
axes[0].set_xlabel("Throttle [%]")
axes[0].set_ylabel("Thrust [N]")
axes[0].set_title("Thrust")
axes[0].grid(True, alpha=0.3)

axes[1].plot(throttles * 100, motor_effs, "r-o", ms=4, label="Motor η")
axes[1].plot(throttles * 100, prop_effs,  "g-o", ms=4, label="Propulsive η")
axes[1].set_xlabel("Throttle [%]")
axes[1].set_ylabel("Efficiency [%]")
axes[1].set_title("Efficiency")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(throttles * 100, currents, "m-o", ms=4)
axes[2].axhline(sunnysky_x2216_1250.max_current, color="r", ls="--", label="I_max")
axes[2].set_xlabel("Throttle [%]")
axes[2].set_ylabel("Current [A]")
axes[2].set_title("Battery current")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Cell 7 — markdown:**
```markdown
## Endurance comparison

Compare endurance for different battery choices at 60% cruise throttle.
```

**Cell 8 — code (endurance comparison):**
```python
from aerisplane.catalog.batteries import (
    tattu_4s_3300, tattu_4s_5200, gens_ace_4s_4000, turnigy_nano_tech_4s_5000
)

battery_options = [
    ("Tattu 4S 3300", tattu_4s_3300),
    ("Tattu 4S 5200", tattu_4s_5200),
    ("Gens Ace 4S 4000", gens_ace_4s_4000),
    ("Nano-Tech 4S 5000", turnigy_nano_tech_4s_5000),
]

endurances, labels, masses = [], [], []

for label, batt in battery_options:
    prop_sys = PropulsionSystem(
        motor=sunnysky_x2216_1250,
        propeller=apc_10x4_7sf,
        battery=batt,
        esc=ESC(max_current=60.0, resistance=0.002),
    )
    ac = ap.Aircraft(name="test", wings=[wing], propulsion=prop_sys)
    r = prop_analyze(ac, cond, throttle=0.6)
    endurances.append(r.battery_endurance_s / 60)
    labels.append(label)
    masses.append(batt.mass * 1000)

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(labels, endurances, color="steelblue")
ax.bar_label(bars, fmt="%.1f min", padding=3)
ax2 = ax.twinx()
ax2.plot(labels, masses, "ro--", label="Battery mass [g]")
ax2.set_ylabel("Battery mass [g]", color="red")
ax.set_ylabel("Endurance [min]")
ax.set_title("Endurance by battery (60% throttle, V=16 m/s)")
plt.tight_layout()
plt.show()
```

**Cell 9 — markdown:**
```markdown
## Integration with mission analysis

The propulsion model feeds into `mission.analyze()` for accurate range and
endurance estimates accounting for different flight segments.
```

**Cell 10 — code (mission integration):**
```python
from aerisplane.weights import analyze as weight_analyze
from aerisplane.mission import analyze as mission_analyze
from aerisplane.mission.segments import Mission, Climb, Cruise, Loiter

wr = weight_analyze(aircraft)

mission = Mission(
    start_altitude=0.0,
    segments=[
        Climb(name="climb",  velocity=14.0, climb_rate=2.0, to_altitude=100.0),
        Cruise(name="cruise", velocity=18.0, altitude=100.0, distance=8000.0),
        Loiter(name="loiter", velocity=16.0, altitude=100.0, duration=600.0),
        Cruise(name="return", velocity=18.0, altitude=100.0, distance=8000.0),
    ]
)

mr = mission_analyze(aircraft, wr, mission)
print(mr.report())
```

- [ ] **Step 2: Write and execute the notebook** (same approach as Task 14)

Generate the `.ipynb` using a script, execute it to store outputs, commit.

```bash
jupyter nbconvert --to notebook --execute --inplace docs/tutorials/08_propulsion_and_endurance.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/08_propulsion_and_endurance.ipynb
git commit -m "docs: add Tutorial 08 — propulsion and endurance walkthrough"
```

---

## Task 16: Full site build verification and push

**Files:** none new — verification only.

- [ ] **Step 1: Run full site build**

```bash
mkdocs build --strict 2>&1 | tail -20
```

Expected: `INFO - Documentation built in X.X seconds` with no ERRORs or unresolved
references.

- [ ] **Step 2: Serve locally and spot-check all nav items**

```bash
mkdocs serve
```

Check each nav section manually:
- Home, Installation, Getting Started load correctly
- All 9 tutorial notebooks render with code outputs
- All 10 API reference pages render with mkdocstrings blocks
- Architecture, Contributing, Changelog pages load

- [ ] **Step 3: Push to GitHub**

```bash
git push origin feature/vendor-asb-aero
```

Create a pull request to `main`. After merging to `main`, the GitHub Actions
`docs.yml` workflow will automatically deploy the site to GitHub Pages at
`https://kasparstepan.github.io/AerisPlane`.

- [ ] **Step 4: Enable GitHub Pages on the repository**

In the repository settings on GitHub:
- Go to **Settings → Pages**
- Set **Source** to `Deploy from a branch`
- Set **Branch** to `gh-pages` / `(root)`

After the first `docs.yml` run completes, the site will be live.

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by task |
|---|---|
| MkDocs site + Material theme + GitHub Pages | Task 1, 16 |
| mkdocstrings blocks | Tasks 8–13 |
| mkdocs-jupyter tutorials | Task 1 (config) + 14, 15 |
| GitHub Actions CI (tests + docs) | Task 5 |
| pyproject.toml URLs + docs dep | Task 4 |
| CONTRIBUTING.md, CHANGELOG.md | Task 3 |
| Stale docs fix | Task 3 |
| API docs for missing modules (core, structures, control, propulsion, mdo, catalog) | Tasks 9–13 |
| Rename existing api-*.md → *.md | Task 8 |
| Tutorial 07 MDO | Task 14 |
| Tutorial 08 Propulsion | Task 15 |
| Docstrings: Airfoil, ControlSurface, Motor/Battery/Propeller | Task 6 |
| Docstrings: mission.analyze(), propulsion.analyze() | Task 7 |
| landing pages: index, installation, getting-started | Task 2 |

All spec requirements accounted for. ✓
