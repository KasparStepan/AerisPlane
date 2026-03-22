# UAVDesign: Conceptual RC/UAV MDO Tool Architecture

This document describes the architecture of `uavdesign`, a Python‑based conceptual multidisciplinary design and optimization (MDAO) tool for fixed‑wing and optionally VTOL RC‑scale UAVs. The architecture is headless, library‑like, and designed to support future high‑fidelity aerodynamics, 3D‑print‑ready geometry, and CAD integration.

---

## 1. Design goals

### 1.1 Conceptual‑only, RC‑UAV‑focused

`uavdesign` is a **conceptual design / MDO environment**, not a full‑fidelity CFD/FEA or HPC framework. It targets:

- Low‑Re UAVs (1–20 kg, electric, hand‑ or catapult‑launched).
- Parametric configuration sizing: spans, chords, tail volumes, aspect ratios, propulsion sizing, and mission feasibility.
- Exploration of design space via global and gradient‑based optimization.

High‑fidelity local effects (e.g., separations, acoustics, detailed stress concentrations) are handled later in external tools such as CFD/FEA or CAD/3D printing.

### 1.2 “Pure Python” and extensible

- Zero external GUI apps in the core pipeline (e.g., no OpenVSP inside `AircraftConfig`).
- Aero backends such as AeroSandbox, OpenVSP, native VLM, or SU2 appear only behind **adapters**, not in the core model.
- The architecture is designed so that you can add CAD/STL exporters, FEA integrators, and flight dynamics simulators as optional modules later.

### 1.3 Data‑oriented and optimization‑friendly

- The aircraft is a **rich, parameterized object model** that can be vectorized into a design vector for optimization.
- The interface to each discipline (aero, weights, propulsion, mission, flight dynamics) is simple, functional, and transparent.

---

## 2. High‑level organization

`uavdesign` is organized as a Python package with discipline‑separated modules:

```text
uavdesign/
  core/          # basic data types, state, units, design variables
  geometry/      # aircraft geometry model: Wing, Fuselage, Tail, etc.
  aero/          # 2D/3D aerodynamics; adapters to AeroSandbox, native VLM, etc.
  weights/       # mass, CG, inertia, and basic structure models
  propulsion/    # motor, propeller, battery, ESC, and power/energy models
  mission/       # mission segments, range/endurance, constraints
  flightdyn/     # trim, flight dynamics, stability, and control
  mdo/           # optimization, constraints, and MDO orchestration
  export/        # future CAD/STL exporters (e.g., OpenCascade, OpenVSP, Blender)
  io/            # configuration formats (YAML/JSON) and logging
```

Dependencies:

- NumPy, SciPy, and optionally OpenMDAO (as a pluggable driver, not a core dependency).
- AeroSandbox and NeuralFoil appear only in `aero/aerosandbox_backend.py` and possibly `weights/`, not in the core model.

---

## 3. Core data model (`core/`)

`core/` contains the minimal data types needed across disciplines.

### 3.1 `Environment` and `FlightCondition`

```python
class Environment:
    """Atmosphere, gravity, and environment constants."""
    def __init__(self, altitude, temperature, pressure, density):
        ...

class FlightCondition:
    """State point for a mission segment."""
    def __init__(self,
                 v: float,      # airspeed [m/s]
                 h: float,      # altitude [m]
                 alpha: float,  # angle of attack [rad]
                 beta: float,   # sideslip [rad]
                 p: float,      # roll rate [rad/s]
                 q: float,      # pitch rate [rad/s]
                 r: float,      # yaw rate [rad/s]
                 df_ail: float,  # aileron deflection [rad]
                 df_elev: float, # elevator deflection [rad]
                 df_rudd: float, # rudder deflection [rad]
                 ):
        ...
```

`FlightCondition` serves as a standardized interface between aero, propulsion, and flight dynamics.

### 3.2 `DesignVariables` and `DesignPoint`

```python
class DesignVariables:
    """List of design variables with metadata."""
    def __init__(self, names: List[str], lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        ...

class DesignPoint:
    """Current values of design variables."""
    def __init__(self, values: np.ndarray, variables: DesignVariables):
        ...
```

These allow the `mdo/` module to treat the aircraft as a vector for MDO without tying it to any specific solver.

---

## 4. Aircraft geometry model (`geometry/`)

### 4.1 Overall design: `AircraftConfig`

The aircraft is a single, rich object that encapsulates the configuration:

```python
from typing import List
from .wing import Wing
from .fuselage import Fuselage
from .tail import Tail        # or just Wing with role flags
from .propulsion import PropulsionSystem
from .payload import Payload
from .environment import Environment

class AircraftConfig:
    def __init__(self):
        self.wings:      List[Wing]           = []   # main + canard, etc.
        self.fuselages:  List[Fuselage]       = []   # fuselage, centerbody, etc.
        self.tails:      List[Tail]           = []   # horizontal, vertical
        self.propulsion: PropulsionSystem     = PropulsionSystem(...)
        self.payload:    Payload              = Payload(...)
        self.environment: Environment         = Environment(...)
        self.design_vars: Optional[DesignVariables] = None

    def to_numpy(self) -> np.ndarray:
        """Return design vector as 1D array."""
        ...

    def from_numpy(self, x: np.ndarray):
        """Update configuration from design vector."""
        ...
```

`AircraftConfig` is **your single source of truth** for the aircraft. All other modules (`aero/`, `weights/`, `mission/`, `flightdyn/`, `mdo/`) depend only on this class and its helper methods, not on external geometry types.

### 4.2 Wings and fuselages: `Wing`, `Fuselage`, `WingXSec`, `FuselageXSec`

The geometry classes are strongly inspired by AeroSandbox’s `Airplane/WingXSec/FuselageXSec` structure but live in your own namespace.

```python
class WingXSec:
    """Spanwise cross‑section of a wing."""
    def __init__(self,
                 y_le: float,        # local y at leading edge
                 z_le: float,        # local z offset
                 chord: float,
                 twist: float,       # [rad]
                 airfoil: "Airfoil", # your own Airfoil object
                 control_surfaces: List["ControlSurface"] = None,
                 ):
        ...
    def area_contribution(self) -> float:
        ...
```

```python
class FuselageXSec:
    """Cross‑section of a fuselage."""
    def __init__(self,
                 x: float,            # distance from nose or ref point
                 shape_type: str,     # "circle", "ellipse", "rectangle"
                 radius_or_dims: tuple,  # e.g., (r,) or (w, h)
                 ):
        ...
    def area(self) -> float:
        ...
```

```python
class Wing:
    """A lifting surface composed of WingXSecs."""
    def __init__(self,
                 x_le: float,         # root x
                 y_le: float,         # root y
                 z_le: float,         # root z
                 symmetric: bool,     # mirror across y=0?
                 xsecs: List[WingXSec],
                 name: str = "wing",
                 ):
        self.x_le = x_le
        self.y_le = y_le
        self.z_le = z_le
        self.symmetric = symmetric
        self.xsecs = xsecs
        self.name = name

    def span(self) -> float:
        """Total span."""
        ...

    def area(self) -> float:
        """Planform area."""
        ...

    def mesh_panels(self, n_spanwise: int, n_chordwise: int) -> List["Panel3D"]:
        """Generate 3D panels for VLM."""
        ...
```

```python
class Fuselage:
    """Fuselage composed of FuselageXSecs."""
    def __init__(self,
                 x_le: float,         # nose x
                 y_le: float,         # center y
                 z_le: float,         # center z
                 xsecs: List[FuselageXSec],
                 name: str = "fuselage",
                 ):
        self.x_le = x_le
        self.y_le = y_le
        self.z_le = z_le
        self.xsecs = xsecs
        self.name = name

    def length(self) -> float:
        ...
    def volume_approx(self) -> float:
        ...
```

### 4.3 `Airfoil` and `ControlSurface`

```python
class Airfoil:
    def __init__(self, coords: np.ndarray, name: str):
        self.coords = coords       # (N, 2) airfoil coordinates
        self.name = name

    def repanel(self, n: int) -> np.ndarray:
        """Return re‑panelled coordinates."""
        ...
    def thickness(self) -> float:
        ...
    def chord_leading_edge(self) -> float:
        ...
```

```python
class ControlSurface:
    def __init__(self,
                 type_: str,      # "aileron", "flap", "elevator", "rudder"
                 span_min: float, # fraction of span, 0–1
                 span_max: float, # fraction of span, 0–1
                 deflection_axis_local: np.ndarray,  # local vector
                 max_deflection: float,   # [rad]
                 ):
        ...
```

---

## 5. Aerodynamics module (`aero/`)

### 5.1 Aerodynamics backends

You can plug in:

- AeroSandbox (LLT, VLM, NeuralFoil, AVL, etc.).
- Your own low‑order models (LLT, VLM).
- (Optional, future) OpenVSP/VSPAERO, SU2, XFLR5, etc.

Structure:

```text
uavdesign/aero/
  __init__.py
  llt.py
  vlm.py
  neuralfoil_backend.py
  aerosandbox_backend.py
  openvsp_backend.py
  interfaces.py
```

### 5.2 Unified aero interface

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class AeroLoads:
    CL: float
    CD: float
    CM: float
    CFx: float   # forces in body axes?
    CFy: float
    CFz: float
    CMx: float
    CMy: float
    CMz: float
    CL_dist: np.ndarray  # spanwise CL?
    CDi_dist: np.ndarray  # induced drag dist?

def compute_aero(
    aircraft: AircraftConfig,
    condition: FlightCondition,
    backend: str = "aerosandbox",
) -> AeroLoads:
    if backend == "aerosandbox":
        return aero_aerosandbox.compute_aero(aircraft, condition)
    elif backend == "native_vlm":
        return aero_vlm.compute_aero(aircraft, condition)
    elif backend == "openvsp":
        return aero_openvsp.compute_aero(aircraft, condition)
    else:
        raise ValueError(f"Unknown aero backend: {backend}")
```

### 5.3 AeroSandbox adapter (`aero/aerosandbox_backend.py`)

```python
import aerosandbox as asb

def to_aerosandbox_airplane(aircraft: AircraftConfig) -> asb.Airplane:
    ...

def compute_aero(
    aircraft: AircraftConfig,
    condition: FlightCondition,
) -> AeroLoads:
    airplane = to_aerosandbox_airplane(aircraft)
    # Choose method: LLT, VLM, etc.
    aero_result = airplane.get_aero(...)
    ...
```

### 5.4 Native VLM / LLT (`aero/vlm.py`, `aero/llt.py`)

- `aero/llt.py`: nonlinear lifting‑line using NeuralFoil polars.
- `aero/vlm.py`: 3D VLM over panels generated from `Wing.mesh_panels`.

---

## 6. Weights, CG, and inertia (`weights/`)

```python
class WeightModel:
    def __init__(self, aircraft: AircraftConfig):
        self.aircraft = aircraft

    def total_mass(self) -> float:
        ...

    def cg(self) -> np.ndarray:
        ...

    def inertia_tensor(self) -> np.ndarray:
        ...
```

Implementation uses simple geometric primitives and calibrated densities to approximate component masses, CG, and inertias.

---

## 7. Propulsion module (`propulsion/`)

```python
class Motor:
    def __init__(self, kv: float, resistance: float, max_current: float):
        ...

class Propeller:
    def __init__(self, diameter: float, pitch: float, diameter_units: str):
        ...

class Battery:
    def __init__(self, capacity_ah: float, nominal_voltage: float, c_rating: float):
        ...

class PropulsionSystem:
    def __init__(self, motor: Motor, prop: Propeller, battery: Battery, esc: ESC):
        ...

    def thrust(self, rpm: float, v: float) -> float:
        ...
    def power_electrical(self, rpm: float, v: float) -> float:
        ...
    def available_thrust(self, conditions: List[FlightCondition]) -> float:
        ...
```

---

## 8. Mission and performance (`mission/`)

```python
class MissionSegment:
    def __init__(self, duration: float, start: FlightCondition, name: str):
        ...

    def update(self, aircraft: AircraftConfig, step: float) -> FlightCondition:
        ...
```

```python
class MissionProfile:
    def __init__(self, segments: List[MissionSegment]):
        self.segments = segments

    def run(self, aircraft: AircraftConfig) -> dict:
        ...
```

---

## 9. Flight dynamics, trim, and stability (`flightdyn/`)

```python
def trim(aircraft: AircraftConfig, condition: FlightCondition, aero_backend: str) -> FlightCondition:
    ...

def linearize(aircraft: AircraftConfig, trimmed_condition: FlightCondition, aero_backend: str) -> dict:
    ...
```

```python
class FlightDynamicsModel:
    def __init__(self, aircraft: AircraftConfig, weights: WeightModel, aero_backend: str):
        ...

    def simulate_6dof(self, initial_state: np.ndarray, time: np.ndarray, controls: np.ndarray):
        ...
```

---

## 10. MDO and optimization (`mdo/`)

The `mdo/` module orchestrates optimization over `AircraftConfig`:

```python
def solve_mdo(
    aircraft: AircraftConfig,
    objective_fn: Callable[[AircraftConfig], float],
    constraints: List[Callable[[AircraftConfig], float]],
    design_vars: DesignVariables,
    method: str = "scipy",
    options: dict = None,
):
    """Top-level MDO entry point."""
    ...
```

It translates the design vector into `AircraftConfig`, runs mission/aero/weights/constraints, and passes results to optimizers (SciPy, pyOptSparse, pygmo, or OpenMDAO drivers).

---

# Repository Layout and Folder Structure

Below is a modern, Python‑centric folder layout that follows current best practices (PEP 517, `pyproject.toml`, tests, CI, docs, and examples).

---

## 1. Repository root layout

```text
repo-root/
├── pyproject.toml              # build + deps + linting tools (ruff, mypy, pytest, etc.)
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml     # (if you use pre-commit)
├── docs/                       # Sphinx docs, architecture, user guides
├── notebooks/                  # Jupyter/interactive notebooks for exploration
├── examples/                   # Short, self‑contained example scripts
├── src/aerisplane/             # main Python package
│   ├── __init__.py
│   ├── core/                   # data types, state, design vars
│   ├── geometry/               # AircraftConfig, Wing, Fuselage, etc.
│   ├── aero/                   # aero backends (AeroSandbox, VLM, NeuralFoil, etc.)
│   ├── weights/                # mass, CG, inertia
│   ├── propulsion/             # motor/prop/battery models
│   ├── mission/                # mission segments and profiles
│   ├── flightdyn/              # trim, linearization, flight dynamics
│   ├── mdo/                    # MDO orchestration and optimization
│   ├── export/                 # CAD/STL exporters
│   ├── io/                     # config I/O, logging, serialization
│   └── tests/                  # optional local tests
└── tests/                      # recommended root-level tests
    ├── conftest.py
    ├── test_*.py
    ├── geometry/
    ├── aero/
    ├── weights/
    └── ...
```

---

## 2. Within `src/aerisplane/` – module structure

```text
uavdesign/
├── __init__.py                 # expose main public API
│   from uavdesign.core import AircraftConfig, FlightCondition, Environment
│   from uavdesign.geometry import Wing, Fuselage, WingXSec, FuselageXSec
│   from uavdesign.aero import compute_aero, AeroLoads
│   from uavdesign.weights import WeightModel
│   from uavdesign.propulsion import PropulsionSystem
│   from uavdesign.mission import MissionProfile
│   from uavdesign.flightdyn import trim, linearize
│   from uavdesign.mdo import solve_mdo
│
├── core/
│   ├── __init__.py
│   ├── base.py                 # AircraftConfig, Environment, FlightCondition
│   └── variables.py            # DesignVariables, DesignPoint
│
├── geometry/
│   ├── __init__.py             # Wing, Fuselage, WingXSec, FuselageXSec, Airfoil
│   ├── wing.py
│   ├── fuselage.py
│   ├── airfoil.py
│   └── control_surface.py
│
├── aero/
│   ├── __init__.py             # compute_aero, AeroLoads, backend selectors
│   ├── interfaces.py           # AeroLoads dataclass, core API
│   ├── neuralfoil_backend.py   # NeuralFoil integration
│   ├── aerosandbox_backend.py  # asb.Airplane ↔ AeroLoads
│   ├── native_vlm.py           # your own VLM implementation
│   ├── llt.py                  # lifting‑line
│   └── utils.py                # paneling and helper functions
│
├── weights/
│   ├── __init__.py
│   └── weight_model.py
│
├── propulsion/
│   ├── __init__.py
│   └── components.py
│
├── mission/
│   ├── __init__.py
│   └── mission.py
│
├── flightdyn/
│   ├── __init__.py
│   └── flightdyn.py
│
├── mdo/
│   ├── __init__.py
│   ├── problem.py              # MDO problem definition, constraints
│   └── optimizers.py           # SciPy / pyOptSparse / pygmo wrappers
│
├── export/
│   ├── __init__.py
│   ├── stl_opencascade.py      # CAD kernel + STL export (future)
│   ├── openscad_export.py      # OpenSCAD backend (future)
│   └── openvsp_export.py       # asb → .vsp (optional)
│
└── io/
    ├── __init__.py
    └── config.py               # YAML/JSON config I/O
```

---

## 3. Tests layout

```text
tests/
├── conftest.py                 # fixtures, e.g., sample_aircraft()
├── test_core.py
├── test_geometry.py
├── geometry/
│   ├── test_wing.py
│   ├── test_fuselage.py
│   └── test_airfoil.py
├── aero/
│   ├── test_neuralfoil_backend.py
│   ├── test_aerosandbox_backend.py
│   ├── test_native_vlm.py
│   └── test_llt.py
├── weights/
│   ├── test_weight_model.py
├── propulsion/
│   ├── test_propulsion.py
├── mission/
│   ├── test_mission.py
├── flightdyn/
│   ├── test_trim.py
│   ├── test_linearize.py
└── mdo/
    ├── test_mdo_problem.py
    └── test_optimizers.py
```

---

## 4. Examples and notebooks

```text
examples/
├── basic_tutorial.py            # Build an RC UAV, run mission, optimize
├── neuralfoil_usage.py
├── aero_backend_compare.py      # Compare AeroSandbox vs native VLM
├── mdo_example.py               # CLI-style MDO run
└── 3dprint_export_demo.py       # conceptual → CAD → STL (future)

notebooks/
├── 01_aircraft_assembly.ipynb
├── 02_aero_sweep.ipynb
└── 03_mdo_walkthrough.ipynb
```

---

## 5. CI / dev tools layout

```text
.github/
  └── workflows/
      ├── test.yml         # Run tests on push
      └── lint.yml         # Run ruff, mypy, etc.

.ruff.toml                 # linting rules
mypy.ini                   # static typing config
pytest.ini                 # pytest configuration
tox.ini                    # optional: multi‑Python testing
```

---

## 6. Docs layout (Sphinx)

```text
docs/
├── conf.py
├── index.rst
├── architecture.rst           # this architecture document
├── tutorial.rst
├── api/
│   ├── core.rst
│   ├── geometry.rst
│   ├── aero.rst
│   └── ...
└── _build/ (ignored in git)
```

This combined architecture + layout document should be a good starting point for your repo.  
Save it as `docs/ARCHITECTURE.md` or similar, and evolve it as you implement.