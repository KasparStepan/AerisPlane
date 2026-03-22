# UAVDesign: Conceptual RC/UAV Design and Optimization Tool

## Overview

UAVDesign is a Python-based conceptual design and multidisciplinary optimization (MDO) toolkit for fixed-wing and VTOL-capable RC-scale UAVs. It focuses on early-stage configuration and sizing rather than detailed CAD or CFD/FEA, with an emphasis on clean architecture, extensibility, and integration with external analysis tools.

The core idea is a unified, parametric aircraft model (`AircraftConfig`) that can be analyzed by multiple backends (AeroSandbox, native VLM/LLT, future CFD/FEA, CAD exporters) via thin adapter layers. This keeps the main codebase pure-Python and solver-agnostic while allowing you to plug in higher-fidelity tools as the project grows.

## Key Features

- **Unified aircraft model**
  - Object-oriented representation of wings, tails, fuselages, propulsion, and payloads.
  - Cross-section-based geometry (WingXSec, FuselageXSec) suitable for both aero and future CAD/3D-print export.
  - Easy mapping between design variables and geometric/physical properties.

- **Modular discipline modules**
  - `geometry/`: parametric aircraft geometry, panel meshing utilities.
  - `aero/`: 2D and 3D aerodynamics with multiple backends (NeuralFoil, AeroSandbox, native LLT/VLM; OpenVSP/SU2 optional later).
  - `weights/`: mass, CG, and inertia models using analytical/empirical formulas.
  - `propulsion/`: electric propulsion (motor–prop–battery–ESC) performance and energy models.
  - `mission/`: segment-based mission analysis (climb, cruise, loiter, return) with range/endurance and constraint checks.
  - `flightdyn/`: trim, linearization, and basic flight dynamics/stability analysis.
  - `mdo/`: optimization orchestration (SciPy, pyOptSparse, pygmo, and optionally OpenMDAO drivers).

- **Backend-agnostic aerodynamics**
  - Core API `compute_aero(aircraft, condition, backend=...)` that returns aerodynamic loads independent of the solver.
  - Adapters to:
    - AeroSandbox (for fast 3D aero using its Airplane/VLM tools).
    - NeuralFoil (for 2D section polars in LLT/VLM).
    - Native future VLM/LLT implementations or external solvers.

- **Conceptual-focus with path to manufacturing**
  - Designed for conceptual sizing and MDO: configuration choice, tail volume sizing, endurance/range optimization, and stability margins.
  - Geometry representation is rich enough to support later CAD/3D-print backends (e.g., OpenCascade, OpenSCAD, OpenVSP export) without changing the core model.

## Intended Workflow

1. **Define configuration**
   - Use the high-level Python API to construct an `AircraftConfig` with wings, tails, fuselage, propulsion, and payload.
   - Optionally load configurations from YAML/JSON via the `io/` module.

2. **Run analyses**
   - Use `aero/`, `weights/`, `propulsion/`, `mission/`, and `flightdyn/` to evaluate performance, stability, and constraints at one or more design points.

3. **Optimize**
   - Define design variables and objectives (e.g., maximize endurance with stall speed and battery constraints).
   - Run optimization with the `mdo/` module using your preferred backend (SciPy, pygmo, or OpenMDAO-based workflows).

4. **Export (future)**
   - Optionally export configurations to CAD or STL via the `export/` module for detailed design and 3D printing.

## Project Philosophy

- **Separation of concerns:** geometry, physics, and optimization are clearly separated, making it easy to swap solvers or extend capabilities.
- **Solver-agnostic core:** the aircraft model and mission logic do not depend on any specific aero or CAD package.
- **RC/UAV realism:** models and defaults are tuned for RC-scale aircraft and UAV use cases, not just transport aircraft.
- **Research-friendly:** architecture and APIs are designed so you can plug in new physics models (e.g., advanced LLT/VLM, free-wake, CFD, structural sizing) as the research evolves.
