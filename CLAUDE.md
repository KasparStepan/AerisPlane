# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AerisPlane (`aerisplane`) is a Python MDO (multidisciplinary design optimization) toolkit for fixed-wing RC/UAV conceptual design (1-20 kg class). No code exists yet — the project is in the specification phase. The authoritative design spec is `docs/superpowers/specs/2026-03-22-aerisplane-framework-design.md`.

## Architecture

**Layered, discipline-separated package** at `src/aerisplane/`:

- `core/` — Pure data model (dataclasses). Depends on **numpy only**. No AeroSandbox, no matplotlib, no solver imports. Classes: `Aircraft`, `Wing`, `WingXSec`, `Fuselage`, `FuselageXSec`, `Airfoil`, `ControlSurface`, `Servo`, `Spar`, `Skin`, `Material`, `TubeSection`, `Motor`, `Propeller`, `Battery`, `ESC`, `PropulsionSystem`, `Payload`, `FlightCondition`.
- `aero/` — Aerodynamics. Backend adapters translate `core/` objects to solver-specific types. AeroSandbox translation (`aircraft_to_asb()`) lives **only** in `aero/aerosandbox_backend.py`.
- `weights/` — Component-based mass buildup from geometry + hardware catalog.
- `structures/` — Beam solver (fast, for optimizer) and OpenAeroStruct adapter (detailed).
- `stability/` — Numerical stability derivatives via central finite differences (5 aero calls).
- `control/` — Control authority analysis (roll rate, pitch, rudder authority, servo loads).
- `mission/` — Point-performance energy budget per mission segment.
- `mdo/` — Optimization orchestration. String-path design variables, evaluation caching, external optimizer wrappers (SciPy, pygmo).
- `catalog/` — Hardware database as Python module-level instances (motors, batteries, servos, materials).
- `utils/` — ISA atmosphere, unit conversions, matplotlib plotting style.

**Dependency flow:** `core/` ← `catalog/` ← discipline modules ← `mdo/` (top). Nothing depends on `mdo/`.

**Backend pattern:** Solvers are lazy-imported behind string-based backend selection (`backend="aerosandbox"` or `backend="openaerostruct"`). Each adapter: translate in → call solver → translate out to result dataclass.

**Every discipline module follows the same pattern:** `__init__.py` (entry `analyze()` function), `result.py` (result dataclass with `.plot()` and `.report()`), implementation files.

## Key Design Rules

- `core/` must never import AeroSandbox, matplotlib, or any solver library
- `Wing` is used for both wings and tail surfaces — no separate `Tail` class
- `Material` holds intrinsic properties only; cross-section geometry is in `TubeSection`
- Control surface deflections are passed via `FlightCondition.deflections` dict, not by mutating `Aircraft`
- MDO `evaluate()` caches results keyed on the design vector to avoid redundant discipline chain runs
- `MDOProblem.validate()` resolves all string paths at construction time, before optimizer starts

## Build & Development Commands

Not yet configured. When `pyproject.toml` is created, the expected setup is:

```bash
pip install -e ".[dev]"         # editable install with dev dependencies
pip install -e ".[oas]"         # with OpenAeroStruct support
pip install -e ".[all]"         # everything
pytest tests/                   # run all tests
pytest tests/test_aero/         # run single discipline tests
ruff check src/                 # lint
```

## Repository Layout

- `src/aerisplane/` — Python package source
- `docs/` — Design specs and documentation
- `planes/CoreFly/` — Aircraft project workspace (definitions, configs, optimization scripts, results). Not part of the Python package.
- `tutorials/` — Jupyter notebooks and example scripts (planned)
- `tests/` — pytest tests mirroring `src/` structure (planned)

## User Context

The author is a mechanical engineer (PhD) — not a professional developer. Code should be readable, explicit, and self-documenting with engineering-domain naming. Avoid clever abstractions; prefer straightforward code that an engineer can follow.
