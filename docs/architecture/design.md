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
