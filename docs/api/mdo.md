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

# ... define aircraft with propulsion ...

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
        DesignVar("wings[0].xsecs[1].chord",     lower=0.08, upper=0.22, scale=0.15),
        DesignVar("wings[0].xsecs[1].xyz_le[1]", lower=0.8,  upper=1.8,  scale=1.2),
    ],
    constraints=[
        Constraint("structures.wings[0].bending_margin", lower=0.0),
        Constraint("stability.static_margin", lower=0.05, upper=0.20),
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
| `"scipy_de"` | SciPy | Differential Evolution — global, robust, handles integers |
| `"scipy_minimize"` | SciPy | Local gradient-free or gradient-based |
| `"scipy_shgo"` | SciPy | Simplicial Homology Global Optimisation |
| `"pygmo_de"` | pygmo | Differential Evolution (requires `pip install pygmo`) |
| `"pygmo_sade"` | pygmo | Self-Adaptive DE |
| `"pygmo_nsga2"` | pygmo | Multi-objective NSGA-II |

Recommended default: `"scipy_de"`.

---

## Checkpoint and resume

```python
result = problem.optimize(
    method="scipy_de",
    checkpoint_path="runs/opt_run1",
    checkpoint_interval=50,
)
# Resume: pass the same checkpoint_path — automatically resumes
result = problem.optimize(
    method="scipy_de",
    checkpoint_path="runs/opt_run1",
)
```

---

## Sensitivity analysis

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
