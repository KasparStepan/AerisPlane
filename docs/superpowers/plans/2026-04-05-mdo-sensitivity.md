# MDO Sensitivity Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `MDOProblem.sensitivity(x)` → `SensitivityResult` that computes forward finite-difference gradients of the objective and constraints with respect to each design variable, and ranks them by normalized sensitivity.

**Architecture:** New file `src/aerisplane/mdo/sensitivity.py` holds the `SensitivityResult` dataclass and `compute_sensitivity()` function. `MDOProblem` gains a `sensitivity()` method that delegates to it. Finite differences use the existing `evaluate()` + `objective_function()` + `constraint_functions()` cache — each perturbed evaluation is cached so it does not re-run if the optimizer happens to visit the same point later.

**Tech Stack:** numpy, existing `MDOProblem.evaluate()` infrastructure.

---

### Task 1: `SensitivityResult` dataclass

**Files:**
- Create: `src/aerisplane/mdo/sensitivity.py`
- Create: `tests/test_mdo/test_sensitivity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mdo/test_sensitivity.py
import numpy as np
import pytest
from aerisplane.mdo.sensitivity import SensitivityResult


def _make_result(n_dvars=3, n_constraints=2):
    paths = [f"wings[0].xsecs[{i}].chord" for i in range(n_dvars)]
    grad_obj = np.array([0.5, -0.3, 0.8])[:n_dvars]
    grad_con = {
        "stability.static_margin": np.array([0.1, 0.2, -0.05])[:n_dvars],
        "structures.wings[0].bending_margin": np.array([-0.4, 0.1, 0.3])[:n_dvars],
    }
    norm_obj = np.abs(grad_obj)
    norm_con = {k: np.abs(v) for k, v in grad_con.items()}
    return SensitivityResult(
        dvar_paths=paths,
        grad_objective=grad_obj,
        grad_constraints=grad_con,
        normalized_objective=norm_obj,
        normalized_constraints=norm_con,
        x=np.ones(n_dvars) * 0.25,
        objective_value=0.850,
        step_size=1e-4,
    )


def test_result_fields():
    r = _make_result()
    assert len(r.dvar_paths) == 3
    assert r.grad_objective.shape == (3,)
    assert "stability.static_margin" in r.grad_constraints


def test_report_is_string():
    r = _make_result()
    report = r.report()
    assert isinstance(report, str)
    assert "Objective" in report
    assert "wings[0].xsecs[0].chord" in report


def test_report_sorted_by_normalized_sensitivity():
    r = _make_result()
    report = r.report()
    lines = [l for l in report.splitlines() if "wings[0].xsecs" in l]
    # Most sensitive variable (index 2, norm=0.8) should appear first
    assert lines[0].startswith("  1.")


def test_plot_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = _make_result()
    fig = r.plot()
    assert fig is not None
    plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_mdo/test_sensitivity.py -v
```

Expected: FAIL with ImportError

- [ ] **Step 3: Implement `SensitivityResult` and `compute_sensitivity()` in `sensitivity.py`**

```python
# src/aerisplane/mdo/sensitivity.py
"""Sensitivity analysis for MDOProblem.

Computes forward finite-difference gradients of the objective and all
constraints with respect to each design variable and ranks them by
normalized (elastic) sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class SensitivityResult:
    """Sensitivity analysis result.

    Parameters
    ----------
    dvar_paths : list of str
        Design variable paths in order.
    grad_objective : ndarray, shape (n,)
        dObjective/dx_i for each design variable.
    grad_constraints : dict[str, ndarray]
        dConstraint/dx_i per constraint path.
    normalized_objective : ndarray, shape (n,)
        Normalized (elastic) sensitivity = |dObj/dx_i| * |x_i / obj|.
    normalized_constraints : dict[str, ndarray]
        Same normalization applied to each constraint gradient.
    x : ndarray
        Design vector at which sensitivity was computed.
    objective_value : float
        Objective value at x.
    step_size : float
        Finite-difference step h used.
    """

    dvar_paths: list[str]
    grad_objective: np.ndarray
    grad_constraints: dict[str, np.ndarray]
    normalized_objective: np.ndarray
    normalized_constraints: dict[str, np.ndarray]
    x: np.ndarray
    objective_value: float
    step_size: float

    def report(self) -> str:
        """Ranked sensitivity table as a human-readable string."""
        n = len(self.dvar_paths)
        order = np.argsort(-self.normalized_objective)  # descending

        lines = ["Sensitivity Analysis — Objective Gradients (normalized, ranked)"]
        lines.append(f"  Evaluated at x={np.round(self.x, 4).tolist()}")
        lines.append(f"  Objective value: {self.objective_value:.5g}")
        lines.append(f"  Step size: {self.step_size:.2e}")
        lines.append("")
        lines.append("  Objective:")
        for rank, idx in enumerate(order, start=1):
            lines.append(
                f"  {rank:2d}. {self.dvar_paths[idx]:<55} "
                f"dObj/dx={self.grad_objective[idx]:+.4g}  "
                f"|norm|={self.normalized_objective[idx]:.4g}"
            )

        for cname, grad in self.grad_constraints.items():
            norm = self.normalized_constraints.get(cname, np.zeros(n))
            con_order = np.argsort(-norm)
            lines.append("")
            lines.append(f"  Constraint: {cname}")
            for rank, idx in enumerate(con_order, start=1):
                lines.append(
                    f"  {rank:2d}. {self.dvar_paths[idx]:<55} "
                    f"dCon/dx={grad[idx]:+.4g}  "
                    f"|norm|={norm[idx]:.4g}"
                )

        return "\n".join(lines)

    def plot(self):
        """Horizontal bar chart of normalized objective sensitivities."""
        import matplotlib.pyplot as plt

        n = len(self.dvar_paths)
        order = np.argsort(self.normalized_objective)  # ascending for horizontal bar
        labels = [self.dvar_paths[i] for i in order]
        values = self.normalized_objective[order]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * n + 1)))
        ax.barh(labels, values)
        ax.set_xlabel("Normalized sensitivity |dObj/dx| * |x/f|")
        ax.set_title("Objective Sensitivity")
        fig.tight_layout()
        return fig


def compute_sensitivity(
    problem,
    x: np.ndarray,
    step: float = 1e-4,
) -> SensitivityResult:
    """Compute forward finite-difference sensitivity at x.

    Parameters
    ----------
    problem : MDOProblem
        The optimization problem.
    x : ndarray
        Design vector (scaled, in optimizer space).
    step : float
        Finite-difference step size (in scaled space).

    Returns
    -------
    SensitivityResult
    """
    n = len(x)
    obj0 = problem.objective_function(x)
    con0 = problem.constraint_functions(x)

    grad_obj = np.zeros(n)
    grad_con = np.zeros((len(con0), n)) if len(con0) > 0 else np.zeros((0, n))

    for i in range(n):
        xp = x.copy()
        xp[i] += step
        # Clamp to bounds
        lo, hi = problem.get_bounds()
        xp[i] = min(xp[i], hi[i])
        obj_p = problem.objective_function(xp)
        grad_obj[i] = (obj_p - obj0) / step

        if len(con0) > 0:
            con_p = problem.constraint_functions(xp)
            grad_con[:, i] = (con_p - con0) / step

    # Normalized sensitivities: |df/dx_i| * |x_i / f|
    abs_obj = abs(obj0) if abs(obj0) > 1e-30 else 1.0
    norm_obj = np.abs(grad_obj) * (np.abs(x) / abs_obj)

    constraint_paths = [c.path for c in problem._constraints]
    grad_con_dict: dict[str, np.ndarray] = {}
    norm_con_dict: dict[str, np.ndarray] = {}
    for j, cpath in enumerate(constraint_paths):
        g = grad_con[j] if j < len(grad_con) else np.zeros(n)
        abs_con = abs(con0[j]) if j < len(con0) and abs(con0[j]) > 1e-30 else 1.0
        grad_con_dict[cpath] = g
        norm_con_dict[cpath] = np.abs(g) * (np.abs(x) / abs_con)

    return SensitivityResult(
        dvar_paths=[dv.path for dv in problem._dvars],
        grad_objective=grad_obj,
        grad_constraints=grad_con_dict,
        normalized_objective=norm_obj,
        normalized_constraints=norm_con_dict,
        x=x.copy(),
        objective_value=float(obj0),
        step_size=step,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_mdo/test_sensitivity.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aerisplane/mdo/sensitivity.py tests/test_mdo/test_sensitivity.py
git commit -m "feat(mdo): add SensitivityResult dataclass and compute_sensitivity()"
```

---

### Task 2: `MDOProblem.sensitivity()` method

**Files:**
- Modify: `src/aerisplane/mdo/problem.py`
- Modify: `src/aerisplane/mdo/__init__.py`
- Modify: `tests/test_mdo/test_sensitivity.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mdo/test_sensitivity.py`:

```python
import numpy as np
import aerisplane as ap
from aerisplane.mdo import MDOProblem, DesignVar, Constraint, Objective


@pytest.fixture
def simple_mdo_problem():
    """Minimal MDOProblem with a mock evaluate that uses real Aircraft."""
    from aerisplane.catalog.materials import carbon_fiber_tube, petg
    spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.015, wall_thickness=0.001),
    )
    skin = ap.Skin(material=petg, thickness=0.0008)
    wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.26,
                        airfoil=ap.Airfoil("naca2412"), spar=spar, skin=skin),
            ap.WingXSec(xyz_le=[0.03, 0.75, 0.05], chord=0.15,
                        airfoil=ap.Airfoil("naca2412")),
        ],
        symmetric=True,
    )
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.85, 0.0, 0.0], chord=0.12,
                        airfoil=ap.Airfoil("naca0012")),
            ap.WingXSec(xyz_le=[0.87, 0.25, 0.01], chord=0.08,
                        airfoil=ap.Airfoil("naca0012")),
        ],
        symmetric=True,
    )
    aircraft = ap.Aircraft(name="test", wings=[wing, htail])
    condition = ap.FlightCondition(velocity=16.0, altitude=80.0, alpha=4.0)
    return MDOProblem(
        aircraft=aircraft,
        condition=condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.20, upper=0.32),
            DesignVar("wings[0].xsecs[1].chord", lower=0.10, upper=0.20),
        ],
        constraints=[Constraint("stability.static_margin", lower=0.04)],
        objective=Objective("weights.total_mass", maximize=False),
        alpha=4.0,
    )


def test_sensitivity_returns_result(simple_mdo_problem):
    from aerisplane.mdo.sensitivity import SensitivityResult
    lo, hi = simple_mdo_problem.get_bounds()
    x = (lo + hi) / 2.0
    result = simple_mdo_problem.sensitivity(x)
    assert isinstance(result, SensitivityResult)


def test_sensitivity_grad_shape(simple_mdo_problem):
    lo, hi = simple_mdo_problem.get_bounds()
    x = (lo + hi) / 2.0
    result = simple_mdo_problem.sensitivity(x)
    assert result.grad_objective.shape == (2,)
    assert "stability.static_margin" in result.grad_constraints


def test_sensitivity_report_and_plot(simple_mdo_problem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = simple_mdo_problem.get_bounds()
    x = (lo + hi) / 2.0
    result = simple_mdo_problem.sensitivity(x)
    report = result.report()
    assert isinstance(report, str)
    fig = result.plot()
    assert fig is not None
    plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_mdo/test_sensitivity.py -v -k mdo_problem
```

Expected: FAIL with AttributeError (no `sensitivity` method on MDOProblem)

- [ ] **Step 3: Add `sensitivity()` method to `MDOProblem`**

In `src/aerisplane/mdo/problem.py`, add as the last method of the `MDOProblem` class (before the end of the class body):

```python
def sensitivity(
    self,
    x: "np.ndarray",
    step: float = 1e-4,
) -> "SensitivityResult":
    """Compute finite-difference sensitivity at a given design point.

    Parameters
    ----------
    x : ndarray
        Design vector in scaled optimizer space (as returned by get_bounds).
    step : float
        Forward finite-difference step in scaled space. Default 1e-4.

    Returns
    -------
    SensitivityResult
        Gradients and normalized sensitivities for objective and constraints,
        ranked by influence on the objective.
    """
    from aerisplane.mdo.sensitivity import compute_sensitivity
    return compute_sensitivity(self, x, step=step)
```

- [ ] **Step 4: Export `SensitivityResult` from `mdo/__init__.py`**

In `src/aerisplane/mdo/__init__.py`, add:

```python
from aerisplane.mdo.sensitivity import SensitivityResult
```

And add `"SensitivityResult"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_mdo/test_sensitivity.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -x -q --ignore=tests/test_mdo/test_integration.py
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/aerisplane/mdo/problem.py src/aerisplane/mdo/__init__.py \
        src/aerisplane/mdo/sensitivity.py tests/test_mdo/test_sensitivity.py
git commit -m "feat(mdo): add MDOProblem.sensitivity() method"
```
