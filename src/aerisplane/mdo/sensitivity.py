"""Sensitivity analysis for MDOProblem.

Computes forward finite-difference gradients of the objective and all
constraints with respect to each design variable and ranks them by
normalized (elastic) sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensitivityResult:
    """Sensitivity analysis result.

    Parameters
    ----------
    dvar_paths : list of str — design variable paths
    grad_objective : ndarray shape (n,) — dObjective/dx_i
    grad_constraints : dict[str, ndarray] — dConstraint/dx_i per constraint path
    normalized_objective : ndarray shape (n,) — |dObj/dx_i| * |x_i / obj|
    normalized_constraints : dict[str, ndarray] — same for each constraint
    x : ndarray — design vector at evaluation
    objective_value : float — objective value at x
    step_size : float — finite-difference step used
    """

    dvar_paths: list
    grad_objective: np.ndarray
    grad_constraints: dict
    normalized_objective: np.ndarray
    normalized_constraints: dict
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
        order = np.argsort(self.normalized_objective)  # ascending for barh
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
    x : ndarray — design vector in scaled optimizer space
    step : float — finite-difference step in scaled space

    Returns
    -------
    SensitivityResult
    """
    n = len(x)
    obj0 = problem.objective_function(x)
    con0 = problem.constraint_functions(x)
    n_con = len(con0)

    grad_obj = np.zeros(n)
    grad_con = np.zeros((n_con, n)) if n_con > 0 else np.zeros((0, n))

    lo, hi = problem.get_bounds()

    for i in range(n):
        xp = x.copy()
        xp[i] += step
        xp[i] = min(xp[i], hi[i])  # clamp to bounds
        obj_p = problem.objective_function(xp)
        grad_obj[i] = (obj_p - obj0) / step

        if n_con > 0:
            con_p = problem.constraint_functions(xp)
            grad_con[:, i] = (con_p - con0) / step

    # Normalized: |df/dx_i| * |x_i / f|
    abs_obj = abs(obj0) if abs(obj0) > 1e-30 else 1.0
    norm_obj = np.abs(grad_obj) * (np.abs(x) / abs_obj)

    # Map each constraint to its row slice in grad_con.
    # constraint_functions() appends one row per bound (lower, upper, equals),
    # so a constraint with both lower and upper contributes 2 rows.
    constraint_paths = [c.path for c in problem._constraints]
    grad_con_dict: dict = {}
    norm_con_dict: dict = {}
    row = 0
    for j, c in enumerate(problem._constraints):
        n_rows = 0
        if c.lower is not None:
            n_rows += 1
        if c.upper is not None:
            n_rows += 1
        if c.equals is not None:
            n_rows += 1

        if n_rows > 0 and row + n_rows <= len(grad_con):
            g_rows = grad_con[row : row + n_rows]
            # Use the mean across bound rows as the representative gradient
            g = g_rows.mean(axis=0)
            # Pick the violation entry with the largest absolute value for normalisation
            abs_con = max(abs(float(con0[row + k])) for k in range(n_rows))
            if abs_con < 1e-30:
                abs_con = 1.0
        else:
            g = np.zeros(n)
            abs_con = 1.0

        grad_con_dict[c.path] = g
        norm_con_dict[c.path] = np.abs(g) * (np.abs(x) / abs_con)
        row += n_rows

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
