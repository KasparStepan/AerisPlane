"""Optimisation result and progress snapshot dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class OptimisationSnapshot:
    """Progress snapshot passed to the user callback after each evaluation.

    Parameters
    ----------
    n_evals : int
        Total number of discipline chain evaluations so far.
    objective : float
        Current best objective value (raw, before sign flip).
    objective_initial : float
        Objective at the initial design vector.
    improvement_pct : float
        (objective − objective_initial) / |objective_initial| × 100.
    improvement_last_100 : float
        Relative improvement over the last 100 evaluations.
    x_best : ndarray
        Best design vector (scaled space).
    constraints_satisfied : bool
        True if all constraints are satisfied at x_best.
    constraint_values : dict
        Mapping constraint path → current value at x_best.
    elapsed_s : float
        Wall time since optimize() was called [s].
    history : list of float
        Objective value at every evaluation up to this point.
    """
    n_evals: int
    objective: float
    objective_initial: float
    improvement_pct: float
    improvement_last_100: float
    x_best: np.ndarray
    constraints_satisfied: bool
    constraint_values: dict
    elapsed_s: float
    history: list


@dataclass
class OptimizationResult:
    """Result of a completed optimisation run.

    Parameters
    ----------
    x_initial : ndarray
        Starting design vector (scaled).
    x_optimal : ndarray
        Optimal design vector (scaled).
    objective_initial : float
        Objective value at x_initial.
    objective_optimal : float
        Objective value at x_optimal.
    constraints_satisfied : bool
        True if all constraints pass at x_optimal.
    n_evaluations : int
        Total number of discipline chain evaluations.
    convergence_history : list of float
        Best objective value per evaluation.
    variables : dict
        Mapping path → (initial_value, optimal_value) in physical units.
    aero, weights, structures, stability, control, mission
        Full discipline results at x_optimal (None if discipline not run).
    aircraft : Aircraft
        Optimised aircraft at x_optimal.
    pareto_front : list or None
        For multi-objective runs: list of (x, objectives) tuples.
        None for single-objective.
    """
    x_initial: np.ndarray
    x_optimal: np.ndarray
    objective_initial: float
    objective_optimal: float
    constraints_satisfied: bool
    n_evaluations: int
    convergence_history: list
    variables: dict
    aero: Any
    weights: Any
    structures: Any
    stability: Any
    control: Any
    mission: Any
    aircraft: Any
    pareto_front: Optional[list] = None

    def report(self) -> str:
        """Formatted plain-text optimisation summary.

        Sections:
        - Objective: initial, optimal, % change, eval count, constraint status
        - Design Variables: table of path / initial / optimal values
        - Pareto Front: solution count (only if pareto_front is not None)
        """
        lines = ["AerisPlane Optimisation Result", "=" * 60]

        lines += ["", "Objective", "-" * 40]
        improvement = (
            (self.objective_optimal - self.objective_initial)
            / (abs(self.objective_initial) + 1e-30) * 100
        )
        lines.append(f"  Initial  : {self.objective_initial:.6g}")
        lines.append(f"  Optimal  : {self.objective_optimal:.6g}")
        lines.append(f"  Change   : {improvement:+.1f}%")
        lines.append(f"  Evals    : {self.n_evaluations}")
        lines.append(f"  Constraints satisfied: {'YES' if self.constraints_satisfied else 'NO'}")

        lines += ["", "Design Variables", "-" * 40]
        lines.append(f"  {'Path':<45} {'Initial':>10} {'Optimal':>10}")
        lines.append(f"  {'-'*45} {'-'*10} {'-'*10}")
        for path, (init, opt) in self.variables.items():
            lines.append(f"  {path:<45} {init:>10.5g} {opt:>10.5g}")

        if self.pareto_front is not None:
            lines += ["", f"Pareto Front: {len(self.pareto_front)} solutions"]

        return "\n".join(lines)

    def plot(self):
        """Return a 1×2 matplotlib figure.

        Left panel: convergence history (objective vs evaluation number),
        with a rolling average overlay when history has ≥10 points.
        Right panel: horizontal bar chart comparing initial and optimal
        design variable values.
        """
        import matplotlib.pyplot as plt

        fig, (ax_conv, ax_vars) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: convergence history
        evals = np.arange(1, len(self.convergence_history) + 1)
        ax_conv.plot(evals, self.convergence_history, linewidth=1.2, alpha=0.6,
                     label="per eval")
        if len(self.convergence_history) >= 10:
            window = max(1, len(self.convergence_history) // 20)
            smooth = np.convolve(self.convergence_history,
                                 np.ones(window) / window, mode="valid")
            ax_conv.plot(np.arange(window, len(self.convergence_history) + 1),
                         smooth, linewidth=2, label=f"rolling {window}")
        ax_conv.set_xlabel("Evaluation")
        ax_conv.set_ylabel("Objective")
        ax_conv.set_title("Convergence History")
        ax_conv.legend(fontsize=8)
        ax_conv.grid(True, alpha=0.3)

        # Right: design variable comparison
        if self.variables:
            names = list(self.variables.keys())
            inits = np.array([v[0] for v in self.variables.values()])
            opts  = np.array([v[1] for v in self.variables.values()])
            y = np.arange(len(names))
            ax_vars.barh(y + 0.2, inits, height=0.35, label="Initial", alpha=0.7)
            ax_vars.barh(y - 0.2, opts,  height=0.35, label="Optimal", alpha=0.7)
            ax_vars.set_yticks(y)
            ax_vars.set_yticklabels([n.split(".")[-1] for n in names], fontsize=8)
            ax_vars.set_xlabel("Value")
            ax_vars.set_title("Design Variables: Initial vs Optimal")
            ax_vars.legend(fontsize=8)
            ax_vars.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_pareto(self):
        """Plot the Pareto front for multi-objective results.

        Returns a scatter figure of objective 1 vs objective 2.
        Returns None and emits a UserWarning if pareto_front is None
        (single-objective run).
        """
        if self.pareto_front is None:
            import warnings
            warnings.warn("No Pareto front available (single-objective run).")
            return None
        if not self.pareto_front:
            import warnings
            warnings.warn("Pareto front is empty — nothing to plot.")
            return None

        import matplotlib.pyplot as plt

        pts = np.array([obj for _, obj in self.pareto_front])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(pts[:, 0], pts[:, 1], s=40, zorder=5)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title(f"Pareto Front ({len(self.pareto_front)} solutions)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
