"""Alpha sweep — CL, CD, Cm vs angle of attack with per-component breakdown.

Each component coefficient is normalised by the *aircraft* reference area
(and c_ref for Cm), so curves from different components sit on the same
scale and approximately sum to the total.

Method vs per-component availability
--------------------------------------
"aero_buildup"
    Each wing and each fuselage resolved independently → full breakdown.
"vlm"
    Per-wing breakdown by summing panel forces per wing_record entry.
    No fuselage (VLM is a lifting-surface-only solver).
"lifting_line", "nonlinear_lifting_line"
    Total aircraft only.

Example
-------
>>> import numpy as np
>>> from aerisplane.aero.alpha_sweep import alpha_sweep
>>> alphas = np.linspace(-5, 15, 21)
>>> result = alpha_sweep(aircraft, condition, alphas, method="aero_buildup",
...                      xyz_ref=weight_result.cg.tolist())
>>> result.plot()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentCurve:
    """CL, CD, Cm vs alpha for a single aircraft component.

    All coefficients are normalised by the *aircraft* S_ref (and S_ref*c_ref
    for Cm) so they share the same scale as the total-aircraft curves.
    """
    name: str
    alphas: np.ndarray   # [deg]
    CL: np.ndarray
    CD: np.ndarray
    Cm: np.ndarray


@dataclass
class AlphaSweepResult:
    """Result of a CL/CD/Cm angle-of-attack sweep.

    Attributes
    ----------
    alphas : np.ndarray
        Angle-of-attack values swept [deg].
    CL, CD, Cm : np.ndarray
        Total aircraft coefficients.
    components : dict[str, ComponentCurve]
        Per-component polars keyed by component name.
        Empty dict when the solver does not support per-component output.
    method : str
        Solver method used.
    s_ref, c_ref, b_ref : float
        Reference geometry (echoed from the aircraft).
    """

    alphas: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    Cm: np.ndarray
    method: str
    components: dict = field(default_factory=dict)
    s_ref: float = 0.0
    c_ref: float = 0.0
    b_ref: float = 0.0

    # ── Public plot methods ───────────────────────────────────────────────

    def plot(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """Four-panel aerodynamic polar: CL-α, Cm-α, drag polar, L/D.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax_cl, ax_cm = axes[0, 0], axes[0, 1]
        ax_polar, ax_ld = axes[1, 0], axes[1, 1]

        self._draw_cl_alpha(ax_cl)
        self._draw_cm_alpha(ax_cm)

        ax_polar.plot(self.CD, self.CL, "o-", color="k", lw=2, ms=3, label="Total")
        ax_polar.set_xlabel("CD")
        ax_polar.set_ylabel("CL")
        ax_polar.set_title("Drag Polar", fontsize=11, fontweight="bold")
        ax_polar.grid(True, alpha=0.3)
        ax_polar.legend(fontsize=9)

        ld = np.where(np.abs(self.CD) > 1e-9, self.CL / self.CD, np.nan)
        ax_ld.plot(self.alphas, ld, "o-", color="k", lw=2, ms=3, label="Total")
        ax_ld.axhline(0, color="gray", lw=0.7, ls="--")
        ax_ld.set_xlabel("α [deg]")
        ax_ld.set_ylabel("L/D")
        ax_ld.set_title("Lift-to-Drag Ratio", fontsize=11, fontweight="bold")
        ax_ld.grid(True, alpha=0.3)
        ax_ld.legend(fontsize=9)

        fig.suptitle(
            f"Aerodynamic Polar — method: {self.method}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(pad=1.2)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_lift_curves(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """CL vs α: per-component curves plus bold total line.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)
        fig, ax = plt.subplots(figsize=(8, 5))
        self._draw_cl_alpha(ax)
        fig.tight_layout(pad=1.0)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_moment_curves(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """Cm vs α: per-component curves plus bold total line.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)
        fig, ax = plt.subplots(figsize=(8, 5))
        self._draw_cm_alpha(ax)
        fig.tight_layout(pad=1.0)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    # ── Internal drawing helpers ──────────────────────────────────────────

    def _draw_cl_alpha(self, ax) -> None:
        from aerisplane.utils.plotting import PALETTE
        for i, (name, comp) in enumerate(self.components.items()):
            ax.plot(comp.alphas, comp.CL, "o-",
                    color=PALETTE[i % len(PALETTE)], lw=1.5, ms=3, label=name)
        ax.plot(self.alphas, self.CL, "o-",
                color="k", lw=2.5, ms=4, label="Total")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel("α [deg]")
        ax.set_ylabel("CL")
        ax.set_title("Lift Curves", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_cm_alpha(self, ax) -> None:
        from aerisplane.utils.plotting import PALETTE
        for i, (name, comp) in enumerate(self.components.items()):
            ax.plot(comp.alphas, comp.Cm, "o-",
                    color=PALETTE[i % len(PALETTE)], lw=1.5, ms=3, label=name)
        ax.plot(self.alphas, self.Cm, "o-",
                color="k", lw=2.5, ms=4, label="Total")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel("α [deg]")
        ax.set_ylabel("Cm")
        ax.set_title("Pitching Moment Curves", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def alpha_sweep(
    aircraft: Aircraft,
    condition: FlightCondition,
    alpha_range,
    method: str = "aero_buildup",
    xyz_ref: Optional[list] = None,
    spanwise_resolution: int = 8,
    chordwise_resolution: int = 4,
    model_size: str = "medium",
    verbose: bool = False,
) -> AlphaSweepResult:
    """Run a CL/CD/Cm angle-of-attack sweep with optional per-component breakdown.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry.
    condition : FlightCondition
        Baseline condition; ``alpha`` is overridden at each sweep point.
        All other fields (velocity, altitude, beta, …) stay constant.
    alpha_range : array-like
        Angles of attack [deg].
    method : str
        Aero solver.  ``"aero_buildup"`` (default) gives wings + fuselages
        separately; ``"vlm"`` gives per-wing only; lifting-line methods give
        total aircraft only.
    xyz_ref : list[float] or None
        Moment reference [x, y, z] [m].  Defaults to ``aircraft.xyz_ref``.
        Pass CG position for stability-referenced moments.
    spanwise_resolution : int
        Spanwise panels/stations per wing section (VLM / LL).
    chordwise_resolution : int
        Chordwise panels per section (VLM only).
    model_size : str
        NeuralFoil model size for aero_buildup / LL.
    verbose : bool
        Print progress.

    Returns
    -------
    AlphaSweepResult
    """
    alphas = np.asarray(alpha_range, dtype=float)
    n = len(alphas)
    ref = list(xyz_ref) if xyz_ref is not None else list(aircraft.xyz_ref)

    S = aircraft.reference_area()
    c = aircraft.reference_chord()
    b = aircraft.reference_span()

    total_CL = np.zeros(n)
    total_CD = np.zeros(n)
    total_Cm = np.zeros(n)
    comp_data: dict[str, dict] = {}   # name → {"CL": [], "CD": [], "Cm": []}

    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"  alpha = {alpha:+.1f}°  ({i + 1}/{n})")
        cond = condition.copy()
        cond.alpha = float(alpha)

        if method == "aero_buildup":
            _step_buildup(aircraft, cond, ref, S, c, i,
                          total_CL, total_CD, total_Cm, comp_data, model_size)
        elif method == "vlm":
            _step_vlm(aircraft, cond, ref, S, c, i,
                      total_CL, total_CD, total_Cm, comp_data,
                      spanwise_resolution, chordwise_resolution)
        else:
            from aerisplane.aero import analyze as _analyze
            r = _analyze(aircraft, cond, method=method,
                         spanwise_resolution=spanwise_resolution,
                         model_size=model_size)
            total_CL[i] = r.CL
            total_CD[i] = r.CD
            total_Cm[i] = r.Cm

    components = {
        name: ComponentCurve(
            name=name,
            alphas=alphas.copy(),
            CL=np.array(v["CL"]),
            CD=np.array(v["CD"]),
            Cm=np.array(v["Cm"]),
        )
        for name, v in comp_data.items()
    }

    return AlphaSweepResult(
        alphas=alphas,
        CL=total_CL, CD=total_CD, Cm=total_Cm,
        method=method,
        components=components,
        s_ref=S, c_ref=c, b_ref=b,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Solver-specific step functions
# ─────────────────────────────────────────────────────────────────────────────

def _step_buildup(aircraft, cond, ref, S, c, idx,
                  total_CL, total_CD, total_Cm, comp_data, model_size):
    from aerisplane.aero.solvers.aero_buildup import AeroBuildup
    ac = copy.deepcopy(aircraft)
    ac.xyz_ref = ref
    out = AeroBuildup(ac, cond, xyz_ref=ref, model_size=model_size).run()

    q = cond.dynamic_pressure()
    qS  = q * S
    qSc = qS * c if c > 0 else 1.0

    total_CL[idx] = float(out["CL"])
    total_CD[idx] = float(out["CD"])
    total_Cm[idx] = float(out["Cm"])

    for j, comp in enumerate(out["wing_aero_components"]):
        _append(aircraft.wings[j].name, comp, cond, qS, qSc, comp_data)
    for j, comp in enumerate(out["fuselage_aero_components"]):
        _append(aircraft.fuselages[j].name, comp, cond, qS, qSc, comp_data)


def _append(name, comp, cond, qS, qSc, comp_data):
    """Append one AeroComponentResults to comp_data dict."""
    if name not in comp_data:
        comp_data[name] = {"CL": [], "CD": [], "Cm": []}
    comp_data[name]["CL"].append(float(comp.L) / qS if qS > 0 else 0.0)
    comp_data[name]["CD"].append(float(comp.D) / qS if qS > 0 else 0.0)
    comp_data[name]["Cm"].append(float(comp.M_b[1]) / qSc if qSc > 0 else 0.0)


def _step_vlm(aircraft, cond, ref, S, c, idx,
              total_CL, total_CD, total_Cm, comp_data,
              spanwise_resolution, chordwise_resolution):
    from aerisplane.aero.solvers.vlm import VortexLatticeMethod
    solver = VortexLatticeMethod(
        aircraft=aircraft, condition=cond,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        verbose=False,
    )
    out = solver.run()

    q = cond.dynamic_pressure()
    qS  = q * S
    qSc = qS * c if c > 0 else 1.0
    ref_arr = np.array(ref)

    total_CL[idx] = float(out["CL"])
    total_CD[idx] = float(out["CD"])

    # Recompute Cm about the requested reference (VLM defaults to origin).
    r_all = solver.vortex_centers - ref_arr.reshape(1, 3)
    M_g_tot = np.cross(r_all, solver.forces_geometry).sum(axis=0)
    M_b_tot = cond.convert_axes(*M_g_tot, from_axes="geometry", to_axes="body")
    total_Cm[idx] = float(M_b_tot[1]) / qSc if qSc > 0 else float(out["Cm"])

    for rec in solver.wing_records:
        name = rec["wing"].name
        s, n = rec["panel_start"], rec["n_panels"]

        F_g = solver.forces_geometry[s:s + n].sum(axis=0)
        r_w = solver.vortex_centers[s:s + n] - ref_arr.reshape(1, 3)
        M_g = np.cross(r_w, solver.forces_geometry[s:s + n]).sum(axis=0)

        F_w = cond.convert_axes(*F_g, from_axes="geometry", to_axes="wind")
        M_b = cond.convert_axes(*M_g, from_axes="geometry", to_axes="body")

        if name not in comp_data:
            comp_data[name] = {"CL": [], "CD": [], "Cm": []}
        comp_data[name]["CL"].append(float(-F_w[2]) / qS if qS > 0 else 0.0)
        comp_data[name]["CD"].append(float(-F_w[0]) / qS if qS > 0 else 0.0)
        comp_data[name]["Cm"].append(float(M_b[1]) / qSc if qSc > 0 else 0.0)


__all__ = ["alpha_sweep", "AlphaSweepResult", "ComponentCurve"]
