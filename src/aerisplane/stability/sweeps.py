"""Stability sweep analyses: beta sweep (Cn, Cl) and rate sweeps (Cl(p̂), Cn(r̂)).

Beta sweep
----------
Runs the aero solver at a range of sideslip angles β while holding α fixed.
Returns Cl(β) and Cn(β) curves — the primary lateral static stability plots.
Per-component breakdown available with "aero_buildup" (wings + fuselages) or
"vlm" (wings only).

Rate sweeps
-----------
Vary the nondimensional roll rate p̂ = pb/(2V) or yaw rate r̂ = rb/(2V) and
record Cl(p̂) or Cn(r̂).  These give the roll and yaw damping curves.
Total aircraft only — component breakdown adds little design insight here.

Example
-------
>>> import numpy as np
>>> from aerisplane.stability.sweeps import beta_sweep, rate_sweep
>>>
>>> betas = np.linspace(-20, 20, 21)
>>> bs = beta_sweep(aircraft, condition, betas, method="aero_buildup",
...                 xyz_ref=weight_result.cg.tolist())
>>> bs.plot()
>>>
>>> rates = np.linspace(-0.15, 0.15, 21)
>>> rs_p = rate_sweep(aircraft, condition, rates, rate_type="p",
...                   xyz_ref=weight_result.cg.tolist())
>>> rs_p.plot()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition


# ─────────────────────────────────────────────────────────────────────────────
# Beta sweep
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BetaSweepResult:
    """Cl(β) and Cn(β) sweep result with optional per-component curves.

    Attributes
    ----------
    betas : np.ndarray
        Sideslip angles swept [deg].
    Cl, Cn : np.ndarray
        Total aircraft rolling- and yawing-moment coefficients.
    CY : np.ndarray
        Total aircraft side-force coefficient.
    components_Cl, components_Cn : dict[str, np.ndarray]
        Per-component Cl and Cn arrays keyed by component name.
        Empty when the solver does not support per-component output.
    method : str
        Aero solver used.
    Cl_beta, Cn_beta : float
        Slope dCl/dβ, dCn/dβ at β = 0 [1/deg] estimated by linear regression
        over the central ±5° range (or full range if narrower).
    """

    betas: np.ndarray
    Cl: np.ndarray
    Cn: np.ndarray
    CY: np.ndarray
    method: str
    components_Cl: dict = field(default_factory=dict)
    components_Cn: dict = field(default_factory=dict)
    Cl_beta: float = float("nan")
    Cn_beta: float = float("nan")

    def plot(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """Three-panel figure: Cl(β), Cn(β), CY(β) with per-component breakdown.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, (ax_cl, ax_cn, ax_cy) = plt.subplots(1, 3, figsize=(15, 5))

        self._draw_beta_panel(ax_cl, self.Cl, self.components_Cl,
                              "Cl", "Dihedral Effect  Cl(β)",
                              slope=self.Cl_beta, slope_label="Cl_β")
        self._draw_beta_panel(ax_cn, self.Cn, self.components_Cn,
                              "Cn", "Yaw Stability  Cn(β)",
                              slope=self.Cn_beta, slope_label="Cn_β")

        # CY — total only
        ax_cy.plot(self.betas, self.CY, "o-", color="k", lw=2, ms=3, label="Total")
        ax_cy.axhline(0, color="gray", lw=0.7, ls="--")
        ax_cy.axvline(0, color="gray", lw=0.7, ls=":")
        ax_cy.set_xlabel("β [deg]")
        ax_cy.set_ylabel("CY")
        ax_cy.set_title("Side Force  CY(β)", fontsize=11, fontweight="bold")
        ax_cy.legend(fontsize=9)
        ax_cy.grid(True, alpha=0.3)

        fig.suptitle(
            f"Beta Sweep — method: {self.method}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(pad=1.2)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def _draw_beta_panel(self, ax, total, components, ylabel, title, slope, slope_label):
        from aerisplane.utils.plotting import PALETTE
        for i, (name, arr) in enumerate(components.items()):
            ax.plot(self.betas, arr, "o-",
                    color=PALETTE[i % len(PALETTE)], lw=1.5, ms=3, label=name)
        ax.plot(self.betas, total, "o-", color="k", lw=2.5, ms=4, label="Total")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.axvline(0, color="gray", lw=0.7, ls=":")
        if not np.isnan(slope):
            ax.text(
                0.03, 0.97, f"{slope_label} = {slope:+.5f} /deg",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
            )
        ax.set_xlabel("β [deg]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)


def beta_sweep(
    aircraft: Aircraft,
    condition: FlightCondition,
    beta_range,
    method: str = "aero_buildup",
    xyz_ref: Optional[list] = None,
    spanwise_resolution: int = 8,
    chordwise_resolution: int = 4,
    model_size: str = "medium",
    verbose: bool = False,
) -> BetaSweepResult:
    """Run Cl(β), Cn(β), CY(β) sweep with optional per-component breakdown.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition
        Baseline condition; ``beta`` is overridden at each sweep point.
    beta_range : array-like
        Sideslip angles [deg].
    method : str
        Aero solver.  ``"aero_buildup"`` gives full per-component breakdown;
        ``"vlm"`` per-wing only; LL methods give total only.
    xyz_ref : list[float] or None
        Moment reference [x, y, z] [m].  Defaults to ``aircraft.xyz_ref``.
    """
    betas = np.asarray(beta_range, dtype=float)
    n = len(betas)
    ref = list(xyz_ref) if xyz_ref is not None else list(aircraft.xyz_ref)

    S = aircraft.reference_area()
    b = aircraft.reference_span()
    c = aircraft.reference_chord()

    total_Cl = np.zeros(n)
    total_Cn = np.zeros(n)
    total_CY = np.zeros(n)
    comp_Cl: dict[str, list] = {}
    comp_Cn: dict[str, list] = {}

    for i, beta in enumerate(betas):
        if verbose:
            print(f"  beta = {beta:+.1f}°  ({i + 1}/{n})")
        cond = condition.copy()
        cond.beta = float(beta)

        if method == "aero_buildup":
            _beta_step_buildup(aircraft, cond, ref, S, b, c, i,
                               total_Cl, total_Cn, total_CY,
                               comp_Cl, comp_Cn, model_size)
        elif method == "vlm":
            _beta_step_vlm(aircraft, cond, ref, S, b, c, i,
                           total_Cl, total_Cn, total_CY,
                           comp_Cl, comp_Cn,
                           spanwise_resolution, chordwise_resolution)
        else:
            from aerisplane.aero import analyze as _analyze
            r = _analyze(aircraft, cond, method=method,
                         spanwise_resolution=spanwise_resolution,
                         model_size=model_size)
            total_Cl[i] = r.Cl
            total_Cn[i] = r.Cn
            total_CY[i] = r.CY

    Cl_beta, Cn_beta = _fit_slope_at_zero(betas, total_Cl, total_Cn)

    return BetaSweepResult(
        betas=betas,
        Cl=total_Cl, Cn=total_Cn, CY=total_CY,
        method=method,
        components_Cl={k: np.array(v) for k, v in comp_Cl.items()},
        components_Cn={k: np.array(v) for k, v in comp_Cn.items()},
        Cl_beta=Cl_beta,
        Cn_beta=Cn_beta,
    )


def _fit_slope_at_zero(betas, Cl, Cn, window_deg: float = 5.0):
    """Linear regression slope at β = 0 using points within ±window_deg."""
    mask = np.abs(betas) <= window_deg
    if mask.sum() < 2:
        mask = np.ones(len(betas), dtype=bool)
    b_sel = betas[mask]
    Cl_slope = float(np.polyfit(b_sel, Cl[mask], 1)[0]) if len(b_sel) >= 2 else float("nan")
    Cn_slope = float(np.polyfit(b_sel, Cn[mask], 1)[0]) if len(b_sel) >= 2 else float("nan")
    return Cl_slope, Cn_slope


def _beta_step_buildup(aircraft, cond, ref, S, b, c, idx,
                       total_Cl, total_Cn, total_CY,
                       comp_Cl, comp_Cn, model_size):
    from aerisplane.aero.solvers.aero_buildup import AeroBuildup
    ac = copy.deepcopy(aircraft)
    ac.xyz_ref = ref
    out = AeroBuildup(ac, cond, xyz_ref=ref, model_size=model_size).run()

    q = cond.dynamic_pressure()
    qSb = q * S * b if b > 0 else 1.0

    total_Cl[idx] = float(out["Cl"])
    total_Cn[idx] = float(out["Cn"])
    total_CY[idx] = float(out["CY"])

    for j, comp in enumerate(out["wing_aero_components"]):
        name = aircraft.wings[j].name
        M_b = cond.convert_axes(*comp.M_g, from_axes="geometry", to_axes="body")
        _append_lat(name, M_b, qSb, comp_Cl, comp_Cn)
    for j, comp in enumerate(out["fuselage_aero_components"]):
        name = aircraft.fuselages[j].name
        M_b = cond.convert_axes(*comp.M_g, from_axes="geometry", to_axes="body")
        _append_lat(name, M_b, qSb, comp_Cl, comp_Cn)


def _append_lat(name, M_b, qSb, comp_Cl, comp_Cn):
    if name not in comp_Cl:
        comp_Cl[name] = []
        comp_Cn[name] = []
    comp_Cl[name].append(float(M_b[0]) / qSb if qSb > 0 else 0.0)
    comp_Cn[name].append(float(M_b[2]) / qSb if qSb > 0 else 0.0)


def _beta_step_vlm(aircraft, cond, ref, S, b, c, idx,
                   total_Cl, total_Cn, total_CY,
                   comp_Cl, comp_Cn,
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
    qSb = q * S * b if b > 0 else 1.0
    ref_arr = np.array(ref)

    total_Cl[idx] = float(out["Cl"])
    total_Cn[idx] = float(out["Cn"])
    total_CY[idx] = float(out["CY"])

    for rec in solver.wing_records:
        name = rec["wing"].name
        s, n = rec["panel_start"], rec["n_panels"]
        r_w = solver.vortex_centers[s:s + n] - ref_arr.reshape(1, 3)
        M_g = np.cross(r_w, solver.forces_geometry[s:s + n]).sum(axis=0)
        M_b = cond.convert_axes(*M_g, from_axes="geometry", to_axes="body")
        _append_lat(name, M_b, qSb, comp_Cl, comp_Cn)


# ─────────────────────────────────────────────────────────────────────────────
# Rate sweep
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RateSweepResult:
    """Cl(p̂) or Cn(r̂) rate-sweep result — total aircraft only.

    Attributes
    ----------
    rates : np.ndarray
        Nondimensional rate values swept (p̂ = pb/2V or r̂ = rb/2V) [-].
    Cl : np.ndarray
        Rolling-moment coefficient (meaningful for roll-rate sweep).
    Cn : np.ndarray
        Yawing-moment coefficient (meaningful for yaw-rate sweep).
    rate_type : str
        ``"p"`` for roll-rate sweep or ``"r"`` for yaw-rate sweep.
    slope_Cl, slope_Cn : float
        Linear slope dCl/dp̂ or dCn/dr̂ from central regression [per unit rate].
    method : str
    """

    rates: np.ndarray
    Cl: np.ndarray
    Cn: np.ndarray
    rate_type: str
    slope_Cl: float = float("nan")
    slope_Cn: float = float("nan")
    method: str = "aero_buildup"

    def plot(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """Plot Cl and Cn vs nondimensional rate.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        if self.rate_type == "p":
            xlabel = r"$\hat{p} = pb/(2V)$"
            titles = ("Roll Damping  Cl(p̂)", "Cross coupling  Cn(p̂)")
            slope_labels = (
                f"Cl_p = {self.slope_Cl:+.3f}",
                f"Cn_p = {self.slope_Cn:+.3f}",
            )
        else:
            xlabel = r"$\hat{r} = rb/(2V)$"
            titles = ("Cross coupling  Cl(r̂)", "Yaw Damping  Cn(r̂)")
            slope_labels = (
                f"Cl_r = {self.slope_Cl:+.3f}",
                f"Cn_r = {self.slope_Cn:+.3f}",
            )

        fig, (ax_cl, ax_cn) = plt.subplots(1, 2, figsize=(11, 5))
        for ax, data, title, slabel, color in zip(
            [ax_cl, ax_cn],
            [self.Cl, self.Cn],
            titles,
            slope_labels,
            [PALETTE[0], PALETTE[1]],
        ):
            ax.plot(self.rates, data, "o-", color=color, lw=2, ms=4)
            ax.axhline(0, color="gray", lw=0.7, ls="--")
            ax.axvline(0, color="gray", lw=0.7, ls=":")
            if not np.isnan(self.slope_Cl):
                ax.text(
                    0.03, 0.97, slabel,
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
                )
            ax.set_xlabel(xlabel)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        ax_cl.set_ylabel("Cl")
        ax_cn.set_ylabel("Cn")

        fig.suptitle(
            f"Rate Sweep (type={self.rate_type}) — method: {self.method}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(pad=1.2)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig


def rate_sweep(
    aircraft: Aircraft,
    condition: FlightCondition,
    rate_range,
    rate_type: str = "p",
    method: str = "aero_buildup",
    xyz_ref: Optional[list] = None,
    spanwise_resolution: int = 8,
    chordwise_resolution: int = 4,
    model_size: str = "medium",
    verbose: bool = False,
) -> RateSweepResult:
    """Run Cl(p̂) or Cn(r̂) nondimensional-rate sweep.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition
        Baseline; rotation rates are overridden per sweep point.
    rate_range : array-like
        Nondimensional rates to sweep (p̂ = pb/2V or r̂ = rb/2V).
    rate_type : str
        ``"p"`` for roll-rate sweep, ``"r"`` for yaw-rate sweep.
    method : str
        Aero solver for the sweep evaluations.
    xyz_ref : list or None
        Moment reference point [m].

    Returns
    -------
    RateSweepResult
    """
    rates = np.asarray(rate_range, dtype=float)
    n = len(rates)
    ref = list(xyz_ref) if xyz_ref is not None else list(aircraft.xyz_ref)

    V = float(condition.velocity)
    b = aircraft.reference_span()
    S = aircraft.reference_area()

    Cl_arr = np.zeros(n)
    Cn_arr = np.zeros(n)

    for i, r_hat in enumerate(rates):
        if verbose:
            print(f"  {rate_type}̂ = {r_hat:+.4f}  ({i + 1}/{n})")

        cond = condition.copy()
        # Convert nondim rate → dimensional [rad/s]
        rate_dim = float(r_hat) * (2.0 * V) / b if b > 0 else 0.0
        if rate_type == "p":
            cond.p = rate_dim
        else:
            cond.r = rate_dim

        result = _run_total_only(
            aircraft, cond, ref, method,
            spanwise_resolution, chordwise_resolution, model_size
        )
        Cl_arr[i] = result["Cl"]
        Cn_arr[i] = result["Cn"]

    # Fit central slopes
    mask = np.abs(rates) <= 0.06
    if mask.sum() < 2:
        mask = np.ones(n, dtype=bool)
    r_sel = rates[mask]
    slope_Cl = float(np.polyfit(r_sel, Cl_arr[mask], 1)[0]) if len(r_sel) >= 2 else float("nan")
    slope_Cn = float(np.polyfit(r_sel, Cn_arr[mask], 1)[0]) if len(r_sel) >= 2 else float("nan")

    return RateSweepResult(
        rates=rates,
        Cl=Cl_arr, Cn=Cn_arr,
        rate_type=rate_type,
        slope_Cl=slope_Cl, slope_Cn=slope_Cn,
        method=method,
    )


def _run_total_only(aircraft, cond, ref, method,
                    spanwise_resolution, chordwise_resolution, model_size):
    """Run one aero evaluation and return dict with Cl, Cn, CY (total only)."""
    if method == "aero_buildup":
        from aerisplane.aero.solvers.aero_buildup import AeroBuildup
        ac = copy.deepcopy(aircraft)
        ac.xyz_ref = ref
        out = AeroBuildup(ac, cond, xyz_ref=ref, model_size=model_size).run()
        return {"Cl": float(out["Cl"]), "Cn": float(out["Cn"]), "CY": float(out["CY"])}
    else:
        from aerisplane.aero import analyze as _analyze
        r = _analyze(aircraft, cond, method=method,
                     spanwise_resolution=spanwise_resolution,
                     model_size=model_size)
        return {"Cl": r.Cl, "Cn": r.Cn, "CY": r.CY}


__all__ = ["beta_sweep", "rate_sweep", "BetaSweepResult", "RateSweepResult"]
