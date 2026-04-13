"""LateralResult dataclass — all lateral-directional plots and structured report.

All figures follow the same visual language: per-component curves in colour,
total aircraft in a thick black line, zero-lines in grey.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aerisplane.stability.derivatives import DerivativeResult
from aerisplane.stability.sweeps import BetaSweepResult, RateSweepResult
from aerisplane.stability.lateral_model import LateralModes, TimeResponse
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.weights.result import WeightResult


@dataclass
class LateralResult:
    """Complete lateral-directional stability analysis result.

    Attributes
    ----------
    aircraft_name : str
    condition : FlightCondition
    weight_result : WeightResult
    deriv : DerivativeResult
        Point stability derivatives at the analysis condition.
    beta_sweep : BetaSweepResult
        Cl(β) and Cn(β) sweeps with optional component breakdown.
    rate_sweep_p : RateSweepResult
        Cl(p̂) and Cn(p̂) roll-rate sweep.
    rate_sweep_r : RateSweepResult
        Cl(r̂) and Cn(r̂) yaw-rate sweep.
    modes : LateralModes
        Eigenvalues and mode properties.
    responses : dict[str, TimeResponse]
        Time-history responses keyed by scenario name.
    A : np.ndarray
        The 4×4 lateral A-matrix used for the analysis.
    aero_method : str
        Aero solver used for sweeps.
    """

    aircraft_name: str
    condition: FlightCondition
    weight_result: WeightResult
    deriv: DerivativeResult
    beta_sweep: BetaSweepResult
    rate_sweep_p: RateSweepResult
    rate_sweep_r: RateSweepResult
    modes: LateralModes
    responses: dict = field(default_factory=dict)
    A: Optional[np.ndarray] = None
    aero_method: str = "aero_buildup"

    # ═════════════════════════════════════════════════════════════════════
    # Structured text report
    # ═════════════════════════════════════════════════════════════════════

    def report(self) -> str:
        """Multi-section structured stability report.

        Sections
        --------
        1. Header — aircraft, condition, reference geometry, method
        2. Lateral-directional derivatives table
        3. Mode summary — eigenvalues, frequencies, damping, time constants
        4. Handling-quality checks
        5. Reference block

        Returns
        -------
        str
            Plain-text report suitable for printing or saving to a file.
        """
        lines = []
        W = 62  # column width

        def sep(char="="):
            lines.append(char * W)

        def hdr(text):
            lines.append("")
            lines.append(text)
            sep("-")

        # ── 1. Header ────────────────────────────────────────────────────
        sep()
        lines.append("AerisPlane  Lateral-Directional Stability Report")
        sep()
        lines.append(f"  Aircraft   : {self.aircraft_name}")
        lines.append(f"  Date       : {_today()}")
        cond = self.condition
        V    = float(cond.velocity)
        h    = float(cond.altitude)
        mach = cond.mach()
        lines.append(
            f"  Condition  : V = {V:.1f} m/s,  h = {h:.0f} m,"
            f"  Mach = {mach:.3f}"
        )
        lines.append(f"               α = {float(cond.alpha):.2f}°,  β = {float(cond.beta):.2f}°")
        cg = self.weight_result.cg
        lines.append(
            f"  CG         : [{cg[0]*1000:.1f},  {cg[1]*1000:.1f},"
            f"  {cg[2]*1000:.1f}] mm"
        )
        mass = self.weight_result.total_mass
        lines.append(f"  Mass       : {mass*1000:.0f} g")
        ac  = self.weight_result   # WeightResult has no aircraft ref
        S   = self.deriv.baseline.s_ref
        b   = self.deriv.baseline.b_ref
        c   = self.deriv.baseline.c_ref
        lines.append(
            f"  Reference  : S = {S:.4f} m²,  b = {b:.3f} m,"
            f"  c = {c:.4f} m"
        )
        lines.append(f"  Method     : {self.aero_method}")

        # ── 2. Derivatives table ─────────────────────────────────────────
        hdr("LATERAL-DIRECTIONAL DERIVATIVES  [about CG, nondimensional]")
        _FMT = "  {:<10} {:>+12.5f}  1/deg    {}"
        d = self.deriv

        def _sign_note(val, positive_good, name):
            if positive_good:
                return f"{'OK (+)' if val > 0 else 'WARN (−)'}   {name}"
            else:
                return f"{'OK (−)' if val < 0 else 'WARN (+)'}   {name}"

        lines.append(_FMT.format("CY_beta", d.CY_beta, "side force / sideslip"))
        lines.append(_FMT.format("Cl_beta", d.Cl_beta,
                                  _sign_note(d.Cl_beta, False, "dihedral effect")))
        lines.append(_FMT.format("Cn_beta", d.Cn_beta,
                                  _sign_note(d.Cn_beta, True,  "weathercock stability")))

        def _rate_row(name, val, positive_good, note):
            if val is None:
                return f"  {name:<10} {'N/A':>13}           {note}"
            return _FMT.format(name, val, _sign_note(val, positive_good, note))

        lines.append(_rate_row("CY_p", d.CY_p, None, "side force / roll rate"))
        lines.append(_rate_row("Cl_p", d.Cl_p, False, "roll damping"))
        lines.append(_rate_row("Cn_p", d.Cn_p, None, "adverse yaw"))
        lines.append(_rate_row("CY_r", d.CY_r, None, "side force / yaw rate"))
        lines.append(_rate_row("Cl_r", d.Cl_r, None, "roll due to yaw"))
        lines.append(_rate_row("Cn_r", d.Cn_r, False, "yaw damping"))

        # ── 3. Mode summary ───────────────────────────────────────────────
        hdr("LATERAL-DIRECTIONAL MODES")
        modes = self.modes

        def _mode_block(mode, label):
            lines.append(f"  {label}")
            ev = mode.eigenvalue
            lines.append(f"    Eigenvalue     {_fmt_ev(ev)}")
            if np.isfinite(mode.time_constant):
                lines.append(f"    Time constant  {mode.time_constant:.3f} s")
            if label == "Dutch Roll" and np.isfinite(mode.frequency):
                T = 2 * math.pi / mode.frequency if mode.frequency > 0 else float("nan")
                lines.append(f"    ω_n            {mode.frequency:.3f} rad/s"
                              f"  ({mode.frequency/(2*math.pi):.3f} Hz)")
                lines.append(f"    Period         {T:.2f} s")
                lines.append(f"    ζ              {mode.damping:+.4f}")
            lines.append(f"    Status         {'STABLE' if mode.stable else 'UNSTABLE'}")

        _mode_block(modes.roll,       "Roll Subsidence")
        lines.append("")
        _mode_block(modes.dutch_roll, "Dutch Roll")
        lines.append("")
        _mode_block(modes.spiral,     "Spiral")

        # ── 4. Handling-quality checks ───────────────────────────────────
        hdr("HANDLING-QUALITY CHECKS")
        checks = _hq_checks(self.deriv, self.modes)
        for name, passed, note in checks:
            flag = "PASS" if passed else "FAIL"
            lines.append(f"  [{flag}]  {name:<42} {note}")

        # ── 5. Reference ─────────────────────────────────────────────────
        hdr("REFERENCE")
        lines.append("  Axes       : body (x forward, y right, z down)")
        lines.append("  CL / CD    : wind axes,  Cm / Cl / Cn : body axes")
        lines.append("  Moments    : normalised by q·S·b")
        lines.append("  Rates      : normalised by b/(2V)")
        lines.append("  Derivatives: central finite differences  Δα/β = 0.5/1.0 deg")
        lines.append("               rate Δ = 0.001 nondimensional")

        return "\n".join(lines)

    # ═════════════════════════════════════════════════════════════════════
    # Individual plot methods
    # ═════════════════════════════════════════════════════════════════════

    def plot_static_curves(self, show: bool = True, save_path=None):
        """Three-panel: Cl(β), Cn(β), CY(β) with per-component breakdown."""
        return self.beta_sweep.plot(show=show, save_path=save_path)

    def plot_rate_curves(self, show: bool = True, save_path=None):
        """Two-row figure: Cl(p̂)/Cn(p̂) and Cl(r̂)/Cn(r̂)."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        _fill_rate_ax(axes[0, 0], self.rate_sweep_p.rates, self.rate_sweep_p.Cl,
                      r"$\hat{p}$", "Cl", "Roll Damping  Cl(p̂)",
                      f"Cl_p = {self.rate_sweep_p.slope_Cl:+.3f}",
                      PALETTE[0])
        _fill_rate_ax(axes[0, 1], self.rate_sweep_p.rates, self.rate_sweep_p.Cn,
                      r"$\hat{p}$", "Cn", "Cross Coupling  Cn(p̂)",
                      f"Cn_p = {self.rate_sweep_p.slope_Cn:+.3f}",
                      PALETTE[1])
        _fill_rate_ax(axes[1, 0], self.rate_sweep_r.rates, self.rate_sweep_r.Cl,
                      r"$\hat{r}$", "Cl", "Cross Coupling  Cl(r̂)",
                      f"Cl_r = {self.rate_sweep_r.slope_Cl:+.3f}",
                      PALETTE[2])
        _fill_rate_ax(axes[1, 1], self.rate_sweep_r.rates, self.rate_sweep_r.Cn,
                      r"$\hat{r}$", "Cn", "Yaw Damping  Cn(r̂)",
                      f"Cn_r = {self.rate_sweep_r.slope_Cn:+.3f}",
                      PALETTE[3])

        fig.suptitle("Rate Derivatives", fontsize=13, fontweight="bold")
        fig.tight_layout(pad=1.2)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_derivatives_bar(self, show: bool = True, save_path=None):
        """Bar chart of all 9 lateral-directional derivative values."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        d = self.deriv
        names = [
            "CY_β", "Cl_β", "Cn_β",
            "CY_p", "Cl_p", "Cn_p",
            "CY_r", "Cl_r", "Cn_r",
        ]
        values = [
            d.CY_beta, d.Cl_beta, d.Cn_beta,
            d.CY_p or 0.0, d.Cl_p or 0.0, d.Cn_p or 0.0,
            d.CY_r or 0.0, d.Cl_r or 0.0, d.Cn_r or 0.0,
        ]
        colors = [
            PALETTE[1] if v < 0 else PALETTE[0] for v in values
        ]

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(names))
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1)

        # Value labels on bars
        for bar, val in zip(bars, values):
            offset = 0.002 if val >= 0 else -0.002
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:+.4f}", ha="center", va=va, fontsize=8)

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("Coefficient [1/deg or per unit rate]")
        ax.set_title(
            "Lateral-Directional Derivatives",
            fontsize=12, fontweight="bold",
        )
        # Colour legend
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color=PALETTE[1], label="Negative"),
                     Patch(color=PALETTE[0], label="Positive")],
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(pad=1.0)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_poles(self, show: bool = True, save_path=None):
        """Eigenvalue (pole) map in the complex plane."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, ax = plt.subplots(figsize=(7, 6))

        m = self.modes
        _plot_pole(ax, m.roll.eigenvalue,      "Roll",      PALETTE[0], "o")
        _plot_pole(ax, m.spiral.eigenvalue,    "Spiral",    PALETTE[2], "s")
        _plot_pole(ax, m.dutch_roll.eigenvalue,"Dutch Roll (×2)", PALETTE[1], "D")
        _plot_pole(ax, m.dutch_roll.eigenvalue.conjugate(), None, PALETTE[1], "D")

        ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.5)
        ax.axhline(0, color="gray",  lw=0.7, ls=":")
        ax.set_xlabel("Re(λ)  [1/s]")
        ax.set_ylabel("Im(λ)  [1/s]")
        ax.set_title("Lateral-Directional Poles", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=1.0)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_mode_shapes(self, show: bool = True, save_path=None):
        """Bar plots of eigenvector components for each lateral mode."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        state_labels = ["β", "p", "r", "φ"]
        evecs = self.modes.eigenvectors  # (4, 4) complex

        _mode_cols = [
            (0, "Roll Subsidence",    PALETTE[0]),
            (1, "Dutch Roll",         PALETTE[1]),
            (3, "Spiral",             PALETTE[2]),
        ]
        for ax, (col, title, color) in zip(axes, _mode_cols):
            vec = evecs[:, col]
            mag = np.abs(vec)
            norm = mag.max() if mag.max() > 0 else 1.0
            mag_norm = mag / norm
            ax.bar(state_labels, mag_norm, color=color, edgecolor="white", linewidth=1)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Normalised |eigenvector component|")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Lateral Mode Shapes", fontsize=13, fontweight="bold")
        fig.tight_layout(pad=1.2)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_time_responses(self, show: bool = True, save_path=None):
        """Time-history responses for Dutch roll, roll subsidence, and spiral.

        Each scenario gets its own row of subplots.  Control step responses
        (aileron / rudder) are appended if available.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.9)

        scenario_keys = ["dutch_roll", "roll_subsidence", "spiral"]
        if "aileron_step" in self.responses:
            scenario_keys.append("aileron_step")
        if "rudder_step" in self.responses:
            scenario_keys.append("rudder_step")

        n_rows = len(scenario_keys)
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.2 * n_rows), squeeze=False)

        signal_specs = [
            ("beta",  "β [deg]",    PALETTE[0]),
            ("p",     "p [deg/s]",  PALETTE[1]),
            ("r",     "r [deg/s]",  PALETTE[2]),
            ("phi",   "φ [deg]",    PALETTE[3]),
        ]

        titles = {
            "dutch_roll":    "Dutch Roll",
            "roll_subsidence": "Roll Subsidence",
            "spiral":        "Spiral Mode",
            "aileron_step":  "Aileron Step  δa = 10°",
            "rudder_step":   "Rudder Step  δr = 10°",
        }

        for row, key in enumerate(scenario_keys):
            if key not in self.responses:
                continue
            resp = self.responses[key]
            row_title = titles.get(key, key)

            for col, (attr, ylabel, color) in enumerate(signal_specs):
                ax = axes[row, col]
                ax.plot(resp.t, getattr(resp, attr), color=color, lw=1.8)
                ax.axhline(0, color="gray", lw=0.7, ls="--")
                ax.set_xlabel("t [s]")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_title(row_title, fontsize=10, fontweight="bold")

        fig.suptitle("Lateral-Directional Time Responses", fontsize=13, fontweight="bold")
        fig.tight_layout(pad=1.0)
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot(self, show: bool = True, save_path_prefix: Optional[str] = None):
        """Generate all lateral-directional figures.

        Parameters
        ----------
        save_path_prefix : str or None
            If given, each figure is saved to
            ``<prefix>_<name>.png``.

        Returns
        -------
        dict[str, matplotlib.figure.Figure]
        """
        def _path(name):
            return f"{save_path_prefix}_{name}.png" if save_path_prefix else None

        figs = {}
        figs["static_curves"]    = self.plot_static_curves(show=show, save_path=_path("static_curves"))
        figs["rate_curves"]      = self.plot_rate_curves(show=show, save_path=_path("rate_curves"))
        figs["derivatives_bar"]  = self.plot_derivatives_bar(show=show, save_path=_path("derivatives_bar"))
        figs["poles"]            = self.plot_poles(show=show, save_path=_path("poles"))
        figs["mode_shapes"]      = self.plot_mode_shapes(show=show, save_path=_path("mode_shapes"))
        figs["time_responses"]   = self.plot_time_responses(show=show, save_path=_path("time_responses"))
        return figs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def _fmt_ev(ev: complex) -> str:
    re = ev.real
    im = ev.imag
    if abs(im) < 1e-6:
        return f"{re:+.4f}  [1/s]"
    return f"{re:+.4f} ± {abs(im):.4f}j  [1/s]"


def _plot_pole(ax, ev: complex, label, color, marker):
    kw = {"color": color, "zorder": 5, "markersize": 10}
    if label:
        kw["label"] = label
    ax.plot(ev.real, ev.imag, marker, **kw)


def _fill_rate_ax(ax, rates, data, xlabel, ylabel, title, slope_text, color):
    ax.plot(rates, data, "o-", color=color, lw=2, ms=4)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.text(0.03, 0.97, slope_text, transform=ax.transAxes, fontsize=9, va="top",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)


def _hq_checks(deriv: DerivativeResult, modes: LateralModes):
    """Return list of (name, passed, note) tuples."""
    checks = []

    # Static
    checks.append((
        "Dihedral effect  Cl_beta < 0",
        deriv.Cl_beta < 0,
        f"Cl_β = {deriv.Cl_beta:+.5f} /deg",
    ))
    checks.append((
        "Weathercock stability  Cn_beta > 0",
        deriv.Cn_beta > 0,
        f"Cn_β = {deriv.Cn_beta:+.5f} /deg",
    ))
    if deriv.Cl_p is not None:
        checks.append((
            "Roll damping  Cl_p < 0",
            deriv.Cl_p < 0,
            f"Cl_p = {deriv.Cl_p:+.4f}",
        ))
    if deriv.Cn_r is not None:
        checks.append((
            "Yaw damping  Cn_r < 0",
            deriv.Cn_r < 0,
            f"Cn_r = {deriv.Cn_r:+.4f}",
        ))

    # Dynamic
    checks.append(("Roll mode stable", modes.roll.stable, ""))
    checks.append(("Spiral mode stable", modes.spiral.stable,
                   f"τ = {modes.spiral.time_constant:.1f} s"
                   if np.isfinite(modes.spiral.time_constant) else ""))
    checks.append(("Dutch roll stable", modes.dutch_roll.stable, ""))
    if np.isfinite(modes.dutch_roll.damping):
        checks.append((
            "Dutch roll  ζ ≥ 0.08  (MIL-F-8785C)",
            modes.dutch_roll.damping >= 0.08,
            f"ζ = {modes.dutch_roll.damping:.3f}",
        ))
    if np.isfinite(modes.dutch_roll.frequency):
        checks.append((
            "Dutch roll  ω_n ≥ 0.4 rad/s",
            modes.dutch_roll.frequency >= 0.4,
            f"ω_n = {modes.dutch_roll.frequency:.3f} rad/s",
        ))
    return checks


__all__ = ["LateralResult"]
