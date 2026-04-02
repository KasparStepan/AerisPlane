"""Stability analysis result dataclass with plotting and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StabilityResult:
    """Static and dynamic stability analysis result.

    Parameters
    ----------
    Longitudinal static stability
    -----------------------------
    static_margin : float
        Static margin as fraction of MAC (positive = stable).
    neutral_point : float
        Neutral point x-position [m] from aircraft origin.
    Cm_alpha : float
        Pitch moment derivative dCm/dalpha [1/deg]. Negative = stable.
    CL_alpha : float
        Lift curve slope dCL/dalpha [1/deg].

    Lateral-directional static stability
    -------------------------------------
    Cl_beta : float
        Roll-due-to-sideslip dCl/dbeta [1/deg]. Negative = stable (dihedral effect).
    Cn_beta : float
        Yaw-due-to-sideslip dCn/dbeta [1/deg]. Positive = stable (weathercock).

    Rate derivatives  (None if compute_rate_derivatives=False)
    -----------------------------------------------------------
    CL_q, Cm_q : float or None
        Lift and pitch moment due to pitch rate q_hat = qc/2V.
        Cm_q is the pitch damping derivative — negative = damped.
    Cl_p, Cn_p, CY_p : float or None
        Roll, yaw, and side force due to roll rate p_hat = pb/2V.
        Cl_p is the roll damping derivative — negative = damped.
    Cn_r, Cl_r, CY_r : float or None
        Yaw, roll, and side force due to yaw rate r_hat = rb/2V.
        Cn_r is the yaw damping derivative — negative = damped.

    Dynamic stability modes  (None if rate derivatives not computed)
    ----------------------------------------------------------------
    sp_frequency : float or None
        Short-period natural frequency [rad/s].
    sp_damping : float or None
        Short-period damping ratio (positive = damped).
    ph_frequency : float or None
        Phugoid natural frequency [rad/s] (Lanchester approximation).
    ph_damping : float or None
        Phugoid damping ratio (Lanchester: CD/CL/√2).

    Trim
    ----
    trim_alpha : float
        Angle of attack for Cm = 0 at current CG [deg].
    trim_elevator : float
        Elevator deflection for trimmed level flight [deg]. NaN if no elevator
        found or insufficient authority.

    Tail volume coefficients
    -----------------------
    Vh : float
        Horizontal tail volume coefficient. NaN if no htail found.
    Vv : float
        Vertical tail volume coefficient. NaN if no vtail found.

    CG envelope
    -----------
    cg_forward_limit : float
        Forward CG limit as fraction of MAC from wing LE.
    cg_aft_limit : float
        Aft CG limit as fraction of MAC from wing LE.

    Reference
    ---------
    cg_x : float
        CG x-position used for analysis [m].
    mac : float
        Mean aerodynamic chord [m].
    mac_le_x : float
        MAC leading edge x-position [m].

    Baseline coefficients
    --------------------
    CL_baseline : float
        Lift coefficient at the analysis condition.
    Cm_baseline : float
        Pitching moment coefficient at the analysis condition.
    """

    # Longitudinal static
    static_margin: float
    neutral_point: float
    Cm_alpha: float
    CL_alpha: float

    # Lateral-directional static
    Cl_beta: float
    Cn_beta: float

    # Rate derivatives (None when not computed)
    CL_q: Optional[float] = None
    Cm_q: Optional[float] = None
    Cl_p: Optional[float] = None
    Cn_p: Optional[float] = None
    CY_p: Optional[float] = None
    Cn_r: Optional[float] = None
    Cl_r: Optional[float] = None
    CY_r: Optional[float] = None

    # Dynamic stability modes (None when not computed)
    sp_frequency: Optional[float] = None   # short-period [rad/s]
    sp_damping: Optional[float] = None     # short-period damping ratio
    ph_frequency: Optional[float] = None   # phugoid [rad/s]
    ph_damping: Optional[float] = None     # phugoid damping ratio

    # Trim
    trim_alpha: float = float("nan")
    trim_elevator: float = float("nan")

    # Tail volume coefficients
    Vh: float = float("nan")
    Vv: float = float("nan")

    # CG envelope
    cg_forward_limit: float = 0.0
    cg_aft_limit: float = 0.0

    # Reference
    cg_x: float = 0.0
    mac: float = 0.0
    mac_le_x: float = 0.0

    # Baseline
    CL_baseline: float = 0.0
    Cm_baseline: float = 0.0

    # Optional Cm-vs-alpha sweep data for plotting
    _alpha_sweep: Optional[np.ndarray] = None
    _Cm_sweep: Optional[np.ndarray] = None

    def report(self) -> str:
        """Formatted stability analysis report."""
        lines = []
        lines.append("AerisPlane Stability Analysis")
        lines.append("=" * 60)

        lines.append("")
        lines.append("Longitudinal Static Stability")
        lines.append("-" * 40)
        lines.append(f"  CL_alpha        {self.CL_alpha:>+10.5f}  1/deg")
        lines.append(f"  Cm_alpha        {self.Cm_alpha:>+10.5f}  1/deg")
        lines.append(f"  Neutral point   {self.neutral_point * 1000:>10.1f}  mm from nose")
        lines.append(f"  CG position     {self.cg_x * 1000:>10.1f}  mm from nose")
        lines.append(f"  Static margin   {self.static_margin * 100:>10.1f}  % MAC")

        status = "STABLE" if self.Cm_alpha < 0 else "UNSTABLE"
        lines.append(f"  Status          {status:>10s}")

        lines.append("")
        lines.append("Lateral-Directional Static Stability")
        lines.append("-" * 40)
        lines.append(f"  Cl_beta         {self.Cl_beta:>+10.5f}  1/deg")
        lines.append(f"  Cn_beta         {self.Cn_beta:>+10.5f}  1/deg")

        dihedral_ok = "OK (negative)" if self.Cl_beta < 0 else "WARN (positive)"
        weathercock_ok = "OK (positive)" if self.Cn_beta > 0 else "WARN (negative)"
        lines.append(f"  Dihedral effect {dihedral_ok:>16s}")
        lines.append(f"  Weathercock     {weathercock_ok:>16s}")

        # Rate derivatives (only shown when computed)
        if self.Cm_q is not None:
            lines.append("")
            lines.append("Rate Derivatives")
            lines.append("-" * 40)
            lines.append(f"  CL_q            {self.CL_q:>+10.4f}  (per qc/2V)")
            lines.append(f"  Cm_q            {self.Cm_q:>+10.4f}  (per qc/2V)  pitch damping")
            if self.Cl_p is not None:
                lines.append(f"  Cl_p            {self.Cl_p:>+10.4f}  (per pb/2V)  roll damping")
                lines.append(f"  Cn_p            {self.Cn_p:>+10.4f}  (per pb/2V)  adverse yaw")
            if self.Cn_r is not None:
                lines.append(f"  Cn_r            {self.Cn_r:>+10.4f}  (per rb/2V)  yaw damping")
                lines.append(f"  Cl_r            {self.Cl_r:>+10.4f}  (per rb/2V)")

        # Dynamic modes (only shown when computed)
        if self.sp_frequency is not None:
            lines.append("")
            lines.append("Dynamic Stability Modes")
            lines.append("-" * 40)
            nan = float("nan")
            if not np.isnan(self.sp_frequency):
                sp_period = 2 * np.pi / self.sp_frequency if self.sp_frequency > 0 else nan
                lines.append(f"  Short-period ω  {self.sp_frequency:>10.3f}  rad/s")
                lines.append(f"  Short-period T  {sp_period:>10.3f}  s")
                lines.append(f"  Short-period ζ  {self.sp_damping:>+10.4f}")
                sp_ok = "OK (damped)" if self.sp_damping > 0 else "WARN (undamped)"
                lines.append(f"  Short-period    {sp_ok:>16s}")
            else:
                lines.append(f"  Short-period        N/A  (check Cm_alpha, Cm_q signs)")
            if not np.isnan(self.ph_frequency):
                ph_period = 2 * np.pi / self.ph_frequency if self.ph_frequency > 0 else nan
                lines.append(f"  Phugoid ω       {self.ph_frequency:>10.4f}  rad/s")
                lines.append(f"  Phugoid T       {ph_period:>10.2f}  s")
                lines.append(f"  Phugoid ζ       {self.ph_damping:>+10.4f}")

        lines.append("")
        lines.append("Trim")
        lines.append("-" * 40)
        lines.append(f"  Trim alpha      {self.trim_alpha:>10.2f}  deg")
        if np.isnan(self.trim_elevator):
            lines.append( "  Trim elevator       N/A  (no elevator or authority insufficient)")
        else:
            lines.append(f"  Trim elevator   {self.trim_elevator:>10.2f}  deg")

        lines.append("")
        lines.append("Tail Volume Coefficients")
        lines.append("-" * 40)
        if np.isnan(self.Vh):
            lines.append( "  Vh                  N/A  (no htail identified)")
        else:
            lines.append(f"  Vh              {self.Vh:>10.3f}")
        if np.isnan(self.Vv):
            lines.append( "  Vv                  N/A  (no vtail identified)")
        else:
            lines.append(f"  Vv              {self.Vv:>10.3f}")

        lines.append("")
        lines.append("CG Envelope (fraction of MAC from LE)")
        lines.append("-" * 40)
        lines.append(f"  Forward limit   {self.cg_forward_limit * 100:>10.1f}  % MAC")
        lines.append(f"  Aft limit       {self.cg_aft_limit * 100:>10.1f}  % MAC")
        lines.append(f"  Current CG      {self._cg_fraction() * 100:>10.1f}  % MAC")

        lines.append("")
        lines.append("Baseline Coefficients")
        lines.append("-" * 40)
        lines.append(f"  CL              {self.CL_baseline:>+10.4f}")
        lines.append(f"  Cm              {self.Cm_baseline:>+10.4f}")

        return "\n".join(lines)

    def _cg_fraction(self) -> float:
        """CG position as fraction of MAC from MAC leading edge."""
        if self.mac == 0:
            return 0.0
        return (self.cg_x - self.mac_le_x) / self.mac

    def plot(self):
        """Static stability summary plot.

        Returns a matplotlib Figure with:
        1. Cm vs alpha curve with trim point and stability slope
        2. CG envelope on a MAC bar diagram
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        fig, (ax_cm, ax_cg) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={"width_ratios": [1.2, 1]},
        )

        # --- Left panel: Cm vs alpha ---
        if self._alpha_sweep is not None and self._Cm_sweep is not None:
            ax_cm.plot(
                self._alpha_sweep, self._Cm_sweep,
                "o-", color=PALETTE[0], linewidth=2, markersize=4,
                label="Cm(alpha)",
            )
        else:
            # Show tangent line from baseline
            da = 3.0  # degrees each side
            alpha_base = self.trim_alpha if not np.isnan(self.trim_alpha) else 0.0
            alpha_line = np.array([alpha_base - da, alpha_base + da])
            Cm_line = self.Cm_baseline + self.Cm_alpha * (alpha_line - alpha_base)
            ax_cm.plot(
                alpha_line, Cm_line,
                "-", color=PALETTE[0], linewidth=2,
                label=f"Cm_alpha = {self.Cm_alpha:+.4f}/deg",
            )

        ax_cm.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        # Mark trim point
        if not np.isnan(self.trim_alpha):
            ax_cm.axvline(
                self.trim_alpha, color=PALETTE[1], linewidth=1.5, linestyle=":",
                label=f"Trim alpha = {self.trim_alpha:.1f} deg",
            )

        ax_cm.set_xlabel("Angle of Attack [deg]")
        ax_cm.set_ylabel("Cm (pitch moment coefficient)")
        ax_cm.set_title("Pitch Stability", fontsize=11, fontweight="bold")
        ax_cm.legend(fontsize=9)
        ax_cm.grid(True, alpha=0.3)

        # --- Right panel: CG envelope on MAC bar ---
        cg_frac = self._cg_fraction()
        bar_y = 0.5
        bar_height = 0.3

        # MAC bar
        ax_cg.barh(
            bar_y, 1.0, height=bar_height,
            color="#E8E8E8", edgecolor="#666666", linewidth=1.5,
        )

        # Acceptable CG range
        cg_range = self.cg_aft_limit - self.cg_forward_limit
        ax_cg.barh(
            bar_y, cg_range, left=self.cg_forward_limit, height=bar_height,
            color=PALETTE[2], alpha=0.4, edgecolor=PALETTE[2], linewidth=1.5,
            label="Allowable CG range",
        )

        # Current CG marker
        ax_cg.plot(
            cg_frac, bar_y, "v", color=PALETTE[0],
            markersize=14, zorder=5, label=f"CG = {cg_frac * 100:.1f}% MAC",
        )

        # Neutral point marker
        np_frac = (self.neutral_point - self.mac_le_x) / self.mac if self.mac > 0 else 0
        ax_cg.plot(
            np_frac, bar_y, "D", color=PALETTE[1],
            markersize=10, zorder=5, label=f"NP = {np_frac * 100:.1f}% MAC",
        )

        # Forward / aft limit lines
        ax_cg.axvline(
            self.cg_forward_limit, color="#CC0000", linewidth=1.5, linestyle="--",
            label=f"Fwd limit = {self.cg_forward_limit * 100:.0f}% MAC",
        )
        ax_cg.axvline(
            self.cg_aft_limit, color="#CC0000", linewidth=1.5, linestyle="-.",
            label=f"Aft limit = {self.cg_aft_limit * 100:.0f}% MAC",
        )

        ax_cg.set_xlim(-0.05, 1.1)
        ax_cg.set_ylim(0, 1)
        ax_cg.set_xlabel("Fraction of MAC")
        ax_cg.set_title("CG Envelope", fontsize=11, fontweight="bold")
        ax_cg.legend(fontsize=8, loc="upper right")
        ax_cg.set_yticks([])
        ax_cg.grid(axis="x", alpha=0.3)

        # Label LE / TE
        ax_cg.text(0, bar_y - 0.2, "LE", ha="center", fontsize=9, fontweight="bold")
        ax_cg.text(1, bar_y - 0.2, "TE", ha="center", fontsize=9, fontweight="bold")

        fig.suptitle(
            f"Static Margin = {self.static_margin * 100:.1f}% MAC",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout(pad=1.0)
        return fig
