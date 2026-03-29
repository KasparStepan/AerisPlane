"""Flight envelope computation across altitudes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.mission.performance import (
    DragPolar,
    best_endurance_speed,
    best_range_speed,
    fit_drag_polar,
    glide_performance,
    max_endurance,
    max_level_speed,
    max_range,
    max_rate_of_climb,
    power_available,
    power_required,
    stall_speed,
)
from aerisplane.weights.result import WeightResult


@dataclass
class EnvelopeResult:
    """Flight envelope across altitudes.

    All arrays are indexed by altitude.
    """

    altitudes: np.ndarray           # [m]
    stall_speeds: np.ndarray        # [m/s]
    best_endurance_speeds: np.ndarray
    best_range_speeds: np.ndarray
    max_speeds: np.ndarray          # [m/s], NaN where flight impossible
    max_rocs: np.ndarray            # [m/s]
    best_climb_speeds: np.ndarray   # V_y [m/s]

    # Scalar summaries (at sea level / reference altitude)
    ld_max: float
    endurance_s: float              # max endurance [s]
    range_m: float                  # max range [m]
    service_ceiling: float          # altitude [m] where ROC = 0.5 m/s
    absolute_ceiling: float         # altitude [m] where ROC = 0

    # Glide
    best_glide_ratio: float
    best_glide_speed: float         # at reference altitude [m/s]
    min_sink_rate: float            # [m/s]

    # Drag polar at reference altitude
    polar: DragPolar

    # Power curve data at reference altitude (for plotting)
    _v_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    _pr_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    _pa_grid: np.ndarray = field(default_factory=lambda: np.array([]))

    def report(self) -> str:
        lines = []
        lines.append("AerisPlane Flight Performance Envelope")
        lines.append("=" * 60)

        lines.append("")
        lines.append("Drag Polar")
        lines.append("-" * 40)
        lines.append(f"  CD0             {self.polar.CD0:>10.5f}")
        lines.append(f"  k               {self.polar.k:>10.5f}")
        lines.append(f"  L/D max         {self.ld_max:>10.1f}")

        lines.append("")
        lines.append("Characteristic Speeds (sea level)")
        lines.append("-" * 40)
        lines.append(f"  Stall           {self.stall_speeds[0]:>10.1f}  m/s")
        lines.append(f"  Best endurance  {self.best_endurance_speeds[0]:>10.1f}  m/s")
        lines.append(f"  Best range      {self.best_range_speeds[0]:>10.1f}  m/s")
        v_max_sl = self.max_speeds[0]
        if np.isnan(v_max_sl):
            lines.append(f"  Max speed            N/A")
        else:
            lines.append(f"  Max speed       {v_max_sl:>10.1f}  m/s")

        lines.append("")
        lines.append("Climb Performance")
        lines.append("-" * 40)
        lines.append(f"  Max ROC (SL)    {self.max_rocs[0]:>10.1f}  m/s")
        lines.append(f"  V_y (SL)        {self.best_climb_speeds[0]:>10.1f}  m/s")
        lines.append(f"  Service ceiling {self.service_ceiling:>10.0f}  m")
        lines.append(f"  Absolute ceiling{self.absolute_ceiling:>10.0f}  m")

        lines.append("")
        lines.append("Glide Performance")
        lines.append("-" * 40)
        lines.append(f"  Best glide L/D  {self.best_glide_ratio:>10.1f}")
        lines.append(f"  Best glide V    {self.best_glide_speed:>10.1f}  m/s")
        lines.append(f"  Min sink rate   {self.min_sink_rate:>10.2f}  m/s")

        lines.append("")
        lines.append("Endurance & Range")
        lines.append("-" * 40)
        lines.append(f"  Max endurance   {self.endurance_s / 60:>10.1f}  min")
        lines.append(f"  Max range       {self.range_m / 1000:>10.1f}  km")

        return "\n".join(lines)

    def plot(self):
        """Flight envelope summary: 4 subplots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --- Top-left: Power curves at sea level ---
        ax = axes[0, 0]
        if len(self._v_grid) > 0:
            ax.plot(self._v_grid, self._pr_grid, label="P required", color=PALETTE[0], lw=2)
            ax.plot(self._v_grid, self._pa_grid, label="P available", color=PALETTE[1], lw=2)
            ax.set_xlabel("Airspeed [m/s]")
            ax.set_ylabel("Power [W]")
            ax.set_title("Power Curves (Sea Level)", fontweight="bold")
            ax.legend()
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        # --- Top-right: Speed envelope vs altitude ---
        ax = axes[0, 1]
        ax.plot(self.stall_speeds, self.altitudes, label="Stall", color=PALETTE[3], lw=2)
        valid_max = ~np.isnan(self.max_speeds)
        if valid_max.any():
            ax.plot(self.max_speeds[valid_max], self.altitudes[valid_max],
                    label="Max speed", color=PALETTE[1], lw=2)
        ax.fill_betweenx(self.altitudes, self.stall_speeds,
                         np.where(valid_max, self.max_speeds, self.stall_speeds),
                         alpha=0.15, color=PALETTE[0], label="Flyable region")
        ax.set_xlabel("Airspeed [m/s]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title("Speed Envelope", fontweight="bold")
        ax.legend(fontsize=8)

        # --- Bottom-left: ROC vs altitude ---
        ax = axes[1, 0]
        ax.plot(self.max_rocs, self.altitudes, color=PALETTE[2], lw=2)
        ax.axvline(0.5, color=PALETTE[3], ls="--", lw=1, label="Service ceiling (0.5 m/s)")
        ax.axvline(0, color="gray", ls="-", lw=0.8)
        ax.set_xlabel("Max Rate of Climb [m/s]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title("Climb Performance", fontweight="bold")
        ax.legend(fontsize=8)

        # --- Bottom-right: Summary text ---
        ax = axes[1, 1]
        ax.axis("off")
        summary = (
            f"L/D max: {self.ld_max:.1f}\n"
            f"Stall (SL): {self.stall_speeds[0]:.1f} m/s\n"
            f"Best range V: {self.best_range_speeds[0]:.1f} m/s\n"
            f"Best endurance V: {self.best_endurance_speeds[0]:.1f} m/s\n"
            f"Max ROC (SL): {self.max_rocs[0]:.1f} m/s\n"
            f"Service ceiling: {self.service_ceiling:.0f} m\n"
            f"Best glide ratio: {self.best_glide_ratio:.1f}\n"
            f"Min sink: {self.min_sink_rate:.2f} m/s\n"
            f"Endurance: {self.endurance_s / 60:.1f} min\n"
            f"Range: {self.range_m / 1000:.1f} km"
        )
        ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=12,
                verticalalignment="center", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
        ax.set_title("Summary", fontweight="bold")

        fig.suptitle("Flight Performance Envelope", fontsize=14, fontweight="bold")
        fig.tight_layout(pad=1.0)
        return fig


def compute_envelope(
    aircraft: Aircraft,
    weight_result: WeightResult,
    CL_max: float = 1.4,
    altitudes: np.ndarray | None = None,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> EnvelopeResult:
    """Compute flight performance envelope across altitudes.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft definition (must have propulsion system).
    weight_result : WeightResult
        Weight analysis result.
    CL_max : float
        Maximum lift coefficient for stall speed estimation.
    altitudes : array or None
        Altitude array [m]. Default: 0 to 3000 m in 200 m steps.
    aero_method : str
        Aero solver for drag polar fitting.
    """
    if altitudes is None:
        altitudes = np.arange(0, 3200, 200, dtype=float)

    mass = weight_result.total_mass
    S = aircraft.reference_area()
    prop = aircraft.propulsion

    # Fit drag polar at sea level (assume weak altitude dependence for CD0, k)
    polar = fit_drag_polar(
        aircraft, weight_result, altitude=0.0,
        aero_method=aero_method, **aero_kwargs,
    )

    n_alt = len(altitudes)
    stall_speeds_arr = np.zeros(n_alt)
    ve_arr = np.zeros(n_alt)
    vr_arr = np.zeros(n_alt)
    vmax_arr = np.full(n_alt, np.nan)
    roc_arr = np.zeros(n_alt)
    vy_arr = np.zeros(n_alt)

    for i, alt in enumerate(altitudes):
        stall_speeds_arr[i] = stall_speed(mass, S, CL_max, alt)
        ve_arr[i] = best_endurance_speed(polar, mass, alt)
        vr_arr[i] = best_range_speed(polar, mass, alt)

        if prop is not None:
            vm = max_level_speed(polar, mass, prop, alt)
            vmax_arr[i] = vm if vm is not None else np.nan
            roc_max, v_y = max_rate_of_climb(polar, mass, prop, alt)
            roc_arr[i] = roc_max
            vy_arr[i] = v_y

    # Ceilings
    service_ceiling = float(altitudes[-1])
    absolute_ceiling = float(altitudes[-1])
    for i, alt in enumerate(altitudes):
        if roc_arr[i] <= 0.5 and i > 0:
            # Interpolate
            if roc_arr[i - 1] > 0.5:
                frac = (roc_arr[i - 1] - 0.5) / (roc_arr[i - 1] - roc_arr[i])
                service_ceiling = altitudes[i - 1] + frac * (alt - altitudes[i - 1])
            break
    else:
        service_ceiling = float(altitudes[-1])

    for i, alt in enumerate(altitudes):
        if roc_arr[i] <= 0 and i > 0:
            frac = roc_arr[i - 1] / (roc_arr[i - 1] - roc_arr[i])
            absolute_ceiling = altitudes[i - 1] + frac * (alt - altitudes[i - 1])
            break
    else:
        absolute_ceiling = float(altitudes[-1])

    # Glide and endurance/range at sea level
    gp = glide_performance(polar, mass, altitude=0.0)

    endurance = 0.0
    range_m = 0.0
    if prop is not None:
        endurance = max_endurance(polar, mass, prop, altitude=0.0)
        range_m = max_range(polar, mass, prop, altitude=0.0)

    # Power curve data for plotting
    v_grid = np.linspace(max(stall_speeds_arr[0] * 0.8, 3.0), 40.0, 80)
    pr_grid = np.array([power_required(v, polar, mass, 0.0) for v in v_grid])
    pa_grid = np.zeros_like(v_grid)
    if prop is not None:
        pa_grid = np.array([power_available(prop, v, 0.0) for v in v_grid])

    return EnvelopeResult(
        altitudes=altitudes,
        stall_speeds=stall_speeds_arr,
        best_endurance_speeds=ve_arr,
        best_range_speeds=vr_arr,
        max_speeds=vmax_arr,
        max_rocs=roc_arr,
        best_climb_speeds=vy_arr,
        ld_max=polar.ld_max(),
        endurance_s=endurance,
        range_m=range_m,
        service_ceiling=service_ceiling,
        absolute_ceiling=absolute_ceiling,
        best_glide_ratio=gp.best_glide_ratio,
        best_glide_speed=gp.best_glide_speed,
        min_sink_rate=gp.min_sink_rate,
        polar=polar,
        _v_grid=v_grid,
        _pr_grid=pr_grid,
        _pa_grid=pa_grid,
    )
