"""Aerodynamic analysis result dataclass with plotting and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AeroResult:
    """Full aerodynamic analysis result from a single operating point.

    Forces are in wind axes (L, D, Y). Moments are in body axes (l_b, m_b, n_b).
    Coefficients follow the same axis convention.

    Parameters
    ----------
    method : str
        Solver used: "vlm", "lifting_line", "nonlinear_lifting_line", "aero_buildup".

    Forces (wind axes) [N]
    ----------------------
    L : float
        Lift force [N].
    D : float
        Drag force [N].
    Y : float
        Side force [N].

    Moments (body axes) [Nm]
    ------------------------
    l_b : float
        Rolling moment [Nm]. Positive = roll right.
    m_b : float
        Pitching moment [Nm]. Positive = pitch up.
    n_b : float
        Yawing moment [Nm]. Positive = nose right.

    Force vectors [N]
    -----------------
    F_g, F_b, F_w : list[float]
        [x, y, z] force components in geometry, body, wind axes.
    M_g, M_b, M_w : list[float]
        [x, y, z] moment components in geometry, body, wind axes.

    Coefficients
    ------------
    CL, CD, CY : float
        Force coefficients (wind axes).
    Cl, Cm, Cn : float
        Moment coefficients (body axes).
    CDi : float or None
        Induced drag coefficient. None when not available.
    CDp : float or None
        Parasitic (viscous) drag coefficient. Available from aero_buildup only.

    Operating condition (echoed)
    ----------------------------
    alpha : float
        Angle of attack [deg].
    beta : float
        Sideslip angle [deg].
    velocity : float
        True airspeed [m/s].
    altitude : float
        Altitude [m].
    dynamic_pressure : float
        Dynamic pressure q [Pa].
    reynolds_number : float
        Reynolds number based on reference chord.

    Reference geometry (echoed)
    ---------------------------
    s_ref : float
        Reference area [m^2].
    c_ref : float
        Reference chord [m].
    b_ref : float
        Reference span [m].
    """

    method: str

    # Forces (wind axes) [N]
    L: float = 0.0
    D: float = 0.0
    Y: float = 0.0

    # Moments (body axes) [Nm]
    l_b: float = 0.0
    m_b: float = 0.0
    n_b: float = 0.0

    # Force / moment vectors
    F_g: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    F_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    F_w: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    M_g: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    M_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    M_w: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Force coefficients (wind axes)
    CL: float = 0.0
    CD: float = 0.0
    CY: float = 0.0

    # Moment coefficients (body axes)
    Cl: float = 0.0
    Cm: float = 0.0
    Cn: float = 0.0

    # Drag breakdown
    CDi: Optional[float] = None
    CDp: Optional[float] = None

    # Operating condition (echoed)
    alpha: float = 0.0
    beta: float = 0.0
    velocity: float = 0.0
    altitude: float = 0.0
    dynamic_pressure: float = 0.0
    reynolds_number: float = 0.0

    # Reference geometry (echoed)
    s_ref: float = 0.0
    c_ref: float = 0.0
    b_ref: float = 0.0

    # Internal — ASB solver and airplane objects (not included in repr / compare)
    _solver: Any = field(default=None, repr=False, compare=False)
    _airplane: Any = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def report(self) -> None:
        """Print a formatted summary of the aerodynamic result."""
        sep = "-" * 52
        print(sep)
        print(f"  AeroResult — method: {self.method}")
        print(sep)
        print(f"  Operating point:")
        print(f"    alpha          = {self.alpha:>10.4f}  deg")
        print(f"    beta           = {self.beta:>10.4f}  deg")
        print(f"    velocity       = {self.velocity:>10.4f}  m/s")
        print(f"    altitude       = {self.altitude:>10.1f}  m")
        print(f"    dyn. pressure  = {self.dynamic_pressure:>10.2f}  Pa")
        print(f"    Reynolds (MAC) = {self.reynolds_number:>10.0f}")
        print(sep)
        print(f"  Reference geometry:")
        print(f"    S_ref = {self.s_ref:.4f} m^2   "
              f"c_ref = {self.c_ref:.4f} m   "
              f"b_ref = {self.b_ref:.4f} m")
        print(sep)
        print(f"  Force coefficients (wind axes):")
        print(f"    CL   = {self.CL:>10.6f}")
        print(f"    CD   = {self.CD:>10.6f}")
        if self.CDi is not None:
            print(f"    CDi  = {self.CDi:>10.6f}  (induced)")
        if self.CDp is not None:
            print(f"    CDp  = {self.CDp:>10.6f}  (parasitic)")
        print(f"    CY   = {self.CY:>10.6f}")
        print(sep)
        print(f"  Moment coefficients (body axes):")
        print(f"    Cl   = {self.Cl:>10.6f}  (roll)")
        print(f"    Cm   = {self.Cm:>10.6f}  (pitch)")
        print(f"    Cn   = {self.Cn:>10.6f}  (yaw)")
        print(sep)
        print(f"  Forces (wind axes):")
        print(f"    L    = {self.L:>10.3f}  N")
        print(f"    D    = {self.D:>10.3f}  N")
        print(f"    Y    = {self.Y:>10.3f}  N")
        print(sep)
        print(f"  Moments (body axes):")
        print(f"    l_b  = {self.l_b:>10.3f}  Nm  (roll)")
        print(f"    m_b  = {self.m_b:>10.3f}  Nm  (pitch)")
        print(f"    n_b  = {self.n_b:>10.3f}  Nm  (yaw)")
        print(sep)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_polar(
        results: list[AeroResult],
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a CL-alpha, CD-alpha, Cm-alpha polar from a list of AeroResult objects.

        Parameters
        ----------
        results : list of AeroResult
            Results at different alpha values (sorted by alpha automatically).
        show : bool
            Whether to call plt.show().
        save_path : str or None
            If given, saves the figure to this path.
        """
        import matplotlib.pyplot as plt

        results_sorted = sorted(results, key=lambda r: r.alpha)
        alphas = [r.alpha for r in results_sorted]
        CLs = [r.CL for r in results_sorted]
        CDs = [r.CD for r in results_sorted]
        Cms = [r.Cm for r in results_sorted]
        method = results_sorted[0].method if results_sorted else "unknown"

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Aerodynamic Polar — method: {method}")

        axes[0].plot(alphas, CLs, "o-")
        axes[0].set_xlabel("alpha [deg]")
        axes[0].set_ylabel("CL")
        axes[0].grid(True)
        axes[0].set_title("Lift curve")

        axes[1].plot(CDs, CLs, "o-")
        axes[1].set_xlabel("CD")
        axes[1].set_ylabel("CL")
        axes[1].grid(True)
        axes[1].set_title("Drag polar")

        axes[2].plot(alphas, Cms, "o-")
        axes[2].set_xlabel("alpha [deg]")
        axes[2].set_ylabel("Cm")
        axes[2].grid(True)
        axes[2].set_title("Pitching moment")

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()

    def plot_spanwise_loading(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot spanwise Cl distribution from a VLM result.

        Uses per-panel data stored from the AeroSandbox VLM solver.
        Only available when method='vlm' and the result was produced by
        aerisplane.aero.analyze().

        Parameters
        ----------
        show : bool
            Whether to call plt.show().
        save_path : str or None
            If given, save the figure to this path.

        Raises
        ------
        RuntimeError
            If solver data is not available (non-VLM methods or externally
            constructed AeroResult).
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if self._solver is None:
            raise RuntimeError(
                "Solver data not available. "
                "plot_spanwise_loading() requires method='vlm' and an AeroResult "
                "produced by aerisplane.aero.analyze()."
            )

        pts = np.array(self._solver.collocation_points)   # (N, 3)
        fg  = np.array(self._solver.forces_geometry)      # (N, 3)
        areas = np.array(self._solver.areas)               # (N,)

        y_panels  = pts[:, 1]
        fz_panels = fg[:, 2]    # z in geometry frame = lift
        q = self.dynamic_pressure

        # Per-panel local Cl
        cl_panels = fz_panels / (q * areas)

        # Group into spanwise stations: round y to 6 decimal places, then unique
        y_rounded = np.round(y_panels, 6)
        y_stations = np.unique(y_rounded)

        y_plot  = []
        cl_plot = []
        for y_s in y_stations:
            mask = y_rounded == y_s
            y_plot.append(y_s)
            cl_plot.append(cl_panels[mask].mean())

        y_plot  = np.array(y_plot)
        cl_plot = np.array(cl_plot)

        # Normalise to semi-span fraction for x-axis label
        b_half = self.b_ref / 2.0
        eta = y_plot / b_half if b_half > 0 else y_plot

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eta, cl_plot, "o-", lw=1.8, ms=4)
        ax.axhline(self.CL, ls="--", lw=1.0, color="grey", label=f"CL (total) = {self.CL:.3f}")
        ax.set_xlabel("η = 2y / b  (span fraction)")
        ax.set_ylabel("Local Cl")
        ax.set_title(
            f"Spanwise lift distribution — VLM\n"
            f"α = {self.alpha:.1f}°,  V = {self.velocity:.1f} m/s,  "
            f"h = {self.altitude:.0f} m"
        )
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
