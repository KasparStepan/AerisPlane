"""Weight buildup result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComponentMass:
    """One component's mass contribution to the weight buildup.

    Parameters
    ----------
    name : str
        Component identifier (e.g., "main_wing_spar", "battery").
    mass : float
        Component mass [kg].
    cg : numpy array
        Center of gravity position [x, y, z] in aircraft frame [m].
    source : str
        How the mass was determined: "computed" or "override".
    """

    name: str
    mass: float
    cg: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    source: str = "computed"

    def __post_init__(self) -> None:
        self.cg = np.asarray(self.cg, dtype=float)


@dataclass
class ComponentOverride:
    """User-provided measured mass and CG for a component.

    Use this to replace a computed estimate with a measured value,
    or to add a component that the buildup doesn't know about
    (e.g., "gps_module", "receiver").

    Parameters
    ----------
    mass : float
        Measured mass [kg].
    cg : array-like or None
        Measured CG position [x, y, z] in aircraft frame [m].
        If None, keeps the computed CG (only valid when overriding
        an existing component).
    """

    mass: float
    cg: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.cg is not None:
            self.cg = np.asarray(self.cg, dtype=float)


@dataclass
class WeightResult:
    """Complete mass buildup result.

    Parameters
    ----------
    total_mass : float
        Total aircraft mass [kg].
    cg : numpy array
        Overall center of gravity [x, y, z] in aircraft frame [m].
    inertia_tensor : numpy array
        3x3 inertia tensor about the CG [kg*m^2].
    components : dict
        Component breakdown keyed by name.
    wing_loading : float
        Wing loading [g/dm^2].
    """

    total_mass: float
    cg: np.ndarray
    inertia_tensor: np.ndarray
    components: dict[str, ComponentMass]
    wing_loading: float

    def report(self) -> str:
        """Formatted table of mass breakdown.

        Returns a human-readable string with columns:
        Component | Mass [g] | CG_x [mm] | CG_z [mm] | Source | %
        """
        lines = []
        lines.append("AerisPlane Weight Buildup")
        lines.append("=" * 75)
        header = (
            f"{'Component':<24} {'Mass [g]':>9} {'CG_x [mm]':>10} "
            f"{'CG_z [mm]':>10} {'Source':<10} {'%':>5}"
        )
        lines.append(header)
        lines.append("-" * 75)

        sorted_components = sorted(
            self.components.values(), key=lambda c: c.mass, reverse=True
        )

        for comp in sorted_components:
            pct = (comp.mass / self.total_mass * 100.0) if self.total_mass > 0 else 0.0
            lines.append(
                f"{comp.name:<24} {comp.mass * 1000:>9.1f} {comp.cg[0] * 1000:>10.1f} "
                f"{comp.cg[2] * 1000:>10.1f} {comp.source:<10} {pct:>5.1f}"
            )

        lines.append("-" * 75)
        lines.append(
            f"{'TOTAL':<24} {self.total_mass * 1000:>9.1f}"
        )
        lines.append(
            f"CG position:  [{self.cg[0] * 1000:.1f}, "
            f"{self.cg[1] * 1000:.1f}, {self.cg[2] * 1000:.1f}] mm"
        )
        lines.append(f"Wing loading: {self.wing_loading:.1f} g/dm²")

        return "\n".join(lines)

    def plot(self):
        """Mass breakdown pie chart and side-view CG diagram.

        Returns a matplotlib Figure with two subplots:
        1. Pie chart of component masses
        2. Side-view (x-z) CG diagram with component positions

        Raises ImportError if matplotlib is not available.
        """
        import matplotlib.pyplot as plt

        fig, (ax_pie, ax_cg) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Pie chart ---
        sorted_comps = sorted(
            self.components.values(), key=lambda c: c.mass, reverse=True
        )
        names = [c.name for c in sorted_comps]
        masses = [c.mass for c in sorted_comps]

        ax_pie.pie(
            masses,
            labels=names,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax_pie.set_title("Mass Breakdown")

        # --- Side-view CG diagram ---
        for comp in sorted_comps:
            ax_cg.scatter(
                comp.cg[0] * 1000,
                comp.cg[2] * 1000,
                s=comp.mass * 5000,  # circle size proportional to mass
                alpha=0.5,
                zorder=2,
            )
            ax_cg.annotate(
                comp.name,
                (comp.cg[0] * 1000, comp.cg[2] * 1000),
                fontsize=7,
                ha="center",
                va="bottom",
            )

        # Overall CG
        ax_cg.scatter(
            self.cg[0] * 1000,
            self.cg[2] * 1000,
            s=200,
            c="red",
            marker="x",
            linewidths=3,
            zorder=3,
            label=f"CG ({self.cg[0] * 1000:.1f}, {self.cg[2] * 1000:.1f}) mm",
        )
        ax_cg.set_xlabel("x [mm]")
        ax_cg.set_ylabel("z [mm]")
        ax_cg.set_title("Side View — Component CG Positions")
        ax_cg.legend()
        ax_cg.set_aspect("equal")
        ax_cg.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
