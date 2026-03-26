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
        """Mass breakdown horizontal bar chart and side-view CG diagram.

        Returns a matplotlib Figure with two subplots:
        1. Horizontal bar chart of component masses (sorted, with % labels)
        2. Side-view (x-z) CG diagram with sized bubbles and annotations

        Uses AerisPlane/AeroSandbox-style formatting for clean, readable output.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        fig, (ax_bar, ax_cg) = plt.subplots(
            1, 2, figsize=(15, 6),
            gridspec_kw={"width_ratios": [1, 1.2]},
        )

        sorted_comps = sorted(
            self.components.values(), key=lambda c: c.mass, reverse=False
        )
        names = [c.name.replace("_", " ") for c in sorted_comps]
        masses_g = [c.mass * 1000 for c in sorted_comps]

        # --- Horizontal bar chart ---
        colors = sns.color_palette("husl", n_colors=len(sorted_comps))
        bars = ax_bar.barh(names, masses_g, color=colors, edgecolor="white", linewidth=0.5)

        # Add mass labels at bar ends
        for bar, comp in zip(bars, sorted_comps):
            pct = comp.mass / self.total_mass * 100 if self.total_mass > 0 else 0
            label = f" {comp.mass * 1000:.1f}g ({pct:.0f}%)"
            ax_bar.text(
                bar.get_width(), bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8, color="#333333",
            )

        ax_bar.set_xlabel("Mass [g]")
        ax_bar.set_title(
            f"Mass Breakdown — {self.total_mass * 1000:.0f}g total, "
            f"{self.wing_loading:.1f} g/dm²",
            fontsize=11, fontweight="bold",
        )
        # Extend x-axis to fit labels
        x_max = max(masses_g) * 1.45 if masses_g else 1
        ax_bar.set_xlim(0, x_max)
        ax_bar.grid(axis="x", alpha=0.3, linewidth=0.8)
        ax_bar.grid(axis="y", alpha=0)

        # Mark override vs computed
        for i, comp in enumerate(sorted_comps):
            if comp.source == "override":
                ax_bar.text(
                    -2, i, "*", fontsize=10, fontweight="bold",
                    color="#EA4335", ha="right", va="center",
                )

        # --- Side-view CG diagram ---
        # Bubble sizes: scale so the largest component is a readable circle
        max_mass = max(c.mass for c in sorted_comps) if sorted_comps else 1
        for idx, comp in enumerate(sorted_comps):
            size = (comp.mass / max_mass) * 600 + 30  # min 30, max 630
            ax_cg.scatter(
                comp.cg[0] * 1000,
                comp.cg[2] * 1000,
                s=size,
                color=colors[idx],
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )

        # Label only the top components (> 3% of total) to avoid clutter
        threshold = self.total_mass * 0.03
        for idx, comp in enumerate(sorted_comps):
            if comp.mass >= threshold:
                ax_cg.annotate(
                    comp.name.replace("_", " "),
                    (comp.cg[0] * 1000, comp.cg[2] * 1000),
                    fontsize=7.5,
                    ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                    color="#333333",
                )

        # Overall CG crosshair
        ax_cg.scatter(
            self.cg[0] * 1000, self.cg[2] * 1000,
            s=250, c="#EA4335", marker="+", linewidths=3, zorder=4,
        )
        ax_cg.annotate(
            f"CG\n({self.cg[0] * 1000:.0f}, {self.cg[2] * 1000:.0f}) mm",
            (self.cg[0] * 1000, self.cg[2] * 1000),
            fontsize=9, fontweight="bold", color="#EA4335",
            ha="left", va="top",
            xytext=(10, -5), textcoords="offset points",
        )

        ax_cg.set_xlabel("x [mm]", fontsize=10)
        ax_cg.set_ylabel("z [mm]", fontsize=10)
        ax_cg.set_title("Side View — Component Positions", fontsize=11, fontweight="bold")
        ax_cg.set_aspect("equal")
        ax_cg.grid(True, which="major", alpha=0.3, linewidth=0.8)
        ax_cg.grid(True, which="minor", alpha=0.15, linewidth=0.4)
        ax_cg.minorticks_on()

        fig.tight_layout(pad=0.8)
        return fig
