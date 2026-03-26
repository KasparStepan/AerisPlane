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

    def plot_cg(self):
        """Standalone side-view CG bubble diagram with generous vertical space.

        Returns a matplotlib Figure showing component CG positions as
        mass-proportional bubbles in the x-z plane, with the overall CG
        marked as a red crosshair. Bubbles are spread vertically to avoid
        overlap when z-values are clustered.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        fig, ax = plt.subplots(figsize=(13, 9))

        sorted_comps = sorted(
            self.components.values(), key=lambda c: c.mass, reverse=True
        )
        colors = sns.color_palette("husl", n_colors=max(len(sorted_comps), 1))

        max_mass = max(c.mass for c in sorted_comps) if sorted_comps else 1
        z_vals = [c.cg[2] * 1000 for c in sorted_comps]
        z_span = max(z_vals) - min(z_vals) if len(z_vals) > 1 else 0
        x_vals = [c.cg[0] * 1000 for c in sorted_comps]
        x_span = max(x_vals) - min(x_vals) if len(x_vals) > 1 else 100

        for idx, comp in enumerate(sorted_comps):
            size = (comp.mass / max_mass) * 900 + 50
            ax.scatter(
                comp.cg[0] * 1000,
                comp.cg[2] * 1000,
                s=size,
                color=colors[idx],
                alpha=0.7,
                edgecolors="white",
                linewidth=1,
                zorder=2,
            )

        # Label components > 2% of total, alternate above/below to reduce overlap
        threshold = self.total_mass * 0.02
        label_idx = 0
        for idx, comp in enumerate(sorted_comps):
            if comp.mass >= threshold:
                pct = comp.mass / self.total_mass * 100 if self.total_mass > 0 else 0
                # Alternate label placement above/below
                y_offset = 14 if label_idx % 2 == 0 else -14
                va = "bottom" if y_offset > 0 else "top"
                ax.annotate(
                    f"{comp.name.replace('_', ' ')}\n{comp.mass * 1000:.1f}g ({pct:.0f}%)",
                    (comp.cg[0] * 1000, comp.cg[2] * 1000),
                    fontsize=8.5,
                    ha="center", va=va,
                    xytext=(0, y_offset), textcoords="offset points",
                    color="#333333",
                    arrowprops={"arrowstyle": "-", "color": "#cccccc", "linewidth": 0.5},
                )
                label_idx += 1

        # Overall CG crosshair
        ax.scatter(
            self.cg[0] * 1000, self.cg[2] * 1000,
            s=400, c="#EA4335", marker="+", linewidths=3.5, zorder=4,
        )
        ax.annotate(
            f"Overall CG\n({self.cg[0] * 1000:.0f}, {self.cg[2] * 1000:.0f}) mm",
            (self.cg[0] * 1000, self.cg[2] * 1000),
            fontsize=10, fontweight="bold", color="#EA4335",
            ha="left", va="top",
            xytext=(14, -10), textcoords="offset points",
        )

        ax.set_xlabel("x [mm]", fontsize=11)
        ax.set_ylabel("z [mm]", fontsize=11)
        ax.set_title(
            f"Side View — Component CG Positions ({self.total_mass * 1000:.0f}g total)",
            fontsize=12, fontweight="bold",
        )

        # Add generous z-padding so bubbles aren't squished into a line
        z_pad = max(x_span * 0.25, 30) if z_span < x_span * 0.3 else z_span * 0.3
        z_mid = (max(z_vals) + min(z_vals)) / 2
        ax.set_ylim(z_mid - z_pad, z_mid + z_pad)

        ax.grid(True, which="major", alpha=0.3, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.15, linewidth=0.4)
        ax.minorticks_on()
        fig.tight_layout(pad=1.0)
        return fig

    def plot_cg_bars(self):
        """CG position bar chart showing each component's x-position.

        Returns a matplotlib Figure with horizontal bars indicating the
        CG x-position of each component, colored by mass fraction.
        Bar width encodes mass. Overall CG shown as a vertical red line.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        fig, ax = plt.subplots(figsize=(12, 7))

        sorted_comps = sorted(
            self.components.values(), key=lambda c: c.cg[0], reverse=False
        )
        names = [c.name.replace("_", " ") for c in sorted_comps]
        cg_x_mm = [c.cg[0] * 1000 for c in sorted_comps]
        masses_g = [c.mass * 1000 for c in sorted_comps]

        # Color by mass fraction
        max_mass = max(c.mass for c in sorted_comps) if sorted_comps else 1
        cmap = plt.cm.get_cmap("YlOrRd")
        bar_colors = [cmap(0.2 + 0.7 * c.mass / max_mass) for c in sorted_comps]

        # Bar height proportional to mass (normalized)
        bar_heights = [0.3 + 0.6 * (m / max(masses_g)) for m in masses_g] if masses_g else [0.5]

        for i, (name, x, mass_g, h, color) in enumerate(
            zip(names, cg_x_mm, masses_g, bar_heights, bar_colors)
        ):
            ax.barh(
                i, x, height=h, color=color,
                edgecolor="white", linewidth=0.8, zorder=2,
            )
            pct = mass_g / (self.total_mass * 1000) * 100 if self.total_mass > 0 else 0
            ax.text(
                x + 3, i,
                f" {x:.0f} mm  ({mass_g:.1f}g, {pct:.0f}%)",
                va="center", ha="left", fontsize=8, color="#333333",
            )

        # Overall CG vertical line
        cg_x = self.cg[0] * 1000
        ax.axvline(cg_x, color="#EA4335", linewidth=2.5, linestyle="--", zorder=3, alpha=0.8)
        ax.text(
            cg_x, -0.8,
            f"  CG = {cg_x:.0f} mm",
            fontsize=10, fontweight="bold", color="#EA4335",
            ha="left", va="top",
        )

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("CG$_x$ position [mm]", fontsize=11)
        ax.set_title(
            "Component CG$_x$ Positions (bar width ∝ mass)",
            fontsize=12, fontweight="bold",
        )
        x_max = max(cg_x_mm) * 1.35 if cg_x_mm else 1
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", alpha=0.3, linewidth=0.8)
        ax.grid(axis="y", alpha=0)
        fig.tight_layout(pad=1.0)
        return fig

    def plot_distribution(self):
        """Donut/ring chart of mass distribution with callout labels.

        Returns a matplotlib Figure with a ring-shaped pie chart.
        Labels are placed in two columns (left and right) with connecting
        lines to their slices, keeping the chart clean and readable.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        fig, ax = plt.subplots(figsize=(14, 8))

        sorted_comps = sorted(
            self.components.values(), key=lambda c: c.mass, reverse=True
        )
        masses_g = [c.mass * 1000 for c in sorted_comps]
        names = [c.name.replace("_", " ") for c in sorted_comps]

        colors = sns.color_palette("husl", n_colors=max(len(sorted_comps), 1))

        # Draw the donut
        wedges, _ = ax.pie(
            masses_g,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5},
        )

        # Collect labels that need placement, split into left/right
        right_labels = []  # tx > 0 (right side of chart)
        left_labels = []   # tx <= 0 (left side)

        for wedge, name, mass_g, comp, color in zip(
            wedges, names, masses_g, sorted_comps, colors
        ):
            angle = (wedge.theta2 + wedge.theta1) / 2
            angle_rad = np.radians(angle)

            pct = comp.mass / self.total_mass * 100 if self.total_mass > 0 else 0
            if pct < 1.5:
                continue

            # Anchor on outer edge of ring
            ring_r = 0.775
            anchor_x = -ring_r * np.cos(angle_rad)
            anchor_y = ring_r * np.sin(angle_rad)

            # Natural text direction
            tx_sign = -1 if np.cos(angle_rad) > 0 else 1
            label_text = f"{name}  {mass_g:.1f}g ({pct:.0f}%)"

            entry = (anchor_x, anchor_y, tx_sign, label_text, color, angle)
            if tx_sign > 0:
                right_labels.append(entry)
            else:
                left_labels.append(entry)

        # Distribute labels evenly in each column to avoid overlap
        # Use min spacing to determine vertical extent needed
        min_spacing = 0.22  # minimum y-gap between labels

        def _place_labels(labels, side, ax):
            if not labels:
                return
            # Sort by anchor y-position (top to bottom)
            labels.sort(key=lambda e: -e[1])
            n = len(labels)
            # Center the label stack, with min_spacing between each
            total_height = (n - 1) * min_spacing
            y_top = total_height / 2
            text_x = 1.6 if side == "right" else -1.6
            ha = "left" if side == "right" else "right"

            for i, (ax_x, ax_y, _, label_text, color, _angle) in enumerate(labels):
                ty = y_top - i * min_spacing

                ax.annotate(
                    label_text,
                    xy=(ax_x, ax_y),
                    xytext=(text_x, ty),
                    fontsize=8.5,
                    color="#333333",
                    ha=ha, va="center",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": color,
                        "linewidth": 1.0,
                        "connectionstyle": "arc3,rad=0.05",
                    },
                )

        _place_labels(right_labels, "right", ax)
        _place_labels(left_labels, "left", ax)

        # Center text
        ax.text(
            0, 0,
            f"{self.total_mass * 1000:.0f}g\n{self.wing_loading:.1f} g/dm²",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333",
        )

        ax.set_title(
            "Mass Distribution",
            fontsize=12, fontweight="bold", pad=20,
        )

        # Compute y-limits to fit all labels
        all_labels = right_labels + left_labels
        max_n_side = max(len(right_labels), len(left_labels), 1)
        y_extent = max((max_n_side - 1) * min_spacing / 2 + 0.3, 1.3)
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-y_extent, y_extent)
        ax.set_aspect("equal")
        fig.tight_layout(pad=0.5)
        return fig
