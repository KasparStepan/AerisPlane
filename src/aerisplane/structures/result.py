# src/aerisplane/structures/result.py
"""Structural analysis result dataclasses with reporting and plotting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WingStructureResult:
    """Structural analysis result for one wing semi-span.

    Parameters
    ----------
    wing_name : str
    y : ndarray
        Spanwise stations [m].
    shear_force : ndarray
        V(y) [N].
    bending_moment : ndarray
        M(y) [N*m].
    deflection : ndarray
        delta(y) [m].
    tip_deflection : float
        delta at tip [m].
    tip_deflection_ratio : float
        delta_tip / semispan [-].
    root_bending_moment : float
        M at root [N*m].
    root_shear_force : float
        V at root [N].
    bending_margin : float
        Margin of safety against bending yield at root.
    shear_margin : float
        Margin of safety against shear yield at root.
    buckling_margin : float
        Margin of safety against shell buckling at root.
    spar_fits : bool
        True if spar OD <= available airfoil height at root.
    divergence_speed : float
        Torsional divergence speed [m/s] (inf if no risk).
    design_load_factor : float
        Ultimate load factor used for this analysis.
    """

    wing_name: str
    y: np.ndarray
    shear_force: np.ndarray
    bending_moment: np.ndarray
    deflection: np.ndarray
    tip_deflection: float
    tip_deflection_ratio: float
    root_bending_moment: float
    root_shear_force: float
    bending_margin: float
    shear_margin: float
    buckling_margin: float
    spar_fits: bool
    divergence_speed: float
    design_load_factor: float

    @property
    def is_safe(self) -> bool:
        """True if all margins are non-negative and spar fits in airfoil."""
        return (
            self.bending_margin >= 0.0
            and self.shear_margin >= 0.0
            and self.buckling_margin >= 0.0
            and self.spar_fits
        )

    def report(self) -> str:
        lines = [
            f"Wing: {self.wing_name}",
            f"  Design load factor:   {self.design_load_factor:.2f} g (ultimate)",
            f"  Root bending moment:  {self.root_bending_moment:.2f} N*m",
            f"  Root shear force:     {self.root_shear_force:.2f} N",
            f"  Tip deflection:       {self.tip_deflection * 1000:.1f} mm"
            f"  ({self.tip_deflection_ratio * 100:.1f}% semispan)",
            f"  Bending MoS (root):   {self.bending_margin:+.3f}"
            f"  {'PASS' if self.bending_margin >= 0 else 'FAIL'}",
            f"  Shear MoS (root):     {self.shear_margin:+.3f}"
            f"  {'PASS' if self.shear_margin >= 0 else 'FAIL'}",
            f"  Buckling MoS (root):  {self.buckling_margin:+.3f}"
            f"  {'PASS' if self.buckling_margin >= 0 else 'FAIL'}",
            f"  Spar fits in airfoil: {'PASS' if self.spar_fits else 'FAIL'}",
            f"  Divergence speed:     "
            + (f"{self.divergence_speed:.1f} m/s"
               if np.isfinite(self.divergence_speed) else "inf (no risk)"),
            f"  Overall: {'SAFE' if self.is_safe else 'FAILED'}",
        ]
        return "\n".join(lines)


@dataclass
class StructureResult:
    """Complete structural analysis result for all wings.

    Parameters
    ----------
    wings : list of WingStructureResult
        One entry per structural wing (wings without a spar are excluded).
    design_load_factor : float
        Ultimate load factor applied to all wings.
    """

    wings: list[WingStructureResult]
    design_load_factor: float

    @property
    def is_safe(self) -> bool:
        """True if all analysed wings pass all checks."""
        return all(w.is_safe for w in self.wings)

    def report(self) -> str:
        header = [
            "Structural Analysis",
            "=" * 60,
            f"Design load factor: {self.design_load_factor:.2f} g (ultimate)",
            f"Overall: {'SAFE' if self.is_safe else 'FAILED'}",
            "",
        ]
        wing_reports = [w.report() for w in self.wings]
        return "\n".join(header + wing_reports)

    def plot(self, show: bool = True, save_path: str | None = None) -> None:
        """Plot spanwise shear, moment, and deflection for all wings."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        n = len(self.wings)
        if n == 0:
            return

        fig, axes = plt.subplots(3, n, figsize=(5 * n, 9), squeeze=False)
        fig.suptitle(f"Structural Analysis - n_design = {self.design_load_factor:.2f} g")

        for col, wr in enumerate(self.wings):
            y = wr.y
            axes[0, col].plot(y, wr.shear_force)
            axes[0, col].set_ylabel("Shear [N]")
            axes[0, col].set_title(wr.wing_name)
            axes[0, col].grid(True, alpha=0.4)

            axes[1, col].plot(y, wr.bending_moment)
            axes[1, col].set_ylabel("Moment [N*m]")
            axes[1, col].grid(True, alpha=0.4)

            axes[2, col].plot(y, wr.deflection * 1000)
            axes[2, col].set_xlabel("Span y [m]")
            axes[2, col].set_ylabel("Deflection [mm]")
            axes[2, col].grid(True, alpha=0.4)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
