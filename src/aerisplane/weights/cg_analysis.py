"""CG envelope analysis and ballast calculation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.weights.result import ComponentOverride, WeightResult


@dataclass
class CGEnvelopeCase:
    """One loading configuration in a CG envelope study.

    Parameters
    ----------
    name : str
        Configuration description (e.g., "no payload", "heavy battery").
    overrides : dict
        Component overrides that define this configuration.
    result : WeightResult
        The weight result for this configuration.
    """

    name: str
    overrides: dict[str, ComponentOverride]
    result: WeightResult


@dataclass
class CGEnvelope:
    """CG range across multiple loading configurations.

    Parameters
    ----------
    cases : list of CGEnvelopeCase
        All computed loading configurations.
    """

    cases: list[CGEnvelopeCase] = field(default_factory=list)

    @property
    def cg_x_min(self) -> float:
        """Most forward CG_x across all cases [m]."""
        return min(c.result.cg[0] for c in self.cases)

    @property
    def cg_x_max(self) -> float:
        """Most aft CG_x across all cases [m]."""
        return max(c.result.cg[0] for c in self.cases)

    @property
    def cg_x_range(self) -> float:
        """CG_x travel distance [m]."""
        return self.cg_x_max - self.cg_x_min

    @property
    def mass_min(self) -> float:
        """Lightest configuration [kg]."""
        return min(c.result.total_mass for c in self.cases)

    @property
    def mass_max(self) -> float:
        """Heaviest configuration [kg]."""
        return max(c.result.total_mass for c in self.cases)

    def report(self, mac: float = 0.0, mac_le_x: float = 0.0) -> str:
        """Formatted CG envelope summary.

        Parameters
        ----------
        mac : float
            Mean aerodynamic chord [m]. If >0, CG is also shown as %MAC.
        mac_le_x : float
            MAC leading edge x-position [m].
        """
        lines = []
        lines.append("CG Envelope Analysis")
        lines.append("=" * 70)

        header = f"{'Configuration':<30s} {'Mass [g]':>9s} {'CG_x [mm]':>10s}"
        if mac > 0:
            header += f" {'CG %MAC':>8s}"
        lines.append(header)
        lines.append("-" * 70)

        for case in self.cases:
            line = (
                f"{case.name:<30s} "
                f"{case.result.total_mass * 1000:>9.1f} "
                f"{case.result.cg[0] * 1000:>10.1f}"
            )
            if mac > 0:
                pct = (case.result.cg[0] - mac_le_x) / mac * 100
                line += f" {pct:>7.1f}%"
            lines.append(line)

        lines.append("-" * 70)
        lines.append(
            f"CG_x range: {self.cg_x_min * 1000:.1f} — "
            f"{self.cg_x_max * 1000:.1f} mm "
            f"(travel: {self.cg_x_range * 1000:.1f} mm)"
        )
        lines.append(
            f"Mass range: {self.mass_min * 1000:.1f} — "
            f"{self.mass_max * 1000:.1f} g"
        )

        return "\n".join(lines)

    def plot(self):
        """Plot CG_x vs total mass for all configurations.

        Returns a matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        for case in self.cases:
            ax.scatter(
                case.result.cg[0] * 1000,
                case.result.total_mass * 1000,
                s=100, zorder=3,
            )
            ax.annotate(
                case.name,
                (case.result.cg[0] * 1000, case.result.total_mass * 1000),
                fontsize=8, ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points",
            )

        ax.set_xlabel("CG_x [mm]")
        ax.set_ylabel("Total mass [g]")
        ax.set_title("CG Envelope — Loading Configurations")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def compute_cg_envelope(
    aircraft: Aircraft,
    configurations: dict[str, dict[str, ComponentOverride]],
    base_overrides: dict[str, ComponentOverride] | None = None,
) -> CGEnvelope:
    """Compute CG positions for multiple loading configurations.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft to analyze.
    configurations : dict
        Maps configuration name to a dict of component overrides.
        Each configuration is analyzed independently.
    base_overrides : dict or None
        Overrides applied to ALL configurations (e.g., measured structural masses).
        Configuration-specific overrides take priority over base overrides.

    Returns
    -------
    CGEnvelope
        CG positions and masses for all configurations.
    """
    from aerisplane.weights import analyze

    cases = []
    for name, config_overrides in configurations.items():
        # Merge base + config overrides
        merged = dict(base_overrides) if base_overrides else {}
        merged.update(config_overrides)

        result = analyze(aircraft, overrides=merged if merged else None)
        cases.append(CGEnvelopeCase(name=name, overrides=config_overrides, result=result))

    return CGEnvelope(cases=cases)


def compute_ballast(
    weight_result: WeightResult,
    target_cg_x: float,
    ballast_position_x: float,
    ballast_position_y: float = 0.0,
    ballast_position_z: float = 0.0,
) -> float:
    """Compute the ballast mass needed to shift CG to a target x-position.

    Parameters
    ----------
    weight_result : WeightResult
        Current weight result (before ballast).
    target_cg_x : float
        Desired CG x-position [m].
    ballast_position_x : float
        X-position where ballast will be added [m].
    ballast_position_y : float
        Y-position of ballast [m]. Default 0 (centerline).
    ballast_position_z : float
        Z-position of ballast [m]. Default 0.

    Returns
    -------
    float
        Required ballast mass [kg]. Positive means ballast is needed.
        Returns 0.0 if the CG is already at or past the target
        in the direction of the ballast position.
    """
    current_cg_x = weight_result.cg[0]
    current_mass = weight_result.total_mass

    # Moment balance: (M * cg_x + m_b * x_b) / (M + m_b) = target_cg_x
    # Solving for m_b:
    # M * cg_x + m_b * x_b = target_cg_x * (M + m_b)
    # M * cg_x + m_b * x_b = target_cg_x * M + target_cg_x * m_b
    # m_b * (x_b - target_cg_x) = M * (target_cg_x - cg_x)
    # m_b = M * (target_cg_x - cg_x) / (x_b - target_cg_x)

    denominator = ballast_position_x - target_cg_x
    if abs(denominator) < 1e-10:
        # Ballast at the target position can't shift CG
        return 0.0

    ballast_mass = current_mass * (target_cg_x - current_cg_x) / denominator

    # Negative ballast means CG is already past the target
    return max(0.0, ballast_mass)
