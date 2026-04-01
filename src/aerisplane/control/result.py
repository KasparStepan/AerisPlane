"""Control authority analysis result dataclass with plotting and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControlResult:
    """Control authority analysis result.

    Parameters
    ----------
    Roll
    ----
    max_roll_rate : float
        Steady-state roll rate at max aileron deflection [deg/s].
    aileron_authority : float
        Normalized 0-1 (1.0 = meets 180 deg/s requirement).
    Cl_delta_a : float
        Roll moment per degree aileron [1/deg].

    Pitch
    -----
    elevator_authority : float
        Normalized 0-1.
    Cm_delta_e : float
        Pitch moment per degree elevator [1/deg].
    max_pitch_acceleration : float
        Max pitch angular acceleration from full elevator [deg/s^2].

    Yaw
    ---
    rudder_authority : float
        Normalized 0-1 (1.0 = can hold 5 m/s crosswind).
    Cn_delta_r : float
        Yaw moment per degree rudder [1/deg].
    max_crosswind : float
        Max crosswind for coordinated flight [m/s].

    Servo loads
    -----------
    aileron_hinge_moment : float or None
        Hinge moment at max deflection [N*m]. None if no servo.
    elevator_hinge_moment : float or None
        Hinge moment at max deflection [N*m]. None if no servo.
    rudder_hinge_moment : float or None
        Hinge moment at max deflection [N*m]. None if no servo.

    Servo adequacy
    -------------
    aileron_servo_margin : float or None
        Servo torque / hinge moment (>1 = OK). None if no servo.
    elevator_servo_margin : float or None
        Servo torque / hinge moment (>1 = OK). None if no servo.
    rudder_servo_margin : float or None
        Servo torque / hinge moment (>1 = OK). None if no servo.
    """

    # Roll
    max_roll_rate: float
    aileron_authority: float
    Cl_delta_a: float

    # Pitch
    elevator_authority: float
    Cm_delta_e: float
    max_pitch_acceleration: float

    # Yaw
    rudder_authority: float
    Cn_delta_r: float
    max_crosswind: float

    # Servo loads
    aileron_hinge_moment: Optional[float] = None
    elevator_hinge_moment: Optional[float] = None
    rudder_hinge_moment: Optional[float] = None

    # Servo adequacy
    aileron_servo_margin: Optional[float] = None
    elevator_servo_margin: Optional[float] = None
    rudder_servo_margin: Optional[float] = None

    def report(self) -> str:
        """Formatted control authority report."""
        lines = []
        lines.append("AerisPlane Control Authority Analysis")
        lines.append("=" * 60)

        # --- Control derivatives ---
        lines.append("")
        lines.append("Control Derivatives")
        lines.append("-" * 40)
        lines.append(f"  Cl_delta_a      {self.Cl_delta_a:>+10.5f}  1/deg")
        lines.append(f"  Cm_delta_e      {self.Cm_delta_e:>+10.5f}  1/deg")
        lines.append(f"  Cn_delta_r      {self.Cn_delta_r:>+10.5f}  1/deg")

        # --- Roll ---
        lines.append("")
        lines.append("Roll Authority")
        lines.append("-" * 40)
        lines.append(f"  Max roll rate   {self.max_roll_rate:>10.1f}  deg/s")
        lines.append(f"  Authority       {self.aileron_authority:>10.2f}  (1.0 = 180 deg/s)")
        _status_line(lines, "  Status", self.aileron_authority)

        # --- Pitch ---
        lines.append("")
        lines.append("Pitch Authority")
        lines.append("-" * 40)
        lines.append(f"  Max pitch accel {self.max_pitch_acceleration:>10.1f}  deg/s^2")
        lines.append(f"  Authority       {self.elevator_authority:>10.2f}")
        _status_line(lines, "  Status", self.elevator_authority)

        # --- Yaw ---
        lines.append("")
        lines.append("Yaw / Crosswind Authority")
        lines.append("-" * 40)
        if np.isinf(self.max_crosswind):
            lines.append(f"  Max crosswind        inf  m/s (directionally unstable)")
        else:
            lines.append(f"  Max crosswind   {self.max_crosswind:>10.1f}  m/s")
        lines.append(f"  Authority       {self.rudder_authority:>10.2f}  (1.0 = 5 m/s)")
        _status_line(lines, "  Status", self.rudder_authority)

        # --- Servo loads ---
        lines.append("")
        lines.append("Servo Loads")
        lines.append("-" * 40)
        _servo_line(lines, "Aileron", self.aileron_hinge_moment, self.aileron_servo_margin)
        _servo_line(lines, "Elevator", self.elevator_hinge_moment, self.elevator_servo_margin)
        _servo_line(lines, "Rudder", self.rudder_hinge_moment, self.rudder_servo_margin)

        return "\n".join(lines)

    def plot(self):
        """Control authority summary plot.

        Returns a matplotlib Figure with:
        1. Authority bar chart (roll, pitch, yaw) with threshold line
        2. Servo torque margin for each surface with servo
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        from aerisplane.utils.plotting import PALETTE

        sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95)

        has_servos = any(
            m is not None
            for m in [
                self.aileron_servo_margin,
                self.elevator_servo_margin,
                self.rudder_servo_margin,
            ]
        )

        if has_servos:
            fig, (ax_auth, ax_servo) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax_auth = plt.subplots(figsize=(8, 6))
            ax_servo = None

        # --- Left panel: Authority bars ---
        axes_names = ["Roll", "Pitch", "Yaw"]
        authorities = [
            self.aileron_authority,
            self.elevator_authority,
            self.rudder_authority,
        ]

        colors = []
        for a in authorities:
            if a >= 1.0:
                colors.append("#34A853")  # green
            elif a >= 0.5:
                colors.append("#FBBC04")  # yellow
            else:
                colors.append("#EA4335")  # red

        bars = ax_auth.barh(axes_names, authorities, color=colors, edgecolor="white", height=0.5)

        # Requirement line
        ax_auth.axvline(1.0, color="#333333", linewidth=2, linestyle="--", label="Requirement")

        # Value labels
        for bar, val in zip(bars, authorities):
            ax_auth.text(
                bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=10, fontweight="bold",
            )

        ax_auth.set_xlabel("Authority (1.0 = meets requirement)")
        ax_auth.set_title("Control Authority", fontsize=11, fontweight="bold")
        ax_auth.set_xlim(0, max(max(authorities) * 1.3, 1.3))
        ax_auth.legend(fontsize=9)
        ax_auth.grid(axis="x", alpha=0.3)

        # --- Right panel: Servo margins ---
        if ax_servo is not None:
            servo_names = []
            margins = []
            margin_colors = []

            for name, margin in [
                ("Aileron", self.aileron_servo_margin),
                ("Elevator", self.elevator_servo_margin),
                ("Rudder", self.rudder_servo_margin),
            ]:
                if margin is not None:
                    servo_names.append(name)
                    margins.append(margin)
                    if margin >= 1.5:
                        margin_colors.append("#34A853")
                    elif margin >= 1.0:
                        margin_colors.append("#FBBC04")
                    else:
                        margin_colors.append("#EA4335")

            if servo_names:
                s_bars = ax_servo.barh(
                    servo_names, margins, color=margin_colors,
                    edgecolor="white", height=0.5,
                )

                ax_servo.axvline(
                    1.0, color="#EA4335", linewidth=2, linestyle="--",
                    label="Minimum (1.0)",
                )
                ax_servo.axvline(
                    1.5, color="#FBBC04", linewidth=1.5, linestyle=":",
                    label="Target (1.5)",
                )

                for bar, val in zip(s_bars, margins):
                    ax_servo.text(
                        bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va="center", ha="left", fontsize=10, fontweight="bold",
                    )

                ax_servo.set_xlabel("Servo Margin (torque / hinge moment)")
                ax_servo.set_title("Servo Adequacy", fontsize=11, fontweight="bold")
                ax_servo.set_xlim(0, max(max(margins) * 1.3, 2.0))
                ax_servo.legend(fontsize=9)
                ax_servo.grid(axis="x", alpha=0.3)

        fig.tight_layout(pad=1.0)
        return fig


def _status_line(lines: list, prefix: str, authority: float) -> None:
    """Append a pass/fail status line based on authority value."""
    if authority >= 1.0:
        lines.append(f"{prefix}          {'ADEQUATE':>10s}")
    elif authority >= 0.5:
        lines.append(f"{prefix}          {'MARGINAL':>10s}")
    else:
        lines.append(f"{prefix}          {'INSUFFICIENT':>10s}")


def _servo_line(
    lines: list, name: str, hinge_moment: float | None, margin: float | None
) -> None:
    """Append a servo load summary line."""
    if hinge_moment is None:
        lines.append(f"  {name:<12s}  no servo assigned")
    else:
        status = "OK" if margin and margin >= 1.0 else "OVERLOADED"
        lines.append(
            f"  {name:<12s}  hinge = {abs(hinge_moment):.3f} N*m, "
            f"margin = {margin:.2f}x  [{status}]"
        )
