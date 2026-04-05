"""Propulsion analysis result dataclass."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PropulsionResult:
    """Result of a propulsion operating-point analysis.

    Parameters
    ----------
    thrust_n : float  — Thrust [N]
    current_a : float  — Battery current [A]
    rpm : float  — Motor/propeller RPM
    motor_efficiency : float  — Electrical-to-shaft efficiency [-]
    propulsive_efficiency : float  — T*V/P_shaft [-]
    electrical_power_w : float  — Total electrical power [W]
    shaft_power_w : float  — Shaft power into propeller [W]
    battery_endurance_s : float  — Time to battery depletion [s]
    c_rate : float  — Instantaneous C-rate = current / capacity_ah [-]
    over_current : bool  — True if current > min(motor.max_current, esc.max_current)
    throttle : float  — Throttle [0-1]
    velocity_ms : float  — Flight velocity [m/s]
    """
    thrust_n: float
    current_a: float
    rpm: float
    motor_efficiency: float
    propulsive_efficiency: float
    electrical_power_w: float
    shaft_power_w: float
    battery_endurance_s: float
    c_rate: float
    over_current: bool
    throttle: float
    velocity_ms: float

    def report(self) -> str:
        flag = "  *** OVER-CURRENT ***" if self.over_current else ""
        return (
            f"Propulsion Analysis (throttle={self.throttle:.0%}, V={self.velocity_ms:.1f} m/s)\n"
            f"  Thrust               : {self.thrust_n:.2f} N\n"
            f"  Current              : {self.current_a:.2f} A{flag}\n"
            f"  RPM                  : {self.rpm:.0f}\n"
            f"  Motor efficiency     : {self.motor_efficiency:.1%}\n"
            f"  Propulsive efficiency: {self.propulsive_efficiency:.1%}\n"
            f"  Electrical power     : {self.electrical_power_w:.1f} W\n"
            f"  Shaft power          : {self.shaft_power_w:.1f} W\n"
            f"  C-rate               : {self.c_rate:.1f} C\n"
            f"  Battery endurance    : {self.battery_endurance_s / 60:.1f} min\n"
        )

    def plot(self):
        import matplotlib.pyplot as plt
        labels = ["Thrust (N)", "Current (A)", "RPM/100", "η_motor (%)", "η_prop (%)"]
        values = [
            self.thrust_n, self.current_a, self.rpm / 100.0,
            self.motor_efficiency * 100.0, self.propulsive_efficiency * 100.0,
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values)
        if self.over_current:
            bars[1].set_color("red")
        ax.set_title(f"Propulsion — throttle {self.throttle:.0%}, V={self.velocity_ms:.1f} m/s")
        ax.set_ylabel("Value")
        fig.tight_layout()
        return fig
