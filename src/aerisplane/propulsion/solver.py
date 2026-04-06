"""Operating-point solver."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from aerisplane.core.propulsion import PropulsionSystem


def solve_operating_point(
    system: PropulsionSystem,
    throttle: float,
    velocity: float,
    rho: float,
) -> tuple[float, float]:
    """Find equilibrium RPM and current.

    Returns (rpm, current). Returns (0.0, 0.0) if throttle <= 0.
    """
    if throttle <= 0.0:
        return 0.0, 0.0

    motor = system.motor
    prop = system.propeller
    battery = system.battery

    V_eff = throttle * battery.nominal_voltage
    Kt = 30.0 / (math.pi * motor.kv)

    def torque_residual(rpm: float) -> float:
        back_emf_v = rpm / motor.kv  # back-EMF volts
        I = (V_eff - back_emf_v) / motor.resistance
        I = max(0.0, min(I, motor.max_current))
        Q_motor = Kt * (I - motor.no_load_current)

        n = rpm / 60.0
        J = prop.advance_ratio(velocity, rpm)
        if prop.performance_data is None:
            cp = prop._parametric_cp(J)
        else:
            cp = float(np.interp(J, prop.performance_data.J, prop.performance_data.CP))
        Q_prop = cp * rho * n**2 * prop.diameter**5 / (2.0 * math.pi)
        return Q_motor - Q_prop

    rpm_max = motor.kv * V_eff

    f_low = torque_residual(1.0)
    f_high = torque_residual(rpm_max)

    if f_low * f_high > 0:
        rpms = np.linspace(1.0, rpm_max, 200)
        residuals = np.array([abs(torque_residual(r)) for r in rpms])
        rpm_eq = float(rpms[np.argmin(residuals)])
    else:
        rpm_eq = brentq(torque_residual, 1.0, rpm_max, xtol=0.1)

    back_emf_v = rpm_eq / motor.kv
    current = (V_eff - back_emf_v) / motor.resistance
    current = max(0.0, min(current, motor.max_current))
    return float(rpm_eq), float(current)
