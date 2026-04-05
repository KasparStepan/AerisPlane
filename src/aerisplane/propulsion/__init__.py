"""Propulsion discipline module.

Public API
----------
analyze(aircraft, condition, throttle) -> PropulsionResult
"""
from __future__ import annotations
from aerisplane.propulsion.result import PropulsionResult

__all__ = ["PropulsionResult", "analyze"]


def analyze(aircraft, condition, throttle: float = 1.0) -> PropulsionResult:
    """Compute propulsion operating point.

    Parameters
    ----------
    aircraft : Aircraft — must have a propulsion attribute (PropulsionSystem)
    condition : FlightCondition
    throttle : float — [0-1]

    Raises ValueError if aircraft has no PropulsionSystem.
    """
    from aerisplane.propulsion.solver import solve_operating_point
    from aerisplane.utils.atmosphere import isa

    propulsion = getattr(aircraft, "propulsion", None)
    if propulsion is None:
        raise ValueError(
            f"Aircraft '{aircraft.name}' has no PropulsionSystem — "
            "cannot run propulsion analysis."
        )

    _T, _p, rho, _mu = isa(condition.altitude)
    velocity = condition.velocity

    rpm, current = solve_operating_point(propulsion, throttle, velocity, rho)

    motor = propulsion.motor
    prop = propulsion.propeller
    battery = propulsion.battery
    esc = propulsion.esc

    thrust = prop.thrust(rpm, velocity, rho)
    shaft_power = prop.power(rpm, velocity, rho)
    motor_eff = motor.efficiency(throttle * battery.nominal_voltage, current)
    prop_eff = prop.efficiency(rpm, velocity, rho)

    electrical_power = throttle * battery.nominal_voltage * current if current > 0 else 0.0
    endurance = battery.energy() / electrical_power if electrical_power > 0 else float("inf")
    c_rate = current / battery.capacity_ah if battery.capacity_ah > 0 else 0.0

    max_allowed = min(motor.max_current, esc.max_current)
    over_current = bool(current > max_allowed)

    return PropulsionResult(
        thrust_n=thrust,
        current_a=current,
        rpm=rpm,
        motor_efficiency=motor_eff,
        propulsive_efficiency=prop_eff,
        electrical_power_w=electrical_power,
        shaft_power_w=shaft_power,
        battery_endurance_s=endurance,
        c_rate=c_rate,
        over_current=over_current,
        throttle=throttle,
        velocity_ms=velocity,
    )
