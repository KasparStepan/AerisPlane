"""Control authority analysis module.

Public API
----------
analyze(aircraft, condition, weight_result, stability_result, aero_method="vlm")
    Run a full control authority analysis and return a ControlResult.

Computes control derivatives for each surface type (aileron, elevator,
rudder) via forward finite differences, then derives roll rate, pitch
acceleration, crosswind capability, and servo load margins.

Example
-------
>>> from aerisplane.control import analyze
>>> from aerisplane.weights import analyze as weight_analyze
>>> from aerisplane.stability import analyze as stab_analyze
>>> wr = weight_analyze(aircraft)
>>> sr = stab_analyze(aircraft, condition, wr)
>>> result = analyze(aircraft, condition, wr, sr, aero_method="vlm")
>>> print(result.report())
"""

from __future__ import annotations

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.control.authority import (
    CROSSWIND_REQUIREMENT,
    ROLL_RATE_REQUIREMENT,
    compute_control_derivatives,
    compute_max_crosswind,
    compute_pitch_acceleration,
    compute_roll_rate,
    compute_servo_margin,
    estimate_hinge_moment,
    estimate_roll_damping,
)
from aerisplane.control.result import ControlResult
from aerisplane.stability.result import StabilityResult
from aerisplane.weights.result import WeightResult


def analyze(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    stability_result: StabilityResult,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> ControlResult:
    """Run a full control authority analysis.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry definition.
    condition : FlightCondition
        Operating point (velocity, altitude, alpha, beta).
    weight_result : WeightResult
        Weight analysis result (mass, CG, inertia tensor).
    stability_result : StabilityResult
        Stability analysis result (Cn_beta for crosswind calculation).
    aero_method : str
        Aero solver method (default "vlm").
    **aero_kwargs
        Additional keyword arguments passed to aero.analyze().

    Returns
    -------
    ControlResult
        Complete control authority analysis result.
    """
    # Step 1: Control derivatives
    deriv = compute_control_derivatives(
        aircraft, condition, weight_result, aero_method, **aero_kwargs
    )

    # Step 2: Roll damping from strip theory
    Cl_p = estimate_roll_damping(aircraft, condition)

    # Step 3: Roll rate
    aileron_info = deriv.surfaces.get("aileron")
    delta_a_max = aileron_info.surface.max_deflection if aileron_info else 0.0
    max_roll_rate = compute_roll_rate(
        Cl_delta_a=deriv.Cl_delta_a,
        delta_a_max=delta_a_max,
        Cl_p=Cl_p,
        velocity=condition.velocity,
        span=aircraft.reference_span(),
    )

    # Step 4: Pitch acceleration
    elevator_info = deriv.surfaces.get("elevator")
    delta_e_max = elevator_info.surface.max_deflection if elevator_info else 0.0
    I_yy = float(weight_result.inertia_tensor[1, 1])
    max_pitch_acceleration = compute_pitch_acceleration(
        Cm_delta_e=deriv.Cm_delta_e,
        delta_e_max=delta_e_max,
        dynamic_pressure=condition.dynamic_pressure(),
        S_ref=aircraft.reference_area(),
        c_ref=aircraft.reference_chord(),
        I_yy=I_yy,
    )

    # Step 5: Max crosswind
    rudder_info = deriv.surfaces.get("rudder")
    delta_r_max = rudder_info.surface.max_deflection if rudder_info else 0.0
    max_crosswind = compute_max_crosswind(
        Cn_delta_r=deriv.Cn_delta_r,
        delta_r_max=delta_r_max,
        Cn_beta=stability_result.Cn_beta,
        velocity=condition.velocity,
    )

    # Step 6: Hinge moments and servo margins
    q = condition.dynamic_pressure()

    aileron_hinge = None
    aileron_margin = None
    elevator_hinge = None
    elevator_margin = None
    rudder_hinge = None
    rudder_margin = None

    if aileron_info:
        aileron_hinge = estimate_hinge_moment(aileron_info.surface, aileron_info.wing, q)
        aileron_margin = compute_servo_margin(aileron_hinge, aileron_info.surface)

    if elevator_info:
        elevator_hinge = estimate_hinge_moment(elevator_info.surface, elevator_info.wing, q)
        elevator_margin = compute_servo_margin(elevator_hinge, elevator_info.surface)

    if rudder_info:
        rudder_hinge = estimate_hinge_moment(rudder_info.surface, rudder_info.wing, q)
        rudder_margin = compute_servo_margin(rudder_hinge, rudder_info.surface)

    # Step 7: Normalize authority
    aileron_authority = min(max_roll_rate / ROLL_RATE_REQUIREMENT, 1.0) if ROLL_RATE_REQUIREMENT > 0 else 0.0
    rudder_authority = min(max_crosswind / CROSSWIND_REQUIREMENT, 1.0) if CROSSWIND_REQUIREMENT > 0 else 0.0

    # Elevator authority: can the elevator trim over a ±10° alpha range?
    # Required Cm change = Cm_alpha * 10 deg. Elevator provides Cm_de * de_max.
    Cm_alpha = stability_result.Cm_alpha
    if abs(Cm_alpha) > 1e-10 and abs(deriv.Cm_delta_e) > 1e-10:
        alpha_range_available = abs(deriv.Cm_delta_e * delta_e_max / Cm_alpha)
        elevator_authority = min(alpha_range_available / 10.0, 1.0)
    else:
        elevator_authority = 0.0

    return ControlResult(
        # Roll
        max_roll_rate=max_roll_rate,
        aileron_authority=aileron_authority,
        Cl_delta_a=deriv.Cl_delta_a,
        # Pitch
        elevator_authority=elevator_authority,
        Cm_delta_e=deriv.Cm_delta_e,
        max_pitch_acceleration=max_pitch_acceleration,
        # Yaw
        rudder_authority=rudder_authority,
        Cn_delta_r=deriv.Cn_delta_r,
        max_crosswind=max_crosswind,
        # Servo loads
        aileron_hinge_moment=aileron_hinge,
        elevator_hinge_moment=elevator_hinge,
        rudder_hinge_moment=rudder_hinge,
        # Servo margins
        aileron_servo_margin=aileron_margin,
        elevator_servo_margin=elevator_margin,
        rudder_servo_margin=rudder_margin,
    )


__all__ = ["analyze", "ControlResult"]
