"""Control authority computation.

Computes control derivatives via forward finite differences, estimates
roll damping from strip theory, and derives physical authority metrics
(roll rate, pitch acceleration, crosswind capability, hinge moments).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# _trapz was added in NumPy 2.0; fall back to np.trapz for NumPy <2.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

from aerisplane.aero import analyze as aero_analyze
from aerisplane.aero.result import AeroResult
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.control_surface import ControlSurface
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.wing import Wing
from aerisplane.weights.result import WeightResult


# Surface type classification keywords
_AILERON_KEYS = ("aileron", "ail")
_ELEVATOR_KEYS = ("elevator", "elev")
_RUDDER_KEYS = ("rudder", "rud")

# Authority normalization targets
ROLL_RATE_REQUIREMENT = 180.0   # deg/s
CROSSWIND_REQUIREMENT = 5.0     # m/s

# Thin-airfoil hinge moment coefficient slope [1/rad]
_CH_DELTA = -0.6


@dataclass
class SurfaceInfo:
    """Identified control surface with its parent wing."""

    surface: ControlSurface
    wing: Wing
    surface_type: str  # "aileron", "elevator", "rudder"


@dataclass
class ControlDerivatives:
    """Raw control derivatives from finite-difference computation."""

    Cl_delta_a: float = 0.0
    Cm_delta_e: float = 0.0
    Cn_delta_r: float = 0.0

    baseline: AeroResult | None = None

    # Identified surfaces (for hinge moment computation)
    surfaces: dict[str, SurfaceInfo] = field(default_factory=dict)


def find_control_surfaces(aircraft: Aircraft) -> dict[str, SurfaceInfo]:
    """Scan aircraft wings and classify control surfaces by type.

    Returns a dict keyed by surface type ("aileron", "elevator", "rudder").
    If multiple surfaces of the same type exist, the first one found is used
    for derivative computation (they are all deflected together).
    """
    surfaces: dict[str, SurfaceInfo] = {}

    for wing in aircraft.wings:
        for cs in getattr(wing, "control_surfaces", []):
            name_lower = cs.name.lower()

            if any(k in name_lower for k in _AILERON_KEYS):
                stype = "aileron"
            elif any(k in name_lower for k in _ELEVATOR_KEYS):
                stype = "elevator"
            elif any(k in name_lower for k in _RUDDER_KEYS):
                stype = "rudder"
            else:
                continue  # skip flaps and unknown surfaces

            if stype not in surfaces:
                surfaces[stype] = SurfaceInfo(
                    surface=cs, wing=wing, surface_type=stype,
                )

    return surfaces


def compute_control_derivatives(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> ControlDerivatives:
    """Compute control derivatives via forward finite differences.

    For each control surface type found (aileron, elevator, rudder):
    1. Run baseline aero (all surfaces at zero)
    2. Run with the surface deflected to max_deflection
    3. Derivative = delta_coefficient / max_deflection

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry.
    condition : FlightCondition
        Operating point.
    weight_result : WeightResult
        Weight result (provides CG for moment reference).
    aero_method : str
        Aero solver method.
    **aero_kwargs
        Extra kwargs for aero.analyze().

    Returns
    -------
    ControlDerivatives
        Control derivatives and identified surfaces.
    """
    # Deep copy to avoid mutating caller's aircraft
    ac = copy.deepcopy(aircraft)
    cg = weight_result.cg
    ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]

    surfaces = find_control_surfaces(ac)

    # Baseline: all surfaces at zero
    cond_base = condition.copy()
    cond_base.deflections = {}
    baseline = aero_analyze(ac, cond_base, method=aero_method, **aero_kwargs)

    result = ControlDerivatives(baseline=baseline)

    # Map surface types back to original aircraft surfaces for hinge moments
    orig_surfaces = find_control_surfaces(aircraft)
    result.surfaces = orig_surfaces

    # Deflect each surface type and compute derivative
    for stype, info in surfaces.items():
        cs = info.surface
        delta_max = cs.max_deflection  # degrees

        if abs(delta_max) < 1e-6:
            continue

        cond_defl = condition.copy()
        cond_defl.deflections = {cs.name: delta_max}

        defl_result = aero_analyze(ac, cond_defl, method=aero_method, **aero_kwargs)

        if stype == "aileron":
            result.Cl_delta_a = (defl_result.Cl - baseline.Cl) / delta_max
        elif stype == "elevator":
            result.Cm_delta_e = (defl_result.Cm - baseline.Cm) / delta_max
        elif stype == "rudder":
            result.Cn_delta_r = (defl_result.Cn - baseline.Cn) / delta_max

    return result


def estimate_roll_damping(aircraft: Aircraft, condition: FlightCondition) -> float:
    """Estimate roll damping derivative Cl_p from strip theory.

    Cl_p = dCl / d(p_hat) where p_hat = p * b / (2V) is the non-dimensional
    roll rate.

    Uses the main wing chord distribution:
        Cl_p = -(CL_alpha_2d / 4) * integral[ (c/c_ref) * eta^2 ] d_eta

    where eta = 2y/b is the non-dimensional span coordinate.

    Returns
    -------
    float
        Roll damping derivative Cl_p [negative = damping].
    """
    main_wing = aircraft.main_wing()
    if main_wing is None:
        return -1.0  # fallback

    y = main_wing._y_stations()  # semispan stations [m]
    c = main_wing._chords()       # chords at each station [m]
    b_semi = main_wing.semispan()
    c_ref = main_wing.mean_aerodynamic_chord()

    if b_semi == 0 or c_ref == 0:
        return -1.0

    # Non-dimensional span coordinate eta = y / b_semi (0 to 1)
    eta = y / b_semi

    # CL_alpha for a 2D airfoil (thin airfoil theory)
    CL_alpha_2d = 2.0 * np.pi  # per radian

    # Strip theory integrand: (c / c_ref) * eta^2
    integrand = (c / c_ref) * eta**2

    integral = float(_trapz(integrand, eta))

    # Cl_p for the full wing (both sides contribute equally)
    # The factor accounts for both sides of a symmetric wing
    Cl_p = -CL_alpha_2d / 4.0 * integral

    # For a symmetric wing the integral covers one side; both sides contribute
    if getattr(main_wing, "symmetric", True):
        Cl_p *= 2.0  # account for missing side in eta integral

    return Cl_p


def compute_roll_rate(
    Cl_delta_a: float,
    delta_a_max: float,
    Cl_p: float,
    velocity: float,
    span: float,
) -> float:
    """Steady-state roll rate from aileron deflection.

    At steady state, aileron moment = damping moment:
        Cl_da * da = -Cl_p * (p * b / 2V)

    So:  p = -Cl_da * da / Cl_p * (2V / b)

    Parameters
    ----------
    Cl_delta_a : float
        Roll control derivative [1/deg].
    delta_a_max : float
        Maximum aileron deflection [deg].
    Cl_p : float
        Roll damping derivative (negative).
    velocity : float
        Airspeed [m/s].
    span : float
        Wingspan [m].

    Returns
    -------
    float
        Steady-state roll rate [deg/s]. Always positive.
    """
    if abs(Cl_p) < 1e-12 or span < 1e-6:
        return 0.0

    # Roll rate in rad/s
    p_rad = -Cl_delta_a * delta_a_max / Cl_p * (2.0 * velocity / span)

    return abs(np.degrees(p_rad))


def compute_pitch_acceleration(
    Cm_delta_e: float,
    delta_e_max: float,
    dynamic_pressure: float,
    S_ref: float,
    c_ref: float,
    I_yy: float,
) -> float:
    """Maximum pitch angular acceleration from elevator.

    alpha_ddot = (Cm_de * de_max * q * S * c) / I_yy

    Parameters
    ----------
    Cm_delta_e : float
        Pitch control derivative [1/deg].
    delta_e_max : float
        Maximum elevator deflection [deg].
    dynamic_pressure : float
        Dynamic pressure [Pa].
    S_ref : float
        Reference area [m^2].
    c_ref : float
        Reference chord [m].
    I_yy : float
        Pitch moment of inertia [kg*m^2].

    Returns
    -------
    float
        Max pitch acceleration [deg/s^2]. Always positive.
    """
    if I_yy < 1e-12:
        return 0.0

    # Moment in N*m
    moment = Cm_delta_e * delta_e_max * dynamic_pressure * S_ref * c_ref

    # Angular acceleration in rad/s^2
    alpha_ddot = moment / I_yy

    return abs(np.degrees(alpha_ddot))


def compute_max_crosswind(
    Cn_delta_r: float,
    delta_r_max: float,
    Cn_beta: float,
    velocity: float,
) -> float:
    """Maximum crosswind the rudder can counteract.

    beta_max = (Cn_dr * dr_max) / Cn_beta
    crosswind = V * sin(beta_max)

    Parameters
    ----------
    Cn_delta_r : float
        Yaw control derivative [1/deg].
    delta_r_max : float
        Maximum rudder deflection [deg].
    Cn_beta : float
        Weathercock stability derivative [1/deg] from stability analysis.
    velocity : float
        Airspeed [m/s].

    Returns
    -------
    float
        Maximum crosswind speed [m/s].
    """
    if abs(Cn_delta_r) < 1e-12:
        return 0.0

    if abs(Cn_beta) < 1e-12:
        # Directionally unstable or neutral — rudder is not the limiting factor
        return float("inf")

    beta_max_deg = abs(Cn_delta_r * delta_r_max / Cn_beta)
    beta_max_rad = np.radians(beta_max_deg)

    return abs(velocity * np.sin(beta_max_rad))


def estimate_hinge_moment(
    cs: ControlSurface,
    wing: Wing,
    dynamic_pressure: float,
) -> float | None:
    """Estimate hinge moment using thin-airfoil theory.

    H = Ch * q * S_cs * c_cs

    where Ch ≈ -0.6 * deflection [rad] (thin airfoil approximation).

    Parameters
    ----------
    cs : ControlSurface
        Control surface definition.
    wing : Wing
        Parent wing.
    dynamic_pressure : float
        Dynamic pressure [Pa].

    Returns
    -------
    float or None
        Hinge moment [N*m] at max deflection. None if no servo assigned.
    """
    if cs.servo is None:
        return None

    # Compute control surface area
    y = wing._y_stations()
    c = wing._chords()
    b_semi = wing.semispan()

    if b_semi == 0:
        return 0.0

    # Span positions of the surface in meters
    y_start = cs.span_start * b_semi
    y_end = cs.span_end * b_semi

    # Interpolate chords at surface boundaries and integrate
    c_start = float(np.interp(y_start, y, c))
    c_end = float(np.interp(y_end, y, c))

    # Trapezoidal area of the surface (single side)
    span_extent = y_end - y_start
    S_cs = cs.chord_fraction * 0.5 * (c_start + c_end) * span_extent

    # Mean chord of the control surface
    c_cs = cs.chord_fraction * 0.5 * (c_start + c_end)

    # Hinge moment coefficient at max deflection
    delta_rad = np.radians(cs.max_deflection)
    Ch = _CH_DELTA * delta_rad

    # Hinge moment [N*m]
    hinge_moment = Ch * dynamic_pressure * S_cs * c_cs

    return abs(hinge_moment)


def compute_servo_margin(
    hinge_moment: float | None,
    cs: ControlSurface,
) -> float | None:
    """Compute servo torque margin.

    margin = servo_torque / hinge_moment

    Returns None if no servo assigned or hinge moment is None/zero.
    """
    if hinge_moment is None or cs.servo is None:
        return None

    if abs(hinge_moment) < 1e-12:
        return float("inf")

    return cs.servo.torque / abs(hinge_moment)
