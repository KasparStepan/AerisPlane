"""Point performance equations for level flight, climb, and glide.

Implements the methods from the UAV flight envelope research document.
All equations assume steady, unaccelerated flight conditions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aerisplane.aero import analyze as aero_analyze
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.propulsion import PropulsionSystem
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult


G = 9.81  # gravitational acceleration [m/s^2]


@dataclass
class DragPolar:
    """Parabolic drag polar: CD = CD0 + k * CL^2.

    Fitted from a small number of aero evaluations at different speeds.

    Parameters
    ----------
    CD0 : float
        Zero-lift drag coefficient.
    k : float
        Induced drag factor: k = 1 / (pi * AR * e).
    S_ref : float
        Reference wing area [m^2].
    """

    CD0: float
    k: float
    S_ref: float

    def cd(self, cl: float) -> float:
        """Total drag coefficient at given CL."""
        return self.CD0 + self.k * cl**2

    def ld_max(self) -> float:
        """Maximum lift-to-drag ratio."""
        return 1.0 / (2.0 * math.sqrt(self.CD0 * self.k))

    def cl_for_ld_max(self) -> float:
        """CL at maximum L/D (minimum drag speed)."""
        return math.sqrt(self.CD0 / self.k)

    def cl_for_min_power(self) -> float:
        """CL at minimum power required (best endurance)."""
        return math.sqrt(3.0 * self.CD0 / self.k)


def fit_drag_polar(
    aircraft: Aircraft,
    weight_result: WeightResult,
    altitude: float = 0.0,
    aero_method: str = "vlm",
    speeds: tuple[float, ...] = (10.0, 15.0, 20.0),
    **aero_kwargs,
) -> DragPolar:
    """Fit a parabolic drag polar from aero evaluations at several speeds.

    Runs the aero solver at each speed, computes CL and CD, then fits
    CD = CD0 + k * CL^2 via least squares.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft definition.
    weight_result : WeightResult
        Weight result for CG (moment reference).
    altitude : float
        Altitude for the analysis [m MSL].
    aero_method : str
        Aero solver method.
    speeds : tuple of float
        Airspeeds to evaluate [m/s]. At least 2 required.
    **aero_kwargs
        Extra kwargs for aero.analyze().

    Returns
    -------
    DragPolar
        Fitted drag polar.
    """
    import copy

    ac = copy.deepcopy(aircraft)
    cg = weight_result.cg
    ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]

    S = aircraft.reference_area()
    W = weight_result.total_mass * G
    _, _, rho, _ = isa(altitude)

    cls = []
    cds = []

    for V in speeds:
        # Compute alpha for level flight: CL_required = 2W / (rho * V^2 * S)
        cl_req = 2.0 * W / (rho * V**2 * S) if (rho * V**2 * S) > 0 else 0.0
        # Estimate alpha from thin-airfoil CL_alpha ~ 2*pi*AR/(AR+2) per radian
        AR = aircraft.main_wing().aspect_ratio() if aircraft.main_wing() else 6.0
        cl_alpha_rad = 2.0 * np.pi * AR / (AR + 2.0)
        alpha_est = np.degrees(cl_req / cl_alpha_rad) if cl_alpha_rad > 0 else 2.0

        cond = FlightCondition(velocity=V, altitude=altitude, alpha=alpha_est)
        result = aero_analyze(ac, cond, method=aero_method, **aero_kwargs)

        cls.append(result.CL)
        cds.append(result.CD)

    cls = np.array(cls)
    cds = np.array(cds)

    # Fit CD = CD0 + k * CL^2 via least squares
    # A @ [CD0, k]^T = cds  where A = [[1, CL1^2], [1, CL2^2], ...]
    A = np.column_stack([np.ones_like(cls), cls**2])
    coeffs, _, _, _ = np.linalg.lstsq(A, cds, rcond=None)
    CD0_fit = max(coeffs[0], 0.0)
    k = max(coeffs[1], 0.01)  # enforce positive

    # VLM is inviscid — add parasitic drag estimate if CD0 is too small.
    # Use flat-plate turbulent skin friction: Cf ~ 0.455 / log10(Re)^2.58
    # Assume S_wet ~ 2.5 * S_ref for a typical RC aircraft.
    if CD0_fit < 0.005:
        Re = rho * speeds[1] * (S / (aircraft.main_wing().aspect_ratio() if aircraft.main_wing() else 6.0))**0.5 / (isa(altitude)[3])
        Cf = 0.455 / (math.log10(max(Re, 1e3)))**2.58
        S_wet_ratio = 2.5  # wetted area / reference area
        CD0_parasitic = Cf * S_wet_ratio
        CD0 = max(CD0_fit + CD0_parasitic, 0.005)
    else:
        CD0 = CD0_fit

    return DragPolar(CD0=CD0, k=k, S_ref=S)


def power_required(
    velocity: float,
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> float:
    """Power required for steady level flight [W].

    P_R = 0.5 * rho * V^3 * S * CD0 + 2 * k * W^2 / (rho * V * S)

    Parameters
    ----------
    velocity : float
        True airspeed [m/s].
    polar : DragPolar
        Fitted drag polar.
    mass : float
        Aircraft mass [kg].
    altitude : float
        Altitude [m MSL].

    Returns
    -------
    float
        Power required [W].
    """
    _, _, rho, _ = isa(altitude)
    W = mass * G
    S = polar.S_ref

    if velocity <= 0:
        return float("inf")

    # Parasitic + induced power
    P_parasitic = 0.5 * rho * S * polar.CD0 * velocity**3
    P_induced = 2.0 * polar.k * W**2 / (rho * S * velocity)

    return P_parasitic + P_induced


def power_available(
    propulsion: PropulsionSystem,
    velocity: float,
    altitude: float = 0.0,
    throttle: float = 1.0,
) -> float:
    """Power available from the propulsion system [W].

    Computes thrust * velocity at the motor-prop operating point.

    Parameters
    ----------
    propulsion : PropulsionSystem
        Propulsion system.
    velocity : float
        True airspeed [m/s].
    altitude : float
        Altitude [m MSL].
    throttle : float
        Throttle setting 0-1.

    Returns
    -------
    float
        Power available (thrust power) [W].
    """
    _, _, rho, _ = isa(altitude)

    # ESC voltage: throttle * battery voltage
    V_bat = propulsion.battery.voltage_under_load(0.0)  # approximate
    V_esc = throttle * V_bat

    # Motor no-load RPM at this voltage
    rpm_max = propulsion.motor.kv * V_esc
    if rpm_max <= 0:
        return 0.0

    # Find operating RPM: motor torque = prop torque
    # Use max current as upper bound
    max_current = min(
        propulsion.motor.max_current,
        propulsion.esc.max_current,
        propulsion.battery.max_current(),
    )
    V_bat_loaded = propulsion.battery.voltage_under_load(max_current)
    V_esc_loaded = throttle * V_bat_loaded
    rpm_at_max_I = propulsion.motor.rpm(V_esc_loaded, max_current)

    # Thrust at this operating point
    rpm = max(rpm_at_max_I, 0.0)
    thrust = propulsion.propeller.thrust(rpm, velocity, rho)

    return max(thrust * velocity, 0.0)


# --- Characteristic speeds ---


def stall_speed(
    mass: float,
    S: float,
    CL_max: float = 1.4,
    altitude: float = 0.0,
) -> float:
    """1-g stall speed [m/s].

    V_stall = sqrt(2W / (rho * S * CL_max))
    """
    _, _, rho, _ = isa(altitude)
    W = mass * G
    return math.sqrt(2.0 * W / (rho * S * CL_max))


def best_range_speed(
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> float:
    """Speed for maximum L/D (best range) [m/s].

    V* = sqrt(2W / (rho * S * CL*))  where CL* = sqrt(CD0 / k)
    """
    _, _, rho, _ = isa(altitude)
    W = mass * G
    cl_star = polar.cl_for_ld_max()
    return math.sqrt(2.0 * W / (rho * polar.S_ref * cl_star))


def best_endurance_speed(
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> float:
    """Speed for minimum power required (best endurance) [m/s].

    V_mp = 3^(-1/4) * V*  ~= 0.76 * V*
    """
    v_star = best_range_speed(polar, mass, altitude)
    return v_star * 3.0**(-0.25)


def max_level_speed(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    v_max_search: float = 60.0,
    dv: float = 0.5,
) -> float | None:
    """Maximum level flight speed [m/s].

    Scans speed range to find largest V where P_A >= P_R.
    Returns None if P_A < P_R everywhere (cannot fly).
    """
    v_max_found = None

    for v_int in range(int(5 / dv), int(v_max_search / dv)):
        v = v_int * dv
        pr = power_required(v, polar, mass, altitude)
        pa = power_available(propulsion, v, altitude)
        if pa >= pr:
            v_max_found = v

    return v_max_found


# --- Climb and glide ---


@dataclass
class GlidePerformance:
    """Glide performance summary."""
    best_glide_ratio: float    # L/D max
    best_glide_speed: float    # m/s
    min_sink_speed: float      # m/s (speed for minimum sink rate)
    min_sink_rate: float       # m/s (vertical speed at min sink speed)


def rate_of_climb(
    velocity: float,
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
) -> float:
    """Rate of climb at given speed [m/s].

    ROC = (P_A - P_R) / W
    """
    W = mass * G
    if W <= 0:
        return 0.0
    pr = power_required(velocity, polar, mass, altitude)
    pa = power_available(propulsion, velocity, altitude)
    return (pa - pr) / W


def max_rate_of_climb(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    v_min: float = 5.0,
    v_max: float = 40.0,
    dv: float = 0.5,
) -> tuple[float, float]:
    """Maximum rate of climb and speed for best climb [m/s].

    Returns (ROC_max, V_y).
    """
    best_roc = -float("inf")
    best_v = v_min

    for v_int in range(int(v_min / dv), int(v_max / dv)):
        v = v_int * dv
        roc = rate_of_climb(v, polar, mass, propulsion, altitude)
        if roc > best_roc:
            best_roc = roc
            best_v = v

    return best_roc, best_v


def glide_range(
    polar: DragPolar,
    from_altitude: float,
) -> float:
    """Still-air glide range from given altitude [m].

    R = (L/D)_max * altitude
    """
    return polar.ld_max() * from_altitude


def glide_performance(
    polar: DragPolar,
    mass: float,
    altitude: float = 0.0,
) -> GlidePerformance:
    """Compute glide performance summary."""
    ld_max = polar.ld_max()
    v_bg = best_range_speed(polar, mass, altitude)
    v_ms = best_endurance_speed(polar, mass, altitude)

    # Min sink rate: P_R_min / W (sink rate = D*V / W at V_mp)
    pr_min = power_required(v_ms, polar, mass, altitude)
    W = mass * G
    min_sink = pr_min / W if W > 0 else 0.0

    return GlidePerformance(
        best_glide_ratio=ld_max,
        best_glide_speed=v_bg,
        min_sink_speed=v_ms,
        min_sink_rate=min_sink,
    )


# --- Endurance and range ---


def max_endurance(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    eta_total: float | None = None,
) -> float:
    """Maximum endurance at best endurance speed [seconds].

    E = E_battery * eta / P_R_min

    Parameters
    ----------
    eta_total : float or None
        Total propulsive efficiency. If None, estimated from motor and prop.
    """
    v_mp = best_endurance_speed(polar, mass, altitude)
    pr_min = power_required(v_mp, polar, mass, altitude)

    if eta_total is None:
        eta_total = _estimate_efficiency(propulsion, v_mp, altitude)

    E_bat = propulsion.battery.energy()

    if pr_min <= 0:
        return float("inf")

    return E_bat * eta_total / pr_min


def max_range(
    polar: DragPolar,
    mass: float,
    propulsion: PropulsionSystem,
    altitude: float = 0.0,
    eta_total: float | None = None,
) -> float:
    """Maximum range at best range speed [meters].

    R = E_battery * eta * (L/D)_max / W
    """
    W = mass * G

    if eta_total is None:
        v_star = best_range_speed(polar, mass, altitude)
        eta_total = _estimate_efficiency(propulsion, v_star, altitude)

    E_bat = propulsion.battery.energy()
    ld_max = polar.ld_max()

    if W <= 0:
        return 0.0

    return E_bat * eta_total * ld_max / W


def _estimate_efficiency(
    propulsion: PropulsionSystem,
    velocity: float,
    altitude: float,
) -> float:
    """Estimate total propulsive efficiency (motor * prop).

    Uses the propulsion model at the given operating point.
    Falls back to 0.5 if computation fails.
    """
    _, _, rho, _ = isa(altitude)

    # Get operating RPM at max throttle
    V_bat = propulsion.battery.voltage_under_load(0.0)
    rpm = propulsion.motor.kv * V_bat

    if rpm <= 0 or velocity <= 0:
        return 0.5

    # Prop efficiency
    eta_prop = propulsion.propeller.efficiency(rpm, velocity, rho)

    # Motor efficiency (approximate at mid current)
    mid_current = 0.5 * propulsion.motor.max_current
    eta_motor = propulsion.motor.efficiency(V_bat, mid_current)

    eta_total = eta_prop * eta_motor

    # Clamp to reasonable range
    return max(min(eta_total, 0.85), 0.1) if eta_total > 0 else 0.5
