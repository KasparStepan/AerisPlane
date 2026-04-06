# src/aerisplane/structures/loads.py
"""Design load factor computation for RC/UAV structural sizing.

Computes limit and ultimate load factors from both the pilot-commanded
maneuver case and the CS-VLA gust case.  The design load factor is the
maximum of both, multiplied by the safety factor.
"""
from __future__ import annotations

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult

_G = 9.81  # m/s²


def maneuver_load_factor(n_limit: float = 3.5) -> float:
    """Pilot-commanded limit load factor.

    Parameters
    ----------
    n_limit : float
        Limit load factor (default 3.5 — aerobatic RC per CS-VLA).

    Returns
    -------
    float
    """
    return float(n_limit)


def gust_load_factor(
    velocity: float,
    altitude: float,
    cl_alpha_per_rad: float,
    wing_loading: float,
    U_de: float = 9.0,
) -> float:
    """Load factor increment from a discrete vertical gust (CS-VLA).

    .. math::
        \\Delta n = \\frac{\\rho V U_{de} a}{2 (W/S)}

    Parameters
    ----------
    velocity : float
        Airspeed [m/s].
    altitude : float
        Altitude [m].
    cl_alpha_per_rad : float
        Lift curve slope dCL/dα [1/rad].
    wing_loading : float
        Wing loading W/S [Pa].
    U_de : float
        Design gust velocity [m/s]. Default 9 m/s (CS-VLA cruise gust).

    Returns
    -------
    float
        Total load factor n = 1 + Δn.
    """
    _, _, rho, _ = isa(altitude)
    delta_n = rho * velocity * U_de * cl_alpha_per_rad / (2.0 * wing_loading)
    return 1.0 + float(delta_n)


def design_load_factor(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    stability_result=None,
    n_limit: float = 3.5,
    safety_factor: float = 1.5,
    gust_velocity: float = 9.0,
) -> float:
    """Ultimate design load factor: max(maneuver, gust) × safety_factor.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition
        Sizing flight condition (typically max-speed cruise).
    weight_result : WeightResult
        Provides total mass for wing loading.
    stability_result : StabilityResult or None
        If provided, uses CL_alpha from stability analysis [1/deg].
        If None, uses the default CL_alpha ≈ 5.5 1/rad.
    n_limit : float
        Limit maneuver load factor (default 3.5).
    safety_factor : float
        Applied to the limit load factor (default 1.5 → ultimate).
    gust_velocity : float
        Design gust velocity U_de [m/s] (default 9.0 per CS-VLA).

    Returns
    -------
    float
        Ultimate design load factor.
    """
    S = aircraft.reference_area() or 1.0
    W = weight_result.total_mass * _G
    wing_loading = W / S

    if stability_result is not None:
        # CL_alpha is stored in [1/deg] — convert to 1/rad
        cl_alpha_rad = float(stability_result.CL_alpha) * (180.0 / np.pi)
    else:
        cl_alpha_rad = 5.5  # typical for moderate AR wing

    n_maneuver = maneuver_load_factor(n_limit)
    n_gust = gust_load_factor(
        velocity=condition.velocity,
        altitude=condition.altitude,
        cl_alpha_per_rad=cl_alpha_rad,
        wing_loading=wing_loading,
        U_de=gust_velocity,
    )
    return float(max(n_maneuver, n_gust) * safety_factor)
