# src/aerisplane/structures/__init__.py
"""Structures discipline module — Euler-Bernoulli wing beam solver.

Public API
----------
analyze(aircraft, aero_result, weight_result, ...) -> StructureResult
"""
from __future__ import annotations

import numpy as np

from aerisplane.aero.result import AeroResult
from aerisplane.core.aircraft import Aircraft
from aerisplane.structures.beam import WingBeam
from aerisplane.structures.checks import (
    bending_margin,
    buckling_margin,
    divergence_speed,
    fits_in_airfoil,
    shear_margin,
)
from aerisplane.structures.loads import design_load_factor
from aerisplane.structures.result import StructureResult, WingStructureResult
from aerisplane.utils.atmosphere import isa
from aerisplane.weights.result import WeightResult


def analyze(
    aircraft: Aircraft,
    aero_result: AeroResult,
    weight_result: WeightResult,
    n_limit: float = 3.5,
    safety_factor: float = 1.5,
    gust_velocity: float = 9.0,
    stability_result=None,
    n_stations: int = 50,
) -> StructureResult:
    """Run structural beam analysis for all wings with a spar.

    Parameters
    ----------
    aircraft : Aircraft
    aero_result : AeroResult
        Aerodynamic result at the sizing condition (provides total lift L).
    weight_result : WeightResult
        Weight analysis result (provides total mass for wing loading).
    n_limit : float
        Limit maneuver load factor (default 3.5).
    safety_factor : float
        Safety factor applied to limit load factor (default 1.5 -> ultimate).
    gust_velocity : float
        Design gust velocity U_de [m/s] (default 9.0 per CS-VLA).
    stability_result : StabilityResult or None
        If provided, CL_alpha from stability analysis is used for gust n.
    n_stations : int
        Number of spanwise beam integration stations per wing (default 50).

    Returns
    -------
    StructureResult
        Per-wing structural results, margins, and deflections.
    """
    sizing_condition = _condition_from_aero_result(aero_result)

    n_design = design_load_factor(
        aircraft, sizing_condition, weight_result,
        stability_result=stability_result,
        n_limit=n_limit,
        safety_factor=safety_factor,
        gust_velocity=gust_velocity,
    )

    # Attribute total lift to wings by area fraction
    total_area = sum(w.area() for w in aircraft.wings) or 1.0
    total_lift = aero_result.L  # [N]

    # Air density at sizing condition (for divergence speed)
    _, _, rho, _ = isa(aero_result.altitude)

    wing_results = []
    for wing in aircraft.wings:
        # Skip wings with no spar at any cross-section
        if not any(xs.spar is not None for xs in wing.xsecs):
            continue

        # Lift attributed to this wing (proportional to planform area)
        wing_lift = total_lift * (wing.area() / total_area)

        beam = WingBeam(wing, n_stations=n_stations)
        beam_result = beam.solve(
            total_lift=wing_lift,
            load_factor=n_design,
            inertia_relief=True,
        )

        semispan = wing.semispan()
        tip_ratio = (beam_result.tip_deflection / semispan
                     if semispan > 0.0 else 0.0)

        # Checks at root (index 0 -- highest M and V)
        root_xsec = wing.xsecs[0]
        root_spar = root_xsec.spar
        root_airfoil = root_xsec.airfoil
        root_chord = root_xsec.chord

        M_root = beam_result.root_bending_moment
        V_root = beam_result.root_shear_force

        mos_bending = (bending_margin(root_spar, M_root)
                       if root_spar is not None else float("inf"))
        mos_shear = (shear_margin(root_spar, V_root)
                     if root_spar is not None else float("inf"))
        mos_buckling = (buckling_margin(root_spar, M_root)
                        if root_spar is not None else float("inf"))
        spar_ok = (
            fits_in_airfoil(root_airfoil, root_spar, root_chord)
            if root_spar is not None and root_airfoil is not None
            else True
        )

        # Torsional divergence
        GJ_root = float(beam.GJ[0])
        if stability_result is not None:
            cl_alpha_rad = float(stability_result.CL_alpha) * (180.0 / np.pi)
        else:
            cl_alpha_rad = 5.5
        e = (0.25 - root_spar.position) * root_chord if root_spar is not None else 0.0
        V_div = divergence_speed(
            GJ_root=GJ_root,
            cl_alpha_per_rad=cl_alpha_rad,
            wing_area=wing.area(),
            density=rho,
            e=e,
        )

        wing_results.append(WingStructureResult(
            wing_name=wing.name,
            y=beam_result.y,
            shear_force=beam_result.V,
            bending_moment=beam_result.M,
            deflection=beam_result.delta,
            tip_deflection=beam_result.tip_deflection,
            tip_deflection_ratio=tip_ratio,
            root_bending_moment=M_root,
            root_shear_force=V_root,
            bending_margin=mos_bending,
            shear_margin=mos_shear,
            buckling_margin=mos_buckling,
            spar_fits=spar_ok,
            divergence_speed=V_div,
            design_load_factor=n_design,
        ))

    return StructureResult(wings=wing_results, design_load_factor=n_design)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _condition_from_aero_result(aero_result: AeroResult):
    """Reconstruct a FlightCondition from AeroResult."""
    from aerisplane.core.flight_condition import FlightCondition
    return FlightCondition(
        velocity=aero_result.velocity,
        altitude=aero_result.altitude,
        alpha=aero_result.alpha,
    )
