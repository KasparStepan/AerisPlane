"""Wing-body interference drag and carryover factor models.

References
----------
- Hoerner, *Fluid-Dynamic Drag*, Ch. 8 (junction drag).
- Schlichting & Truckenbrodt, *Aerodynamics of the Airplane*, Eq. 8.31.
- Multhopp, *Aerodynamics of the Fuselage in Combination with Wings* (1941).
"""
from __future__ import annotations

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition


# ---------------------------------------------------------------------------
# Primitive helpers (pure math, no geometry types)
# ---------------------------------------------------------------------------

def CDA_junction(
    wing_root_thickness: float,
    fuselage_radius: float,
    Re_root: float,
    fillet_radius: float = 0.0,
) -> float:
    """Junction drag area [m^2] (Hoerner Ch. 8).

    Parameters
    ----------
    wing_root_thickness : float
        Physical root thickness *t* [m].
    fuselage_radius : float
        Fuselage radius at the wing junction [m].
    Re_root : float
        Reynolds number based on root chord.
    fillet_radius : float
        Fillet radius at the junction [m].  Zero means no fillet.

    Returns
    -------
    float
        Drag area D/q [m^2].
    """
    t = wing_root_thickness
    if t <= 0.0:
        return 0.0

    x_junction = np.pi * fuselage_radius
    delta_star = 0.37 * x_junction / Re_root**0.2  # turbulent BL displacement thickness
    cda = 0.8 * t * delta_star

    if fillet_radius > 0.0:
        cda *= np.exp(-2.0 * fillet_radius / t)

    return float(cda)


def wing_carryover_lift_factor(
    fuselage_diameter: float,
    wing_span: float,
) -> float:
    """Lift carryover factor K_L (Schlichting & Truckenbrodt Eq. 8.31).

    Parameters
    ----------
    fuselage_diameter : float
        Fuselage diameter [m].
    wing_span : float
        Total wing span [m].

    Returns
    -------
    float
        Multiplicative factor on lift coefficient (>= 1).
    """
    if fuselage_diameter <= 0.0 or wing_span <= 0.0:
        return 1.0
    d_over_b = fuselage_diameter / wing_span
    return 1.0 + 0.41 * d_over_b**2


def wing_carryover_drag_factor(
    fuselage_diameter: float,
    wing_span: float,
) -> float:
    """Drag carryover factor K_D (Multhopp).

    Parameters
    ----------
    fuselage_diameter : float
        Fuselage diameter [m].
    wing_span : float
        Total wing span [m].

    Returns
    -------
    float
        Multiplicative factor on induced drag (>= 1).
    """
    if fuselage_diameter <= 0.0 or wing_span <= 0.0:
        return 1.0
    d_over_b = fuselage_diameter / wing_span
    denom = max(1.0 - d_over_b**2, 0.01)
    return 1.0 / denom


# ---------------------------------------------------------------------------
# Aircraft-level helper
# ---------------------------------------------------------------------------

def total_junction_drag(aircraft: Aircraft, condition: FlightCondition) -> float:
    """Total wing-fuselage junction drag force [N].

    Iterates over every (wing, fuselage) pair, interpolates the fuselage
    radius at the wing root x-position, and sums the junction drag
    contributions.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft configuration.
    condition : FlightCondition
        Flight condition (velocity, altitude).

    Returns
    -------
    float
        Junction drag force [N].
    """
    if not aircraft.fuselages or not aircraft.wings:
        return 0.0

    _, _, rho, mu = condition.atmosphere()
    V = condition.velocity
    q = condition.dynamic_pressure()

    total_drag = 0.0

    for wing in aircraft.wings:
        if len(wing.xsecs) < 1:
            continue

        root_xsec = wing.xsecs[0]
        root_x = root_xsec.xyz_le[0]  # x-position of wing root LE
        root_chord = root_xsec.chord

        # Root airfoil thickness ratio
        airfoil = root_xsec.airfoil
        if airfoil is not None:
            tc = airfoil.thickness()
        else:
            tc = 0.0
        if tc <= 0.0:
            tc = 0.12  # default

        root_thickness = tc * root_chord  # physical thickness [m]
        Re_root = rho * V * root_chord / mu

        n_junctions = 2 if wing.symmetric else 1

        for fuse in aircraft.fuselages:
            if len(fuse.xsecs) < 2:
                continue

            # Interpolate fuselage radius at wing root x-position
            # (x-positions are in fuselage-local coords, offset by fuse.x_le)
            fuse_x_stations = np.array([xs.x + fuse.x_le for xs in fuse.xsecs])
            fuse_radii = np.array([xs.equivalent_radius() for xs in fuse.xsecs])

            # Clamp to fuselage extent
            if root_x < fuse_x_stations[0] or root_x > fuse_x_stations[-1]:
                continue

            radius_at_root = float(np.interp(root_x, fuse_x_stations, fuse_radii))
            if radius_at_root <= 0.0:
                continue

            cda = CDA_junction(
                wing_root_thickness=root_thickness,
                fuselage_radius=radius_at_root,
                Re_root=Re_root,
            )
            total_drag += cda * q * n_junctions

    return float(total_drag)
