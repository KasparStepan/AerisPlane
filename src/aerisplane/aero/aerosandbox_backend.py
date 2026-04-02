"""Translate AerisPlane core objects to AeroSandbox equivalents.

Public API
----------
aircraft_to_asb(aircraft) -> asb.Airplane
condition_to_asb(condition) -> asb.OperatingPoint

The leading underscore variant ``_condition_to_asb`` is kept as an alias for
notebook compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aerisplane.core.aircraft import Aircraft
    from aerisplane.core.flight_condition import FlightCondition


def _wing_to_asb(wing):
    """Convert an AerisPlane Wing to an asb.Wing."""
    import aerosandbox as asb

    asb_xsecs = []
    for xsec in wing.xsecs:
        # Build ASB control surfaces that apply at this section
        asb_controls = []
        for cs in wing.control_surfaces:
            asb_controls.append(
                asb.ControlSurface(
                    name=cs.name,
                    symmetric=cs.symmetric,
                    deflection=0.0,  # deflections applied later via condition
                    hinge_point=1.0 - cs.chord_fraction,
                )
            )

        asb_xsecs.append(
            asb.WingXSec(
                xyz_le=list(xsec.xyz_le),
                chord=xsec.chord,
                twist=xsec.twist,
                airfoil=asb.Airfoil(name=xsec.airfoil.name),
                control_surfaces=asb_controls if asb_controls else None,
            )
        )

    return asb.Wing(
        name=wing.name,
        xsecs=asb_xsecs,
        symmetric=wing.symmetric,
    )


def _fuselage_xsec_to_asb(xsec, fuselage):
    """Convert a single FuselageXSec to an asb.FuselageXSec."""
    import aerosandbox as asb

    center = [fuselage.x_le + xsec.x, fuselage.y_le, fuselage.z_le]

    if xsec.shape == "ellipse" and xsec.width is not None and xsec.height is not None:
        return asb.FuselageXSec(
            xyz_c=center,
            width=xsec.width,
            height=xsec.height,
            shape=2.0,  # ellipse
        )
    elif xsec.shape == "rectangle" and xsec.width is not None and xsec.height is not None:
        return asb.FuselageXSec(
            xyz_c=center,
            width=xsec.width,
            height=xsec.height,
            shape=100.0,  # high exponent ≈ rectangle
        )
    else:
        # Circular: use radius for both width and height
        r = xsec.radius or 0.0
        return asb.FuselageXSec(
            xyz_c=center,
            width=2.0 * r,
            height=2.0 * r,
            shape=2.0,
        )


def _fuselage_to_asb(fuselage):
    """Convert an AerisPlane Fuselage to an asb.Fuselage."""
    import aerosandbox as asb

    return asb.Fuselage(
        name=fuselage.name,
        xsecs=[_fuselage_xsec_to_asb(xs, fuselage) for xs in fuselage.xsecs],
    )


def aircraft_to_asb(aircraft: Aircraft):
    """Convert an AerisPlane Aircraft to an ``aerosandbox.Airplane``.

    Parameters
    ----------
    aircraft : Aircraft
        AerisPlane aircraft definition.

    Returns
    -------
    asb.Airplane
    """
    import aerosandbox as asb

    return asb.Airplane(
        name=aircraft.name,
        xyz_ref=aircraft.xyz_ref,
        wings=[_wing_to_asb(w) for w in aircraft.wings],
        fuselages=[_fuselage_to_asb(f) for f in aircraft.fuselages],
        s_ref=aircraft.reference_area() or None,
        c_ref=aircraft.reference_chord() or None,
        b_ref=aircraft.reference_span() or None,
    )


def condition_to_asb(condition: FlightCondition):
    """Convert an AerisPlane FlightCondition to an ``aerosandbox.OperatingPoint``.

    Parameters
    ----------
    condition : FlightCondition

    Returns
    -------
    asb.OperatingPoint
    """
    import aerosandbox as asb

    return asb.OperatingPoint(
        atmosphere=asb.Atmosphere(altitude=condition.altitude),
        velocity=condition.velocity,
        alpha=condition.alpha,
        beta=getattr(condition, "beta", 0.0),
    )


# Alias for notebook compatibility
_condition_to_asb = condition_to_asb
