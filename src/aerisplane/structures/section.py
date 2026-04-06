# src/aerisplane/structures/section.py
"""Wing cross-section structural properties derived from airfoil geometry.

Provides effective bending stiffness EI accounting for both the spar tube
and the surface skin, using the transformed-section (homogenisation) method.
"""
from __future__ import annotations

import numpy as np

from aerisplane.core.airfoil import Airfoil
from aerisplane.core.structures import Skin, Spar


def airfoil_spar_height(
    airfoil: Airfoil,
    spar_position: float,
    chord: float,
) -> float:
    """Physical height available for a spar at the given chordwise position [m].

    Height = (y_upper − y_lower) at x = spar_position, scaled by chord.

    Parameters
    ----------
    airfoil : Airfoil
    spar_position : float
        Chordwise position as fraction of chord [0..1].
    chord : float
        Physical chord length [m].

    Returns
    -------
    float
        Available spar diameter [m].
    """
    upper = airfoil.upper_coordinates()  # LE→TE, normalised
    lower = airfoil.lower_coordinates()  # LE→TE, normalised
    y_up = float(np.interp(spar_position, upper[:, 0], upper[:, 1]))
    y_lo = float(np.interp(spar_position, lower[:, 0], lower[:, 1]))
    return (y_up - y_lo) * chord


def skin_second_moment_of_area(
    airfoil: Airfoil,
    chord: float,
    skin_thickness: float,
) -> float:
    """Second moment of area of the skin about the chord line [m^4].

    Uses the contour integral:  I_skin = t · ∫ y(s)² ds
    where s is the arc-length parameter along the airfoil contour.
    The neutral axis is approximated as the chord line (y = 0 on the
    normalised airfoil).  This is exact for symmetric airfoils and
    introduces only a small error for cambered sections.

    Parameters
    ----------
    airfoil : Airfoil
    chord : float
        Physical chord length [m].
    skin_thickness : float
        Skin wall thickness t [m].

    Returns
    -------
    float
    """
    if skin_thickness == 0.0:
        return 0.0
    if airfoil.coordinates is None:
        return 0.0
    xs = airfoil.coordinates[:, 0] * chord
    ys = airfoil.coordinates[:, 1] * chord
    ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    y_mid = 0.5 * (ys[:-1] + ys[1:])
    return float(skin_thickness * np.sum(y_mid ** 2 * ds))


def effective_EI(
    airfoil: Airfoil,
    chord: float,
    spar: Spar,
    skin: Skin | None = None,
) -> float:
    """Effective bending stiffness EI_eff [N·m²] of the wing cross-section.

    Uses the transformed-section (Voigt) method to combine spar and skin:

        EI_eff = E_spar · I_spar + E_skin · I_skin

    Parameters
    ----------
    airfoil : Airfoil
    chord : float
        Physical chord [m].
    spar : Spar
    skin : Skin or None
        If None, only the spar contribution is included.

    Returns
    -------
    float
    """
    EI = spar.material.E * spar.section.second_moment_of_area()
    if skin is not None:
        I_skin = skin_second_moment_of_area(airfoil, chord, skin.thickness)
        EI += skin.material.E * I_skin
    return float(EI)


def spar_fits_in_airfoil(airfoil: Airfoil, spar: Spar, chord: float) -> bool:
    """True if the spar outer diameter fits within the airfoil at the spar position.

    Parameters
    ----------
    airfoil : Airfoil
    spar : Spar
    chord : float
        Physical chord [m].

    Returns
    -------
    bool
    """
    h = airfoil_spar_height(airfoil, spar.position, chord)
    return spar.section.outer_diameter <= h
