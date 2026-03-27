# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/aerodynamics/aero_3D/aero_buildup_submodels/
#   fuselage_aerodynamics_utilities.py
#   softmax_scalefree.py
"""Fuselage aerodynamics utilities used by AeroBuildup."""
from __future__ import annotations

from typing import List

import numpy as np
from aerisplane.aero._np_compat import blend, softmax


def critical_mach(fineness_ratio_nose: float) -> float:
    """Transonic critical Mach number for a streamlined fuselage.

    Fit to Raymer "Aircraft Design" 2nd Ed., Fig. 12.28.

    Args:
        fineness_ratio_nose: Nose fineness ratio = 2 * L_n / d, where L_n is
            the nose length to where the cross-section first becomes constant
            and d is the diameter there.

    Returns: Critical Mach number.
    """
    p = {"a": 11.087202397070559, "b": 13.469755774708842, "c": 4.034476257077558}
    mach_dd = 1 - (p["a"] / (2 * fineness_ratio_nose + p["b"])) ** p["c"]
    mach_crit = mach_dd - (0.1 / 80) ** (1 / 3)
    return mach_crit


def jorgensen_eta(fineness_ratio: float) -> float:
    """Crossflow lift multiplier eta for a fuselage (Jorgensen 1977, NASA TR R-474).

    Args:
        fineness_ratio: Fuselage fineness ratio (length / diameter).

    Returns: Eta parameter [-].
    """
    x = fineness_ratio
    p = {
        "1scl": 23.009059965179222,
        "1cen": -122.76900250914575,
        "2scl": 13.006453125841258,
        "2cen": -24.367562906887436,
    }
    return 1 - p["1scl"] / (x - p["1cen"]) - (p["2scl"] / (x - p["2cen"])) ** 2


def fuselage_base_drag_coefficient(mach: float) -> float:
    """Fuselage base drag coefficient vs Mach number.

    Fit to MIL-HDBK-762 Fig. 5-140.

    Args:
        mach: Mach number [-].

    Returns: Base drag coefficient.
    """
    m = mach
    p = {
        "a": 0.18024110740341143,
        "center_sup": -0.21737019935624047,
        "m_trans": 0.9985447737532848,
        "pc_sub": 0.15922582283573747,
        "pc_sup": 0.04698820458826384,
        "scale_sup": 0.34978926411193456,
        "trans_str": 9.999987483414937,
    }
    return blend(
        p["trans_str"] * (m - p["m_trans"]),
        p["pc_sup"] + p["a"] * np.exp(-((p["scale_sup"] * (m - p["center_sup"])) ** 2)),
        p["pc_sub"],
    )


def fuselage_form_factor(
    fineness_ratio: float,
    ratio_of_corner_radius_to_body_width: float = 0.5,
) -> float:
    """Form factor of a fuselage with rounded-square cross-section.

    From Götten et al. (2021), "Improved Form Factor for Drag Estimation of
    Fuselages with Various Cross Sections", AIAA J. Aircraft, DOI 10.2514/1.C036032.

    Args:
        fineness_ratio: length / diameter.
        ratio_of_corner_radius_to_body_width: 0 = square, 0.5 (default) = circle.

    Returns: Form factor F, used in C_D = C_f * F * (S_wet / S_ref).
    """
    fr = fineness_ratio
    r = 2 * ratio_of_corner_radius_to_body_width

    cs1 = -0.825885 * r**0.411795 + 4.0001
    cs2 = -0.340977 * r**7.54327 - 2.27920
    cs3 = -0.013846 * r**1.34253 + 1.11029

    return cs1 * fr**cs2 + cs3


def softmax_scalefree(x: List[float]) -> float:
    """Scale-free softmax: automatically sets softness to 1% of the largest value.

    Args:
        x: List of values to take the smooth maximum of.

    Returns: Smooth maximum.
    """
    if len(x) == 1:
        return x[0]
    softness = float(np.max(np.array([1e-6] + x))) * 0.01
    return softmax(*x, softness=softness)
