"""Plain trailing-edge flap aerodynamic corrections via Glauert thin-airfoil theory.

Reference: Glauert, "The Elements of Aerofoil and Airscrew Theory", Ch. VII.

The formulae give the 2-D lift and quarter-chord pitching-moment increments
due to a plain trailing-edge control surface deflected by angle δ:

    θ_f = arccos(1 − 2·c_f)
    ΔCl / δ  = 2(π − θ_f + sin θ_f)          [per rad]
    ΔCm_qc / δ = −(sin θ_f + sin(2θ_f) / 2) / 2   [per rad, about c/4]

where c_f = hinge-to-trailing-edge chord / total chord (same as ControlSurface.chord_fraction).

These increments are added on top of the baseline NeuralFoil polar for any
section covered by a deflected control surface.  The model is linear in δ and
does not saturate — the caller is responsible for passing physically sensible
deflection values (within ControlSurface.min_deflection … max_deflection).

Sign convention (matches VLM Rodrigues rotation):
  δ > 0  →  trailing edge deflects DOWN  →  ΔCl > 0  (more lift)
  δ > 0  →  ΔCm_qc < 0  (nose-down pitching moment for aft surface)
"""

from __future__ import annotations

import math


def plain_flap_delta(chord_fraction: float, deflection_deg: float) -> tuple[float, float]:
    """Thin-airfoil lift and moment increments for a plain trailing-edge flap.

    Parameters
    ----------
    chord_fraction : float
        Fraction of the local chord occupied by the control surface, measured
        from the hinge to the trailing edge.  Must be in (0, 1).
        Corresponds to ``ControlSurface.chord_fraction``.
    deflection_deg : float
        Control surface deflection [deg].
        Positive = trailing edge down.

    Returns
    -------
    delta_cl : float
        2-D lift coefficient increment ΔCl (additive to NeuralFoil CL).
    delta_cm_qc : float
        2-D quarter-chord pitching moment increment ΔCm (additive to NeuralFoil CM).
        Negative for positive (TE-down) deflection on an aft control surface.
    """
    if chord_fraction <= 0.0 or chord_fraction >= 1.0:
        return 0.0, 0.0
    if deflection_deg == 0.0:
        return 0.0, 0.0

    delta_rad = math.radians(deflection_deg)

    # Glauert flap angle: maps chord fraction to angular coordinate on unit circle
    theta_f = math.acos(1.0 - 2.0 * chord_fraction)

    # Lift increment per radian (Glauert Eq. 7.12)
    dCl_ddelta = 2.0 * (math.pi - theta_f + math.sin(theta_f))

    # Quarter-chord moment increment per radian (Glauert Eq. 7.14)
    dCm_ddelta = -(math.sin(theta_f) + math.sin(2.0 * theta_f) / 2.0) / 2.0

    delta_cl = dCl_ddelta * delta_rad
    delta_cm_qc = dCm_ddelta * delta_rad

    return delta_cl, delta_cm_qc


def section_cs_corrections(
    wing,
    deflections: dict,
    section_idx: int,
    is_mirrored: bool = False,
) -> tuple[float, float]:
    """Sum ΔCl and ΔCm contributions from all control surfaces covering a section.

    Iterates over ``wing.control_surfaces`` and applies ``plain_flap_delta``
    for each surface whose spanwise range overlaps the section midpoint.

    Parameters
    ----------
    wing : Wing
        Wing object (has ``control_surfaces`` list and ``semispan()`` method).
    deflections : dict
        Mapping of control surface name → deflection angle [deg].
        Typically ``FlightCondition.deflections``.
    section_idx : int
        Index of the section between ``wing.xsecs[section_idx]`` and
        ``wing.xsecs[section_idx + 1]``.
    is_mirrored : bool
        True for the reflected (left) side of a symmetric wing.
        Asymmetric control surfaces (e.g. ailerons) have their deflection
        negated on the mirrored side.

    Returns
    -------
    delta_cl : float
        Summed ΔCl from all active control surfaces on this section.
    delta_cm : float
        Summed ΔCm_qc from all active control surfaces on this section.
    """
    if not wing.control_surfaces or not deflections:
        return 0.0, 0.0

    semispan = wing.semispan()
    if semispan <= 0.0:
        return 0.0, 0.0

    xsec_a = wing.xsecs[section_idx]
    xsec_b = wing.xsecs[section_idx + 1]
    y_a = float(xsec_a.xyz_le[1])
    y_b = float(xsec_b.xyz_le[1])
    span_frac_mid = (abs(y_a) + abs(y_b)) / 2.0 / semispan

    total_dcl = 0.0
    total_dcm = 0.0

    for cs in wing.control_surfaces:
        deflection = deflections.get(cs.name, 0.0)
        if deflection == 0.0:
            continue
        if not (cs.span_start <= span_frac_mid <= cs.span_end):
            continue

        # Asymmetric surfaces (ailerons): negate deflection on mirrored side
        effective_deflection = deflection
        if is_mirrored and not cs.symmetric:
            effective_deflection = -deflection

        dcl, dcm = plain_flap_delta(cs.chord_fraction, effective_deflection)
        total_dcl += dcl
        total_dcm += dcm

    return total_dcl, total_dcm
