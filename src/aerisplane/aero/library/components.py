# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/library/aerodynamics/components.py
"""Component drag area models: control linkages, gaps, bolts, joints."""
from __future__ import annotations

import numpy as np
from typing import Literal


def CDA_control_linkage(
    Re_l: float | np.ndarray,
    linkage_length: float | np.ndarray,
    is_covered: bool | np.ndarray = False,
    is_top: bool | np.ndarray = False,
) -> float | np.ndarray:
    """Drag area [m^2] of a typical RC-airplane control linkage.

    Data from Hepperle, "Drag of Linkages", citing Würz (1989).

    Args:
        Re_l: Reynolds number with linkage length as reference.
        linkage_length: Linkage length [m].
        is_covered: True if an aerodynamic fairing surrounds the linkage.
        is_top: True if the linkage is on the top wing surface.

    Returns: Drag area D/q [m^2].
    """
    p = {
        "CD0": 7.833083680086374e-05,
        "CD1": 0.0001216877860785463,
        "c_length": 30.572471745477774,
        "covered_drag_ratio": 0.7520722978405192,
        "top_drag_ratio": 1.1139040832208857,
    }

    side_drag_multiplier = np.where(is_top, p["top_drag_ratio"], 1)
    covered_drag_multiplier = np.where(is_covered, p["covered_drag_ratio"], 1)
    linkage_length_multiplier = 1 + p["c_length"] * linkage_length
    CDA_raw = p["CD1"] / (Re_l / 1e5) + p["CD0"]

    return side_drag_multiplier * covered_drag_multiplier * linkage_length_multiplier * CDA_raw


def CDA_control_surface_gaps(
    local_chord: float,
    control_surface_span: float,
    local_thickness_over_chord: float = 0.12,
    control_surface_hinge_x: float = 0.75,
    n_side_gaps: int = 2,
    side_gap_width: float | None = None,
    hinge_gap_width: float | None = None,
) -> float:
    """Drag area [m^2] of control surface hinge and side gaps (Hoerner 1965).

    Args:
        local_chord: Local chord at control surface midpoint [m].
        control_surface_span: Span of the control surface [m].
        local_thickness_over_chord: Local t/c ratio [-].
        control_surface_hinge_x: Hinge-line x/c location [-].
        n_side_gaps: Number of chordwise side gaps (typically 2).
        side_gap_width: Width of side gaps [m]; computed if None.
        hinge_gap_width: Width of spanwise hinge gap [m]; computed if None.

    Returns: Drag area D/q [m^2].
    """
    if side_gap_width is None:
        side_gap_width = np.maximum(
            np.maximum(0.002, 0.006 * local_chord), control_surface_span * 0.01
        )
    if hinge_gap_width is None:
        hinge_gap_width = 0.03 * local_chord

    CDA_side_gaps = n_side_gaps * (side_gap_width * local_chord * local_thickness_over_chord) * 0.50
    CDA_hinge_gap = 0.025 * hinge_gap_width * control_surface_span

    return CDA_side_gaps + CDA_hinge_gap


def CDA_protruding_bolt_or_rivet(diameter: float, kind: str = "flush_rivet") -> float:
    """Drag area [m^2] of a single bolt or rivet.

    Args:
        diameter: Bolt/rivet diameter [m].
        kind: One of "flush_rivet", "round_rivet", "flat_head_bolt",
              "round_head_bolt", "cylindrical_bolt", "hex_bolt".

    Returns: Drag area D/q [m^2].
    """
    S_ref = np.pi * diameter**2 / 4
    CD_factors = {
        "flush_rivet": 0.002,
        "round_rivet": 0.04,
        "flat_head_bolt": 0.02,
        "round_head_bolt": 0.32,
        "cylindrical_bolt": 0.42,
        "hex_bolt": 0.80,
    }
    try:
        return CD_factors[kind] * S_ref
    except KeyError:
        raise ValueError(f"Invalid `kind` {kind!r}.")


def CDA_perpendicular_sheet_metal_joint(
    joint_width: float,
    sheet_metal_thickness: float,
    kind: Literal[
        "butt_joint_with_inside_joiner",
        "butt_joint_with_inside_weld",
        "butt_joint_with_outside_joiner",
        "butt_joint_with_outside_weld",
        "lap_joint_forward_facing_step",
        "lap_joint_backward_facing_step",
    ] = "butt_joint_with_inside_joiner",
) -> float:
    """Drag area [m^2] of a sheet metal joint perpendicular to the flow.

    Args:
        joint_width: Joint width perpendicular to airflow [m].
        sheet_metal_thickness: Sheet metal thickness [m].
        kind: Joint type string.

    Returns: Drag area D/q [m^2].
    """
    S_ref = joint_width * sheet_metal_thickness
    CD_factors = {
        "butt_joint_with_inside_joiner": 0.01,
        "butt_joint_with_inside_weld": 0.01,
        "butt_joint_with_outside_joiner": 0.70,
        "butt_joint_with_outside_weld": 0.51,
        "lap_joint_forward_facing_step": 0.40,
        "lap_joint_backward_facing_step": 0.22,
        "lap_joint_forward_facing_step_with_bevel": 0.11,
        "lap_joint_backward_facing_step_with_bevel": 0.24,
        "lap_joint_forward_facing_step_with_rounded_bevel": 0.04,
        "lap_joint_backward_facing_step_with_rounded_bevel": 0.16,
        "flush_lap_joint_forward_facing_step": 0.13,
        "flush_lap_joint_backward_facing_step": 0.07,
    }
    try:
        return CD_factors[kind] * S_ref
    except KeyError:
        raise ValueError(f"Invalid `kind` {kind!r}.")
