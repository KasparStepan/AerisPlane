# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/aerodynamics/aero_3D/singularities/
#   uniform_strength_horseshoe_singularities.py
#   point_source.py
"""Potential-flow singularity element functions for 3-D aerodynamic solvers.

All functions are pure NumPy — fully vectorised via broadcasting.
"""
from __future__ import annotations

import numpy as np


def calculate_induced_velocity_horseshoe(
    x_field: float | np.ndarray,
    y_field: float | np.ndarray,
    z_field: float | np.ndarray,
    x_left: float | np.ndarray,
    y_left: float | np.ndarray,
    z_left: float | np.ndarray,
    x_right: float | np.ndarray,
    y_right: float | np.ndarray,
    z_right: float | np.ndarray,
    gamma: float | np.ndarray = 1.0,
    trailing_vortex_direction: np.ndarray | None = None,
    vortex_core_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Induced velocity from a horseshoe vortex (bound leg + two trailing legs).

    Uses the Biot-Savart law with optional Kaufmann vortex core model to
    prevent singularities. Fully vectorised via NumPy broadcasting.

    Parameters
    ----------
    x/y/z_field : field point coordinates (can be broadcast against left/right)
    x/y/z_left  : left vertex of the bound vortex leg
    x/y/z_right : right vertex of the bound vortex leg
    gamma : vortex circulation strength
    trailing_vortex_direction : (3,) unit vector for trailing legs (default: [1,0,0])
    vortex_core_radius : Kaufmann vortex core radius (0 = point vortex)

    Returns
    -------
    u, v, w : induced velocity components
    """
    if trailing_vortex_direction is None:
        trailing_vortex_direction = np.array([1.0, 0.0, 0.0])

    a_x = x_field - x_left
    a_y = y_field - y_left
    a_z = z_field - z_left

    b_x = x_field - x_right
    b_y = y_field - y_right
    b_z = z_field - z_right

    u_x = trailing_vortex_direction[0]
    u_y = trailing_vortex_direction[1]
    u_z = trailing_vortex_direction[2]

    if vortex_core_radius != 0:
        def smoothed_inv(x):
            return x / (x**2 + vortex_core_radius**2)
    else:
        def smoothed_inv(x):
            return 1.0 / x

    # Cross and dot products
    a_cross_b_x = a_y * b_z - a_z * b_y
    a_cross_b_y = a_z * b_x - a_x * b_z
    a_cross_b_z = a_x * b_y - a_y * b_x
    a_dot_b = a_x * b_x + a_y * b_y + a_z * b_z

    a_cross_u_x = a_y * u_z - a_z * u_y
    a_cross_u_y = a_z * u_x - a_x * u_z
    a_cross_u_z = a_x * u_y - a_y * u_x
    a_dot_u = a_x * u_x + a_y * u_y + a_z * u_z

    b_cross_u_x = b_y * u_z - b_z * u_y
    b_cross_u_y = b_z * u_x - b_x * u_z
    b_cross_u_z = b_x * u_y - b_y * u_x
    b_dot_u = b_x * u_x + b_y * u_y + b_z * u_z

    norm_a = (a_x**2 + a_y**2 + a_z**2) ** 0.5
    norm_b = (b_x**2 + b_y**2 + b_z**2) ** 0.5
    norm_a_inv = smoothed_inv(norm_a)
    norm_b_inv = smoothed_inv(norm_b)

    term1 = (norm_a_inv + norm_b_inv) * smoothed_inv(norm_a * norm_b + a_dot_b)
    term2 = norm_a_inv * smoothed_inv(norm_a - a_dot_u)
    term3 = norm_b_inv * smoothed_inv(norm_b - b_dot_u)

    constant = gamma / (4 * np.pi)

    u = constant * (a_cross_b_x * term1 + a_cross_u_x * term2 - b_cross_u_x * term3)
    v = constant * (a_cross_b_y * term1 + a_cross_u_y * term2 - b_cross_u_y * term3)
    w = constant * (a_cross_b_z * term1 + a_cross_u_z * term2 - b_cross_u_z * term3)

    return u, v, w


def calculate_induced_velocity_point_source(
    x_field: float | np.ndarray,
    y_field: float | np.ndarray,
    z_field: float | np.ndarray,
    x_source: float | np.ndarray,
    y_source: float | np.ndarray,
    z_source: float | np.ndarray,
    sigma: float | np.ndarray = 1.0,
    viscous_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Induced velocity from a point source in 3-D potential flow.

    Parameters
    ----------
    x/y/z_field  : field point coordinates
    x/y/z_source : source point coordinates
    sigma : source strength
    viscous_radius : smoothing radius (0 = point source)

    Returns
    -------
    u, v, w : induced velocity components
    """
    dx = x_field - x_source
    dy = y_field - y_source
    dz = z_field - z_source

    r_squared = dx**2 + dy**2 + dz**2

    if viscous_radius != 0:
        def smoothed_x_15_inv(x):
            return x / (x**2.5 + viscous_radius**2.5)
    else:
        def smoothed_x_15_inv(x):
            return x**-1.5

    grad_phi = sigma * smoothed_x_15_inv(r_squared) / (4 * np.pi)

    return grad_phi * dx, grad_phi * dy, grad_phi * dz
