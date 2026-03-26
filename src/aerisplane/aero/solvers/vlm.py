# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.py
"""Vortex Lattice Method solver operating directly on aerisplane core types.

Usage
-----
>>> from aerisplane.aero.solvers.vlm import VortexLatticeMethod
>>> vlm = VortexLatticeMethod(aircraft, condition)
>>> result = vlm.run()   # returns dict with CL, CD, CY, Cl, Cm, Cn, ...
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.utils.spacing import cosspace
from aerisplane.aero.singularities import calculate_induced_velocity_horseshoe


def _tall(a: np.ndarray) -> np.ndarray:
    return a.reshape(-1, 1)


def _wide(a: np.ndarray) -> np.ndarray:
    return a.reshape(1, -1)


class VortexLatticeMethod:
    """3-D Vortex Lattice Method for an aerisplane Aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        aerisplane aircraft definition.
    condition : FlightCondition
        Operating point.
    spanwise_resolution : int
        Sub-divisions per wing section (spanwise).
    chordwise_resolution : int
        Number of chordwise panels.
    spanwise_spacing_function : callable
        Spacing function for spanwise subdivision (default: numpy.linspace).
    chordwise_spacing_function : callable
        Spacing function for chordwise panels (default: cosspace).
    align_trailing_vortices_with_wind : bool
        If True, trailing vortex legs follow the freestream direction.
    vortex_core_radius : float
        Kaufmann vortex core radius for singularity smoothing.
    verbose : bool
        Print solver progress.
    """

    def __init__(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        spanwise_resolution: int = 8,
        chordwise_resolution: int = 4,
        spanwise_spacing_function: Callable | None = None,
        chordwise_spacing_function: Callable | None = None,
        align_trailing_vortices_with_wind: bool = True,
        vortex_core_radius: float = 0.0,
        verbose: bool = False,
    ):
        self.aircraft = aircraft
        self.condition = condition
        self.spanwise_resolution = spanwise_resolution
        self.chordwise_resolution = chordwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function or np.linspace
        self.chordwise_spacing_function = chordwise_spacing_function or cosspace
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind
        self.vortex_core_radius = vortex_core_radius
        self.verbose = verbose

        # Reference geometry
        self.s_ref = aircraft.reference_area()
        self.b_ref = aircraft.reference_span()
        self.c_ref = aircraft.reference_chord()

        self.xyz_ref = [0.0, 0.0, 0.0]

    def run(self) -> dict:
        """Execute the VLM solve and return aerodynamic forces/moments.

        Returns
        -------
        dict with keys: F_g, F_b, F_w, M_g, M_b, M_w,
                        L, D, Y, l_b, m_b, n_b,
                        CL, CD, CY, Cl, Cm, Cn
        """
        if self.verbose:
            print("Meshing...")

        # ── Build panel mesh ──────────────────────────────────────────────
        front_left_verts = []
        back_left_verts = []
        back_right_verts = []
        front_right_verts = []
        is_trailing_edge_list = []

        for wing in self.aircraft.wings:
            w = wing
            if self.spanwise_resolution > 1:
                w = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )
            points, faces = w.mesh_thin_surface(
                chordwise_resolution=self.chordwise_resolution,
                chordwise_spacing_function=self.chordwise_spacing_function,
                add_camber=True,
            )
            front_left_verts.append(points[faces[:, 0], :])
            back_left_verts.append(points[faces[:, 1], :])
            back_right_verts.append(points[faces[:, 2], :])
            front_right_verts.append(points[faces[:, 3], :])
            is_trailing_edge_list.append(
                (np.arange(len(faces)) + 1) % self.chordwise_resolution == 0
            )

        front_left_vertices = np.concatenate(front_left_verts)
        back_left_vertices = np.concatenate(back_left_verts)
        back_right_vertices = np.concatenate(back_right_verts)
        front_right_vertices = np.concatenate(front_right_verts)
        is_trailing_edge = np.concatenate(is_trailing_edge_list)

        # ── Panel geometry ────────────────────────────────────────────────
        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / cross_norm.reshape(-1, 1)
        areas = cross_norm / 2.0

        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2.0
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        collocation_points = 0.5 * (
            0.25 * front_left_vertices + 0.75 * back_left_vertices
        ) + 0.5 * (
            0.25 * front_right_vertices + 0.75 * back_right_vertices
        )

        # Save for later access
        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.is_trailing_edge = is_trailing_edge
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.collocation_points = collocation_points

        # ── Freestream ────────────────────────────────────────────────────
        if self.verbose:
            print("Calculating the freestream influence...")

        steady_freestream_velocity = self.condition.freestream_velocity_geometry_axes()
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(
            steady_freestream_velocity
        )
        rotation_freestream_velocities = self.condition.rotation_velocity_geometry_axes(
            collocation_points
        )
        freestream_velocities = (
            steady_freestream_velocity.reshape(1, 3) + rotation_freestream_velocities
        )
        freestream_influences = np.sum(freestream_velocities * normal_directions, axis=1)

        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        # ── AIC matrix ────────────────────────────────────────────────────
        if self.verbose:
            print("Calculating the collocation influence matrix...")

        u_col, v_col, w_col = calculate_induced_velocity_horseshoe(
            x_field=_tall(collocation_points[:, 0]),
            y_field=_tall(collocation_points[:, 1]),
            z_field=_tall(collocation_points[:, 2]),
            x_left=_wide(left_vortex_vertices[:, 0]),
            y_left=_wide(left_vortex_vertices[:, 1]),
            z_left=_wide(left_vortex_vertices[:, 2]),
            x_right=_wide(right_vortex_vertices[:, 0]),
            y_right=_wide(right_vortex_vertices[:, 1]),
            z_right=_wide(right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1.0, 0.0, 0.0])
            ),
            gamma=1.0,
            vortex_core_radius=self.vortex_core_radius,
        )

        AIC = (
            u_col * _tall(normal_directions[:, 0])
            + v_col * _tall(normal_directions[:, 1])
            + w_col * _tall(normal_directions[:, 2])
        )

        # ── Solve ─────────────────────────────────────────────────────────
        if self.verbose:
            print("Calculating vortex strengths...")

        self.vortex_strengths = np.linalg.solve(AIC, -freestream_influences)

        # ── Forces ────────────────────────────────────────────────────────
        if self.verbose:
            print("Calculating forces on each panel...")

        V_centers = self.get_velocity_at_points(vortex_centers)
        Vi_cross_li = np.cross(V_centers, vortex_bound_leg, axis=1)

        _, _, rho, _ = self.condition.atmosphere()
        forces_geometry = rho * Vi_cross_li * _tall(self.vortex_strengths)
        moments_geometry = np.cross(
            vortex_centers - np.array(self.xyz_ref).reshape(1, 3),
            forces_geometry,
        )

        force_geometry = np.sum(forces_geometry, axis=0)
        moment_geometry = np.sum(moments_geometry, axis=0)

        force_body = np.array(self.condition.convert_axes(
            force_geometry[0], force_geometry[1], force_geometry[2],
            from_axes="geometry", to_axes="body",
        ))
        force_wind = np.array(self.condition.convert_axes(
            force_body[0], force_body[1], force_body[2],
            from_axes="body", to_axes="wind",
        ))
        moment_body = np.array(self.condition.convert_axes(
            moment_geometry[0], moment_geometry[1], moment_geometry[2],
            from_axes="geometry", to_axes="body",
        ))
        moment_wind = np.array(self.condition.convert_axes(
            moment_body[0], moment_body[1], moment_body[2],
            from_axes="body", to_axes="wind",
        ))

        self.forces_geometry = forces_geometry
        self.moments_geometry = moments_geometry
        self.force_geometry = force_geometry
        self.force_body = force_body
        self.force_wind = force_wind
        self.moment_geometry = moment_geometry
        self.moment_body = moment_body
        self.moment_wind = moment_wind

        # ── Coefficients ──────────────────────────────────────────────────
        L = -force_wind[2]
        D = -force_wind[0]
        Y = force_wind[1]
        l_b = moment_body[0]
        m_b = moment_body[1]
        n_b = moment_body[2]

        q = self.condition.dynamic_pressure()
        CL = L / q / self.s_ref
        CD = D / q / self.s_ref
        CY = Y / q / self.s_ref
        Cl = l_b / q / self.s_ref / self.b_ref
        Cm = m_b / q / self.s_ref / self.c_ref
        Cn = n_b / q / self.s_ref / self.b_ref

        return {
            "F_g": force_geometry,
            "F_b": force_body,
            "F_w": force_wind,
            "M_g": moment_geometry,
            "M_b": moment_body,
            "M_w": moment_wind,
            "L": L, "D": D, "Y": Y,
            "l_b": l_b, "m_b": m_b, "n_b": n_b,
            "CL": CL, "CD": CD, "CY": CY,
            "Cl": Cl, "Cm": Cm, "Cn": Cn,
        }

    def get_velocity_at_points(self, points: np.ndarray) -> np.ndarray:
        """Total velocity (freestream + induced) at given points.

        Parameters
        ----------
        points : (N, 3) array

        Returns
        -------
        (N, 3) array of velocity vectors in geometry axes.
        """
        u_induced, v_induced, w_induced = calculate_induced_velocity_horseshoe(
            x_field=_tall(points[:, 0]),
            y_field=_tall(points[:, 1]),
            z_field=_tall(points[:, 2]),
            x_left=_wide(self.left_vortex_vertices[:, 0]),
            y_left=_wide(self.left_vortex_vertices[:, 1]),
            z_left=_wide(self.left_vortex_vertices[:, 2]),
            x_right=_wide(self.right_vortex_vertices[:, 0]),
            y_right=_wide(self.right_vortex_vertices[:, 1]),
            z_right=_wide(self.right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                self.steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1.0, 0.0, 0.0])
            ),
            gamma=_wide(self.vortex_strengths),
            vortex_core_radius=self.vortex_core_radius,
        )
        # When field points coincide with bound-leg endpoints (e.g. vortex
        # centers queried for their own panel), the Biot-Savart denominator
        # is zero producing inf; inf * 0 cross-product → NaN.  The physical
        # self-influence of a bound leg on a point along it is zero, so
        # replacing NaN with 0 is correct.
        u_induced = np.nan_to_num(u_induced, nan=0.0, posinf=0.0, neginf=0.0)
        v_induced = np.nan_to_num(v_induced, nan=0.0, posinf=0.0, neginf=0.0)
        w_induced = np.nan_to_num(w_induced, nan=0.0, posinf=0.0, neginf=0.0)
        u_induced_total = np.sum(u_induced, axis=1)
        v_induced_total = np.sum(v_induced, axis=1)
        w_induced_total = np.sum(w_induced, axis=1)

        V_induced = np.stack([u_induced_total, v_induced_total, w_induced_total], axis=1)
        V_freestream = self.steady_freestream_velocity.reshape(1, 3) + \
            self.condition.rotation_velocity_geometry_axes(points)

        return V_freestream + V_induced
