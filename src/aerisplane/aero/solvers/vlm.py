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


def _rodrigues(points: np.ndarray, axis: np.ndarray, angle_rad: float, anchor: np.ndarray) -> np.ndarray:
    """Rotate *points* around the line (anchor, axis) by angle_rad (Rodrigues formula).

    Parameters
    ----------
    points : (N, 3) or (3,) array
    axis   : (3,) unit-vector direction of rotation axis
    angle_rad : float
    anchor : (3,) any point on the rotation axis

    Returns
    -------
    Rotated points, same shape as input.
    """
    u = axis / np.linalg.norm(axis)
    p = points - anchor
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    if p.ndim == 1:
        return anchor + p * cos_a + np.cross(u, p) * sin_a + u * np.dot(u, p) * (1.0 - cos_a)
    dot = (p * u).sum(axis=1, keepdims=True)
    return anchor + p * cos_a + np.cross(u, p) * sin_a + u * dot * (1.0 - cos_a)


def _apply_control_deflections(
    front_left: np.ndarray,
    back_left: np.ndarray,
    back_right: np.ndarray,
    front_right: np.ndarray,
    wing_records: list,
    deflections: dict,
    chordwise_resolution: int,
) -> None:
    """Rotate trailing-edge panels for each deflected control surface in-place.

    Modifies *front_left*, *back_left*, *back_right*, *front_right* arrays
    directly.  The hinge line is the front edge of the first panel that is
    aft of (1 - chord_fraction) × chord.

    Sign convention (positive deflection = trailing edge DOWN):
    - symmetric surface (elevator, flap): both sides deflect TE-down.
    - asymmetric surface (aileron): right side TE-down, left side TE-up.
    """
    N = chordwise_resolution

    for rec in wing_records:
        wing = rec["wing"]
        if not wing.control_surfaces:
            continue

        panel_start = rec["panel_start"]
        n_strips = rec["n_strips"]
        is_symmetric = rec["is_symmetric"]
        y_root = rec["y_root"]
        y_tip = rec["y_tip"]
        span = y_tip - y_root
        if span <= 0.0:
            continue

        for cs in wing.control_surfaces:
            deflection = deflections.get(cs.name, 0.0)
            if deflection == 0.0:
                continue

            # First chordwise index that is aft of the hinge
            j_hinge = max(0, min(N - 1, int(np.floor(N * (1.0 - cs.chord_fraction)))))

            # Iterate over right side (always present) and optionally mirrored left side
            sides = [("right", False)]
            if is_symmetric:
                sides.append(("left", True))

            for _side_name, is_mirror in sides:
                # Sign of angle.
                # Both the right and left panel hinge directions point in roughly +y
                # (from inboard to outboard on the right side, from outboard to inboard
                # on the mirrored left side — both end up ≈ +y because the mirrored face
                # winding reverses left/right vertex labels).
                # Therefore: same angle sign → same physical TE direction on both sides.
                #   symmetric=True  (elevator, flap): same sign → both TE-down simultaneously
                #   symmetric=False (aileron):        negate for mirror → right TE-down, left TE-up
                if is_mirror:
                    angle_rad = np.radians(deflection) * (1.0 if cs.symmetric else -1.0)
                else:
                    angle_rad = np.radians(deflection)

                strip_offset = (0 if not is_mirror else n_strips) * N

                for s in range(n_strips):
                    # Panel index of the hinge-row panel in this strip
                    k0 = panel_start + strip_offset + s * N + j_hinge

                    # Span fraction: use midpoint along dominant span axis
                    span_ax = rec.get("span_axis", 1)
                    coord_mid = 0.5 * (float(front_left[k0, span_ax]) + float(front_right[k0, span_ax]))
                    span_frac = (abs(coord_mid) - y_root) / span

                    if not (cs.span_start <= span_frac <= cs.span_end):
                        continue

                    # Hinge line: from front_left[k0] to front_right[k0]
                    hl = front_left[k0].copy()
                    hinge_dir = front_right[k0] - front_left[k0]

                    # Rotate all panels at or aft of j_hinge within this strip
                    for j in range(j_hinge, N):
                        k = panel_start + strip_offset + s * N + j
                        # Back vertices always rotate (they are aft of hinge)
                        back_left[k] = _rodrigues(back_left[k], hinge_dir, angle_rad, hl)
                        back_right[k] = _rodrigues(back_right[k], hinge_dir, angle_rad, hl)
                        # Front vertices rotate only if strictly aft of hinge row
                        if j > j_hinge:
                            front_left[k] = _rodrigues(front_left[k], hinge_dir, angle_rad, hl)
                            front_right[k] = _rodrigues(front_right[k], hinge_dir, angle_rad, hl)


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
        Kaufmann vortex core radius for singularity smoothing.  A small
        non-zero value (default 1e-8) regularises the bound-leg self-influence
        at the vortex centre so that term1 → 0 instead of 0/0 = NaN, while
        leaving the trailing-vortex contributions (which drive induced drag)
        unaffected.
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
        vortex_core_radius: float = 1e-8,
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
        wing_records = []   # per-wing info needed for control surface deflections
        panel_offset = 0

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
            flv = points[faces[:, 0], :]
            blv = points[faces[:, 1], :]
            brv = points[faces[:, 2], :]
            frv = points[faces[:, 3], :]
            front_left_verts.append(flv)
            back_left_verts.append(blv)
            back_right_verts.append(brv)
            front_right_verts.append(frv)
            is_trailing_edge_list.append(
                (np.arange(len(faces)) + 1) % self.chordwise_resolution == 0
            )
            n_strips = len(w.xsecs) - 1
            ys = [xsec.xyz_le[1] for xsec in w.xsecs]
            zs = [xsec.xyz_le[2] for xsec in w.xsecs]
            use_z = (max(zs) - min(zs)) > (max(ys) - min(ys))
            span_coords = zs if use_z else ys
            span_axis = 2 if use_z else 1
            wing_records.append({
                "wing": wing,
                "panel_start": panel_offset,
                "n_panels": len(faces),
                "n_strips": n_strips,
                "is_symmetric": wing.symmetric,
                "span_axis": span_axis,
                "y_root": float(min(span_coords)),
                "y_tip": float(max(span_coords)),
            })
            panel_offset += len(faces)

        front_left_vertices = np.concatenate(front_left_verts)
        back_left_vertices = np.concatenate(back_left_verts)
        back_right_vertices = np.concatenate(back_right_verts)
        front_right_vertices = np.concatenate(front_right_verts)
        is_trailing_edge = np.concatenate(is_trailing_edge_list)

        # ── Control surface deflections ───────────────────────────────────
        if self.condition.deflections:
            _apply_control_deflections(
                front_left_vertices, back_left_vertices,
                back_right_vertices, front_right_vertices,
                wing_records, self.condition.deflections,
                self.chordwise_resolution,
            )

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
        self.wing_records = wing_records   # per-wing panel info for post-processing

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

    def get_induced_velocity_at_points(self, points: np.ndarray) -> np.ndarray:
        """Induced-only velocity (no freestream) at given points.

        Parameters
        ----------
        points : (N, 3) array

        Returns
        -------
        (N, 3) array of induced velocity vectors in geometry axes.
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
        # With vortex_core_radius > 0 the Kaufmann model evaluates to 0 at
        # the singularity, so no NaN/inf should appear.  nan_to_num is kept
        # as a defensive guard for degenerate panels (zero-area, etc.).
        u_induced = np.nan_to_num(u_induced, nan=0.0, posinf=0.0, neginf=0.0)
        v_induced = np.nan_to_num(v_induced, nan=0.0, posinf=0.0, neginf=0.0)
        w_induced = np.nan_to_num(w_induced, nan=0.0, posinf=0.0, neginf=0.0)
        return np.stack([
            np.sum(u_induced, axis=1),
            np.sum(v_induced, axis=1),
            np.sum(w_induced, axis=1),
        ], axis=1)

    def get_velocity_at_points(self, points: np.ndarray) -> np.ndarray:
        """Total velocity (freestream + induced) at given points.

        Parameters
        ----------
        points : (N, 3) array

        Returns
        -------
        (N, 3) array of velocity vectors in geometry axes.
        """
        V_induced = self.get_induced_velocity_at_points(points)
        V_freestream = self.steady_freestream_velocity.reshape(1, 3) + \
            self.condition.rotation_velocity_geometry_axes(points)
        return V_freestream + V_induced

    def calculate_streamlines(
        self,
        seed_points: np.ndarray | None = None,
        n_steps: int = 300,
        length: float | None = None,
    ) -> np.ndarray:
        """Trace streamlines from seed points using forward-Euler integration.

        Integrates the total velocity field (freestream + induced) with a fixed
        arc-length step.  Seeds default to the trailing-edge panel midpoints,
        which naturally captures the tip vortices and wake.

        Parameters
        ----------
        seed_points : (N, 3) array or None
            Starting points for streamlines.  Auto-generated from the trailing
            edge if not given.
        n_steps : int
            Number of integration steps.
        length : float or None
            Total arc length of each streamline [m].  Defaults to 5 × c_ref.

        Returns
        -------
        streamlines : (n_seeds, 3, n_steps) array
            Also stored as ``self.streamlines``.
        """
        if length is None:
            length = self.c_ref * 5
        if seed_points is None:
            te_mask = self.is_trailing_edge.astype(bool)
            left_te = self.back_left_vertices[te_mask]
            right_te = self.back_right_vertices[te_mask]
            n_target = 200
            per_panel = max(1, n_target // len(left_te))
            nodes = np.linspace(0, 1, per_panel + 1)
            mids = (nodes[1:] + nodes[:-1]) / 2
            seed_points = np.concatenate([
                t * left_te + (1 - t) * right_te for t in mids
            ])

        n_seeds = len(seed_points)
        streamlines = np.empty((n_seeds, 3, n_steps))
        streamlines[:, :, 0] = seed_points
        step_ds = length / n_steps
        for i in range(1, n_steps):
            V = self.get_velocity_at_points(streamlines[:, :, i - 1])
            speed = np.linalg.norm(V, axis=1, keepdims=True)
            speed = np.where(speed < 1e-10, 1e-10, speed)   # avoid div-by-zero
            streamlines[:, :, i] = streamlines[:, :, i - 1] + step_ds * V / speed

        self.streamlines = streamlines
        return streamlines

    def draw(
        self,
        c: np.ndarray | None = None,
        cmap: str | None = None,
        colorbar_label: str | None = None,
        show: bool = True,
        draw_streamlines: bool = True,
        recalculate_streamlines: bool = False,
        backend: str = "pyvista",
    ) -> object:
        """Draw the solved VLM: wing panels coloured by a scalar + 3-D streamlines.

        Must be called after ``run()``.

        Parameters
        ----------
        c : (N,) array or None
            Scalar field to colour the panels.  Defaults to ``vortex_strengths``.
        cmap : str or None
            Matplotlib colormap name.  Defaults to ``"viridis"``.
        colorbar_label : str or None
            Label for the colour bar.
        show : bool
            Display the figure immediately.
        draw_streamlines : bool
            Overlay 3-D streamlines.
        recalculate_streamlines : bool
            Force re-tracing even if ``self.streamlines`` already exists.
        backend : str
            ``"pyvista"`` (default, interactive 3-D window) or ``"plotly"``
            (inline-friendly HTML figure).

        Returns
        -------
        pyvista Plotter or plotly Figure, depending on backend.
        """
        if c is None:
            c = self.vortex_strengths
            colorbar_label = colorbar_label or "Vortex strength [m²/s]"
        if cmap is None:
            cmap = "viridis"

        if draw_streamlines and (recalculate_streamlines or not hasattr(self, "streamlines")):
            self.calculate_streamlines()

        if backend == "pyvista":
            return self._draw_pyvista(c, cmap, colorbar_label, show, draw_streamlines)
        elif backend == "plotly":
            return self._draw_plotly(c, cmap, colorbar_label, show, draw_streamlines)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'pyvista' or 'plotly'.")

    # ------------------------------------------------------------------ #
    # draw() back-ends
    # ------------------------------------------------------------------ #

    def _draw_pyvista(self, c, cmap, colorbar_label, show, draw_streamlines):
        import pyvista as pv

        fl = self.front_left_vertices
        bl = self.back_left_vertices
        br = self.back_right_vertices
        fr = self.front_right_vertices
        N = len(fl)

        # Build PolyData: vertices stacked in four blocks, one quad per panel
        points_pv = np.concatenate([fl, bl, br, fr])   # (4N, 3)
        r = np.arange(N)
        # pyvista face format: [4, i0, i1, i2, i3, 4, i0, ...]
        faces_pv = np.column_stack([
            np.full(N, 4, dtype=np.intp),
            r, r + N, r + 2 * N, r + 3 * N,
        ]).ravel()
        mesh = pv.PolyData(points_pv, faces_pv)

        plotter = pv.Plotter()
        plotter.title = "AerisPlane VLM"
        plotter.add_axes()
        plotter.show_grid(color="gray")

        scalar_bar_args = {}
        if colorbar_label:
            scalar_bar_args["title"] = colorbar_label
        plotter.add_mesh(
            mesh=mesh,
            scalars=c,
            cmap=cmap,
            show_edges=True,
            show_scalar_bar=True,
            scalar_bar_args=scalar_bar_args,
        )

        if draw_streamlines:
            for i in range(self.streamlines.shape[0]):
                pts = self.streamlines[i, :, :].T   # (n_steps, 3)
                plotter.add_mesh(
                    pv.Spline(pts),
                    color="#88AAFF",
                    opacity=0.6,
                    line_width=1,
                )

        if show:
            plotter.show()
        return plotter

    def _draw_plotly(self, c, cmap, colorbar_label, show, draw_streamlines):
        import plotly.graph_objects as go

        fl = self.front_left_vertices
        bl = self.back_left_vertices
        br = self.back_right_vertices
        fr = self.front_right_vertices
        N = len(fl)

        # Triangulate each quad into 2 triangles for Mesh3d
        all_v = np.vstack([fl, bl, br, fr])   # (4N, 3)
        r = np.arange(N)
        tri_i = np.concatenate([r,       r      ])
        tri_j = np.concatenate([r + N,   r + 2*N])
        tri_k = np.concatenate([r + 2*N, r + 3*N])
        tri_c = np.concatenate([c, c])

        vmax = float(np.percentile(np.abs(c), 98)) or 1.0
        wing_trace = go.Mesh3d(
            x=all_v[:, 0], y=all_v[:, 1], z=all_v[:, 2],
            i=tri_i, j=tri_j, k=tri_k,
            intensity=tri_c,
            colorscale=cmap,
            cmin=-vmax, cmid=0.0, cmax=vmax,
            colorbar=dict(title=dict(text=colorbar_label or ""), x=1.02),
            name="Wing",
            showscale=True,
            opacity=0.95,
            flatshading=True,
        )

        traces = [wing_trace]
        if draw_streamlines:
            for i in range(self.streamlines.shape[0]):
                pts = self.streamlines[i, :, :].T   # (n_steps, 3)
                pts = pts[::3]                       # thin for file size
                traces.append(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="lines",
                    line=dict(color="rgba(136, 170, 255, 0.55)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text="AerisPlane VLM", font=dict(color="white")),
            scene=dict(
                xaxis=dict(title="x [m]"),
                yaxis=dict(title="y [m]"),
                zaxis=dict(title="z [m]"),
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )

        if show:
            fig.show()
        return fig
