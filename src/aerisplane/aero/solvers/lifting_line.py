# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/aerodynamics/aero_3D/lifting_line.py
"""Lifting-line aerodynamics solver with NeuralFoil section polars.

Nonlinear, includes viscous effects. Uses the same horseshoe-vortex AIC matrix
as the VLM, but replaces the no-camber boundary condition with a section CL
pre-computed from NeuralFoil.

Usage
-----
>>> from aerisplane.aero.solvers.lifting_line import LiftingLine
>>> ll = LiftingLine(aircraft, condition)
>>> result = ll.run()   # dict with CL, CD, Cm, ...
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.aero.singularities import (
    calculate_induced_velocity_horseshoe,
    calculate_induced_velocity_point_source,
)
from aerisplane.aero._np_compat import arccosd
from aerisplane.aero.library.control_surface_effects import section_cs_corrections
from aerisplane.utils.spacing import cosspace


# ---------------------------------------------------------------------------
# Broadcasting helpers
# ---------------------------------------------------------------------------

def _tall(array: np.ndarray) -> np.ndarray:
    """Reshape to column vector (N, 1)."""
    return np.reshape(array, (-1, 1))


def _wide(array: np.ndarray) -> np.ndarray:
    """Reshape to row vector (1, N)."""
    return np.reshape(array, (1, -1))


# ---------------------------------------------------------------------------
# LiftingLine solver
# ---------------------------------------------------------------------------

class LiftingLine:
    """Lifting-line aerodynamics with NeuralFoil section polars.

    Nonlinear and viscous.  For each panel the effective angle of attack
    (geometric + induced) is fed to NeuralFoil to obtain CL, CD, and CM.
    The horseshoe-vortex AIC system is solved iteratively via a single linear
    system (linearised CLa = 2π per section).

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft configuration.
    condition : FlightCondition
        Operating point.
    xyz_ref : list of float or None
        Moment reference point [x, y, z] [m].  Defaults to aircraft.xyz_ref.
    model_size : str
        NeuralFoil model size: "xxsmall" … "xxlarge" (default "medium").
    run_symmetric_if_possible : bool
        Not yet implemented.
    verbose : bool
        Print progress messages.
    spanwise_resolution : int
        Number of times each section is subdivided spanwise (default 4).
    spanwise_spacing_function : callable
        Spacing function ``f(start, stop, num) → array``.  Defaults to cosine.
    vortex_core_radius : float
        Kaufmann vortex core radius [m] for regularising singularities.
    align_trailing_vortices_with_wind : bool
        If True, trailing vortices are aligned with the freestream direction
        instead of the x-axis (wake alignment).
    """

    def __init__(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        xyz_ref: List[float] = None,
        model_size: str = "medium",
        run_symmetric_if_possible: bool = False,
        verbose: bool = False,
        spanwise_resolution: int = 4,
        spanwise_spacing_function: Callable = cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
    ):
        if xyz_ref is None:
            xyz_ref = aircraft.xyz_ref

        self.aircraft = aircraft
        self.condition = condition
        self.xyz_ref = xyz_ref
        self.model_size = model_size
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind

        if run_symmetric_if_possible:
            raise NotImplementedError(
                "LiftingLine with symmetry detection is not yet implemented."
            )

    # ------------------------------------------------------------------
    # Inner result dataclass
    # ------------------------------------------------------------------

    @dataclass
    class AeroComponentResults:
        """Forces and moments from a single aero component.

        Parameters
        ----------
        s_ref, c_ref, b_ref : float
            Reference area [m²], chord [m], and span [m].
        condition : FlightCondition
            Operating point (needed for axis conversions).
        F_g : list of float
            [x, y, z] forces in geometry axes [N].
        M_g : list of float
            [x, y, z] moments in geometry axes [N·m].
        """

        s_ref: float
        c_ref: float
        b_ref: float
        condition: FlightCondition
        F_g: List[Union[float, np.ndarray]]
        M_g: List[Union[float, np.ndarray]]

        @property
        def F_b(self) -> List[Union[float, np.ndarray]]:
            """Forces in body axes [N]."""
            return self.condition.convert_axes(
                *self.F_g, from_axes="geometry", to_axes="body"
            )

        @property
        def F_w(self) -> List[Union[float, np.ndarray]]:
            """Forces in wind axes [N]."""
            return self.condition.convert_axes(
                *self.F_g, from_axes="geometry", to_axes="wind"
            )

        @property
        def M_b(self) -> List[Union[float, np.ndarray]]:
            """Moments in body axes [N·m]."""
            return self.condition.convert_axes(
                *self.M_g, from_axes="geometry", to_axes="body"
            )

        @property
        def M_w(self) -> List[Union[float, np.ndarray]]:
            """Moments in wind axes [N·m]."""
            return self.condition.convert_axes(
                *self.M_g, from_axes="geometry", to_axes="wind"
            )

        @property
        def L(self) -> Union[float, np.ndarray]:
            """Lift force [N] (wind axes)."""
            return -self.F_w[2]

        @property
        def Y(self) -> Union[float, np.ndarray]:
            """Side force [N] (wind axes)."""
            return self.F_w[1]

        @property
        def D(self) -> Union[float, np.ndarray]:
            """Drag force [N] (wind axes)."""
            return -self.F_w[0]

        @property
        def l_b(self) -> Union[float, np.ndarray]:
            """Rolling moment [N·m] (body axes, positive roll-right)."""
            return self.M_b[0]

        @property
        def m_b(self) -> Union[float, np.ndarray]:
            """Pitching moment [N·m] (body axes, positive nose-up)."""
            return self.M_b[1]

        @property
        def n_b(self) -> Union[float, np.ndarray]:
            """Yawing moment [N·m] (body axes, positive nose-right)."""
            return self.M_b[2]

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Compute aerodynamic forces and moments.

        Returns
        -------
        dict
            Keys: F_g, F_b, F_w, M_g, M_b, M_w, L, Y, D,
                  l_b, m_b, n_b, CL, CY, CD, Cl, Cm, Cn,
                  wing_aero, fuselage_aero_components.
        """
        # Lazy import to avoid circular at module load time
        from aerisplane.aero.solvers.aero_buildup import AeroBuildup

        wing_aero = self.wing_aerodynamics()

        # Fuselage contributions via AeroBuildup's Jorgensen model
        aerobuildup = AeroBuildup(
            aircraft=self.aircraft,
            condition=self.condition,
            xyz_ref=self.xyz_ref,
        )
        fuselage_aero_components = [
            aerobuildup.fuselage_aerodynamics(fuselage=fuse, include_induced_drag=True)
            for fuse in self.aircraft.fuselages
        ]

        aero_components = [wing_aero] + fuselage_aero_components

        # Sum forces and moments across all components
        F_g_total = [
            sum(comp.F_g[i] for comp in aero_components) for i in range(3)
        ]
        M_g_total = [
            sum(comp.M_g[i] for comp in aero_components) for i in range(3)
        ]

        # Wing-body junction interference drag
        from aerisplane.aero.library.interference import total_junction_drag
        D_junction = total_junction_drag(self.aircraft, self.condition)
        if D_junction > 0:
            D_junc_g = self.condition.convert_axes(
                -D_junction, 0, 0, from_axes="wind", to_axes="geometry"
            )
            for i in range(3):
                F_g_total[i] += D_junc_g[i]

        output: Dict = {
            "F_g": F_g_total,
            "M_g": M_g_total,
        }

        output["F_b"] = self.condition.convert_axes(
            *F_g_total, from_axes="geometry", to_axes="body"
        )
        output["F_w"] = self.condition.convert_axes(
            *F_g_total, from_axes="geometry", to_axes="wind"
        )
        output["M_b"] = self.condition.convert_axes(
            *M_g_total, from_axes="geometry", to_axes="body"
        )
        output["M_w"] = self.condition.convert_axes(
            *M_g_total, from_axes="geometry", to_axes="wind"
        )

        output["L"] = -output["F_w"][2]
        output["Y"] = output["F_w"][1]
        output["D"] = -output["F_w"][0]
        output["l_b"] = output["M_b"][0]
        output["m_b"] = output["M_b"][1]
        output["n_b"] = output["M_b"][2]

        # Nondimensionalise
        qS = self.condition.dynamic_pressure() * self.aircraft.reference_area()
        c = self.aircraft.reference_chord()
        b = self.aircraft.reference_span()

        output["CL"] = output["L"] / qS
        output["CY"] = output["Y"] / qS
        output["CD"] = output["D"] / qS
        output["Cl"] = output["l_b"] / qS / b
        output["Cm"] = output["m_b"] / qS / c
        output["Cn"] = output["n_b"] / qS / b

        output["wing_aero"] = wing_aero
        output["fuselage_aero_components"] = fuselage_aero_components

        return output

    # ------------------------------------------------------------------
    # run_with_stability_derivatives()
    # ------------------------------------------------------------------

    def run_with_stability_derivatives(
        self,
        alpha: bool = True,
        beta: bool = True,
        p: bool = True,
        q: bool = True,
        r: bool = True,
    ) -> Dict:
        """Run with central-finite-difference stability derivatives.

        Uses two perturbed solves per derivative (±delta) for O(Δ²) accuracy.

        Parameters
        ----------
        alpha, beta, p, q, r : bool
            Which derivatives to compute (default all True).

        Returns
        -------
        dict
            All keys from ``run()``, plus e.g. "CLa", "CDa", "Cma",
            "x_np", etc.
        """
        import dataclasses

        b_ref = self.aircraft.reference_span()
        c_ref = self.aircraft.reference_chord()
        V = self.condition.velocity

        do_analysis = {"alpha": alpha, "beta": beta, "p": p, "q": q, "r": r}
        abbreviations = {"alpha": "a", "beta": "b", "p": "p", "q": "q", "r": "r"}
        # Step size in the perturbed variable (radians or rad/s)
        finite_difference_amounts = {
            "alpha": 0.001,
            "beta": 0.001,
            "p": 0.001 * (2 * V) / b_ref,
            "q": 0.001 * (2 * V) / c_ref,
            "r": 0.001 * (2 * V) / b_ref,
        }
        # Scale derivative from (per rad) to (per deg) or nondimensional
        scaling_factors = {
            "alpha": np.degrees(1),
            "beta": np.degrees(1),
            "p": (2 * V) / b_ref,
            "q": (2 * V) / c_ref,
            "r": (2 * V) / b_ref,
        }

        run_base = self.run()

        for d, do in do_analysis.items():
            if not do:
                continue

            delta = finite_difference_amounts[d]

            def _perturbed_run(sign: float, var: str) -> Dict:
                cond = self.condition.copy()
                amt = sign * finite_difference_amounts[var]
                if var == "alpha":
                    cond = dataclasses.replace(cond, alpha=cond.alpha + amt)
                elif var == "beta":
                    cond = dataclasses.replace(cond, beta=cond.beta + amt)
                elif var == "p":
                    cond = dataclasses.replace(cond, p=cond.p + amt)
                elif var == "q":
                    cond = dataclasses.replace(cond, q=cond.q + amt)
                elif var == "r":
                    cond = dataclasses.replace(cond, r=cond.r + amt)
                ll = LiftingLine(
                    aircraft=self.aircraft,
                    condition=cond,
                    xyz_ref=self.xyz_ref,
                    model_size=self.model_size,
                    spanwise_resolution=self.spanwise_resolution,
                    spanwise_spacing_function=self.spanwise_spacing_function,
                    vortex_core_radius=self.vortex_core_radius,
                    align_trailing_vortices_with_wind=self.align_trailing_vortices_with_wind,
                )
                return ll.run()

            run_pos = _perturbed_run(+1.0, d)
            run_neg = _perturbed_run(-1.0, d)

            for num in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
                name = num + abbreviations[d]
                run_base[name] = (
                    (run_pos[num] - run_neg[num]) / (2 * delta) * scaling_factors[d]
                )

            if d == "alpha":
                CLa = np.where(run_base["CLa"] == 0, np.nan, run_base["CLa"])
                run_base["x_np"] = (
                    self.xyz_ref[0] - run_base["Cma"] / CLa * c_ref
                )

            if d == "beta":
                CYb = np.where(run_base["CYb"] == 0, np.nan, run_base["CYb"])
                run_base["x_np_lateral"] = (
                    self.xyz_ref[0] - run_base["Cnb"] / CYb * b_ref
                )

        return run_base

    # ------------------------------------------------------------------
    # wing_aerodynamics()
    # ------------------------------------------------------------------

    def wing_aerodynamics(self) -> "LiftingLine.AeroComponentResults":
        """Compute wing aerodynamics using the lifting-line method.

        Returns an ``AeroComponentResults`` with total forces and moments.
        After calling, panel geometry is also stored as instance attributes
        for post-processing (streamlines, etc.).
        """
        if self.verbose:
            print("Meshing...")

        # ----------------------------------------------------------
        # Build panel arrays from all wings
        # ----------------------------------------------------------
        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        airfoils = []
        control_surfaces_list = []

        for wing in self.aircraft.wings:
            # Subdivide sections for spanwise resolution
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )

            points, faces = wing.mesh_thin_surface(
                chordwise_resolution=1,
                add_camber=False,
            )

            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])

            wing_airfoils = []
            wing_cs = []
            deflections = self.condition.deflections
            for sect_idx, (xsec_a, xsec_b) in enumerate(
                zip(wing.xsecs[:-1], wing.xsecs[1:])
            ):
                wing_airfoils.append(
                    xsec_a.airfoil.blend_with_another_airfoil(
                        airfoil=xsec_b.airfoil,
                        blend_fraction=0.5,
                    )
                )
                wing_cs.append(
                    section_cs_corrections(wing, deflections, sect_idx, is_mirrored=False)
                )

            airfoils.extend(wing_airfoils)
            control_surfaces_list.extend(wing_cs)

            if wing.symmetric:
                airfoils.extend(wing_airfoils)
                # Mirror: asymmetric surfaces (ailerons) get deflection negated.
                mirror_cs = [
                    section_cs_corrections(wing, deflections, s, is_mirrored=True)
                    for s in range(len(wing.xsecs) - 1)
                ]
                control_surfaces_list.extend(mirror_cs)

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)

        # ----------------------------------------------------------
        # Panel geometry
        # ----------------------------------------------------------
        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / _tall(cross_norm)
        areas = cross_norm / 2.0

        # Vortex positions at the 3/4 chord (collocation) and 1/4 chord (bound leg)
        left_vortex_vertices = (
            0.75 * front_left_vertices + 0.25 * back_left_vertices
        )
        right_vortex_vertices = (
            0.75 * front_right_vertices + 0.25 * back_right_vertices
        )
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2.0
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        vortex_bound_leg_norm = np.linalg.norm(vortex_bound_leg, axis=1)
        chord_vectors = (
            (back_left_vertices + back_right_vertices) / 2.0
            - (front_left_vertices + front_right_vertices) / 2.0
        )
        chords = np.linalg.norm(chord_vectors, axis=1)
        wing_directions = vortex_bound_leg / _tall(vortex_bound_leg_norm)
        local_forward_direction = np.cross(normal_directions, wing_directions)

        # Save for post-processing
        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.airfoils = airfoils
        # cs_corrections[i] = (delta_cl, delta_cm) for panel i from deflected CSs
        self.cs_corrections = control_surfaces_list
        self.control_surfaces_list = control_surfaces_list
        self.normal_directions = normal_directions
        self.areas = areas
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.chord_vectors = chord_vectors
        self.chords = chords
        self.local_forward_direction = local_forward_direction
        self.n_panels = areas.shape[0]

        # ----------------------------------------------------------
        # Freestream and rotation velocities
        # ----------------------------------------------------------
        if self.verbose:
            print("Calculating freestream influence...")

        steady_freestream_velocity = (
            self.condition.freestream_velocity_geometry_axes()
        )
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(
            steady_freestream_velocity
        )
        steady_freestream_velocities = np.tile(
            _wide(steady_freestream_velocity), reps=(self.n_panels, 1)
        )
        steady_freestream_directions = np.tile(
            _wide(steady_freestream_direction), reps=(self.n_panels, 1)
        )
        rotation_freestream_velocities = (
            self.condition.rotation_velocity_geometry_axes(
                vortex_centers,
                p=self.condition.p,
                q=self.condition.q,
                r=self.condition.r,
            )
        )
        freestream_velocities = (
            steady_freestream_velocities + rotation_freestream_velocities
        )

        # Add fuselage displacement (upwash) effect on wing panels
        if self.aircraft.fuselages:
            fuselage_vel = self.calculate_fuselage_influences(vortex_centers)
            freestream_velocities = freestream_velocities + fuselage_vel

        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        # ----------------------------------------------------------
        # Section aerodynamic conditions
        # ----------------------------------------------------------
        # Geometric AoA of each section (degrees) — angle between freestream
        # and section normal projected back to alpha
        alpha_geometrics = 90.0 - arccosd(
            np.sum(steady_freestream_directions * normal_directions, axis=1)
        )

        # Sweep correction: component of freestream along chordwise direction
        cos_sweeps = np.sum(
            steady_freestream_directions * -local_forward_direction, axis=1
        )

        machs = self.condition.mach() * cos_sweeps

        # Kinematic viscosity for Reynolds number
        _, _, rho, mu = self.condition.atmosphere()
        nu = mu / rho  # kinematic viscosity [m²/s]
        Res = (self.condition.velocity * chords / nu) * cos_sweeps

        # ----------------------------------------------------------
        # CL at geometric alpha (for the LL RHS)
        # ----------------------------------------------------------
        if self.verbose:
            print("Querying NeuralFoil for section CL at geometric alpha...")

        CLs_at_alpha_geometric = [
            float(af.get_aero_from_neuralfoil(
                alpha=float(alpha_geometrics[i]),
                Re=float(Res[i]),
                model_size=self.model_size,
            )["CL"].flat[0]) + control_surfaces_list[i][0]
            for i, af in enumerate(airfoils)
        ]
        # Linearised lift slope: 2π per section (thin airfoil theory)
        CLas = 2.0 * np.pi * np.ones(len(CLs_at_alpha_geometric))

        # ----------------------------------------------------------
        # Assemble and solve the lifting-line AIC system
        # ----------------------------------------------------------
        if self.verbose:
            print("Assembling AIC matrix...")

        u_c, v_c, w_c = calculate_induced_velocity_horseshoe(
            x_field=_tall(vortex_centers[:, 0]),
            y_field=_tall(vortex_centers[:, 1]),
            z_field=_tall(vortex_centers[:, 2]),
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
            u_c * _tall(normal_directions[:, 0])
            + v_c * _tall(normal_directions[:, 1])
            + w_c * _tall(normal_directions[:, 2])
        )
        # Influence of unit vortex strength [m²/s] on normal velocity [m/s]

        alpha_influence_matrix = AIC / self.condition.velocity
        # Influence of unit vortex strength on induced alpha [rad]

        if self.verbose:
            print("Solving lifting-line system...")

        V_fs_cross_li = np.cross(steady_freestream_velocities, vortex_bound_leg, axis=1)
        V_fs_cross_li_mag = np.linalg.norm(V_fs_cross_li, axis=1)
        V_perp = self.condition.velocity * cos_sweeps  # velocity normal to bound leg

        A = alpha_influence_matrix * np.tile(
            _wide(CLas), (self.n_panels, 1)
        ) - np.diag(
            2.0 * V_fs_cross_li_mag / (V_perp**2) / areas
        )
        b_rhs = -np.array(CLs_at_alpha_geometric)

        vortex_strengths = np.linalg.solve(A, b_rhs)
        self.vortex_strengths = vortex_strengths

        # ----------------------------------------------------------
        # Evaluate section polars at effective (geometric + induced) alpha
        # ----------------------------------------------------------
        alpha_induced = np.degrees(alpha_influence_matrix @ vortex_strengths)
        alphas = alpha_geometrics + alpha_induced  # 1-D (N,)

        if self.verbose:
            print("Querying NeuralFoil for section polars at effective alpha...")

        aeros = [
            af.get_aero_from_neuralfoil(
                alpha=float(alphas[i]),
                Re=float(Res[i]),
                model_size=self.model_size,
            )
            for i, af in enumerate(airfoils)
        ]
        # NeuralFoil returns shape-(1,) arrays; flatten to 1-D float arrays.
        # Add thin-airfoil control surface corrections (delta_cl, delta_cm).
        CLs = np.array([
            float(a["CL"].flat[0]) + control_surfaces_list[i][0]
            for i, a in enumerate(aeros)
        ])
        CDs = np.array([float(a["CD"].flat[0]) for a in aeros])
        CMs = np.array([
            float(a["CM"].flat[0]) + control_surfaces_list[i][1]
            for i, a in enumerate(aeros)
        ])

        # ----------------------------------------------------------
        # Near-field forces: inviscid (Kutta-Joukowski) + viscous profile
        # ----------------------------------------------------------
        if self.verbose:
            print("Computing near-field forces and moments...")

        velocities = self.get_velocity_at_points(
            points=vortex_centers, vortex_strengths=vortex_strengths
        )
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)

        # Inviscid: F = rho * (V × l) * gamma
        Vi_cross_li = np.cross(velocities, vortex_bound_leg, axis=1)
        forces_inviscid = (
            self.condition.density() * Vi_cross_li * _tall(vortex_strengths)
        )
        moments_inviscid = np.cross(
            vortex_centers - _wide(np.array(self.xyz_ref)),
            forces_inviscid,
        )

        # Profile (viscous drag): F = 0.5 * rho * V * |V| * CD * A
        forces_profile = (
            0.5
            * self.condition.density()
            * velocities
            * _tall(velocity_magnitudes)
            * _tall(CDs)
            * _tall(areas)
        )
        moments_profile = np.cross(
            vortex_centers - _wide(np.array(self.xyz_ref)),
            forces_profile,
        )

        # Pitching moment: M = 0.5 * rho * |V|² * CM * c² * bound_leg_YZ
        bound_leg_YZ = vortex_bound_leg.copy()
        bound_leg_YZ[:, 0] = 0.0
        moments_pitching = (
            0.5
            * self.condition.density()
            * _tall(velocity_magnitudes**2)
            * _tall(CMs)
            * _tall(chords**2)
            * bound_leg_YZ
        )

        # Total
        force_total = (
            np.sum(forces_inviscid, axis=0)
            + np.sum(forces_profile, axis=0)
        )
        moment_total = (
            np.sum(moments_inviscid, axis=0)
            + np.sum(moments_profile, axis=0)
            + np.sum(moments_pitching, axis=0)
        )

        return self.AeroComponentResults(
            s_ref=self.aircraft.reference_area(),
            c_ref=self.aircraft.reference_chord(),
            b_ref=self.aircraft.reference_span(),
            condition=self.condition,
            F_g=list(force_total),
            M_g=list(moment_total),
        )

    # ------------------------------------------------------------------
    # Velocity query helpers
    # ------------------------------------------------------------------

    def get_induced_velocity_at_points(
        self,
        points: np.ndarray,
        vortex_strengths: np.ndarray = None,
    ) -> np.ndarray:
        """Induced velocity (vortex contributions only) at arbitrary points.

        Parameters
        ----------
        points : (N, 3) array
            Query points in geometry axes.
        vortex_strengths : (M,) array or None
            Use stored ``self.vortex_strengths`` if not provided.

        Returns
        -------
        (N, 3) array — induced velocity in geometry axes [m/s].
        """
        if vortex_strengths is None:
            if not hasattr(self, "vortex_strengths"):
                raise ValueError(
                    "Vortex strengths not available; call wing_aerodynamics() first "
                    "or pass vortex_strengths explicitly."
                )
            vortex_strengths = self.vortex_strengths

        u, v, w = calculate_induced_velocity_horseshoe(
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
            gamma=_wide(vortex_strengths),
            vortex_core_radius=self.vortex_core_radius,
        )
        return np.stack(
            [np.sum(u, axis=1), np.sum(v, axis=1), np.sum(w, axis=1)], axis=1
        )

    def get_velocity_at_points(
        self,
        points: np.ndarray,
        vortex_strengths: np.ndarray = None,
    ) -> np.ndarray:
        """Total velocity (induced + freestream + rotation) at arbitrary points.

        Parameters
        ----------
        points : (N, 3) array
            Query points in geometry axes.
        vortex_strengths : (M,) array or None
            Use stored ``self.vortex_strengths`` if not provided.

        Returns
        -------
        (N, 3) array — total velocity in geometry axes [m/s].
        """
        V_induced = self.get_induced_velocity_at_points(
            points=points, vortex_strengths=vortex_strengths
        )
        rot_vel = self.condition.rotation_velocity_geometry_axes(
            points,
            p=self.condition.p,
            q=self.condition.q,
            r=self.condition.r,
        )
        freestream = np.add(_wide(self.steady_freestream_velocity), rot_vel)
        return V_induced + freestream

    # ------------------------------------------------------------------
    # Fuselage panel-source influences (optional, for velocity field)
    # ------------------------------------------------------------------

    def calculate_fuselage_influences(self, points: np.ndarray) -> np.ndarray:
        """Point-source fuselage blockage velocity at arbitrary points.

        Parameters
        ----------
        points : (N, 3) array
            Query points in geometry axes.

        Returns
        -------
        (N, 3) array — fuselage-induced velocity [m/s].
        """
        fuse_centers = []
        fuse_radii = []

        for fuse in self.aircraft.fuselages:
            for xsec in fuse.xsecs:
                fuse_centers.append(
                    np.array([fuse.x_le + xsec.x, fuse.y_le, fuse.z_le])
                )
                fuse_radii.append(xsec.equivalent_radius())

        if not fuse_centers:
            return np.zeros((len(points), 3))

        centers = np.stack(fuse_centers, axis=0)
        radii = np.array(fuse_radii)
        areas = np.pi * radii**2
        Vx = self.condition.freestream_velocity_geometry_axes()[0]
        sigmas = Vx * np.diff(areas)

        # Midpoints between consecutive sections as source locations
        source_points = (centers[1:, :] + centers[:-1, :]) / 2.0

        u, v, w = calculate_induced_velocity_point_source(
            x_field=_tall(points[:, 0]),
            y_field=_tall(points[:, 1]),
            z_field=_tall(points[:, 2]),
            x_source=_wide(source_points[:, 0]),
            y_source=_wide(source_points[:, 1]),
            z_source=_wide(source_points[:, 2]),
            sigma=_wide(sigmas),
            viscous_radius=1e-4,
        )
        return np.stack(
            [np.sum(u, axis=1), np.sum(v, axis=1), np.sum(w, axis=1)], axis=1
        )

    # ------------------------------------------------------------------
    # Streamline tracing
    # ------------------------------------------------------------------

    def calculate_streamlines(
        self,
        seed_points: np.ndarray = None,
        n_steps: int = 300,
        length: float = None,
    ) -> np.ndarray:
        """Trace streamlines using forward-Euler integration.

        Parameters
        ----------
        seed_points : (N, 3) array or None
            Start points. Auto-generated from trailing-edge panel vertices if None.
        n_steps : int
            Number of integration steps.
        length : float or None
            Total streamline length [m]. Defaults to 5 × reference chord.

        Returns
        -------
        streamlines : (N, 3, n_steps) array
        """
        if self.verbose:
            print("Calculating streamlines...")

        if length is None:
            length = self.aircraft.reference_chord() * 5.0

        if seed_points is None:
            left_TE = self.back_left_vertices
            right_TE = self.back_right_vertices
            target = 200
            n_per_panel = max(1, target // len(left_TE))
            nodes = np.linspace(0.0, 1.0, n_per_panel + 1)
            mids = (nodes[1:] + nodes[:-1]) / 2.0
            seed_points = np.concatenate(
                [x * left_TE + (1 - x) * right_TE for x in mids]
            )

        streamlines = np.empty((len(seed_points), 3, n_steps))
        streamlines[:, :, 0] = seed_points
        step_length = length / n_steps

        for i in range(1, n_steps):
            V = self.get_velocity_at_points(streamlines[:, :, i - 1])
            V_norm = np.linalg.norm(V, axis=1, keepdims=True)
            streamlines[:, :, i] = (
                streamlines[:, :, i - 1] + step_length * V / V_norm
            )

        self.streamlines = streamlines
        if self.verbose:
            print("Streamlines calculated.")
        return streamlines
