"""Nonlinear lifting-line solver with NeuralFoil section polars.

Wraps the linear LiftingLine solver with a fixed-point iteration that
updates the RHS CL values until vortex strengths converge.  No CasADi
required — plain NumPy fixed-point iteration.

Typical convergence: 10–30 iterations at tolerance 1e-6.

Usage
-----
>>> from aerisplane.aero.solvers.nonlinear_lifting_line import NonlinearLiftingLine
>>> nll = NonlinearLiftingLine(aircraft, condition)
>>> result = nll.run()   # same dict as LiftingLine.run()
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.utils.spacing import cosspace
from aerisplane.aero.singularities import calculate_induced_velocity_horseshoe


def _tall(a: np.ndarray) -> np.ndarray:
    return np.reshape(a, (-1, 1))


def _wide(a: np.ndarray) -> np.ndarray:
    return np.reshape(a, (1, -1))


class NonlinearLiftingLine:
    """Nonlinear lifting-line with NeuralFoil section polars.

    Iterates a fixed-point loop:

    1. Solve linear LL for vortex strengths (with current NeuralFoil RHS).
    2. Compute induced alphas from those vortex strengths.
    3. Query NeuralFoil at **effective** alpha (geometric + induced).
    4. Update RHS with new section CL values.
    5. Repeat until vortex strengths converge (‖Δγ‖ / ‖γ‖ < tolerance).

    The effective alpha is fed back into the RHS on every iteration, unlike
    the linear LL which only queries NeuralFoil at geometric alpha for the
    RHS.  This captures post-stall behaviour and spanwise stall progression.

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
    spanwise_resolution : int
        Spanwise panels per section (default 4).
    spanwise_spacing_function : callable
        Spacing function ``f(start, stop, num) → array``.  Defaults to cosine.
    vortex_core_radius : float
        Kaufmann vortex core radius [m] (default 1e-8).
    align_trailing_vortices_with_wind : bool
        Align trailing vortices with freestream (default False).
    max_iter : int
        Maximum number of fixed-point iterations (default 100).
    tolerance : float
        Convergence criterion: relative change in vortex strengths (default 1e-6).
    relaxation : float
        Under-relaxation factor [0, 1].  Lower values improve stability near
        stall at the cost of more iterations.  Default 0.5.
    verbose : bool
        Print iteration convergence history.
    """

    def __init__(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        xyz_ref: Optional[List[float]] = None,
        model_size: str = "medium",
        spanwise_resolution: int = 4,
        spanwise_spacing_function: Callable = cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        relaxation: float = 0.5,
        verbose: bool = False,
    ):
        self.aircraft = aircraft
        self.condition = condition
        self.xyz_ref = xyz_ref if xyz_ref is not None else aircraft.xyz_ref
        self.model_size = model_size
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.relaxation = relaxation
        self.verbose = verbose

        # Iteration history (filled by run())
        self.residuals: List[float] = []
        self.n_iter: int = 0

    def run(self) -> Dict:
        """Compute aerodynamic forces and moments.

        Returns
        -------
        dict
            Same keys as ``LiftingLine.run()``, plus:
            ``"n_iter"`` — number of iterations taken,
            ``"residuals"`` — per-iteration relative residual history.
        """
        from aerisplane.aero.solvers.lifting_line import LiftingLine
        from aerisplane.aero._np_compat import arccosd

        # Build the underlying linear LL solver — we reuse its mesh
        # and AIC infrastructure; only the RHS changes each iteration.
        ll = LiftingLine(
            aircraft=self.aircraft,
            condition=self.condition,
            xyz_ref=self.xyz_ref,
            model_size=self.model_size,
            spanwise_resolution=self.spanwise_resolution,
            spanwise_spacing_function=self.spanwise_spacing_function,
            vortex_core_radius=self.vortex_core_radius,
            align_trailing_vortices_with_wind=self.align_trailing_vortices_with_wind,
            verbose=False,  # suppress inner verbosity; NLL handles its own
        )

        # ---------------------------------------------------------------
        # Step 1 — Mesh (done once; stores geometry on ll)
        # ---------------------------------------------------------------
        if self.verbose:
            print("NonlinearLiftingLine: meshing...")

        # Trigger mesh build by calling wing_aerodynamics once.
        # We intercept the geometry attributes it stores on `ll`.
        _ = ll.wing_aerodynamics()

        # Retrieve panel geometry from the now-initialised LL
        vortex_centers = ll.vortex_centers
        left_vortex_vertices = ll.left_vortex_vertices
        right_vortex_vertices = ll.right_vortex_vertices
        normal_directions = ll.normal_directions
        areas = ll.areas
        chords = ll.chords
        airfoils = ll.airfoils
        cs_corrections = ll.cs_corrections  # list of (delta_cl, delta_cm) per panel
        control_surfaces_list = ll.control_surfaces_list
        local_forward_direction = ll.local_forward_direction
        vortex_bound_leg = ll.vortex_bound_leg
        vortex_bound_leg_norm = np.linalg.norm(vortex_bound_leg, axis=1)
        freestream_velocities = ll.freestream_velocities
        steady_freestream_velocity = ll.steady_freestream_velocity
        steady_freestream_direction = ll.steady_freestream_direction
        n_panels = ll.n_panels

        # ---------------------------------------------------------------
        # Step 2 — Precompute AIC matrix and section conditions (constant)
        # ---------------------------------------------------------------
        from aerisplane.aero.singularities import calculate_induced_velocity_horseshoe

        def _tall(a): return np.reshape(a, (-1, 1))
        def _wide(a): return np.reshape(a, (1, -1))

        # Per-panel freestream directions (includes fuselage upwash if present)
        freestream_magnitudes = np.linalg.norm(freestream_velocities, axis=1, keepdims=True)
        per_panel_freestream_directions = freestream_velocities / np.maximum(freestream_magnitudes, 1e-12)

        alpha_geometrics = 90.0 - arccosd(
            np.sum(per_panel_freestream_directions * normal_directions, axis=1)
        )

        cos_sweeps = np.sum(
            np.tile(_wide(steady_freestream_direction), (n_panels, 1))
            * -local_forward_direction,
            axis=1,
        )

        _, _, rho, mu = self.condition.atmosphere()
        nu = mu / rho
        Res = (self.condition.velocity * chords / nu) * cos_sweeps
        machs = self.condition.mach() * cos_sweeps

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
        alpha_influence_matrix = AIC / self.condition.velocity

        V_fs_cross_li = np.cross(
            np.tile(_wide(steady_freestream_velocity), (n_panels, 1)),
            vortex_bound_leg,
            axis=1,
        )
        V_fs_cross_li_mag = np.linalg.norm(V_fs_cross_li, axis=1)
        V_perp = self.condition.velocity * cos_sweeps  # velocity ⊥ bound leg

        # Diagonal matrix: maps vortex strength → induced CL term
        diag_D = np.diag(2.0 * V_fs_cross_li_mag / (V_perp**2) / areas)

        # Linear lift slope (thin-airfoil: dCL/dalpha = 2π rad⁻¹)
        CLas = 2.0 * np.pi * np.ones(n_panels)

        # ---------------------------------------------------------------
        # Step 3 — Initial guess from linear LL (already computed above)
        # ---------------------------------------------------------------
        vortex_strengths = ll.vortex_strengths.copy()

        def _query_neuralfoil(alphas_eff: np.ndarray):
            """Query NeuralFoil at each panel's effective alpha.

            Control surface corrections (thin-airfoil ΔCl, ΔCm) are added
            on top of the NeuralFoil baseline at every iteration so the
            fixed-point loop converges with deflected surfaces active.
            """
            CLs_nf = np.empty(n_panels)
            CDs_nf = np.empty(n_panels)
            CMs_nf = np.empty(n_panels)
            for i, af in enumerate(airfoils):
                aero = af.get_aero_from_neuralfoil(
                    alpha=float(alphas_eff[i]),
                    Re=float(Res[i]),
                    mach=float(machs[i]),
                    model_size=self.model_size,
                )
                delta_cl, delta_cm = cs_corrections[i]
                CLs_nf[i] = float(aero["CL"].flat[0]) + delta_cl
                CDs_nf[i] = float(aero["CD"].flat[0])
                CMs_nf[i] = float(aero["CM"].flat[0]) + delta_cm
            return CLs_nf, CDs_nf, CMs_nf

        # ---------------------------------------------------------------
        # Step 4 — Fixed-point iteration
        # ---------------------------------------------------------------
        if self.verbose:
            print(f"{'Iter':>5}  {'Residual':>12}")
            print("-" * 20)

        self.residuals = []
        CLs_eff = np.empty(n_panels)
        CDs_eff = np.empty(n_panels)
        CMs_eff = np.empty(n_panels)

        for iteration in range(self.max_iter):
            # Induced alpha from current vortex strengths [deg]
            alpha_induced = np.degrees(alpha_influence_matrix @ vortex_strengths)
            alphas_eff = alpha_geometrics + alpha_induced

            # Query NeuralFoil at effective alpha
            CLs_new, CDs_new, CMs_new = _query_neuralfoil(alphas_eff)

            # Assemble system matrix and RHS using new NeuralFoil CLs
            A = (
                alpha_influence_matrix * np.tile(_wide(CLas), (n_panels, 1))
                - diag_D
            )
            b_rhs = -CLs_new  # NeuralFoil CL at effective alpha

            # Solve for updated vortex strengths
            gamma_new = np.linalg.solve(A, b_rhs)

            # Convergence check
            gamma_norm = np.linalg.norm(vortex_strengths)
            residual = (
                np.linalg.norm(gamma_new - vortex_strengths) / gamma_norm
                if gamma_norm > 1e-12
                else np.linalg.norm(gamma_new - vortex_strengths)
            )
            self.residuals.append(float(residual))

            if self.verbose:
                print(f"{iteration + 1:>5}  {residual:>12.6e}")

            # Under-relaxation: blend old and new solution
            vortex_strengths = (
                (1.0 - self.relaxation) * vortex_strengths
                + self.relaxation * gamma_new
            )

            # Store effective polar results (will be used for force computation)
            CLs_eff = CLs_new
            CDs_eff = CDs_new
            CMs_eff = CMs_new

            if residual < self.tolerance:
                if self.verbose:
                    print(f"Converged in {iteration + 1} iterations.")
                break
        else:
            if self.verbose:
                print(
                    f"Warning: NonlinearLiftingLine did not converge in "
                    f"{self.max_iter} iterations (final residual={residual:.3e})."
                )

        self.n_iter = len(self.residuals)
        self.vortex_strengths = vortex_strengths

        # ---------------------------------------------------------------
        # Step 5 — Near-field forces from converged solution
        # ---------------------------------------------------------------
        velocities = ll.get_velocity_at_points(
            points=vortex_centers, vortex_strengths=vortex_strengths
        )
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)

        # Inviscid: Kutta-Joukowski
        Vi_cross_li = np.cross(velocities, vortex_bound_leg, axis=1)
        forces_inviscid = (
            self.condition.density() * Vi_cross_li * _tall(vortex_strengths)
        )
        moments_inviscid = np.cross(
            vortex_centers - _wide(np.array(self.xyz_ref)),
            forces_inviscid,
        )

        # Profile drag (viscous)
        forces_profile = (
            0.5
            * self.condition.density()
            * velocities
            * _tall(velocity_magnitudes)
            * _tall(CDs_eff)
            * _tall(areas)
        )
        moments_profile = np.cross(
            vortex_centers - _wide(np.array(self.xyz_ref)),
            forces_profile,
        )

        # Pitching moment
        bound_leg_YZ = vortex_bound_leg.copy()
        bound_leg_YZ[:, 0] = 0.0
        moments_pitching = (
            0.5
            * self.condition.density()
            * _tall(velocity_magnitudes**2)
            * _tall(CMs_eff)
            * _tall(chords**2)
            * bound_leg_YZ
        )

        force_total = np.sum(forces_inviscid, axis=0) + np.sum(forces_profile, axis=0)
        moment_total = (
            np.sum(moments_inviscid, axis=0)
            + np.sum(moments_profile, axis=0)
            + np.sum(moments_pitching, axis=0)
        )

        # ---------------------------------------------------------------
        # Step 6 — Fuselage contributions (via AeroBuildup Jorgensen model)
        # ---------------------------------------------------------------
        from aerisplane.aero.solvers.aero_buildup import AeroBuildup

        aerobuildup = AeroBuildup(
            aircraft=self.aircraft,
            condition=self.condition,
            xyz_ref=self.xyz_ref,
        )
        fuselage_aero_components = [
            aerobuildup.fuselage_aerodynamics(fuselage=fuse, include_induced_drag=True)
            for fuse in self.aircraft.fuselages
        ]
        for comp in fuselage_aero_components:
            for i in range(3):
                force_total[i] += comp.F_g[i]
                moment_total[i] += comp.M_g[i]

        # Wing-body junction interference drag
        from aerisplane.aero.library.interference import total_junction_drag, aircraft_carryover_factors
        D_junction = total_junction_drag(self.aircraft, self.condition)
        if D_junction > 0:
            D_junc_g = self.condition.convert_axes(
                -D_junction, 0, 0, from_axes="wind", to_axes="geometry"
            )
            for i in range(3):
                force_total[i] += D_junc_g[i]

        # Wing-body lift carryover (Schlichting & Truckenbrodt K_L)
        # K_D is not applied here; induced drag is implicit in section polars.
        K_L, _ = aircraft_carryover_factors(self.aircraft)
        if K_L != 1.0:
            _, _, lift_pre = self.condition.convert_axes(*force_total, from_axes="geometry", to_axes="wind")
            delta_L = (K_L - 1.0) * (-lift_pre)
            lift_carryover_g = self.condition.convert_axes(0, 0, -delta_L, from_axes="wind", to_axes="geometry")
            for i in range(3):
                force_total[i] += lift_carryover_g[i]

        # Save panel geometry and freestream for post-processing (streamlines, draw)
        self.front_left_vertices = ll.front_left_vertices
        self.back_left_vertices = ll.back_left_vertices
        self.back_right_vertices = ll.back_right_vertices
        self.front_right_vertices = ll.front_right_vertices
        self.left_vortex_vertices = ll.left_vortex_vertices
        self.right_vortex_vertices = ll.right_vortex_vertices
        self.steady_freestream_velocity = ll.steady_freestream_velocity
        self.steady_freestream_direction = ll.steady_freestream_direction

        # ---------------------------------------------------------------
        # Step 7 — Assemble output dict (same keys as LiftingLine.run())
        # ---------------------------------------------------------------
        output: Dict = {
            "F_g": list(force_total),
            "M_g": list(moment_total),
        }

        output["F_b"] = self.condition.convert_axes(
            *output["F_g"], from_axes="geometry", to_axes="body"
        )
        output["F_w"] = self.condition.convert_axes(
            *output["F_g"], from_axes="geometry", to_axes="wind"
        )
        output["M_b"] = self.condition.convert_axes(
            *output["M_g"], from_axes="geometry", to_axes="body"
        )
        output["M_w"] = self.condition.convert_axes(
            *output["M_g"], from_axes="geometry", to_axes="wind"
        )

        output["L"] = -output["F_w"][2]
        output["Y"] = output["F_w"][1]
        output["D"] = -output["F_w"][0]
        output["l_b"] = output["M_b"][0]
        output["m_b"] = output["M_b"][1]
        output["n_b"] = output["M_b"][2]

        qS = self.condition.dynamic_pressure() * self.aircraft.reference_area()
        c = self.aircraft.reference_chord()
        b = self.aircraft.reference_span()

        output["CL"] = output["L"] / qS
        output["CY"] = output["Y"] / qS
        output["CD"] = output["D"] / qS
        output["Cl"] = output["l_b"] / qS / b
        output["Cm"] = output["m_b"] / qS / c
        output["Cn"] = output["n_b"] / qS / b

        output["fuselage_aero_components"] = fuselage_aero_components
        output["n_iter"] = self.n_iter
        output["residuals"] = self.residuals

        return output

    # ------------------------------------------------------------------
    # Stability derivatives
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

        Uses two perturbed NLL solves per derivative (±delta) for O(Δ²)
        accuracy.  Each perturbed solve runs the full fixed-point iteration,
        so this is more expensive than the linear LL equivalent.

        Parameters
        ----------
        alpha, beta, p, q, r : bool
            Which derivatives to compute (default all True).

        Returns
        -------
        dict
            All keys from ``run()``, plus e.g. "CLa", "CDa", "Cma",
            "x_np", "x_np_lateral", etc.
        """
        import dataclasses

        b_ref = self.aircraft.reference_span()
        c_ref = self.aircraft.reference_chord()
        V = self.condition.velocity

        do_analysis = {"alpha": alpha, "beta": beta, "p": p, "q": q, "r": r}
        abbreviations = {"alpha": "a", "beta": "b", "p": "p", "q": "q", "r": "r"}
        # Step size in the perturbed variable (radians or rad/s)
        fd_amounts = {
            "alpha": 0.001,
            "beta": 0.001,
            "p": 0.001 * (2 * V) / b_ref,
            "q": 0.001 * (2 * V) / c_ref,
            "r": 0.001 * (2 * V) / b_ref,
        }
        # Scale derivative from (per rad) to (per deg) or nondimensional
        scaling = {
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

            def _perturbed_run(sign: float, var: str) -> Dict:
                amt = sign * fd_amounts[var]
                cond = self.condition.copy()
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
                nll = NonlinearLiftingLine(
                    aircraft=self.aircraft,
                    condition=cond,
                    xyz_ref=self.xyz_ref,
                    model_size=self.model_size,
                    spanwise_resolution=self.spanwise_resolution,
                    spanwise_spacing_function=self.spanwise_spacing_function,
                    vortex_core_radius=self.vortex_core_radius,
                    align_trailing_vortices_with_wind=self.align_trailing_vortices_with_wind,
                    max_iter=self.max_iter,
                    tolerance=self.tolerance,
                    relaxation=self.relaxation,
                    verbose=False,
                )
                return nll.run()

            run_pos = _perturbed_run(+1.0, d)
            run_neg = _perturbed_run(-1.0, d)

            for num in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
                name = num + abbreviations[d]
                run_base[name] = (
                    (run_pos[num] - run_neg[num]) / (2 * fd_amounts[d]) * scaling[d]
                )

            if d == "alpha":
                CLa = np.where(run_base["CLa"] == 0, np.nan, run_base["CLa"])
                run_base["x_np"] = self.xyz_ref[0] - run_base["Cma"] / CLa * c_ref

            if d == "beta":
                CYb = np.where(run_base["CYb"] == 0, np.nan, run_base["CYb"])
                run_base["x_np_lateral"] = (
                    self.xyz_ref[0] - run_base["Cnb"] / CYb * b_ref
                )

        return run_base

    # ------------------------------------------------------------------
    # Velocity query helpers
    # ------------------------------------------------------------------

    def get_induced_velocity_at_points(
        self,
        points: np.ndarray,
        vortex_strengths: np.ndarray = None,
    ) -> np.ndarray:
        """Induced-only velocity (no freestream) at given points.

        Parameters
        ----------
        points : (N, 3) array
        vortex_strengths : (M,) array or None
            Uses stored ``self.vortex_strengths`` if not provided.

        Returns
        -------
        (N, 3) array — induced velocity in geometry axes [m/s].
        """
        if vortex_strengths is None:
            if not hasattr(self, "vortex_strengths"):
                raise ValueError(
                    "Call run() before get_induced_velocity_at_points(), "
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
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return np.stack(
            [np.sum(u, axis=1), np.sum(v, axis=1), np.sum(w, axis=1)], axis=1
        )

    def get_velocity_at_points(
        self,
        points: np.ndarray,
        vortex_strengths: np.ndarray = None,
    ) -> np.ndarray:
        """Total velocity (induced + freestream + rotation) at given points.

        Parameters
        ----------
        points : (N, 3) array
        vortex_strengths : (M,) array or None
            Uses stored ``self.vortex_strengths`` if not provided.

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
            Also stored as ``self.streamlines``.
        """
        if self.verbose:
            print("Calculating streamlines...")

        if length is None:
            length = self.aircraft.reference_chord() * 5.0

        if seed_points is None:
            left_te = self.back_left_vertices
            right_te = self.back_right_vertices
            target = 200
            n_per_panel = max(1, target // len(left_te))
            nodes = np.linspace(0.0, 1.0, n_per_panel + 1)
            mids = (nodes[1:] + nodes[:-1]) / 2.0
            seed_points = np.concatenate(
                [t * left_te + (1 - t) * right_te for t in mids]
            )

        streamlines = np.empty((len(seed_points), 3, n_steps))
        streamlines[:, :, 0] = seed_points
        step_length = length / n_steps

        for i in range(1, n_steps):
            V = self.get_velocity_at_points(streamlines[:, :, i - 1])
            speed = np.linalg.norm(V, axis=1, keepdims=True)
            speed = np.where(speed < 1e-10, 1e-10, speed)
            streamlines[:, :, i] = (
                streamlines[:, :, i - 1] + step_length * V / speed
            )

        self.streamlines = streamlines
        if self.verbose:
            print("Streamlines calculated.")
        return streamlines

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(
        self,
        c: np.ndarray = None,
        cmap: str = None,
        colorbar_label: str = None,
        show: bool = True,
        draw_streamlines: bool = True,
        recalculate_streamlines: bool = False,
        backend: str = "pyvista",
    ) -> object:
        """Draw the solved NLL: wing panels coloured by a scalar + 3-D streamlines.

        Must be called after ``run()``.

        Parameters
        ----------
        c : (N,) array or None
            Scalar field for panel colouring. Defaults to ``vortex_strengths``.
        cmap : str or None
            Matplotlib colormap name. Defaults to ``"viridis"``.
        colorbar_label : str or None
            Label for the colour bar.
        show : bool
            Display the figure immediately.
        draw_streamlines : bool
            Overlay 3-D streamlines.
        recalculate_streamlines : bool
            Force re-tracing even if ``self.streamlines`` already exists.
        backend : str
            ``"pyvista"`` (default) or ``"plotly"``.

        Returns
        -------
        pyvista Plotter or plotly Figure.
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

    def _draw_pyvista(self, c, cmap, colorbar_label, show, draw_streamlines):
        import pyvista as pv

        fl = self.front_left_vertices
        bl = self.back_left_vertices
        br = self.back_right_vertices
        fr = self.front_right_vertices
        N = len(fl)

        points_pv = np.concatenate([fl, bl, br, fr])
        r = np.arange(N)
        faces_pv = np.column_stack([
            np.full(N, 4, dtype=np.intp),
            r, r + N, r + 2 * N, r + 3 * N,
        ]).ravel()
        mesh = pv.PolyData(points_pv, faces_pv)

        plotter = pv.Plotter()
        plotter.title = "AerisPlane NonlinearLiftingLine"
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
                pts = self.streamlines[i, :, :].T
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

        all_v = np.vstack([fl, bl, br, fr])
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
                pts = self.streamlines[i, :, :].T
                pts = pts[::3]
                traces.append(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="lines",
                    line=dict(color="rgba(136, 170, 255, 0.55)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text="AerisPlane NonlinearLiftingLine"),
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
