# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/aerodynamics/aero_3D/aero_buildup.py
"""Workbook-style aerodynamic buildup solver.

Uses NeuralFoil for section aerodynamics and Jorgensen slender-body theory
for fuselage contributions.

Usage
-----
>>> from aerisplane.aero.solvers.aero_buildup import AeroBuildup
>>> ab = AeroBuildup(aircraft, condition)
>>> result = ab.run()   # dict with CL, CD, Cm, ...
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.wing import Wing
from aerisplane.core.fuselage import Fuselage

import aerisplane.aero.library as aerolib
from aerisplane.aero.library import transonic
from aerisplane.aero.library.control_surface_effects import section_cs_corrections
from aerisplane.aero.fuselage_aero import (
    critical_mach,
    fuselage_base_drag_coefficient,
    fuselage_form_factor,
    jorgensen_eta,
    softmax_scalefree,
)
from aerisplane.aero.library.interference import total_junction_drag


class AeroBuildup:
    """Workbook-style aerodynamic buildup for an aerisplane Aircraft.

    Combines NeuralFoil section polars (wings) with Jorgensen slender-body
    theory (fuselages) and a global induced-drag model.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft configuration.
    condition : FlightCondition
        Operating point.
    xyz_ref : list of float or None
        Moment reference point [x, y, z] [m].  Defaults to aircraft.xyz_ref.
    model_size : str
        NeuralFoil model size ("small", "medium", "large").
    include_wave_drag : bool
        Whether to include transonic wave drag contributions.
    nose_fineness_ratio : float
        Fuselage nose fineness ratio for wave-drag critical Mach calculation.
    E_wave_drag : float
        Wave drag efficiency factor for fuselages (Raymer E_WD).
    """

    def __init__(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        xyz_ref: list[float] | None = None,
        model_size: str = "large",
        include_wave_drag: bool = True,
        nose_fineness_ratio: float = 3.0,
        E_wave_drag: float = 2.5,
    ):
        self.aircraft = aircraft
        self.condition = condition
        self.xyz_ref = xyz_ref if xyz_ref is not None else list(aircraft.xyz_ref)
        self.model_size = model_size
        self.include_wave_drag = include_wave_drag
        self.nose_fineness_ratio = nose_fineness_ratio
        self.E_wave_drag = E_wave_drag

    # ------------------------------------------------------------------ #
    # Result container
    # ------------------------------------------------------------------ #

    @dataclass
    class AeroComponentResults:
        """Forces and moments from a single aircraft component."""

        s_ref: float
        c_ref: float
        b_ref: float
        condition: FlightCondition
        F_g: list  # [Fx, Fy, Fz] in geometry axes [N]
        M_g: list  # [Mx, My, Mz] in geometry axes [Nm]
        span_effective: float
        oswalds_efficiency: float

        @property
        def F_b(self) -> tuple:
            return self.condition.convert_axes(*self.F_g, from_axes="geometry", to_axes="body")

        @property
        def F_w(self) -> tuple:
            return self.condition.convert_axes(*self.F_g, from_axes="geometry", to_axes="wind")

        @property
        def M_b(self) -> tuple:
            return self.condition.convert_axes(*self.M_g, from_axes="geometry", to_axes="body")

        @property
        def L(self) -> float:
            return -self.F_w[2]

        @property
        def Y(self) -> float:
            return self.F_w[1]

        @property
        def D(self) -> float:
            return -self.F_w[0]

    # ------------------------------------------------------------------ #
    # Main analysis entry point
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Compute aerodynamic forces and moments on the aircraft.

        Returns
        -------
        dict with keys: F_g, F_b, F_w, M_g, M_b, M_w,
                        L, D, Y, l_b, m_b, n_b,
                        CL, CD, CY, Cl, Cm, Cn,
                        D_profile, D_induced,
                        wing_aero_components, fuselage_aero_components
        """
        wing_aero_components = [
            self.wing_aerodynamics(wing=wing, include_induced_drag=False)
            for wing in self.aircraft.wings
        ]
        fuselage_aero_components = [
            self.fuselage_aerodynamics(fuselage=fuse, include_induced_drag=False)
            for fuse in self.aircraft.fuselages
        ]

        aero_components = wing_aero_components + fuselage_aero_components

        F_g_total = [sum(comp.F_g[i] for comp in aero_components) for i in range(3)]
        M_g_total = [sum(comp.M_g[i] for comp in aero_components) for i in range(3)]

        # Wing-body junction interference drag
        D_junction = total_junction_drag(self.aircraft, self.condition)
        if D_junction > 0:
            D_junc_g = self.condition.convert_axes(
                -D_junction, 0, 0, from_axes="wind", to_axes="geometry"
            )
            for i in range(3):
                F_g_total[i] += D_junc_g[i]

        # Induced drag via Trefftz-plane model
        Q = self.condition.dynamic_pressure()
        span_effective_squared = softmax_scalefree(
            [comp.span_effective**2 * comp.oswalds_efficiency for comp in aero_components]
        )

        _, sideforce, lift = self.condition.convert_axes(
            *F_g_total, from_axes="geometry", to_axes="wind"
        )
        D_induced = (lift**2 + sideforce**2) / (Q * np.pi * span_effective_squared)
        D_induced_g = self.condition.convert_axes(-D_induced, 0, 0, from_axes="wind", to_axes="geometry")

        for i in range(3):
            F_g_total[i] += D_induced_g[i]

        output = {"F_g": F_g_total, "M_g": M_g_total}

        output["F_b"] = self.condition.convert_axes(*F_g_total, from_axes="geometry", to_axes="body")
        output["F_w"] = self.condition.convert_axes(*F_g_total, from_axes="geometry", to_axes="wind")
        output["M_b"] = self.condition.convert_axes(*M_g_total, from_axes="geometry", to_axes="body")
        output["M_w"] = self.condition.convert_axes(*M_g_total, from_axes="geometry", to_axes="wind")

        output["L"] = -output["F_w"][2]
        output["Y"] = output["F_w"][1]
        output["D"] = -output["F_w"][0]
        output["l_b"] = output["M_b"][0]
        output["m_b"] = output["M_b"][1]
        output["n_b"] = output["M_b"][2]

        qS = Q * self.aircraft.reference_area()
        c = self.aircraft.reference_chord()
        b = self.aircraft.reference_span()

        output["CL"] = output["L"] / qS
        output["CY"] = output["Y"] / qS
        output["CD"] = output["D"] / qS
        output["Cl"] = output["l_b"] / qS / b
        output["Cm"] = output["m_b"] / qS / c
        output["Cn"] = output["n_b"] / qS / b

        output["wing_aero_components"] = wing_aero_components
        output["fuselage_aero_components"] = fuselage_aero_components
        output["D_profile"] = sum(comp.D for comp in aero_components)
        output["D_induced"] = D_induced

        return output

    def run_with_stability_derivatives(
        self,
        alpha: bool = True,
        beta: bool = True,
        p: bool = True,
        q: bool = True,
        r: bool = True,
    ) -> dict:
        """Run analysis and compute stability derivatives via central finite differences.

        Uses two perturbed solves per derivative (±delta) for O(Δ²) accuracy.

        Parameters
        ----------
        alpha, beta, p, q, r : bool
            Which stability derivatives to compute (set to False to skip).

        Returns
        -------
        dict from run(), plus extra keys like "CLa", "CLb", "CLp", etc.,
        and "x_np" (neutral point) if alpha derivatives are computed.
        """
        import dataclasses

        do_analysis = {"alpha": alpha, "beta": beta, "p": p, "q": q, "r": r}
        abbreviations = {"alpha": "a", "beta": "b", "p": "p", "q": "q", "r": "r"}

        V = self.condition.velocity
        b_ref = self.aircraft.reference_span()
        c_ref = self.aircraft.reference_chord()

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

            def _perturbed_run(sign: float, var: str) -> dict:
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
                ab = copy.copy(self)
                ab.condition = cond
                return ab.run()

            run_pos = _perturbed_run(+1.0, d)
            run_neg = _perturbed_run(-1.0, d)

            for numerator in ["CL", "CD", "CY", "Cl", "Cm", "Cn"]:
                deriv_name = numerator + abbreviations[d]
                run_base[deriv_name] = (
                    (run_pos[numerator] - run_neg[numerator])
                    / (2 * fd_amounts[d])
                    * scaling[d]
                )

            if d == "alpha":
                Cma = run_base["Cma"]
                CLa = np.where(run_base["CLa"] == 0, np.nan, run_base["CLa"])
                run_base["x_np"] = self.xyz_ref[0] - (Cma / CLa * c_ref)

        return run_base

    # ------------------------------------------------------------------ #
    # Wing aerodynamics
    # ------------------------------------------------------------------ #

    def wing_aerodynamics(
        self, wing: Wing, include_induced_drag: bool = True
    ) -> AeroComponentResults:
        """Estimate aerodynamic forces and moments on a wing.

        Uses NeuralFoil for 2-D section polars and a finite-wing lift-curve
        slope correction (Raymer DATCOM).

        Parameters
        ----------
        wing : Wing
        include_induced_drag : bool
            If True, add section induced drag from 2-D CL^2 / (pi * AR * e).
            If False, induced drag is added globally in run().

        Returns
        -------
        AeroComponentResults
        """
        condition = self.condition
        mach = condition.mach()

        wing_MAC = wing.mean_aerodynamic_chord()
        wing_taper = wing.taper_ratio()
        wing_sweep = wing.mean_sweep_angle()
        wing_dihedral = wing.mean_dihedral_angle()

        # Sectional spans (YZ plane), areas, and aerodynamic centers
        sectional_spans = wing.sectional_span_yz()
        half_span = sum(sectional_spans)

        # Inboard distance from root to YZ plane (usually 0 for standard wings)
        span_inboard_to_YZ_plane = float(np.inf)
        for i in range(len(wing.xsecs)):
            span_inboard_to_YZ_plane = float(np.minimum(
                span_inboard_to_YZ_plane,
                abs(wing._compute_xyz_of_WingXSec(i, x_nondim=0.25, z_nondim=0)[1]),
            ))

        xsec_chords = [xsec.chord for xsec in wing.xsecs]
        sectional_chords = [
            (a + b) / 2 for a, b in zip(xsec_chords[:-1], xsec_chords[1:])
        ]
        sectional_areas = [s * c for s, c in zip(sectional_spans, sectional_chords)]
        half_area = sum(sectional_areas)
        area_inboard_to_YZ_plane = span_inboard_to_YZ_plane * wing_MAC

        if wing.symmetric:
            span_0_dihedral = 2 * (half_span + span_inboard_to_YZ_plane * 0.5)
            span_90_dihedral = half_span
            area_0_dihedral = 2 * (half_area + area_inboard_to_YZ_plane * 0.5)
            area_90_dihedral = half_area

            dihedral_factor = np.sin(np.radians(wing_dihedral)) ** 2
            span_effective = span_0_dihedral + (span_90_dihedral - span_0_dihedral) * dihedral_factor
            area_effective = area_0_dihedral + (area_90_dihedral - area_0_dihedral) * dihedral_factor
        else:
            span_effective = half_span
            area_effective = half_area

        AR_effective = span_effective**2 / area_effective
        AR_3D_factor = aerolib.CL_over_Cl(
            aspect_ratio=AR_effective, mach=mach, sweep=wing_sweep, Cl_is_compressible=True
        )
        e = aerolib.oswalds_efficiency(
            taper_ratio=wing_taper, aspect_ratio=AR_effective, sweep=wing_sweep,
            fuselage_diameter_to_span_ratio=0,
        )

        areas = sectional_areas
        aerodynamic_centers = wing.sectional_aerodynamic_centers()

        # Neutral-point shift due to lifting-line unsweep near centerline
        a_ratio = AR_effective / (AR_effective + 2)
        s_sweep = np.radians(wing_sweep)
        t_taper = np.exp(-wing_taper)
        np_shift = (
            -(
                (
                    (3.557726 ** (a_ratio**2.8443985))
                    * ((((s_sweep * a_ratio) + (t_taper * 1.9149417)) + -1.4449639) * s_sweep)
                )
                + (a_ratio + -0.89228547)
            )
            * -0.16073418
        ) * wing_MAC
        aerodynamic_centers = [
            ac + np.array([np_shift, 0, 0]) for ac in aerodynamic_centers
        ]

        xsec_quarter_chords = [
            wing._compute_xyz_of_WingXSec(i, x_nondim=0.25, z_nondim=0)
            for i in range(len(wing.xsecs))
        ]

        def compute_section_aerodynamics(sect_id: int, mirror_across_XZ: bool = False):
            xsec_a = wing.xsecs[sect_id]
            xsec_b = wing.xsecs[sect_id + 1]

            a_weight = xsec_a.chord / (xsec_a.chord + xsec_b.chord)
            b_weight = xsec_b.chord / (xsec_a.chord + xsec_b.chord)
            mean_chord = (xsec_a.chord + xsec_b.chord) / 2

            xg_local, yg_local, zg_local = wing._compute_frame_of_section(sect_id)
            xg_local = list(xg_local)
            yg_local = list(yg_local)
            zg_local = list(zg_local)
            if mirror_across_XZ:
                xg_local[1] *= -1
                yg_local[1] *= -1
                zg_local[1] *= -1

            sect_AC_raw = aerodynamic_centers[sect_id].copy()
            if mirror_across_XZ:
                sect_AC_raw[1] *= -1

            sect_AC = [sect_AC_raw[i] - self.xyz_ref[i] for i in range(3)]

            # Velocity at section AC (freestream + rotation)
            vel_freestream = condition.convert_axes(
                -condition.velocity, 0, 0, from_axes="wind", to_axes="geometry"
            )
            omega_g = condition.convert_axes(
                condition.p, condition.q, condition.r,
                from_axes="body", to_axes="geometry",
            )
            vel_rotation = [
                omega_g[1] * sect_AC[2] - omega_g[2] * sect_AC[1],
                omega_g[2] * sect_AC[0] - omega_g[0] * sect_AC[2],
                omega_g[0] * sect_AC[1] - omega_g[1] * sect_AC[0],
            ]
            vel_vector_g = [vel_freestream[i] + vel_rotation[i] for i in range(3)]
            vel_mag_g = np.sqrt(sum(v**2 for v in vel_vector_g))
            vel_dir_g = [v / vel_mag_g for v in vel_vector_g]
            vel_dot_x = np.dot(vel_dir_g, xg_local)
            vel_dot_z = np.dot(vel_dir_g, zg_local)

            alpha_generalized = np.where(
                vel_dot_x > 0,
                90 - np.degrees(np.arccos(np.clip(vel_dot_z, -1, 1))),
                90 + np.degrees(np.arccos(np.clip(vel_dot_z, -1, 1))),
            )

            alpha_generalized_effective = (
                alpha_generalized
                - (1 - AR_3D_factor**0.8)
                * np.sin(np.radians(2 * alpha_generalized))
                / 2
                * (180 / np.pi)
            )

            # Section sweep angle
            qca = xsec_quarter_chords[sect_id]
            qcb = xsec_quarter_chords[sect_id + 1]
            qc_vec = qcb - qca
            qc_dir = qc_vec / np.linalg.norm(qc_vec)
            vel_dot_qc = np.dot(vel_dir_g, list(qc_dir))
            sweep_rad = np.arcsin(vel_dot_qc)

            Re_a = condition.reynolds(xsec_a.chord)
            Re_b = condition.reynolds(xsec_b.chord)
            mach_normal = mach * np.cos(sweep_rad)

            # Section polars from NeuralFoil
            kwargs = dict(
                alpha=alpha_generalized_effective,
                mach=mach_normal if self.include_wave_drag else 0.0,
                model_size=self.model_size,
            )
            aero_a = xsec_a.airfoil.get_aero_from_neuralfoil(Re=Re_a, **kwargs)
            aero_b = xsec_b.airfoil.get_aero_from_neuralfoil(Re=Re_b, **kwargs)

            # NeuralFoil returns shape-(1,) arrays even for scalar input; squeeze to float
            def _nf(aero, key):
                v = aero[key]
                return float(v.flat[0]) if hasattr(v, "flat") else float(v)

            sect_CL = (_nf(aero_a, "CL") * a_weight + _nf(aero_b, "CL") * b_weight) * AR_3D_factor**0.2
            sect_CDp = _nf(aero_a, "CD") * a_weight + _nf(aero_b, "CD") * b_weight
            sect_CM = _nf(aero_a, "CM") * a_weight + _nf(aero_b, "CM") * b_weight

            # Thin-airfoil control surface corrections (ΔCl, ΔCm)
            delta_cl, delta_cm = section_cs_corrections(
                wing, condition.deflections, sect_id, is_mirrored=mirror_across_XZ
            )
            sect_CL += delta_cl
            sect_CM += delta_cm

            if include_induced_drag:
                sect_CDi = sect_CL**2 / (np.pi * AR_effective * e)
                sect_CD = sect_CDp + sect_CDi
            else:
                sect_CD = sect_CDp

            area = areas[sect_id]
            q_local = 0.5 * condition.density() * vel_mag_g**2
            sect_L = q_local * area * sect_CL
            sect_D = q_local * area * sect_CD
            sect_M = q_local * area * sect_CM * mean_chord

            # Lift direction: section normal projected perpendicular to local freestream
            sign = 1.0 if float(vel_dot_x) > 0 else -1.0
            L_dir_unnorm = [
                sign * (float(zg_local[i]) - float(vel_dot_z) * float(vel_dir_g[i]))
                for i in range(3)
            ]
            L_dir_mag = np.sqrt(sum(v**2 for v in L_dir_unnorm))
            L_dir = [v / float(L_dir_mag) for v in L_dir_unnorm]

            D_dir = [float(v) for v in vel_dir_g]

            sect_F_g = [float(sect_L) * L_dir[i] + float(sect_D) * D_dir[i] for i in range(3)]

            # Moments
            M_g_lift = list(np.cross(
                np.array(sect_AC, dtype=float),
                np.array(sect_F_g, dtype=float),
            ))
            M_pitch_dir = list(np.cross(
                np.array(L_dir, dtype=float),
                np.array(D_dir, dtype=float),
            ))
            M_g_pitch = [M_pitch_dir[i] * float(sect_M) for i in range(3)]
            sect_M_g = [M_g_lift[i] + M_g_pitch[i] for i in range(3)]

            return sect_F_g, sect_M_g

        F_g = [0.0, 0.0, 0.0]
        M_g = [0.0, 0.0, 0.0]

        for sect_id in range(len(wing.xsecs) - 1):
            sect_F_g, sect_M_g = compute_section_aerodynamics(sect_id)
            for i in range(3):
                F_g[i] += sect_F_g[i]
                M_g[i] += sect_M_g[i]

            if wing.symmetric:
                sect_F_g, sect_M_g = compute_section_aerodynamics(sect_id, mirror_across_XZ=True)
                for i in range(3):
                    F_g[i] += sect_F_g[i]
                    M_g[i] += sect_M_g[i]

        return self.AeroComponentResults(
            s_ref=self.aircraft.reference_area(),
            c_ref=self.aircraft.reference_chord(),
            b_ref=self.aircraft.reference_span(),
            condition=condition,
            F_g=F_g,
            M_g=M_g,
            span_effective=span_effective,
            oswalds_efficiency=e,
        )

    # ------------------------------------------------------------------ #
    # Fuselage aerodynamics
    # ------------------------------------------------------------------ #

    def fuselage_aerodynamics(
        self, fuselage: Fuselage, include_induced_drag: bool = True
    ) -> AeroComponentResults:
        """Estimate aerodynamic forces and moments on a fuselage.

        Uses Jorgensen (1977) slender-body theory for inviscid lift, plus
        profile drag from skin friction, base drag, and wave drag.

        Parameters
        ----------
        fuselage : Fuselage
        include_induced_drag : bool

        Returns
        -------
        AeroComponentResults
        """
        condition = self.condition
        length = fuselage.length()
        Re = condition.reynolds(reference_length=length)
        mach = condition.mach()
        q = condition.dynamic_pressure()
        eta = jorgensen_eta(fuselage.fineness_ratio())

        span_effective = softmax_scalefree(
            [xsec.area() ** 0.5 for xsec in fuselage.xsecs]
        )

        F_g = [0.0, 0.0, 0.0]
        M_g = [0.0, 0.0, 0.0]

        # Fuselage cross-section centroids in aircraft frame
        xyz_c = fuselage.xsec_centers()
        xsec_areas = [xsec.area() for xsec in fuselage.xsecs]

        sect_xyz_a = xyz_c[:-1]
        sect_xyz_b = xyz_c[1:]
        sect_xyz_mid = [
            [(a[i] + b[i]) / 2 for i in range(3)]
            for a, b in zip(sect_xyz_a, sect_xyz_b)
        ]
        sect_lengths = [
            float(np.sqrt(sum((b[i] - a[i])**2 for i in range(3))))
            for a, b in zip(sect_xyz_a, sect_xyz_b)
        ]
        sect_directions = [
            [
                np.where(
                    sect_lengths[i] != 0,
                    (sect_xyz_b[i][j] - sect_xyz_a[i][j]) / (sect_lengths[i] + 1e-100),
                    1 if j == 0 else 0,
                )
                for j in range(3)
            ]
            for i in range(len(sect_xyz_a))
        ]
        sect_areas = [
            (a + b + (a * b + 1e-100) ** 0.5) / 3
            for a, b in zip(xsec_areas[:-1], xsec_areas[1:])
        ]

        vel_dir_g = condition.convert_axes(-1, 0, 0, from_axes="wind", to_axes="geometry")

        sin_local_alpha_force = [
            [
                np.dot(s, vel_dir_g) * vel_dir_g[i] - s[i]
                for i in range(3)
            ]
            for s in sect_directions
        ]

        rho_V_sq = condition.density() * condition.velocity**2

        sin_local_alpha_moment = [
            list(np.cross(vel_dir_g, sect_dir))
            for sect_dir in sect_directions
        ]

        # Inviscid lift (Drela FVA Eq. 6.77)
        lift_at_nose = [
            rho_V_sq * xsec_areas[-1] * sin_local_alpha_force[-1][i]
            for i in range(3)
        ]
        # Drela FVA Eq. 6.78 — open-tail moment contribution
        moment_open_tail = [
            -rho_V_sq * sum(sect_lengths) * xsec_areas[-1] * sin_local_alpha_moment[-1][i]
            for i in range(3)
        ]
        # Shape contribution
        moment_shape = [
            rho_V_sq * sum(
                area * moment[i] * slen
                for area, moment, slen in zip(sect_areas, sin_local_alpha_moment, sect_lengths)
            )
            for i in range(3)
        ]

        lift_arm = [xyz_c[0][i] - self.xyz_ref[i] for i in range(3)]
        moment_lift = list(np.cross(lift_arm, lift_at_nose))

        for i in range(3):
            F_g[i] += lift_at_nose[i]
            M_g[i] += moment_open_tail[i] + moment_shape[i] + moment_lift[i]

        # Profile drag: base + skin friction + wave
        base_drag_coeff = fuselage_base_drag_coefficient(mach=mach)
        drag_base = base_drag_coeff * fuselage.area_base() * q

        form_factor = fuselage_form_factor(
            fineness_ratio=fuselage.fineness_ratio(),
            ratio_of_corner_radius_to_body_width=0.5,
        )
        C_f_ideal = (3.46 * np.log10(Re) - 5.6) ** -2
        drag_skin = C_f_ideal * form_factor * fuselage.area_wetted() * q

        if self.include_wave_drag:
            sears_haack_CDA = transonic.sears_haack_drag_from_volume(
                volume=fuselage.volume(), length=length
            )
            C_D_wave = transonic.approximate_CD_wave(
                mach=mach,
                mach_crit=critical_mach(fineness_ratio_nose=self.nose_fineness_ratio),
                CD_wave_at_fully_supersonic=self.E_wave_drag * sears_haack_CDA,
            )
        else:
            C_D_wave = 0.0

        drag_profile = drag_base + drag_skin + C_D_wave * q

        drag_profile_g = condition.convert_axes(-drag_profile, 0, 0, from_axes="wind", to_axes="geometry")
        drag_arm = [(xyz_c[0][i] + xyz_c[-1][i]) / 2 - self.xyz_ref[i] for i in range(3)]
        moment_drag = list(np.cross(drag_arm, drag_profile_g))

        for i in range(3):
            F_g[i] += drag_profile_g[i]
            M_g[i] += moment_drag[i]

        # Viscous crossflow
        sin_alphas = [sum(c**2 for c in s) ** 0.5 for s in sin_local_alpha_force]
        mean_radii = [(area / np.pi + 1e-100) ** 0.5 for area in sect_areas]

        Re_n_sect = [
            s * condition.reynolds(reference_length=2 * r)
            for s, r in zip(sin_alphas, mean_radii)
        ]
        mach_n_sect = [s * mach for s in sin_alphas]

        from aerisplane.aero.library.viscous import Cd_cylinder
        C_d_n = [
            np.where(Re_n_sect[i] != 0, Cd_cylinder(Re_D=Re_n_sect[i], mach=mach_n_sect[i]), 0)
            for i in range(len(fuselage.xsecs) - 1)
        ]

        vel_dot_x = [np.dot(vel_dir_g, d) for d in sect_directions]
        normal_dirs_unnorm = [
            [vel_dir_g[j] - vel_dot_x[i] * sect_directions[i][j] for j in range(3)]
            for i in range(len(sect_directions))
        ]
        for i in range(len(normal_dirs_unnorm)):
            normal_dirs_unnorm[i][2] += 1e-100
        normal_dirs_mag = [(sum(v**2 for v in n) + 1e-100) ** 0.5 for n in normal_dirs_unnorm]
        normal_dirs = [
            [normal_dirs_unnorm[i][j] / normal_dirs_mag[i] for j in range(3)]
            for i in range(len(normal_dirs_unnorm))
        ]

        F_crossflow = [
            sum(
                sect_lengths[i] * rho_V_sq * eta * C_d_n[i]
                * normal_dirs[i][j]
                * sum(c**2 for c in sin_local_alpha_force[i])
                * mean_radii[i]
                for i in range(len(fuselage.xsecs) - 1)
            )
            for j in range(3)
        ]
        M_crossflow = [
            sum(
                np.cross(
                    [sect_xyz_mid[i][k] - self.xyz_ref[k] for k in range(3)],
                    [
                        sect_lengths[i] * rho_V_sq * eta * C_d_n[i]
                        * normal_dirs[i][k]
                        * sum(c**2 for c in sin_local_alpha_force[i])
                        * mean_radii[i]
                        for k in range(3)
                    ],
                )[j]
                for i in range(len(fuselage.xsecs) - 1)
            )
            for j in range(3)
        ]

        for i in range(3):
            F_g[i] += F_crossflow[i]
            M_g[i] += M_crossflow[i]

        if include_induced_drag:
            _, sf, lf = condition.convert_axes(*F_g, from_axes="geometry", to_axes="wind")
            D_ind = (lf**2 + sf**2) / (q * np.pi * span_effective**2)
            D_ind_g = condition.convert_axes(-D_ind, 0, 0, from_axes="wind", to_axes="geometry")
            for i in range(3):
                F_g[i] += D_ind_g[i]

        return self.AeroComponentResults(
            s_ref=self.aircraft.reference_area(),
            c_ref=self.aircraft.reference_chord(),
            b_ref=self.aircraft.reference_span(),
            condition=condition,
            F_g=F_g,
            M_g=M_g,
            span_effective=span_effective,
            oswalds_efficiency=0.95,
        )
