"""Lateral-directional linear model: A-matrix, eigenvalue analysis, time responses.

The standard 4-state small-perturbation lateral model (Nelson, *Flight Stability
and Automatic Control*, 2nd ed.) is used:

    States:  x = [β (rad), p (rad/s), r (rad/s), φ (rad)]

    β_dot = (Yβ·β + Yp·p + Yr·r) / V  + (Yr/V − 1)·r  + g·cosθ₀/V · φ
    p_dot = Γ · (Izz·Lβ·β + Izz·Lp·p + Izz·Lr·r + Ixz·Nβ·β + ...)
    r_dot = Γ · (Ixx·Nβ·β + ...)
    φ_dot = p + r·tan(θ₀)

where Γ = 1/(Ixx·Izz − Ixz²) and L·, N·, Y· are dimensional derivatives.

Dimensional derivatives are computed from the nondimensional stability
coefficients (1/deg or per-unit-rate-hat) via:

    Y_β = q·S · CY_β_rad
    L_β = q·S·b · Cl_β_rad
    N_β = q·S·b · Cn_β_rad
    L_p = q·S·b² · Cl_p / (2V)    (Cl_p is per pb/2V)
    etc.

Control B-matrix (if aileron / rudder found):
    Column 0: aileron δa [rad]
    Column 1: rudder  δr [rad]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aerisplane.stability.derivatives import DerivativeResult
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.weights.result import WeightResult
from aerisplane.core.aircraft import Aircraft


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LateralMode:
    """Properties of a single lateral-directional mode."""
    name: str                        # "roll", "dutch_roll", "spiral"
    eigenvalue: complex              # [1/s]
    frequency: float = float("nan") # natural frequency [rad/s]  (Dutch roll only)
    damping: float = float("nan")   # damping ratio              (Dutch roll only)
    time_constant: float = float("nan")  # |1/Re(λ)| [s]          (real modes)
    stable: bool = True


@dataclass
class LateralModes:
    """All three lateral-directional modes from the A-matrix eigenanalysis."""
    roll: LateralMode
    dutch_roll: LateralMode
    spiral: LateralMode
    eigenvectors: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), complex))
    # Column order: [roll, dutch_roll, dutch_roll_conj, spiral]


@dataclass
class TimeResponse:
    """Time-history response from a linear solve_ivp integration.

    States: β [deg], p [deg/s], r [deg/s], φ [deg]
    """
    t: np.ndarray       # time [s]
    beta: np.ndarray    # sideslip [deg]
    p: np.ndarray       # roll rate [deg/s]
    r: np.ndarray       # yaw rate [deg/s]
    phi: np.ndarray     # bank angle [deg]
    label: str = ""     # description of the excitation


# ─────────────────────────────────────────────────────────────────────────────
# A-matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_lateral_matrix(
    deriv: DerivativeResult,
    condition: FlightCondition,
    weight_result: WeightResult,
    aircraft: Aircraft,
) -> np.ndarray:
    """Build the 4×4 lateral-directional A-matrix.

    Parameters
    ----------
    deriv : DerivativeResult
        Must include rate derivatives (CY_p, Cl_p, Cn_p, CY_r, Cl_r, Cn_r).
    condition : FlightCondition
        Baseline flight condition (provides V, θ₀).
    weight_result : WeightResult
        Provides inertia tensor.
    aircraft : Aircraft
        Provides reference geometry.

    Returns
    -------
    A : (4, 4) ndarray
        State matrix.  States: [β (rad), p (rad/s), r (rad/s), φ (rad)].

    Raises
    ------
    ValueError
        If rate derivatives are not available in deriv.
    """
    _check_rate_derivs(deriv)

    q   = condition.dynamic_pressure()
    V   = float(condition.velocity)
    g   = 9.81
    S   = aircraft.reference_area()
    b   = aircraft.reference_span()
    # Trim pitch angle ≈ trim alpha (small angle, level flight)
    theta0 = np.radians(float(condition.alpha))

    # Inertia (about CG, body axes)
    I = weight_result.inertia_tensor
    Ixx = float(I[0, 0])
    Ixz = float(I[0, 2])
    Izz = float(I[2, 2])

    # Fallback: estimate inertia from geometry when the tensor is zero/degenerate.
    # Simple ellipsoid approximation: Ixx ~ m*(b/2)^2/4, Izz ~ m*L^2/12
    det_I = Ixx * Izz - Ixz**2
    if abs(det_I) < 1e-12 or Ixx <= 0 or Izz <= 0:
        import warnings
        m = float(weight_result.total_mass)
        b_ref = aircraft.reference_span()
        # Estimate aircraft length from the fuselage, or 2× MAC as fallback
        if aircraft.fuselages:
            xsecs = aircraft.fuselages[0].xsecs
            L = abs(xsecs[-1].xyz_c[0] - xsecs[0].xyz_c[0]) if len(xsecs) >= 2 else 2.0 * c
        else:
            L = 2.0 * aircraft.reference_chord()
        Ixx = m * (b_ref / 2.0) ** 2 / 4.0
        Izz = m * L ** 2 / 12.0
        Ixz = 0.0
        det_I = Ixx * Izz
        warnings.warn(
            "Inertia tensor is zero — using geometric estimate: "
            f"Ixx = {Ixx:.4f} kg·m², Izz = {Izz:.4f} kg·m².  "
            "Provide a measured inertia tensor for accurate dynamics.",
            UserWarning,
            stacklevel=3,
        )
    Gamma = 1.0 / det_I

    # Convert static derivatives from 1/deg → 1/rad
    d2r = np.degrees(1)   # 57.296  (multiply to go 1/deg → 1/rad)
    CY_b = deriv.CY_beta * d2r
    Cl_b = deriv.Cl_beta * d2r
    Cn_b = deriv.Cn_beta * d2r

    # Rate derivatives are per unit nondimensional rate (pb/2V or rb/2V)
    # Convert to per rad/s:  X_p_dim = X_p_hat * b/(2V)
    # Then dimensional moment:  L = qSb * Cl;  so L_p = qSb * Cl_p_hat/(2V) * b  = qSb²·Cl_p/(2V)
    # We use the shorthand below:
    CY_p = float(deriv.CY_p) if deriv.CY_p is not None else 0.0
    Cl_p = float(deriv.Cl_p) if deriv.Cl_p is not None else 0.0
    Cn_p = float(deriv.Cn_p) if deriv.Cn_p is not None else 0.0
    CY_r = float(deriv.CY_r) if deriv.CY_r is not None else 0.0
    Cl_r = float(deriv.Cl_r) if deriv.Cl_r is not None else 0.0
    Cn_r = float(deriv.Cn_r) if deriv.Cn_r is not None else 0.0

    # Dimensional side-force derivatives [N / (rad or rad/s)]
    Y_beta = q * S * CY_b
    Y_p    = q * S * b * CY_p / (2.0 * V)
    Y_r    = q * S * b * CY_r / (2.0 * V)

    # Dimensional rolling-moment derivatives [Nm / (rad or rad/s)]
    L_beta = q * S * b * Cl_b
    L_p    = q * S * b**2 * Cl_p / (2.0 * V)
    L_r    = q * S * b**2 * Cl_r / (2.0 * V)

    # Dimensional yawing-moment derivatives [Nm / (rad or rad/s)]
    N_beta = q * S * b * Cn_b
    N_p    = q * S * b**2 * Cn_p / (2.0 * V)
    N_r    = q * S * b**2 * Cn_r / (2.0 * V)

    # Inertia-coupled roll and yaw accelerations per unit moment:
    # [p_dot]   1/(IxxIzz-Ixz²) · [Izz  Ixz] · [L]
    # [r_dot] =                   · [Ixz  Ixx]   [N]

    A = np.zeros((4, 4))

    # Row 0 — β_dot / V
    A[0, 0] = Y_beta / V
    A[0, 1] = Y_p / V
    A[0, 2] = Y_r / V - 1.0
    A[0, 3] = g * np.cos(theta0) / V

    # Row 1 — p_dot (roll)
    A[1, 0] = Gamma * (Izz * L_beta + Ixz * N_beta)
    A[1, 1] = Gamma * (Izz * L_p    + Ixz * N_p)
    A[1, 2] = Gamma * (Izz * L_r    + Ixz * N_r)
    A[1, 3] = 0.0

    # Row 2 — r_dot (yaw)
    A[2, 0] = Gamma * (Ixz * L_beta + Ixx * N_beta)
    A[2, 1] = Gamma * (Ixz * L_p    + Ixx * N_p)
    A[2, 2] = Gamma * (Ixz * L_r    + Ixx * N_r)
    A[2, 3] = 0.0

    # Row 3 — φ_dot
    A[3, 0] = 0.0
    A[3, 1] = 1.0
    A[3, 2] = np.tan(theta0)
    A[3, 3] = 0.0

    return A


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue analysis → mode identification
# ─────────────────────────────────────────────────────────────────────────────

def analyze_modes(A: np.ndarray) -> LateralModes:
    """Compute eigenvalues and identify the three lateral-directional modes.

    Identification heuristic:
    - Roll subsidence: most negative real eigenvalue (fast, real)
    - Dutch roll: complex conjugate pair (positive imaginary part)
    - Spiral: smallest absolute real eigenvalue (slow, real)

    Parameters
    ----------
    A : (4, 4) ndarray

    Returns
    -------
    LateralModes
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Separate real and complex eigenvalues
    real_eigs = [(ev, i) for i, ev in enumerate(eigenvalues)
                 if abs(ev.imag) < 1e-6 * max(abs(ev.real), 1e-10)]
    complex_eigs = [(ev, i) for i, ev in enumerate(eigenvalues) if abs(ev.imag) >= 1e-6]

    # Dutch roll: complex pair
    dutch_roll_ev = complex(0.0)
    dutch_roll_idx = -1
    if complex_eigs:
        # Pick the one with the largest imaginary part
        complex_eigs.sort(key=lambda t: abs(t[0].imag), reverse=True)
        dutch_roll_ev = complex_eigs[0][0]
        dutch_roll_idx = complex_eigs[0][1]

    # Real modes: sort by real part (most negative first)
    real_eigs.sort(key=lambda t: t[0].real)
    roll_ev    = real_eigs[0][0] if len(real_eigs) > 0 else complex(float("nan"))
    spiral_ev  = real_eigs[-1][0] if len(real_eigs) > 1 else real_eigs[0][0]
    roll_idx   = real_eigs[0][1] if len(real_eigs) > 0 else -1
    spiral_idx = real_eigs[-1][1] if len(real_eigs) > 1 else -1

    # Build mode objects
    def _time_const(ev):
        re = float(ev.real)
        return abs(1.0 / re) if abs(re) > 1e-12 else float("inf")

    roll_mode = LateralMode(
        name="roll",
        eigenvalue=roll_ev,
        time_constant=_time_const(roll_ev),
        stable=float(roll_ev.real) < 0,
    )

    dr_omega = float(abs(dutch_roll_ev.imag))
    dr_zeta  = float(-dutch_roll_ev.real / abs(dutch_roll_ev)) if abs(dutch_roll_ev) > 1e-12 else 0.0
    dutch_roll_mode = LateralMode(
        name="dutch_roll",
        eigenvalue=dutch_roll_ev,
        frequency=dr_omega,
        damping=dr_zeta,
        time_constant=_time_const(dutch_roll_ev),
        stable=float(dutch_roll_ev.real) < 0,
    )

    spiral_mode = LateralMode(
        name="spiral",
        eigenvalue=spiral_ev,
        time_constant=_time_const(spiral_ev),
        stable=float(spiral_ev.real) <= 0,
    )

    # Eigenvector matrix in column order [roll, dutch_roll, dutch_roll*, spiral]
    evecs = np.zeros((4, 4), complex)
    if roll_idx >= 0:
        evecs[:, 0] = eigenvectors[:, roll_idx]
    if dutch_roll_idx >= 0:
        evecs[:, 1] = eigenvectors[:, dutch_roll_idx]
        evecs[:, 2] = eigenvectors[:, dutch_roll_idx].conj()
    if spiral_idx >= 0:
        evecs[:, 3] = eigenvectors[:, spiral_idx]

    return LateralModes(
        roll=roll_mode,
        dutch_roll=dutch_roll_mode,
        spiral=spiral_mode,
        eigenvectors=evecs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Control B-matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_control_matrix(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    deriv: DerivativeResult,
    aero_method: str = "aero_buildup",
    delta_deg: float = 2.0,
    **aero_kwargs,
) -> Optional[np.ndarray]:
    """Build the 4×2 control B-matrix via finite differences.

    Columns: [δ_aileron (rad), δ_rudder (rad)]
    Rows:    [β_dot, p_dot, r_dot, φ_dot]

    Returns None if neither aileron nor rudder is found.

    Parameters
    ----------
    delta_deg : float
        Deflection step for finite differences [deg].
    """
    import copy
    from aerisplane.aero import analyze as _aero

    aileron_name = _find_surface(aircraft, ("aileron", "ail"))
    rudder_name  = _find_surface(aircraft, ("rudder", "rud"))

    if aileron_name is None and rudder_name is None:
        return None

    q   = condition.dynamic_pressure()
    V   = float(condition.velocity)
    S   = aircraft.reference_area()
    b   = aircraft.reference_span()

    ac = copy.deepcopy(aircraft)
    ac.xyz_ref = weight_result.cg.tolist()

    def _coeff(surface_name, deflection):
        cond = condition.copy()
        cond.deflections = {**condition.deflections, surface_name: deflection}
        r = _aero(ac, cond, method=aero_method, **aero_kwargs)
        return r.CY, r.Cl, r.Cn

    I = weight_result.inertia_tensor
    Ixx = float(I[0, 0]); Ixz = float(I[0, 2]); Izz = float(I[2, 2])
    det_I = Ixx * Izz - Ixz**2
    if abs(det_I) < 1e-12 or Ixx <= 0 or Izz <= 0:
        m = float(weight_result.total_mass)
        b_ref = aircraft.reference_span()
        if aircraft.fuselages:
            xsecs = aircraft.fuselages[0].xsecs
            L_est = abs(xsecs[-1].xyz_c[0] - xsecs[0].xyz_c[0]) if len(xsecs) >= 2 else 2.0 * aircraft.reference_chord()
        else:
            L_est = 2.0 * aircraft.reference_chord()
        Ixx = m * (b_ref / 2.0) ** 2 / 4.0
        Izz = m * L_est ** 2 / 12.0
        Ixz = 0.0
        det_I = Ixx * Izz
    Gamma = 1.0 / det_I if abs(det_I) > 1e-30 else 0.0

    B = np.zeros((4, 2))

    for col, surface in enumerate([aileron_name, rudder_name]):
        if surface is None:
            continue
        CY_p, Cl_p, Cn_p = _coeff(surface, +delta_deg)
        CY_m, Cl_m, Cn_m = _coeff(surface, -delta_deg)

        dCY = (CY_p - CY_m) / (2.0 * delta_deg)   # 1/deg
        dCl = (Cl_p - Cl_m) / (2.0 * delta_deg)
        dCn = (Cn_p - Cn_m) / (2.0 * delta_deg)

        # Convert to dimensional and per radian of input
        d2r_coeff = np.degrees(1)   # 57.296: convert derivative from /deg to /rad
        Y_d = q * S * dCY * d2r_coeff
        L_d = q * S * b * dCl * d2r_coeff
        N_d = q * S * b * dCn * d2r_coeff

        B[0, col] = Y_d / V
        B[1, col] = Gamma * (Izz * L_d + Ixz * N_d)
        B[2, col] = Gamma * (Ixz * L_d + Ixx * N_d)
        B[3, col] = 0.0

    return B


def _find_surface(aircraft: Aircraft, keywords: tuple) -> Optional[str]:
    for wing in aircraft.wings:
        for cs in getattr(wing, "control_surfaces", []):
            if any(kw in cs.name.lower() for kw in keywords):
                return cs.name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Time responses
# ─────────────────────────────────────────────────────────────────────────────

def simulate_response(
    A: np.ndarray,
    x0_deg,
    t_end: float,
    n_points: int = 500,
    B: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    label: str = "",
) -> TimeResponse:
    """Integrate the lateral linear model from initial conditions x0.

    Parameters
    ----------
    A : (4, 4) ndarray
        State matrix (states in rad/rad-s).
    x0_deg : array-like, length 4
        Initial state [β_deg, p_deg/s, r_deg/s, φ_deg].
    t_end : float
        Simulation duration [s].
    n_points : int
        Number of output time points.
    B : (4, 2) ndarray or None
        Control matrix.  Used only if u is also provided.
    u : (2,) ndarray or None
        Constant control input [δa_rad, δr_rad] for step-response simulations.
    label : str
        Description stored in the result.

    Returns
    -------
    TimeResponse
        States converted back to degrees / deg/s for readability.
    """
    from scipy.integrate import solve_ivp

    x0 = np.radians(np.asarray(x0_deg, dtype=float))

    if B is not None and u is not None:
        u_vec = np.asarray(u, dtype=float)
        def rhs(t, x):
            return A @ x + B @ u_vec
    else:
        def rhs(t, x):
            return A @ x

    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval,
                    method="RK45", rtol=1e-6, atol=1e-8,
                    max_step=t_end / 50.0)

    # Guard against solver divergence (unstable system)
    y = np.nan_to_num(sol.y, nan=0.0, posinf=0.0, neginf=0.0)
    x_deg = np.degrees(y)
    return TimeResponse(
        t=sol.t,
        beta=x_deg[0],
        p=x_deg[1],
        r=x_deg[2],
        phi=x_deg[3],
        label=label,
    )


def compute_standard_responses(
    A: np.ndarray,
    modes: LateralModes,
    B: Optional[np.ndarray] = None,
) -> dict[str, TimeResponse]:
    """Compute Dutch roll, roll subsidence, and spiral standard responses.

    Parameters
    ----------
    A : (4, 4) ndarray
    modes : LateralModes
    B : optional control matrix

    Returns
    -------
    dict with keys "dutch_roll", "roll_subsidence", "spiral",
    and optionally "aileron_step", "rudder_step".
    """
    responses: dict[str, TimeResponse] = {}

    # Dutch roll: initial sideslip β₀ = 5°
    t_end_dr = _safe_period(modes.dutch_roll, n_periods=10, fallback=15.0)
    responses["dutch_roll"] = simulate_response(
        A, [5.0, 0.0, 0.0, 0.0], t_end_dr, label="β₀ = 5°"
    )

    # Roll subsidence: initial roll rate p₀ = 20°/s
    t_end_roll = min(modes.roll.time_constant * 5.0, 5.0) if np.isfinite(modes.roll.time_constant) else 3.0
    responses["roll_subsidence"] = simulate_response(
        A, [0.0, 20.0, 0.0, 0.0], t_end_roll, label="p₀ = 20°/s"
    )

    # Spiral: initial bank angle φ₀ = 5°
    t_end_sp = min(abs(modes.spiral.time_constant) * 2.0, 120.0) if np.isfinite(modes.spiral.time_constant) else 60.0
    responses["spiral"] = simulate_response(
        A, [0.0, 0.0, 0.0, 5.0], t_end_sp, label="φ₀ = 5°"
    )

    # Control step responses (if B-matrix available)
    if B is not None:
        responses["aileron_step"] = simulate_response(
            A, [0.0, 0.0, 0.0, 0.0], t_end_roll * 1.5,
            B=B, u=np.array([np.radians(10.0), 0.0]),
            label="δa = 10°",
        )
        responses["rudder_step"] = simulate_response(
            A, [0.0, 0.0, 0.0, 0.0], t_end_dr * 0.5,
            B=B, u=np.array([0.0, np.radians(10.0)]),
            label="δr = 10°",
        )

    return responses


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_rate_derivs(deriv: DerivativeResult) -> None:
    if deriv.Cl_p is None or deriv.Cn_r is None:
        raise ValueError(
            "Rate derivatives (Cl_p, Cn_r, …) are required for the lateral "
            "A-matrix.  Re-run compute_derivatives() with "
            "compute_rate_derivatives=True."
        )


def _safe_period(mode: LateralMode, n_periods: float, fallback: float) -> float:
    if np.isfinite(mode.frequency) and mode.frequency > 0:
        return n_periods * 2.0 * np.pi / mode.frequency
    return fallback


__all__ = [
    "build_lateral_matrix",
    "analyze_modes",
    "build_control_matrix",
    "simulate_response",
    "compute_standard_responses",
    "LateralMode",
    "LateralModes",
    "TimeResponse",
]
