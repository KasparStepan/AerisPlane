"""Aerodynamic analysis module.

Public API
----------
analyze(aircraft, condition, method="vlm", **kwargs)
    Run a full aerodynamic analysis and return an AeroResult.

plot_geometry(aircraft, style="three_view", show=True, save_path=None)
    Plot aircraft geometry.
    style="three_view" → 4-panel engineering drawing (top/front/side/iso).
    style="wireframe"  → single 3-D matplotlib wireframe.

Supported methods
-----------------
"vlm"
    Vortex Lattice Method (inviscid). Fast, handles arbitrary geometry.
"lifting_line"
    Lifting-line with NeuralFoil section polars. Viscous + nonlinear.
"nonlinear_lifting_line"
    Nonlinear lifting-line: fixed-point iteration updating the LL RHS
    with NeuralFoil CL at effective (geometric + induced) alpha.
    Captures stall and spanwise stall progression.
"aero_buildup"
    Workbook-style AeroBuildup: NeuralFoil wings + Jorgensen fuselage.

Example
-------
>>> from aerisplane.aero import analyze
>>> result = analyze(aircraft, condition, method="vlm")
>>> result.report()
"""

from __future__ import annotations

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.aero.result import AeroResult


def analyze(
    aircraft: Aircraft,
    condition: FlightCondition,
    method: str = "vlm",
    spanwise_resolution: int = 8,
    chordwise_resolution: int = 4,
    model_size: str = "medium",
    verbose: bool = False,
) -> AeroResult:
    """Run an aerodynamic analysis.

    Parameters
    ----------
    aircraft : Aircraft
        aerisplane aircraft definition.
    condition : FlightCondition
        Operating point (velocity, altitude, alpha, beta, deflections).
    method : str
        Solver method: "vlm" (default), "lifting_line",
        "nonlinear_lifting_line", "aero_buildup".
    spanwise_resolution : int
        Spanwise panels / stations per wing section.
        Applies to VLM and LiftingLine. Default 8.
    chordwise_resolution : int
        Chordwise panels per section. VLM only. Default 4.
    model_size : str
        NeuralFoil model size for LiftingLine and AeroBuildup.
        Options: "xxsmall" … "xxlarge". Default "medium".
    verbose : bool
        Print solver progress. Default False.

    Returns
    -------
    AeroResult
        Full aerodynamic result including forces, moments, and coefficients.
    """
    return _run_native(
        aircraft=aircraft,
        condition=condition,
        method=method,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        model_size=model_size,
        verbose=verbose,
    )


def _run_native(
    aircraft: Aircraft,
    condition: FlightCondition,
    method: str,
    spanwise_resolution: int,
    chordwise_resolution: int,
    model_size: str,
    verbose: bool,
) -> AeroResult:
    """Dispatch to vendored native solvers and wrap the result in AeroResult."""

    import numpy as np

    q = condition.dynamic_pressure()
    S = aircraft.reference_area()
    c = aircraft.reference_chord()
    b = aircraft.reference_span()
    qS = q * S

    solver = None

    if method == "vlm":
        from aerisplane.aero.solvers.vlm import VortexLatticeMethod
        solver = VortexLatticeMethod(
            aircraft=aircraft,
            condition=condition,
            spanwise_resolution=spanwise_resolution,
            chordwise_resolution=chordwise_resolution,
            verbose=verbose,
        )
        d = solver.run()
        CDi = float(d["CD"])   # VLM is inviscid; all resolved drag is induced
        CDp = None

    elif method == "aero_buildup":
        from aerisplane.aero.solvers.aero_buildup import AeroBuildup
        solver = AeroBuildup(
            aircraft=aircraft,
            condition=condition,
            model_size=model_size,
        )
        d = solver.run()
        CDi = float(d["D_induced"]) / qS if qS > 0 else 0.0
        CDp = float(d["D_profile"]) / qS if qS > 0 else 0.0

    elif method == "lifting_line":
        from aerisplane.aero.solvers.lifting_line import LiftingLine
        solver = LiftingLine(
            aircraft=aircraft,
            condition=condition,
            model_size=model_size,
            spanwise_resolution=spanwise_resolution,
            verbose=verbose,
        )
        d = solver.run()
        CDi = None
        CDp = None

    elif method == "nonlinear_lifting_line":
        from aerisplane.aero.solvers.nonlinear_lifting_line import NonlinearLiftingLine
        solver = NonlinearLiftingLine(
            aircraft=aircraft,
            condition=condition,
            model_size=model_size,
            spanwise_resolution=spanwise_resolution,
            verbose=verbose,
        )
        d = solver.run()
        CDi = None
        CDp = None

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Supported native methods: 'vlm', 'lifting_line', "
            "'nonlinear_lifting_line', 'aero_buildup'."
        )

    def _to_list(v):
        if hasattr(v, "tolist"):
            return [float(x) for x in v.tolist()]
        return [float(x) for x in v]

    return AeroResult(
        method=method,
        # Forces (wind axes)
        L=float(d["L"]),
        D=float(d["D"]),
        Y=float(d["Y"]),
        # Moments (body axes)
        l_b=float(d["l_b"]),
        m_b=float(d["m_b"]),
        n_b=float(d["n_b"]),
        # Force/moment vectors
        F_g=_to_list(d["F_g"]),
        F_b=_to_list(d["F_b"]),
        F_w=_to_list(d["F_w"]),
        M_g=_to_list(d["M_g"]),
        M_b=_to_list(d["M_b"]),
        M_w=_to_list(d["M_w"]),
        # Coefficients
        CL=float(d["CL"]),
        CD=float(d["CD"]),
        CY=float(d["CY"]),
        Cl=float(d["Cl"]),
        Cm=float(d["Cm"]),
        Cn=float(d["Cn"]),
        # Drag breakdown
        CDi=CDi,
        CDp=CDp,
        # Operating condition
        alpha=float(condition.alpha),
        beta=float(condition.beta),
        velocity=float(condition.velocity),
        altitude=float(condition.altitude),
        dynamic_pressure=float(q),
        reynolds_number=float(condition.reynolds_number(c)) if c > 0 else 0.0,
        # Reference geometry
        s_ref=float(S),
        c_ref=float(c),
        b_ref=float(b),
        # Solver object (for post-processing like spanwise loading plots)
        _solver=solver,
        _airplane=aircraft,
    )


def plot_geometry(aircraft, style="three_view", show=True, save_path=None):
    """Plot aircraft geometry.

    Parameters
    ----------
    aircraft : Aircraft
        aerisplane aircraft to visualise.
    style : str
        ``"three_view"`` (default) — 4-panel top/front/side/isometric view.
        ``"wireframe"`` — single 3-D matplotlib wireframe.
        ``"original"`` — alias for ``"three_view"``.
    show : bool
        Whether to call ``plt.show()`` after drawing.
    save_path : str or None
        If given, save the figure to this path.
    """
    if style == "original":
        style = "three_view"
    from aerisplane.aero.plot import plot_geometry as _plot_geometry
    return _plot_geometry(aircraft=aircraft, style=style, show=show, save_path=save_path)


__all__ = ["analyze", "plot_geometry", "AeroResult"]
