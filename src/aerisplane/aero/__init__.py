"""Aerodynamic analysis module.

Public API
----------
analyze(aircraft, condition, backend="aerosandbox", method="vlm", **kwargs)
    Run a full aerodynamic analysis and return an AeroResult.

plot_geometry(aircraft, style="three_view", show=True, save_path=None)
    Plot aircraft geometry using AeroSandbox's matplotlib drawing routines.
    style="three_view" → 4-panel engineering drawing (top/front/side/iso).
    style="wireframe"  → single 3-D matplotlib wireframe.

Supported backends
------------------
"aerosandbox"
    AeroSandbox vortex lattice, lifting line, nonlinear lifting line,
    and AeroBuildup (NeuralFoil). Default.

"openaerostruct"
    OpenAeroStruct adapter. Not yet implemented.

Example
-------
>>> from aerisplane.aero import analyze, plot_geometry
>>> plot_geometry(aircraft)
>>> result = analyze(aircraft, condition, method="vlm")
>>> result.report()
>>> result.plot_spanwise_loading()
"""

from __future__ import annotations

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.aero.result import AeroResult


def analyze(
    aircraft: Aircraft,
    condition: FlightCondition,
    backend: str = "aerosandbox",
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
    backend : str
        Solver backend. Currently only "aerosandbox" is supported.
    method : str
        Solver method within the backend. For "aerosandbox":
        "vlm" (default), "lifting_line", "nonlinear_lifting_line", "aero_buildup".
    spanwise_resolution : int
        Number of spanwise panels / stations per wing section.
        Applies to VLM, LiftingLine, NonlinearLiftingLine. Default 8.
    chordwise_resolution : int
        Number of chordwise panels per section. VLM only. Default 4.
    model_size : str
        NeuralFoil model size for LiftingLine and AeroBuildup.
        Options: "xxsmall", "xsmall", "small", "medium", "large", "xlarge",
        "xxlarge", "xxxlarge". Default "medium".
    verbose : bool
        Print solver progress. Default False.

    Returns
    -------
    AeroResult
        Full aerodynamic result including forces, moments, and coefficients.

    Raises
    ------
    ValueError
        If an unsupported backend is requested.
    """
    if backend == "aerosandbox":
        from aerisplane.aero.aerosandbox_backend import run_asb

        return run_asb(
            aircraft=aircraft,
            condition=condition,
            method=method,
            spanwise_resolution=spanwise_resolution,
            chordwise_resolution=chordwise_resolution,
            model_size=model_size,
            verbose=verbose,
        )

    elif backend == "openaerostruct":
        raise NotImplementedError(
            "OpenAeroStruct backend is not yet implemented. "
            "Use backend='aerosandbox' for now."
        )

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            "Supported backends: 'aerosandbox'."
        )


def plot_geometry(aircraft, style="three_view", show=True, save_path=None):
    """Plot aircraft geometry using AeroSandbox's matplotlib drawing routines.

    Parameters
    ----------
    aircraft : Aircraft
        aerisplane aircraft to visualise.
    style : str
        ``"three_view"`` (default) — 4-panel top/front/side/isometric view.
        ``"wireframe"`` — single 3-D matplotlib wireframe.
    show : bool
        Whether to call ``plt.show()`` after drawing.
    save_path : str or None
        If given, save the figure to this path.
    """
    from aerisplane.aero.aerosandbox_backend import plot_geometry as _plot_geometry
    return _plot_geometry(aircraft=aircraft, style=style, show=show, save_path=save_path)


__all__ = ["analyze", "plot_geometry", "AeroResult"]
