"""AeroSandbox backend adapter for aerisplane.

Translates aerisplane core objects (Aircraft, FlightCondition) into
AeroSandbox equivalents, runs the selected solver, and returns an AeroResult.

Supported methods
-----------------
"vlm"
    Vortex Lattice Method — inviscid, fast.
    CDi available; CDp = None (no viscous model).

"lifting_line"
    Linear lifting line with NeuralFoil section polars.
    CDi available; CDp available via section integration.

"nonlinear_lifting_line"
    Nonlinear lifting line — iterates until CL_section matches NeuralFoil.
    CDi available; CDp available.

"aero_buildup"
    AeroBuildup (NeuralFoil per section + induced drag). Fastest viscous method.
    Both CDi and CDp available.

Geometry plotting
-----------------
plot_geometry(aircraft, style, show, save_path)
    Translate aircraft to ASB and call AeroSandbox's built-in drawing routines
    directly (draw_wireframe or draw_three_view).  Requires matplotlib.

Control surface note
--------------------
FlightCondition.deflections values are acknowledged but NOT currently passed to
AeroSandbox, because ASB control surface support is broken upstream. The
translator stubs out the control surfaces with an empty list.

When an AeroSandbox fork with working control surfaces is ready:
1. Uncomment _control_surfaces_to_asb() below.
2. Pass the deflections dict into _wing_to_asb().
3. Remove this note.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import aerosandbox as asb

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.fuselage import Fuselage, FuselageXSec
from aerisplane.core.wing import Wing, WingXSec
from aerisplane.utils.atmosphere import isa

from aerisplane.aero.result import AeroResult


# ------------------------------------------------------------------ #
# Translator helpers
# ------------------------------------------------------------------ #

def _airfoil_to_asb(airfoil: Airfoil) -> asb.Airfoil:
    """Translate an aerisplane Airfoil to an AeroSandbox Airfoil.

    If coordinates are available they are passed directly.
    NACA names and database names are passed as-is for ASB to resolve.
    """
    if airfoil.coordinates is not None:
        return asb.Airfoil(name=airfoil.name, coordinates=airfoil.coordinates)
    return asb.Airfoil(name=airfoil.name)


def _xsec_to_asb(xsec: WingXSec) -> asb.WingXSec:
    """Translate a WingXSec to an ASB WingXSec.

    Control surfaces are currently stubbed out — see module docstring.
    """
    return asb.WingXSec(
        xyz_le=xsec.xyz_le,
        chord=xsec.chord,
        twist=xsec.twist,
        airfoil=_airfoil_to_asb(xsec.airfoil),
        control_surfaces=[],   # deferred — see module docstring
    )


def _wing_to_asb(wing: Wing) -> asb.Wing:
    """Translate an aerisplane Wing to an ASB Wing."""
    return asb.Wing(
        name=wing.name,
        xsecs=[_xsec_to_asb(xs) for xs in wing.xsecs],
        symmetric=wing.symmetric,
    )


def _fuselage_xsec_to_asb(
    xsec: FuselageXSec,
    nose_x: float,
    nose_y: float,
    nose_z: float,
) -> asb.FuselageXSec:
    """Translate a FuselageXSec to an ASB FuselageXSec.

    aerisplane stores axial position as distance from nose (xsec.x).
    ASB expects the 3-D center point of the cross-section (xyz_c).
    """
    xyz_c = [nose_x + xsec.x, nose_y, nose_z]

    if xsec.shape == "circle":
        return asb.FuselageXSec(
            xyz_c=xyz_c,
            xyz_normal=[1.0, 0.0, 0.0],
            radius=xsec.radius,
        )
    elif xsec.shape in ("ellipse", "rectangle") and xsec.width and xsec.height:
        # ASB FuselageXSec uses superellipse parameterisation.
        # shape=2 → ellipse, shape=1000 → rectangle (close enough).
        asb_shape = 2.0 if xsec.shape == "ellipse" else 1000.0
        return asb.FuselageXSec(
            xyz_c=xyz_c,
            xyz_normal=[1.0, 0.0, 0.0],
            width=xsec.width,
            height=xsec.height,
            shape=asb_shape,
        )
    else:
        # Fall back to circular with the stored radius.
        return asb.FuselageXSec(
            xyz_c=xyz_c,
            xyz_normal=[1.0, 0.0, 0.0],
            radius=xsec.radius,
        )


def _fuselage_to_asb(fuselage: Fuselage) -> asb.Fuselage:
    """Translate an aerisplane Fuselage to an ASB Fuselage."""
    xsecs = [
        _fuselage_xsec_to_asb(xs, fuselage.x_le, fuselage.y_le, fuselage.z_le)
        for xs in fuselage.xsecs
    ]
    return asb.Fuselage(name=fuselage.name, xsecs=xsecs)


def _condition_to_asb(condition: FlightCondition) -> asb.OperatingPoint:
    """Translate a FlightCondition to an ASB OperatingPoint."""
    atm = asb.Atmosphere(altitude=condition.altitude)
    return asb.OperatingPoint(
        atmosphere=atm,
        velocity=condition.velocity,
        alpha=condition.alpha,
        beta=condition.beta,
    )


def aircraft_to_asb(aircraft: Aircraft) -> asb.Airplane:
    """Translate an aerisplane Aircraft to an ASB Airplane.

    The moment reference point (xyz_ref) is set to the aerodynamic center
    of the main wing (quarter-chord of the mean aerodynamic chord).

    Parameters
    ----------
    aircraft : Aircraft
        aerisplane aircraft to translate.

    Returns
    -------
    asb.Airplane
        Ready-to-use AeroSandbox airplane.
    """
    main_wing = aircraft.main_wing()
    if main_wing is not None:
        xyz_ref = main_wing.aerodynamic_center().tolist()
    else:
        xyz_ref = [0.0, 0.0, 0.0]

    return asb.Airplane(
        name=aircraft.name,
        xyz_ref=xyz_ref,
        wings=[_wing_to_asb(w) for w in aircraft.wings],
        fuselages=[_fuselage_to_asb(f) for f in aircraft.fuselages],
        s_ref=aircraft.reference_area(),
        c_ref=aircraft.reference_chord(),
        b_ref=aircraft.reference_span(),
    )


# ------------------------------------------------------------------ #
# Raw result extraction helpers
# ------------------------------------------------------------------ #

def _float(value: Any) -> float:
    """Safely convert a scalar, 0-d array, or 1-element array to Python float.

    numpy 2.x raises TypeError when calling float() on a 1-D array even if it
    has only one element, so we flatten to a scalar via .item() when needed.
    """
    try:
        arr = np.asarray(value, dtype=float)
        return float(arr.item())
    except Exception:
        return 0.0


def _vec3(value: Any) -> list[float]:
    """Convert an [x, y, z] array-like to a plain Python list of floats."""
    try:
        arr = np.asarray(value, dtype=float).flatten()
        return [float(arr[0]), float(arr[1]), float(arr[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def _extract_drag_breakdown(
    raw: dict[str, Any],
    method: str,
    q_s_ref: float,
) -> tuple[float | None, float | None]:
    """Return (CDi, CDp) from the raw solver dict.

    VLM: CD is purely induced — set CDi = CD, CDp = None.
    LiftingLine / NonlinearLiftingLine: induced only from vortex solution.
    AeroBuildup: provides D_induced and D_profile (forces [N]), converted to
    coefficients by dividing by (q * S_ref).
    """
    cd_total = _float(raw.get("CD", 0.0))

    if method == "aero_buildup":
        d_induced = raw.get("D_induced")
        d_profile = raw.get("D_profile")
        if d_induced is not None and q_s_ref > 0:
            cdi = _float(d_induced) / q_s_ref
        else:
            cdi = None
        if d_profile is not None and q_s_ref > 0:
            cdp = _float(d_profile) / q_s_ref
        else:
            cdp = None
        return cdi, cdp

    if method == "vlm":
        # VLM has no viscous model — all drag is induced.
        return cd_total, None

    # lifting_line and nonlinear_lifting_line: report CDi = CD (no separate CDp key).
    return cd_total, None


# ------------------------------------------------------------------ #
# Main solver entry
# ------------------------------------------------------------------ #

VALID_METHODS = ("vlm", "lifting_line", "nonlinear_lifting_line", "aero_buildup")


def run_asb(
    aircraft: Aircraft,
    condition: FlightCondition,
    method: str = "vlm",
    spanwise_resolution: int = 8,
    chordwise_resolution: int = 4,
    model_size: str = "medium",
    verbose: bool = False,
) -> AeroResult:
    """Run an AeroSandbox analysis and return an AeroResult.

    Parameters
    ----------
    aircraft : Aircraft
        aerisplane aircraft definition.
    condition : FlightCondition
        Operating point (velocity, altitude, alpha, beta, deflections).
    method : str
        Solver to use. One of: "vlm", "lifting_line",
        "nonlinear_lifting_line", "aero_buildup".
    spanwise_resolution : int
        Number of spanwise panels / stations per wing section.
        Applies to VLM, LiftingLine, NonlinearLiftingLine.
    chordwise_resolution : int
        Number of chordwise panels per wing section. VLM only.
    model_size : str
        NeuralFoil model size: "xxsmall", "xsmall", "small", "medium",
        "large", "xlarge", "xxlarge", "xxxlarge".
        Applies to LiftingLine and AeroBuildup.
    verbose : bool
        Whether to print solver progress.

    Returns
    -------
    AeroResult
        Full aerodynamic result.

    Raises
    ------
    ValueError
        If method is not one of the supported options.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {', '.join(VALID_METHODS)}"
        )

    airplane = aircraft_to_asb(aircraft)
    op_point = _condition_to_asb(condition)
    xyz_ref = airplane.xyz_ref

    # ---- Dispatch -------------------------------------------------------
    if method == "vlm":
        solver = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            spanwise_resolution=spanwise_resolution,
            chordwise_resolution=chordwise_resolution,
            verbose=verbose,
        )
        raw = solver.run()

    elif method == "lifting_line":
        solver = asb.LiftingLine(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            model_size=model_size,
            spanwise_resolution=spanwise_resolution,
            verbose=verbose,
        )
        raw = solver.run()

    elif method == "nonlinear_lifting_line":
        solver = asb.NonlinearLiftingLine(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            spanwise_resolution=spanwise_resolution,
            verbose=verbose,
        )
        raw = solver.run()

    elif method == "aero_buildup":
        solver = asb.AeroBuildup(
            airplane=airplane,
            op_point=op_point,
            xyz_ref=xyz_ref,
            model_size=model_size,
        )
        raw = solver.run()

    # ---- Populate result ------------------------------------------------
    _, _, rho, mu = isa(condition.altitude)
    q = 0.5 * rho * condition.velocity ** 2
    re = rho * condition.velocity * aircraft.reference_chord() / mu if mu > 0 else 0.0
    q_s_ref = q * aircraft.reference_area()
    cdi, cdp = _extract_drag_breakdown(raw, method, q_s_ref)

    return AeroResult(
        method=method,
        # Forces
        L=_float(raw.get("L", 0.0)),
        D=_float(raw.get("D", 0.0)),
        Y=_float(raw.get("Y", 0.0)),
        # Moments (body axes)
        l_b=_float(raw.get("l_b", 0.0)),
        m_b=_float(raw.get("m_b", 0.0)),
        n_b=_float(raw.get("n_b", 0.0)),
        # Force / moment vectors
        F_g=_vec3(raw.get("F_g", [0.0, 0.0, 0.0])),
        F_b=_vec3(raw.get("F_b", [0.0, 0.0, 0.0])),
        F_w=_vec3(raw.get("F_w", [0.0, 0.0, 0.0])),
        M_g=_vec3(raw.get("M_g", [0.0, 0.0, 0.0])),
        M_b=_vec3(raw.get("M_b", [0.0, 0.0, 0.0])),
        M_w=_vec3(raw.get("M_w", [0.0, 0.0, 0.0])),
        # Force coefficients
        CL=_float(raw.get("CL", 0.0)),
        CD=_float(raw.get("CD", 0.0)),
        CY=_float(raw.get("CY", 0.0)),
        # Moment coefficients
        Cl=_float(raw.get("Cl", 0.0)),
        Cm=_float(raw.get("Cm", 0.0)),
        Cn=_float(raw.get("Cn", 0.0)),
        # Drag breakdown
        CDi=cdi,
        CDp=cdp,
        # Operating condition (echoed)
        alpha=condition.alpha,
        beta=condition.beta,
        velocity=condition.velocity,
        altitude=condition.altitude,
        dynamic_pressure=q,
        reynolds_number=re,
        # Reference geometry (echoed)
        s_ref=aircraft.reference_area(),
        c_ref=aircraft.reference_chord(),
        b_ref=aircraft.reference_span(),
        # Internal — ASB objects for geometry / VLM plotting
        _solver=solver if method == "vlm" else None,
        _airplane=airplane,
    )


# ------------------------------------------------------------------ #
# Geometry plotting
# ------------------------------------------------------------------ #

def plot_geometry(
    aircraft: Aircraft,
    style: str = "three_view",
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot the aircraft geometry using AeroSandbox's built-in drawing routines.

    Translates the aerisplane Aircraft to an AeroSandbox Airplane and calls
    AeroSandbox's matplotlib-based drawing methods directly.

    Parameters
    ----------
    aircraft : Aircraft
        aerisplane aircraft to visualise.
    style : str
        Drawing style. One of:

        ``"three_view"``
            Standard 4-panel engineering drawing: top, front, side, isometric.
            Uses ``asb.Airplane.draw_three_view()``.

        ``"wireframe"``
            Single 3-D matplotlib wireframe.
            Uses ``asb.Airplane.draw_wireframe()``.

    show : bool
        Whether to call ``plt.show()`` after drawing.
    save_path : str or None
        If given, save the figure to this path before showing.

    Raises
    ------
    ValueError
        If ``style`` is not ``"three_view"`` or ``"wireframe"``.
    """
    import matplotlib.pyplot as plt

    airplane = aircraft_to_asb(aircraft)

    if style == "three_view":
        axs = airplane.draw_three_view(show=False)
        fig = axs[0, 0].get_figure()
        fig.suptitle(aircraft.name, fontsize=13, fontweight="bold", y=1.01)
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    elif style == "wireframe":
        ax = airplane.draw_wireframe(show=False)
        ax.set_title(aircraft.name)
        fig = ax.get_figure()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    else:
        raise ValueError(
            f"Unknown style '{style}'. Choose 'three_view' or 'wireframe'."
        )
