"""Reference aircraft configurations for aerodynamic analysis.

Each factory function returns a fresh :class:`~aerisplane.core.aircraft.Aircraft`
instance representing a typical aircraft in its class.  Geometry is
physically plausible but intentionally generic — these are not models of
certified aircraft.

Usage
-----
>>> from aerisplane.catalog.aircraft import trainer, ultralight
>>> ac = trainer()
>>> cond = trainer_condition()

Categories
----------
small_uav       ~1 kg hand-launch fixed-wing (low Re, UAV)
trainer         ~5 kg conventional RC/UAS trainer
ultralight      ~300 kg light sport aircraft (Cessna-150 class)
glider          ~400 kg 15 m club glider (high AR)
business_jet    ~10 000 kg light business jet (swept wing, M 0.75)
transport       ~70 000 kg narrowbody (double-swept, M 0.80)
"""

from __future__ import annotations

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.control_surface import ControlSurface
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.wing import Wing, WingXSec


# ---------------------------------------------------------------------------
# Shared airfoils
# ---------------------------------------------------------------------------
_NACA2412 = Airfoil("naca2412")
_NACA2414 = Airfoil("naca2414")
_NACA0009 = Airfoil("naca0009")
_NACA0012 = Airfoil("naca0012")


# ---------------------------------------------------------------------------
# Shared control-surface helpers
# ---------------------------------------------------------------------------

def _aileron(span_start: float = 0.55, span_end: float = 0.90,
             chord_fraction: float = 0.25, max_deflection: float = 20.0) -> ControlSurface:
    return ControlSurface(
        name="aileron",
        span_start=span_start, span_end=span_end,
        chord_fraction=chord_fraction,
        symmetric=False,
        max_deflection=max_deflection,
    )


def _elevator(chord_fraction: float = 0.38, max_deflection: float = 25.0) -> ControlSurface:
    return ControlSurface(
        name="elevator",
        span_start=0.0, span_end=1.0,
        chord_fraction=chord_fraction,
        symmetric=True,
        max_deflection=max_deflection,
    )


def _rudder(chord_fraction: float = 0.35, max_deflection: float = 25.0) -> ControlSurface:
    return ControlSurface(
        name="rudder",
        span_start=0.0, span_end=1.0,
        chord_fraction=chord_fraction,
        symmetric=True,
        max_deflection=max_deflection,
    )


# ---------------------------------------------------------------------------
# 1. Small UAV
# ---------------------------------------------------------------------------

def small_uav() -> Aircraft:
    """Hand-launch fixed-wing UAV (~1 kg, 1.4 m span).

    Conventional tractor layout with mild taper and straight LE.
    Represents a ~1 kg class hand-launch UAS operating at low Reynolds numbers
    (Re_chord ~ 1.4 × 10⁵).

    Typical cruise: 14 m/s, 100 m altitude, α = 4°.
    """
    wing = Wing(
        name="main_wing",
        symmetric=True,
        control_surfaces=[_aileron(span_start=0.50, span_end=0.92, chord_fraction=0.28)],
        xsecs=[
            WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.22, airfoil=_NACA2412),
            WingXSec(xyz_le=[0.01, 0.35, 0.00], chord=0.19, airfoil=_NACA2412),
            WingXSec(xyz_le=[0.02, 0.70, 0.00], chord=0.16, airfoil=_NACA2412),
        ],
    )

    htail = Wing(
        name="htail",
        symmetric=True,
        control_surfaces=[_elevator(chord_fraction=0.40)],
        xsecs=[
            WingXSec(xyz_le=[0.72, 0.00, 0.04], chord=0.10, airfoil=_NACA0009),
            WingXSec(xyz_le=[0.74, 0.22, 0.04], chord=0.08, airfoil=_NACA0009),
        ],
    )

    vtail = Wing(
        name="vtail",
        symmetric=False,
        control_surfaces=[_rudder()],
        xsecs=[
            WingXSec(xyz_le=[0.70, 0.00, 0.00], chord=0.12, airfoil=_NACA0009),
            WingXSec(xyz_le=[0.74, 0.00, 0.16], chord=0.08, airfoil=_NACA0009),
        ],
    )

    return Aircraft(
        name="SmallUAV",
        wings=[wing, htail, vtail],
        xyz_ref=[0.055, 0.0, 0.0],  # ≈ 25 % root chord
    )


def small_uav_condition() -> FlightCondition:
    """Typical cruise condition for :func:`small_uav`."""
    return FlightCondition(velocity=14.0, altitude=100.0, alpha=4.0)


# ---------------------------------------------------------------------------
# 2. Trainer
# ---------------------------------------------------------------------------

def trainer() -> Aircraft:
    """Conventional RC/UAS trainer (~5 kg, 2.4 m span).

    High-wing, mild taper and sweep.  Represents a typical 2 m class
    trainer or small survey UAS (Re_chord ~ 3 × 10⁵).

    Typical cruise: 18 m/s, 200 m altitude, α = 4°.
    """
    wing = Wing(
        name="main_wing",
        symmetric=True,
        control_surfaces=[_aileron()],
        xsecs=[
            WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.30, airfoil=_NACA2412),
            WingXSec(xyz_le=[0.04, 0.60, 0.00], chord=0.24, airfoil=_NACA2412),
            WingXSec(xyz_le=[0.13, 1.20, 0.00], chord=0.18, airfoil=_NACA2412),
        ],
    )

    htail = Wing(
        name="htail",
        symmetric=True,
        control_surfaces=[_elevator()],
        xsecs=[
            WingXSec(xyz_le=[1.20, 0.00, 0.08], chord=0.14, airfoil=_NACA0009),
            WingXSec(xyz_le=[1.23, 0.42, 0.08], chord=0.11, airfoil=_NACA0009),
        ],
    )

    vtail = Wing(
        name="vtail",
        symmetric=False,
        control_surfaces=[_rudder()],
        xsecs=[
            WingXSec(xyz_le=[1.15, 0.00, 0.00], chord=0.18, airfoil=_NACA0009),
            WingXSec(xyz_le=[1.22, 0.00, 0.28], chord=0.12, airfoil=_NACA0009),
        ],
    )

    return Aircraft(
        name="Trainer",
        wings=[wing, htail, vtail],
        xyz_ref=[0.075, 0.0, 0.0],
    )


def trainer_condition() -> FlightCondition:
    """Typical cruise condition for :func:`trainer`."""
    return FlightCondition(velocity=18.0, altitude=200.0, alpha=4.0)


# ---------------------------------------------------------------------------
# 3. Ultralight / light sport aircraft
# ---------------------------------------------------------------------------

def ultralight() -> Aircraft:
    """Light sport aircraft (~300 kg MTOW, 10.2 m span).

    Unswept high-wing with rectangular-trapezoidal planform, similar in
    proportions to a Cessna 150.  NACA 2412 throughout
    (which is the actual airfoil on the Cessna 150).

    Typical cruise: 55 m/s, 1 000 m altitude, α = 3°.
    """
    wing = Wing(
        name="main_wing",
        symmetric=True,
        control_surfaces=[
            _aileron(span_start=0.50, span_end=0.88, chord_fraction=0.22, max_deflection=20.0),
            ControlSurface(
                name="flap",
                span_start=0.05, span_end=0.50,
                chord_fraction=0.30, symmetric=True, max_deflection=40.0,
            ),
        ],
        xsecs=[
            WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=1.65, airfoil=_NACA2412),
            WingXSec(xyz_le=[0.06, 2.55, 0.00], chord=1.40, airfoil=_NACA2412),
            WingXSec(xyz_le=[0.12, 5.09, 0.00], chord=1.14, airfoil=_NACA2412),
        ],
    )

    htail = Wing(
        name="htail",
        symmetric=True,
        control_surfaces=[_elevator(chord_fraction=0.35, max_deflection=25.0)],
        xsecs=[
            WingXSec(xyz_le=[4.80, 0.00, 0.20], chord=1.10, airfoil=_NACA0009),
            WingXSec(xyz_le=[4.88, 1.70, 0.20], chord=0.85, airfoil=_NACA0009),
        ],
    )

    vtail = Wing(
        name="vtail",
        symmetric=False,
        control_surfaces=[_rudder(chord_fraction=0.32)],
        xsecs=[
            WingXSec(xyz_le=[4.50, 0.00, 0.00], chord=1.30, airfoil=_NACA0009),
            WingXSec(xyz_le=[4.95, 0.00, 1.42], chord=0.75, airfoil=_NACA0009),
        ],
    )

    return Aircraft(
        name="Ultralight",
        wings=[wing, htail, vtail],
        xyz_ref=[0.41, 0.0, 0.0],  # ≈ 25 % root chord
    )


def ultralight_condition() -> FlightCondition:
    """Typical cruise condition for :func:`ultralight`."""
    return FlightCondition(velocity=55.0, altitude=1000.0, alpha=3.0)


# ---------------------------------------------------------------------------
# 4. Glider
# ---------------------------------------------------------------------------

def glider() -> Aircraft:
    """15 m club glider (~400 kg, AR ≈ 16).

    Three-section wing with mild taper and slight dihedral towards the tip.
    NACA 2414 root / 2412 tip (thickness washout).  Long tail boom with
    T-tail proportions (htail placed high, z = 0.10 m).

    Typical cruise: 24 m/s, 1 500 m altitude, α = 2°.
    """
    wing = Wing(
        name="main_wing",
        symmetric=True,
        control_surfaces=[
            _aileron(span_start=0.60, span_end=0.95, chord_fraction=0.22),
            ControlSurface(
                name="airbrake",
                span_start=0.30, span_end=0.60,
                chord_fraction=0.50, symmetric=True, max_deflection=90.0,
            ),
        ],
        xsecs=[
            WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=1.20, airfoil=_NACA2414),
            WingXSec(xyz_le=[0.00, 3.75, 0.04], chord=0.95, airfoil=_NACA2412),
            WingXSec(xyz_le=[0.02, 7.50, 0.14], chord=0.64, airfoil=_NACA2412),
        ],
    )

    htail = Wing(
        name="htail",
        symmetric=True,
        control_surfaces=[_elevator(chord_fraction=0.38)],
        xsecs=[
            WingXSec(xyz_le=[5.60, 0.00, 0.10], chord=0.65, airfoil=_NACA0009),
            WingXSec(xyz_le=[5.64, 1.80, 0.10], chord=0.50, airfoil=_NACA0009),
        ],
    )

    vtail = Wing(
        name="vtail",
        symmetric=False,
        control_surfaces=[_rudder(chord_fraction=0.30)],
        xsecs=[
            WingXSec(xyz_le=[5.20, 0.00, 0.00], chord=0.90, airfoil=_NACA0009),
            WingXSec(xyz_le=[5.68, 0.00, 1.30], chord=0.48, airfoil=_NACA0009),
        ],
    )

    return Aircraft(
        name="Glider",
        wings=[wing, htail, vtail],
        xyz_ref=[0.30, 0.0, 0.0],  # ≈ 25 % root chord
    )


def glider_condition() -> FlightCondition:
    """Typical cruise condition for :func:`glider`."""
    return FlightCondition(velocity=24.0, altitude=1500.0, alpha=2.0)


# ---------------------------------------------------------------------------
# 5. Business jet
# ---------------------------------------------------------------------------

def business_jet() -> Aircraft:
    """Light business jet (~10 000 kg, 22 m span, 25° LE sweep).

    Three-section swept wing with 3° dihedral.  Represents a Citation /
    Learjet class aircraft at M ≈ 0.74.  Compressibility correction is
    active at this Mach (Prandtl-Glauert factor ≈ 1.48).

    Typical cruise: 220 m/s, 12 000 m altitude, α = 2°  (M ≈ 0.74).
    """
    # Sweep geometry (25° LE sweep, 3° dihedral)
    #   dx_per_m_span = tan(25°) = 0.4663
    #   dz_per_m_span = sin(3°)  = 0.0523
    _sw = np.tan(np.radians(25.0))   # LE sweep slope
    _dh = np.sin(np.radians(3.0))    # dihedral slope
    _z0 = 0.20                        # wing root z above reference line

    wing = Wing(
        name="main_wing",
        symmetric=True,
        control_surfaces=[
            _aileron(span_start=0.55, span_end=0.85, chord_fraction=0.22, max_deflection=15.0),
            ControlSurface(
                name="flap",
                span_start=0.10, span_end=0.55,
                chord_fraction=0.28, symmetric=True, max_deflection=40.0,
            ),
        ],
        xsecs=[
            WingXSec(
                xyz_le=[0.00, 0.00, _z0],
                chord=3.50, airfoil=_NACA2412,
            ),
            WingXSec(
                xyz_le=[_sw * 5.5, 5.50, _z0 + _dh * 5.5],
                chord=2.20, airfoil=_NACA2412,
            ),
            WingXSec(
                xyz_le=[_sw * 11.0, 11.00, _z0 + _dh * 11.0],
                chord=1.20, airfoil=_NACA2412,
            ),
        ],
    )

    # Htail — 20° sweep, 9 m span
    _sw_h = np.tan(np.radians(20.0))
    htail = Wing(
        name="htail",
        symmetric=True,
        control_surfaces=[_elevator(chord_fraction=0.35, max_deflection=20.0)],
        xsecs=[
            WingXSec(xyz_le=[14.00, 0.00, 0.40], chord=1.80, airfoil=_NACA0012),
            WingXSec(xyz_le=[14.00 + _sw_h * 4.5, 4.50, 0.40], chord=1.00, airfoil=_NACA0012),
        ],
    )

    # Vtail — 35° sweep, 3 m height
    _sw_v = np.tan(np.radians(35.0))
    vtail = Wing(
        name="vtail",
        symmetric=False,
        control_surfaces=[_rudder(chord_fraction=0.30)],
        xsecs=[
            WingXSec(xyz_le=[13.00, 0.00, 0.00], chord=2.50, airfoil=_NACA0012),
            WingXSec(xyz_le=[13.00 + _sw_v * 3.0, 0.00, 3.00], chord=1.20, airfoil=_NACA0012),
        ],
    )

    return Aircraft(
        name="BusinessJet",
        wings=[wing, htail, vtail],
        xyz_ref=[0.875, 0.0, _z0],   # ≈ 25 % root chord
    )


def business_jet_condition() -> FlightCondition:
    """Typical cruise condition for :func:`business_jet`.

    M ≈ 0.74 at 12 000 m (speed of sound ≈ 295 m/s).
    """
    return FlightCondition(velocity=220.0, altitude=12000.0, alpha=2.0)


# ---------------------------------------------------------------------------
# 6. Narrowbody transport
# ---------------------------------------------------------------------------

def transport() -> Aircraft:
    """Narrowbody transport aircraft (~70 000 kg, 35.8 m span).

    Double-swept three-section wing: mild inner sweep (26°) transitioning
    to 35° outer sweep with 6° dihedral — representative of a 737/A320
    class aircraft.  At typical cruise (M ≈ 0.80) the Prandtl-Glauert
    correction factor is ≈ 1.67.

    Typical cruise: 240 m/s, 11 000 m altitude, α = 2°  (M ≈ 0.80).
    """
    # Wing geometry: 3 sections
    #   Inner (26° sweep): y = 0 → 5 m
    #   Outer (35° sweep): y = 5 → 17.9 m
    #   6° dihedral throughout
    _sw_in  = np.tan(np.radians(26.0))
    _sw_out = np.tan(np.radians(35.0))
    _dh     = np.sin(np.radians(6.0))
    _z0     = 0.60

    _x_kink = _sw_in * 5.0
    _z_kink = _z0 + _dh * 5.0
    _x_tip  = _x_kink + _sw_out * 12.9
    _z_tip  = _z0 + _dh * 17.9

    wing = Wing(
        name="main_wing",
        symmetric=True,
        control_surfaces=[
            _aileron(span_start=0.70, span_end=0.93, chord_fraction=0.22, max_deflection=12.0),
            ControlSurface(
                name="flap",
                span_start=0.10, span_end=0.70,
                chord_fraction=0.30, symmetric=True, max_deflection=40.0,
            ),
            ControlSurface(
                name="spoiler",
                span_start=0.35, span_end=0.75,
                chord_fraction=0.20, symmetric=True, max_deflection=60.0,
            ),
        ],
        xsecs=[
            WingXSec(xyz_le=[0.00,    0.00,  _z0],     chord=7.50, airfoil=_NACA2412),
            WingXSec(xyz_le=[_x_kink, 5.00,  _z_kink], chord=5.20, airfoil=_NACA2412),
            WingXSec(xyz_le=[_x_tip,  17.90, _z_tip],  chord=1.50, airfoil=_NACA2412),
        ],
    )

    # Htail — 35° sweep, 13 m span
    _sw_h = np.tan(np.radians(35.0))
    htail = Wing(
        name="htail",
        symmetric=True,
        control_surfaces=[_elevator(chord_fraction=0.32, max_deflection=20.0)],
        xsecs=[
            WingXSec(xyz_le=[27.00, 0.00, 1.20], chord=3.80, airfoil=_NACA0012),
            WingXSec(xyz_le=[27.00 + _sw_h * 6.5, 6.50, 1.20], chord=1.90, airfoil=_NACA0012),
        ],
    )

    # Vtail — 45° sweep, 5.8 m height
    vtail = Wing(
        name="vtail",
        symmetric=False,
        control_surfaces=[_rudder(chord_fraction=0.28, max_deflection=25.0)],
        xsecs=[
            WingXSec(xyz_le=[25.00, 0.00, 0.00], chord=6.50, airfoil=_NACA0012),
            WingXSec(xyz_le=[31.80, 0.00, 5.80], chord=3.00, airfoil=_NACA0012),
        ],
    )

    return Aircraft(
        name="Transport",
        wings=[wing, htail, vtail],
        xyz_ref=[1.875, 0.0, _z0],   # ≈ 25 % root chord
    )


def transport_condition() -> FlightCondition:
    """Typical cruise condition for :func:`transport`.

    M ≈ 0.80 at 11 000 m (speed of sound ≈ 295 m/s).
    """
    return FlightCondition(velocity=240.0, altitude=11000.0, alpha=2.0)


# ---------------------------------------------------------------------------
# Convenience listing
# ---------------------------------------------------------------------------

#: All aircraft factory functions, in order from smallest to largest.
AIRCRAFT_CATALOG: dict[str, callable] = {
    "small_uav":    small_uav,
    "trainer":      trainer,
    "ultralight":   ultralight,
    "glider":       glider,
    "business_jet": business_jet,
    "transport":    transport,
}
