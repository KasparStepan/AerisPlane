"""Control surface and servo definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Servo:
    """Servo actuator for a control surface.

    Parameters
    ----------
    name : str
        Model name (e.g., "Hitec D485HW").
    torque : float
        Torque at rated voltage [N*m].
    speed : float
        Speed at rated voltage [deg/s].
    voltage : float
        Rated voltage [V].
    mass : float
        Mass [kg].
    """

    name: str
    torque: float
    speed: float
    voltage: float
    mass: float


@dataclass
class ControlSurface:
    """A control surface on a wing or tail.

    Span fractions are relative to the wing's spanwise extent:
    0.0 = root (first xsec), 1.0 = tip (last xsec).
    This applies to both symmetric and asymmetric wings.

    Parameters
    ----------
    name : str
        Surface identifier: "aileron", "elevator", "rudder", "flap".
    span_start : float
        Start of surface as fraction of wing spanwise extent [0-1].
    span_end : float
        End of surface as fraction of wing spanwise extent [0-1].
    chord_fraction : float
        Fraction of local chord occupied by the surface [0-1].
    max_deflection : float
        Maximum trailing-edge-down deflection [deg].
    min_deflection : float
        Maximum trailing-edge-up deflection [deg] (negative).
    symmetric : bool
        If True (default), both sides of a symmetric wing deflect in the same
        physical direction (elevator, flap).  If False, the mirrored side
        deflects opposite (aileron, spoileron).
    servo : Servo or None
        Assigned servo actuator.
    """

    name: str
    span_start: float
    span_end: float
    chord_fraction: float
    max_deflection: float = 25.0
    min_deflection: float = -25.0
    symmetric: bool = True
    servo: Optional[Servo] = None
