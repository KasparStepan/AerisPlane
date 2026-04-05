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
    """A hinged control surface on a wing.

    Parameters
    ----------
    name : str
        Surface name used to key ``FlightCondition.deflections``.
        Conventional names: ``"elevator"``, ``"aileron"``, ``"rudder"``, ``"flap"``.
    span_start : float
        Start of the surface as a fraction of the wing semi-span [0, 1].
    span_end : float
        End of the surface as a fraction of the wing semi-span [0, 1].
    chord_fraction : float
        Hinge chord fraction [0, 1]. 0.25 means the surface occupies the aft 25%
        of the local chord.
    symmetric : bool
        If True, deflects the same way on both sides (elevator, flap).
        If False, the sign is mirrored on the left side (aileron).
    max_deflection : float
        Maximum deflection magnitude [deg]. Used for authority calculations.

    Notes
    -----
    Sign convention: positive deflection = trailing edge down.
    For ``symmetric=False`` (ailerons): positive = right side TE-down / left side TE-up.

    Examples
    --------
    >>> elevator = ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
    ...                            chord_fraction=0.35, symmetric=True, max_deflection=25.0)
    >>> aileron = ControlSurface(name="aileron", span_start=0.5, span_end=0.9,
    ...                           chord_fraction=0.25, symmetric=False, max_deflection=20.0)
    """

    name: str
    span_start: float
    span_end: float
    chord_fraction: float
    max_deflection: float = 25.0
    min_deflection: float = -25.0
    symmetric: bool = True
    servo: Optional[Servo] = None
