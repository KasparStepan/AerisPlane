"""MDO problem definition: dataclasses and MDOProblem class."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DesignVar:
    """Continuous design variable defined by a dot-bracket path into Aircraft.

    Parameters
    ----------
    path : str
        Dot-bracket path, e.g. ``"wings[0].xsecs[1].chord"``.
    lower : float
        Lower bound (physical units, before scaling).
    upper : float
        Upper bound (physical units, before scaling).
    scale : float
        Optimizer sees ``value / scale``.  Default 1.0 (no scaling).
    """
    path: str
    lower: float
    upper: float
    scale: float = 1.0


@dataclass
class AirfoilPool:
    """Airfoil options for one wing surface.

    Parameters
    ----------
    options : list of str
        Airfoil names resolvable via ``catalog.get_airfoil()``.
    xsecs : list of int or None
        Indices of xsecs whose airfoil is free.  ``None`` → all xsecs.
    """
    options: list[str]
    xsecs: Optional[list[int]] = None


@dataclass
class Constraint:
    """Constraint on a discipline result field.

    Parameters
    ----------
    path : str
        Dot path into the results dict, e.g. ``"stability.static_margin"``
        or ``"structures.wings[0].bending_margin"``.
    lower : float or None
        Value must be >= lower.
    upper : float or None
        Value must be <= upper.
    equals : any or None
        Value must equal this (supports bool for feasibility flags).
    scale : float
        Normalisation factor for the violation vector.
    """
    path: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    equals: Optional[Any] = None
    scale: float = 1.0

    def __post_init__(self):
        if self.lower is None and self.upper is None and self.equals is None:
            raise ValueError(
                f"Constraint '{self.path}': must specify lower, upper, or equals."
            )


@dataclass
class Objective:
    """Optimisation objective pointing at a discipline result field.

    Parameters
    ----------
    path : str
        E.g. ``"mission.endurance_s"`` or ``"weights.total_mass"``.
    maximize : bool
        True → maximise (default).  False → minimise.
    scale : float
        Normalisation factor applied to the raw value.
    """
    path: str
    maximize: bool = True
    scale: float = 1.0
