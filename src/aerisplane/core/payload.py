"""Payload definition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Payload:
    """A payload component with mass and CG location.

    Parameters
    ----------
    mass : float
        Payload mass [kg].
    cg : numpy array
        Center of gravity position [x, y, z] in aircraft frame [m].
    name : str
        Descriptive name.
    """

    mass: float
    cg: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    name: str = "payload"

    def __post_init__(self) -> None:
        self.cg = np.asarray(self.cg, dtype=float)
