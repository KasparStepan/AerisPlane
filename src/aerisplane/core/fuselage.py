"""Fuselage geometry definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aerisplane.core.structures import Material


@dataclass
class FuselageXSec:
    """A cross-section of a fuselage at a given axial station.

    Parameters
    ----------
    x : float
        Axial position from nose [m].
    radius : float
        Cross-section radius [m] (for circular sections).
    shape : str
        Cross-section shape: "circle", "ellipse", "rectangle".
    width : float or None
        Width [m] for ellipse/rectangle shapes.
    height : float or None
        Height [m] for ellipse/rectangle shapes.
    """

    x: float
    radius: float
    shape: str = "circle"
    width: Optional[float] = None
    height: Optional[float] = None

    def area(self) -> float:
        """Cross-sectional area [m^2]."""
        if self.shape == "circle":
            return np.pi * self.radius**2
        elif self.shape == "ellipse" and self.width and self.height:
            return np.pi * (self.width / 2.0) * (self.height / 2.0)
        elif self.shape == "rectangle" and self.width and self.height:
            return self.width * self.height
        return np.pi * self.radius**2

    def perimeter(self) -> float:
        """Cross-section perimeter [m]."""
        if self.shape == "circle":
            return 2.0 * np.pi * self.radius
        elif self.shape == "ellipse" and self.width and self.height:
            a = self.width / 2.0
            b = self.height / 2.0
            # Ramanujan approximation
            return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))
        elif self.shape == "rectangle" and self.width and self.height:
            return 2.0 * (self.width + self.height)
        return 2.0 * np.pi * self.radius


@dataclass
class Fuselage:
    """Fuselage defined by axial cross-sections.

    Parameters
    ----------
    name : str
        Descriptive name.
    xsecs : list of FuselageXSec
        Cross-sections ordered from nose to tail.
    x_le : float
        Nose x-position in aircraft frame [m].
    y_le : float
        Nose y-position in aircraft frame [m].
    z_le : float
        Nose z-position in aircraft frame [m].
    material : Material or None
        Shell material.
    wall_thickness : float
        Shell wall thickness [m].
    """

    name: str = "fuselage"
    xsecs: list[FuselageXSec] = field(default_factory=list)
    x_le: float = 0.0
    y_le: float = 0.0
    z_le: float = 0.0
    material: Optional[Material] = None
    wall_thickness: float = 0.001

    def _x_stations(self) -> np.ndarray:
        """Axial positions of cross-sections."""
        return np.array([xsec.x for xsec in self.xsecs])

    def _areas(self) -> np.ndarray:
        """Cross-section areas at each station."""
        return np.array([xsec.area() for xsec in self.xsecs])

    def _perimeters(self) -> np.ndarray:
        """Cross-section perimeters at each station."""
        return np.array([xsec.perimeter() for xsec in self.xsecs])

    def length(self) -> float:
        """Total fuselage length [m]."""
        if len(self.xsecs) < 2:
            return 0.0
        x = self._x_stations()
        return float(x[-1] - x[0])

    def volume(self) -> float:
        """Approximate internal volume by trapezoidal integration of cross-section areas [m^3]."""
        if len(self.xsecs) < 2:
            return 0.0
        return float(np.trapezoid(self._areas(), self._x_stations()))

    def wetted_area(self) -> float:
        """Approximate wetted (external) area by trapezoidal integration of perimeters [m^2]."""
        if len(self.xsecs) < 2:
            return 0.0
        return float(np.trapezoid(self._perimeters(), self._x_stations()))

    def max_cross_section_area(self) -> float:
        """Maximum cross-section area [m^2]."""
        if not self.xsecs:
            return 0.0
        return float(np.max(self._areas()))

    def fineness_ratio(self) -> float:
        """Fineness ratio: length / max diameter."""
        max_area = self.max_cross_section_area()
        if max_area == 0.0:
            return 0.0
        max_diameter = 2.0 * np.sqrt(max_area / np.pi)
        return self.length() / max_diameter

    def area_base(self) -> float:
        """Cross-section area of the tail (base) station [m^2]."""
        if not self.xsecs:
            return 0.0
        return self.xsecs[-1].area()

    def area_wetted(self) -> float:
        """Wetted (external) surface area [m^2]. Alias for wetted_area()."""
        return self.wetted_area()

    def xsec_centers(self) -> list[np.ndarray]:
        """3-D centre position of each cross-section in aircraft frame [m].

        Returns a list of (3,) arrays [x, y, z], one per xsec.
        Assumes the fuselage is aligned with the x-axis at (x_le, y_le, z_le).
        """
        return [
            np.array([self.x_le + xsec.x, self.y_le, self.z_le])
            for xsec in self.xsecs
        ]
