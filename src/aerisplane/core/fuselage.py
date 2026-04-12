"""Fuselage geometry definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# np.trapezoid was added in NumPy 2.0; np.trapz was removed in NumPy 2.0.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

from aerisplane.core.structures import Material


@dataclass
class FuselageXSec:
    """A cross-section of a fuselage at a given axial station.

    Parameters
    ----------
    x : float
        Axial position from nose [m]. 0 = nose tip.
    width : float
        Cross-section width in the y-direction [m].
    height : float
        Cross-section height in the z-direction [m].
    shape : float
        Superellipse exponent. Controls cross-section shape:
            shape=1.0  → diamond (45° rotated square)
            shape=2.0  → circle / ellipse  (default)
            shape=5.0  → rounded rectangle
            shape→∞   → rectangle
        Use 2.0 for aerodynamic bodies, 4–8 for fuselages with flat sides.
    radius : float or None
        Backward-compat convenience: sets width = height = 2 * radius, shape = 2.0.
        Use width/height directly for non-circular cross-sections.
    """

    x: float
    width: float = 0.0
    height: float = 0.0
    shape: float = 2.0
    radius: Optional[float] = None

    def __post_init__(self) -> None:
        # Backward compat: old string shape values
        if isinstance(self.shape, str):
            _map = {"circle": 2.0, "ellipse": 2.0, "rectangle": 10.0}
            self.shape = float(_map.get(self.shape.lower(), 2.0))

        # Backward compat: radius convenience param
        if self.radius is not None:
            self.width = 2.0 * self.radius
            self.height = 2.0 * self.radius

    def area(self) -> float:
        """Cross-sectional area [m^2].

        Uses closed-form superellipse approximation (error < 0.6% for shape >= 1):

            area = width * height / (shape^-1.8718 + 1)

        Exact for shape=1 (diamond) and shape=2 (circle/ellipse).
        """
        return self.width * self.height / (self.shape ** -1.8717618013591173 + 1.0)

    def perimeter(self) -> float:
        """Cross-section perimeter [m].

        Closed-form approximation derived by symbolic regression (error < 0.2%).
        """
        if self.width == 0.0:
            return 2.0 * self.height
        if self.height == 0.0:
            return 2.0 * self.width

        s = self.shape
        eps = 1e-16
        h = max(
            (self.width + eps) / (self.height + eps),
            (self.height + eps) / (self.width + eps),
        )
        nondim_qp = h + (
            ((s - 0.88487077) * h + 0.2588574 / h) ** np.exp(s / -0.90069205)
            + h + 0.09919785
        ) ** (-1.4812293 / s)
        return 2.0 * nondim_qp * min(self.width, self.height)

    def equivalent_radius(self, preserve: str = "area") -> float:
        """Equivalent circular radius [m] that preserves area or perimeter.

        Parameters
        ----------
        preserve : str
            "area"      : radius = sqrt(area / pi)
            "perimeter" : radius = perimeter / (2 * pi)
        """
        if preserve == "area":
            return float(np.sqrt(self.area() / np.pi + 1e-16))
        elif preserve == "perimeter":
            return self.perimeter() / (2.0 * np.pi)
        else:
            raise ValueError(
                f"preserve must be 'area' or 'perimeter', got '{preserve}'"
            )

    def get_3D_coordinates(
        self,
        theta: np.ndarray,
        xyz_center: np.ndarray,
    ) -> np.ndarray:
        """Sample 3-D points around the cross-section perimeter.

        Parameters
        ----------
        theta : array of float
            Angular positions [rad]. theta=0 → +y (right), theta=pi/2 → +z (up).
        xyz_center : (3,) array
            Center of this cross-section in aircraft frame [m].

        Returns
        -------
        points : (N, 3) array
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        # Superellipse parametric form
        y = (self.width / 2.0) * np.abs(ct) ** (2.0 / self.shape) * np.sign(ct + 1e-32)
        z = (self.height / 2.0) * np.abs(st) ** (2.0 / self.shape) * np.sign(st + 1e-32)
        y = np.where(np.abs(ct) < 1e-10, 0.0, y)
        z = np.where(np.abs(st) < 1e-10, 0.0, z)
        return np.column_stack([
            np.full_like(theta, xyz_center[0]),
            xyz_center[1] + y,
            xyz_center[2] + z,
        ])

    def translate(self, dx: float) -> "FuselageXSec":
        """Return a copy of this cross-section shifted by dx along the x-axis."""
        import copy
        new = copy.copy(self)
        new.x = self.x + dx
        return new


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
        return float(_trapz(self._areas(), self._x_stations()))

    def wetted_area(self) -> float:
        """Approximate wetted (external) area by trapezoidal integration of perimeters [m^2]."""
        if len(self.xsecs) < 2:
            return 0.0
        return float(_trapz(self._perimeters(), self._x_stations()))

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
