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
    color: Optional[str] = None
    """Optional display colour for visualization. Any CSS colour string, e.g. "#808080".

    If None, the visualization module picks from its default palette.
    """

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

    def volume(self, _sectional: bool = False) -> "float | list[float]":
        """Internal volume [m^3].

        Uses the Prismatoid formula per section, which is exact for frustums
        (tapered cylinders) and more accurate than trapz for nose/tail cones:

            V_section = (x_b - x_a) / 3 * (A_a + A_b + sqrt(A_a * A_b))
        """
        if len(self.xsecs) < 2:
            return [] if _sectional else 0.0

        sectional = []
        for xa, xb in zip(self.xsecs[:-1], self.xsecs[1:]):
            sep = xb.x - xa.x
            a_a = xa.area()
            a_b = xb.area()
            sectional.append(sep / 3.0 * (a_a + a_b + (a_a * a_b + 1e-100) ** 0.5))

        return sectional if _sectional else sum(sectional)

    def area_projected(self, type: str = "XY") -> float:
        """Projected area onto XY (top-down) or XZ (side-view) plane [m^2].

        Parameters
        ----------
        type : str
            "XY" uses cross-section width (y-direction extent).
            "XZ" uses cross-section height (z-direction extent).
        """
        if type not in ("XY", "XZ"):
            raise ValueError(f"type must be 'XY' or 'XZ', got '{type}'")
        if len(self.xsecs) < 2:
            return 0.0
        total = 0.0
        for xa, xb in zip(self.xsecs[:-1], self.xsecs[1:]):
            dx = xb.x - xa.x
            if type == "XY":
                avg = (xa.width + xb.width) / 2.0
            else:
                avg = (xa.height + xb.height) / 2.0
            total += avg * dx
        return total

    def x_centroid_projected(self, type: str = "XY") -> float:
        """x-coordinate of the projected-area centroid in aircraft frame [m].

        Parameters
        ----------
        type : str
            "XY" or "XZ" — which projection to use.
        """
        if len(self.xsecs) < 2:
            return self.x_le
        total_x_area = 0.0
        total_area = 0.0
        for xa, xb in zip(self.xsecs[:-1], self.xsecs[1:]):
            x_a = self.x_le + xa.x
            x_b = self.x_le + xb.x
            dx = x_b - x_a
            if type == "XY":
                r_a = xa.width / 2.0
                r_b = xb.width / 2.0
            elif type == "XZ":
                r_a = xa.height / 2.0
                r_b = xb.height / 2.0
            else:
                raise ValueError(f"type must be 'XY' or 'XZ', got '{type}'")
            if (r_a + r_b) > 0:
                x_c = x_a + (r_a + 2.0 * r_b) / (3.0 * (r_a + r_b)) * dx
            else:
                x_c = (x_a + x_b) / 2.0
            section_area = (r_a + r_b) / 2.0 * dx
            total_x_area += x_c * section_area
            total_area += section_area
        if total_area == 0.0:
            return self.x_le
        return total_x_area / total_area

    def mesh_body(
        self, tangential_resolution: int = 36
    ) -> "tuple[np.ndarray, np.ndarray]":
        """3-D quad surface mesh of the fuselage.

        Returns
        -------
        points : (N, 3) float array
            Vertex positions in aircraft frame.
        faces : (M, 4) int array
            Quad face vertex indices [i0, i1, i2, i3].
        """
        theta = np.linspace(0, 2 * np.pi, tangential_resolution + 1)[:-1]
        centers = self.xsec_centers()

        rings = [
            xsec.get_3D_coordinates(theta, center)
            for xsec, center in zip(self.xsecs, centers)
        ]
        points = np.concatenate(rings, axis=0)

        T = tangential_resolution
        faces = []
        for i in range(len(self.xsecs) - 1):
            for j in range(T):
                j1 = (j + 1) % T
                faces.append([i * T + j, i * T + j1, (i + 1) * T + j1, (i + 1) * T + j])

        return points, np.array(faces, dtype=int)

    def mesh_line(
        self,
        y_nondim: float = 0.0,
        z_nondim: float = 0.0,
    ) -> "list[np.ndarray]":
        """3-D points along a line through each cross-section.

        Parameters
        ----------
        y_nondim : float
            Fractional y-offset normalized by half-width. 0 = centreline, ±1 = outer edge.
        z_nondim : float
            Fractional z-offset normalized by half-height.
        """
        centers = self.xsec_centers()
        return [
            np.array([
                center[0],
                center[1] + y_nondim * xsec.width / 2.0,
                center[2] + z_nondim * xsec.height / 2.0,
            ])
            for xsec, center in zip(self.xsecs, centers)
        ]

    def subdivide_sections(
        self,
        ratio: int,
        spacing_function=None,
    ) -> "Fuselage":
        """Return a new Fuselage with each section split into *ratio* sub-sections.

        Parameters
        ----------
        ratio : int
            Number of sub-sections per original section (must be >= 2).
        spacing_function : callable(start, stop, n) → array, optional
            Defaults to numpy.linspace.
        """
        import copy
        if not (isinstance(ratio, int) and ratio >= 2):
            raise ValueError(f"ratio must be an integer >= 2, got {ratio!r}")
        if spacing_function is None:
            spacing_function = np.linspace

        new_xsecs = []
        fracs = spacing_function(0.0, 1.0, ratio + 1)[:-1]

        for xa, xb in zip(self.xsecs[:-1], self.xsecs[1:]):
            for s in fracs:
                aw, bw = 1.0 - float(s), float(s)
                new_xsecs.append(FuselageXSec(
                    x=xa.x * aw + xb.x * bw,
                    width=xa.width * aw + xb.width * bw,
                    height=xa.height * aw + xb.height * bw,
                    shape=xa.shape * aw + xb.shape * bw,
                ))

        new_xsecs.append(copy.copy(self.xsecs[-1]))
        return Fuselage(
            name=self.name,
            xsecs=new_xsecs,
            x_le=self.x_le,
            y_le=self.y_le,
            z_le=self.z_le,
            material=self.material,
            wall_thickness=self.wall_thickness,
        )

    def translate(self, xyz: np.ndarray) -> "Fuselage":
        """Return a copy of this fuselage translated by xyz [m].

        Moves the nose reference point (x_le, y_le, z_le).
        Cross-section local x-coordinates are unchanged.
        """
        import copy
        xyz = np.asarray(xyz, dtype=float)
        new = copy.copy(self)
        new.xsecs = list(self.xsecs)
        new.x_le = self.x_le + float(xyz[0])
        new.y_le = self.y_le + float(xyz[1])
        new.z_le = self.z_le + float(xyz[2])
        return new

    def draw(self, backend: str = "plotly", show: bool = True, **kwargs):
        """Visualize this fuselage. See ``aerisplane.viz.draw`` for full docs."""
        from aerisplane.viz import draw
        return draw(self, backend=backend, show=show, **kwargs)

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
