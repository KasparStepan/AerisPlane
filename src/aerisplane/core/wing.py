"""Wing and wing cross-section definitions with geometry methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from aerisplane.core.airfoil import Airfoil
from aerisplane.core.control_surface import ControlSurface
from aerisplane.core.structures import Skin, Spar


@dataclass
class WingXSec:
    """A spanwise cross-section (station) of a wing.

    Parameters
    ----------
    xyz_le : array-like
        Leading edge position [x, y, z] in aircraft frame [m].
        For a symmetric wing, define the right (y >= 0) side only.
    chord : float
        Chord length [m].
    twist : float
        Twist angle [deg]. Positive = nose up (washout is negative tip twist).
    airfoil : Airfoil or None
        Airfoil at this section. Defaults to NACA 0012.
    spar : Spar or None
        Spar at this section.
    skin : Skin or None
        Skin at this section.
    """

    xyz_le: np.ndarray
    chord: float
    twist: float = 0.0
    airfoil: Optional[Airfoil] = None
    spar: Optional[Spar] = None
    skin: Optional[Skin] = None

    def __post_init__(self) -> None:
        self.xyz_le = np.asarray(self.xyz_le, dtype=float)
        if self.airfoil is None:
            self.airfoil = Airfoil("naca0012")


@dataclass
class Wing:
    """A lifting surface composed of spanwise cross-sections.

    Used for main wings, horizontal tail, and vertical tail — there is no
    separate Tail class. A horizontal stabilizer is just a Wing with a
    different name and position.

    Parameters
    ----------
    name : str
        Descriptive name (e.g., "main_wing", "htail", "vtail").
    xsecs : list of WingXSec
        Ordered from root (y=0 for symmetric) to tip.
    symmetric : bool
        If True, the wing is mirrored about the xz-plane (y=0).
    control_surfaces : list of ControlSurface
        Control surfaces on this wing.
    """

    name: str = "wing"
    xsecs: list[WingXSec] = field(default_factory=list)
    symmetric: bool = True
    control_surfaces: list[ControlSurface] = field(default_factory=list)

    def _y_stations(self) -> np.ndarray:
        """Y-coordinates of each cross-section."""
        return np.array([xsec.xyz_le[1] for xsec in self.xsecs])

    def _chords(self) -> np.ndarray:
        """Chord at each cross-section."""
        return np.array([xsec.chord for xsec in self.xsecs])

    def semispan(self) -> float:
        """Semispan: distance from root to tip along y-axis [m].

        For a non-symmetric wing, this is the full extent from first to last xsec.
        """
        y = self._y_stations()
        return float(np.max(y) - np.min(y))

    def span(self) -> float:
        """Total wingspan [m].

        For symmetric wings: 2 * semispan.
        For non-symmetric wings (e.g., vertical tail): semispan.
        """
        s = self.semispan()
        return 2.0 * s if self.symmetric else s

    def area(self) -> float:
        """Planform area by trapezoidal integration [m^2].

        For symmetric wings, returns the total area (both sides).
        """
        y = self._y_stations()
        c = self._chords()

        # Trapezoidal integration of chord along span
        semi_area = float(np.trapezoid(c, y))
        return 2.0 * semi_area if self.symmetric else semi_area

    def aspect_ratio(self) -> float:
        """Aspect ratio AR = b^2 / S."""
        s = self.area()
        if s == 0.0:
            return 0.0
        return self.span() ** 2 / s

    def mean_aerodynamic_chord(self) -> float:
        """Mean aerodynamic chord by integration: MAC = (1/S_semi) * integral(c^2 dy) [m].

        Reference: Raymer, Aircraft Design, Ch. 7.
        """
        y = self._y_stations()
        c = self._chords()

        semi_area = float(np.trapezoid(c, y))
        if semi_area == 0.0:
            return 0.0

        c_squared = c**2
        return float(np.trapezoid(c_squared, y)) / semi_area

    def mean_aerodynamic_chord_le(self) -> np.ndarray:
        """Leading-edge position of the MAC [x, y, z] in aircraft frame [m].

        Computed as the area-weighted average of section LE positions.
        """
        y = self._y_stations()
        c = self._chords()

        semi_area = float(np.trapezoid(c, y))
        if semi_area == 0.0:
            return np.array([0.0, 0.0, 0.0])

        # Area-weighted LE position
        x_le = np.array([xsec.xyz_le[0] for xsec in self.xsecs])
        z_le = np.array([xsec.xyz_le[2] for xsec in self.xsecs])

        x_mac = float(np.trapezoid(x_le * c, y)) / semi_area
        y_mac = float(np.trapezoid(y * c, y)) / semi_area
        z_mac = float(np.trapezoid(z_le * c, y)) / semi_area

        return np.array([x_mac, y_mac, z_mac])

    def aerodynamic_center(self) -> np.ndarray:
        """Approximate aerodynamic center at quarter-chord of MAC [x, y, z] [m]."""
        mac_le = self.mean_aerodynamic_chord_le()
        mac = self.mean_aerodynamic_chord()
        # AC is at 25% MAC from the LE
        ac = mac_le.copy()
        ac[0] += 0.25 * mac
        return ac

    def taper_ratio(self) -> float:
        """Taper ratio: tip chord / root chord."""
        c = self._chords()
        if len(c) < 2 or c[0] == 0.0:
            return 1.0
        return float(c[-1] / c[0])

    def sweep_le(self) -> float:
        """Leading-edge sweep angle [deg].

        Computed from root and tip LE x-positions.
        """
        if len(self.xsecs) < 2:
            return 0.0
        dx = self.xsecs[-1].xyz_le[0] - self.xsecs[0].xyz_le[0]
        dy = self.xsecs[-1].xyz_le[1] - self.xsecs[0].xyz_le[1]
        if dy == 0.0:
            return 0.0
        return float(np.degrees(np.arctan2(dx, dy)))

    def dihedral(self) -> float:
        """Geometric dihedral angle [deg].

        Computed from root and tip LE z-positions.
        """
        if len(self.xsecs) < 2:
            return 0.0
        dz = self.xsecs[-1].xyz_le[2] - self.xsecs[0].xyz_le[2]
        dy = self.xsecs[-1].xyz_le[1] - self.xsecs[0].xyz_le[1]
        if dy == 0.0:
            return 0.0
        return float(np.degrees(np.arctan2(dz, dy)))

    # ------------------------------------------------------------------ #
    # Geometry computation helpers (used by aero solvers)
    # ------------------------------------------------------------------ #
    # Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
    # https://github.com/peterdsharpe/AeroSandbox

    def _rotation_matrix_3d(self, angle: float, axis: np.ndarray) -> np.ndarray:
        """Rotation matrix for *angle* (radians) about an arbitrary *axis*."""
        axis = axis / np.linalg.norm(axis)
        ux, uy, uz = axis
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c + ux**2 * (1 - c),       ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy**2 * (1 - c),       uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz**2 * (1 - c)],
        ])

    def _compute_frame_of_WingXSec(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local (xg, yg, zg) reference frame of cross-section *index*, in geometry axes.

        xg points chordwise (downstream), yg along the span, zg completes the
        right-hand system. Twist is applied about yg.
        """
        def _proj_yz_normalise(v: np.ndarray) -> np.ndarray:
            yz = np.array([0.0, v[1], v[2]])
            return yz / np.linalg.norm(yz)

        xg_local = np.array([1.0, 0.0, 0.0])

        if index == 0:
            yg_local = _proj_yz_normalise(self.xsecs[1].xyz_le - self.xsecs[0].xyz_le)
            z_scale = 1.0
        elif index >= len(self.xsecs) - 1:
            yg_local = _proj_yz_normalise(self.xsecs[-1].xyz_le - self.xsecs[-2].xyz_le)
            z_scale = 1.0
        else:
            vb = _proj_yz_normalise(self.xsecs[index].xyz_le - self.xsecs[index - 1].xyz_le)
            va = _proj_yz_normalise(self.xsecs[index + 1].xyz_le - self.xsecs[index].xyz_le)
            span_vec = (vb + va) / 2.0
            yg_local = span_vec / np.linalg.norm(span_vec)
            cos_v = float(np.dot(vb, va))
            z_scale = np.sqrt(2.0 / (cos_v + 1.0))

        zg_local = np.cross(xg_local, yg_local) * z_scale

        # Apply twist about yg_local
        twist_rad = np.radians(self.xsecs[index].twist)
        if twist_rad != 0.0:
            rot = self._rotation_matrix_3d(twist_rad, yg_local)
            xg_local = rot @ xg_local
            zg_local = rot @ zg_local

        return xg_local, yg_local, zg_local

    def _compute_xyz_of_WingXSec(
        self, index: int, x_nondim: float, z_nondim: float
    ) -> np.ndarray:
        """3-D point on cross-section *index* at normalised chord (x_nondim) and
        normal (z_nondim) positions."""
        xg, yg, zg = self._compute_frame_of_WingXSec(index)
        c = self.xsecs[index].chord
        return self.xsecs[index].xyz_le + x_nondim * c * xg + z_nondim * c * zg

    def _compute_frame_of_section(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local frame of the section *between* xsecs[index] and xsecs[index+1]."""
        in_front = self.xsecs[index].xyz_le
        in_back = self._compute_xyz_of_WingXSec(index, x_nondim=1, z_nondim=0)
        out_front = self.xsecs[index + 1].xyz_le
        out_back = self._compute_xyz_of_WingXSec(index + 1, x_nondim=1, z_nondim=0)

        diag1 = out_back - in_front
        diag2 = out_front - in_back
        cross = np.cross(diag1, diag2)
        zg_local = cross / np.linalg.norm(cross)

        qc_vec = (0.75 * out_front + 0.25 * out_back) - (0.75 * in_front + 0.25 * in_back)
        qc_vec[0] = 0.0
        yg_local = qc_vec / np.linalg.norm(qc_vec)
        xg_local = np.cross(yg_local, zg_local)

        return xg_local, yg_local, zg_local

    # ------------------------------------------------------------------ #
    # Subdivision and meshing
    # ------------------------------------------------------------------ #

    def subdivide_sections(
        self,
        ratio: int,
        spacing_function: Callable = None,
    ) -> Wing:
        """Split each section into *ratio* sub-sections by interpolation.

        Parameters
        ----------
        ratio : int
            Number of sub-sections per original section (>= 2).
        spacing_function : callable(start, stop, num) → array, optional
            Defaults to ``numpy.linspace``.
        """
        if spacing_function is None:
            spacing_function = np.linspace

        new_xsecs: list[WingXSec] = []
        fracs = spacing_function(0.0, 1.0, ratio + 1)[:-1]

        for xsec_a, xsec_b in zip(self.xsecs[:-1], self.xsecs[1:]):
            for s in fracs:
                a_w, b_w = 1.0 - s, s
                if b_w == 0 or xsec_a.airfoil == xsec_b.airfoil:
                    af = xsec_a.airfoil
                elif a_w == 0:
                    af = xsec_b.airfoil
                else:
                    af = xsec_a.airfoil.blend_with_another_airfoil(
                        airfoil=xsec_b.airfoil, blend_fraction=b_w
                    )
                new_xsecs.append(WingXSec(
                    xyz_le=xsec_a.xyz_le * a_w + xsec_b.xyz_le * b_w,
                    chord=xsec_a.chord * a_w + xsec_b.chord * b_w,
                    twist=xsec_a.twist * a_w + xsec_b.twist * b_w,
                    airfoil=af,
                ))

        new_xsecs.append(self.xsecs[-1])
        return Wing(
            name=self.name,
            xsecs=new_xsecs,
            symmetric=self.symmetric,
            control_surfaces=self.control_surfaces,
        )

    def mesh_line(
        self,
        x_nondim: float | Sequence[float] = 0.25,
        z_nondim: float | Sequence[float] = 0.0,
        add_camber: bool = True,
    ) -> list[np.ndarray]:
        """Return 3-D points at normalised (x_nondim, z_nondim) along each xsec.

        Goes from root to tip. Ignores symmetry (single side only).
        If *add_camber* is True, the camber of each section's airfoil is added to z_nondim.
        """
        points: list[np.ndarray] = []
        for i, xsec in enumerate(self.xsecs):
            xn = x_nondim[i] if isinstance(x_nondim, (list, np.ndarray)) else x_nondim
            zn = z_nondim[i] if isinstance(z_nondim, (list, np.ndarray)) else z_nondim
            if add_camber and xsec.airfoil is not None and xsec.airfoil.coordinates is not None:
                zn = zn + xsec.airfoil.local_camber(xn)
            points.append(self._compute_xyz_of_WingXSec(i, x_nondim=xn, z_nondim=zn))
        return points

    def mesh_thin_surface(
        self,
        chordwise_resolution: int = 4,
        chordwise_spacing_function: Callable | None = None,
        add_camber: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a structured quad mesh of the mean camber surface.

        Parameters
        ----------
        chordwise_resolution : int
            Number of chordwise panels.
        chordwise_spacing_function : callable(start, stop, num) → array
            Defaults to cosine spacing.
        add_camber : bool
            Whether to drape the mesh on the camber line.

        Returns
        -------
        points : (N, 3) array
            Mesh vertices.
        faces : (M, 4) array
            Quad face indices (front-left, back-left, back-right, front-right).
        """
        if chordwise_spacing_function is None:
            from aerisplane.utils.spacing import cosspace
            chordwise_spacing_function = cosspace

        x_nondim = chordwise_spacing_function(0.0, 1.0, chordwise_resolution + 1)

        # Generate spanwise strips — each strip is (num_xsecs, 3)
        spanwise_strips = []
        for xn in x_nondim:
            strip = np.stack(self.mesh_line(x_nondim=xn, z_nondim=0.0, add_camber=add_camber), axis=0)
            spanwise_strips.append(strip)

        points = np.concatenate(spanwise_strips, axis=0)

        num_i = len(spanwise_strips[0])   # spanwise stations
        num_j = len(spanwise_strips)       # chordwise stations

        def idx(i: int, j: int) -> int:
            return i + j * num_i

        faces = []
        for i in range(num_i - 1):
            for j in range(num_j - 1):
                faces.append([idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)])

        if self.symmetric:
            offset = len(points)
            mirrored = points * np.array([[1.0, -1.0, 1.0]])
            points = np.concatenate([points, mirrored], axis=0)
            for i in range(num_i - 1):
                for j in range(num_j - 1):
                    # Reversed winding for left wing
                    faces.append([
                        offset + idx(i + 1, j),
                        offset + idx(i + 1, j + 1),
                        offset + idx(i, j + 1),
                        offset + idx(i, j),
                    ])

        return points, np.array(faces, dtype=int)
