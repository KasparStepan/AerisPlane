"""Wing and wing cross-section definitions with geometry methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

# _trapz was added in NumPy 2.0; fall back to np.trapz for NumPy <2.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

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

    def xsec_area(self) -> float:
        """Cross-sectional area of this wing section [m^2].

        Computed as nondimensional airfoil area * chord^2.
        Returns 0.0 if no airfoil is set.
        """
        if self.airfoil is None:
            return 0.0
        return self.airfoil.nondim_area() * self.chord ** 2


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
        """Spanwise coordinate along the dominant span direction [m].

        For a conventional wing (span in Y) returns the Y coordinates.
        For a vertical tail (span in Z) returns the Z coordinates measured
        from root, so that area() and semispan() give correct values in both
        cases.  The dominant direction is whichever of Y or Z has the larger
        range across the cross-sections.
        """
        ys = np.array([xs.xyz_le[1] for xs in self.xsecs])
        zs = np.array([xs.xyz_le[2] for xs in self.xsecs])
        if np.max(zs) - np.min(zs) > np.max(ys) - np.min(ys):
            # Z-dominant surface (vertical tail): integrate along Z from root
            return zs - zs[0]
        return ys

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

    def area(self, type: str = "planform") -> float:
        """Wing area [m^2].

        Parameters
        ----------
        type : str
            "planform" (default): trapezoidal planform area.
            "wetted": surface area including airfoil thickness, both surfaces.
            "xy": planform area projected onto XY plane (top-down view).
            "xz": area projected onto XZ plane (side view).

        For symmetric wings, returns the total area for both sides in all modes.
        """
        if type == "planform":
            y = self._y_stations()
            c = self._chords()
            semi_area = float(_trapz(c, y))
            return 2.0 * semi_area if self.symmetric else semi_area

        elif type == "wetted":
            spans = self.sectional_span_yz()
            total = 0.0
            for i, span in enumerate(spans):
                xa, xb = self.xsecs[i], self.xsecs[i + 1]
                p_a = xa.airfoil.nondim_perimeter() if xa.airfoil is not None else 2.0
                p_b = xb.airfoil.nondim_perimeter() if xb.airfoil is not None else 2.0
                avg_wetted_chord = (xa.chord * p_a + xb.chord * p_b) / 2.0
                total += span * avg_wetted_chord
            return 2.0 * total if self.symmetric else total

        elif type == "xy":
            total = 0.0
            for xa, xb in zip(self.xsecs[:-1], self.xsecs[1:]):
                dy = abs(xb.xyz_le[1] - xa.xyz_le[1])
                total += dy * (xa.chord + xb.chord) / 2.0
            return 2.0 * total if self.symmetric else total

        elif type == "xz":
            total = 0.0
            for xa, xb in zip(self.xsecs[:-1], self.xsecs[1:]):
                dz = abs(xb.xyz_le[2] - xa.xyz_le[2])
                total += dz * (xa.chord + xb.chord) / 2.0
            return 2.0 * total if self.symmetric else total

        else:
            raise ValueError(
                f"type must be 'planform', 'wetted', 'xy', or 'xz', got '{type!r}'"
            )

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

        semi_area = float(_trapz(c, y))
        if semi_area == 0.0:
            return 0.0

        c_squared = c**2
        return float(_trapz(c_squared, y)) / semi_area

    def mean_aerodynamic_chord_le(self) -> np.ndarray:
        """Leading-edge position of the MAC [x, y, z] in aircraft frame [m].

        Computed as the area-weighted average of section LE positions.
        """
        y = self._y_stations()
        c = self._chords()

        semi_area = float(_trapz(c, y))
        if semi_area == 0.0:
            return np.array([0.0, 0.0, 0.0])

        # Area-weighted LE position (use actual coordinates, arc-length as measure)
        x_le = np.array([xsec.xyz_le[0] for xsec in self.xsecs])
        y_le = np.array([xsec.xyz_le[1] for xsec in self.xsecs])
        z_le = np.array([xsec.xyz_le[2] for xsec in self.xsecs])

        x_mac = float(_trapz(x_le * c, y)) / semi_area
        y_mac = float(_trapz(y_le * c, y)) / semi_area
        z_mac = float(_trapz(z_le * c, y)) / semi_area

        return np.array([x_mac, y_mac, z_mac])

    def aerodynamic_center(self) -> np.ndarray:
        """Approximate aerodynamic center at quarter-chord of MAC ``[x, y, z]`` in metres."""
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

    def mean_sweep_angle(self) -> float:
        """Mean quarter-chord sweep angle [deg].

        Sweep of the line connecting root and tip quarter-chord points,
        measured from the y-axis in the xy-plane.
        """
        if len(self.xsecs) < 2:
            return 0.0
        qc_root = self._compute_xyz_of_WingXSec(0, 0.25, 0)
        qc_tip = self._compute_xyz_of_WingXSec(len(self.xsecs) - 1, 0.25, 0)
        dx = qc_tip[0] - qc_root[0]
        dy = qc_tip[1] - qc_root[1]
        if dy == 0.0:
            return 0.0
        return float(np.degrees(np.arctan2(dx, dy)))

    def mean_dihedral_angle(self) -> float:
        """Mean dihedral angle [deg].

        Dihedral of the line connecting root and tip LE points, in the yz-plane.
        """
        return self.dihedral()

    def mean_geometric_chord(self) -> float:
        """Mean geometric chord: area / span [m]."""
        s = self.span()
        return 0.0 if s == 0.0 else self.area() / s

    def mean_twist_angle(self) -> float:
        """Area-weighted mean twist angle [deg].

        Positive = nose-up. Zero for an untwisted wing.
        """
        areas = self.sectional_areas()
        total_area = sum(areas)
        if total_area == 0.0:
            return 0.0
        twists = [
            (xa.twist + xb.twist) / 2.0
            for xa, xb in zip(self.xsecs[:-1], self.xsecs[1:])
        ]
        return float(sum(t * a for t, a in zip(twists, areas)) / total_area)

    def volume(self) -> float:
        """Total structural volume of the wing [m^3].

        Uses the Prismatoid formula per section (exact for frustum-shaped bodies).
        Includes both sides for symmetric wings.
        """
        spans = self.sectional_span_yz()
        total = 0.0
        for i, span in enumerate(spans):
            a_a = self.xsecs[i].xsec_area()
            a_b = self.xsecs[i + 1].xsec_area()
            total += span / 3.0 * (a_a + a_b + (a_a * a_b + 1e-100) ** 0.5)
        return 2.0 * total if self.symmetric else total

    def sectional_span_yz(self) -> list[float]:
        """Span of each section measured in the YZ plane [m].

        Returns one value per section (len(xsecs) - 1 entries).
        """
        result = []
        for i in range(len(self.xsecs) - 1):
            qc_a = self._compute_xyz_of_WingXSec(i, 0.25, 0)
            qc_b = self._compute_xyz_of_WingXSec(i + 1, 0.25, 0)
            dy = qc_b[1] - qc_a[1]
            dz = qc_b[2] - qc_a[2]
            result.append(float(np.sqrt(dy**2 + dz**2)))
        return result

    def sectional_areas(self) -> list[float]:
        """Planform area of each section [m^2].

        Computed as the average chord times the YZ-plane section span.
        Returns one value per section (len(xsecs) - 1 entries).
        """
        spans = self.sectional_span_yz()
        result = []
        for i, span in enumerate(spans):
            chord_avg = (self.xsecs[i].chord + self.xsecs[i + 1].chord) / 2.0
            result.append(chord_avg * span)
        return result

    def sectional_aerodynamic_centers(self) -> list[np.ndarray]:
        """Approximate aerodynamic center of each section ``[x, y, z]`` in metres.

        Computed as the midpoint of the section's quarter-chord line.
        Returns one (3,) array per section (len(xsecs) - 1 entries).
        """
        result = []
        for i in range(len(self.xsecs) - 1):
            qc_a = self._compute_xyz_of_WingXSec(i, 0.25, 0)
            qc_b = self._compute_xyz_of_WingXSec(i + 1, 0.25, 0)
            result.append((qc_a + qc_b) / 2.0)
        return result

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
