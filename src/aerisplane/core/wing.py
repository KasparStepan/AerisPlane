"""Wing and wing cross-section definitions with geometry methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
