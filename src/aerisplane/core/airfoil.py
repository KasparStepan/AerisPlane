"""Airfoil geometry definition with NACA generators and file I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_AIRFOIL_DB = Path(__file__).parent.parent / "catalog" / "airfoils"


def _load_from_catalog(name: str) -> Optional[np.ndarray]:
    """Try to load airfoil coordinates from the catalog by name.

    Tries exact match first, then case-insensitive match.
    Returns None if not found.
    """
    for candidate in (name, name.lower(), name.upper()):
        path = _AIRFOIL_DB / f"{candidate}.dat"
        if path.exists():
            lines = path.read_text().strip().splitlines()
            coords = []
            for line in lines[1:]:  # skip name header
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        coords.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue
            if coords:
                return np.array(coords)
    return None


@dataclass(eq=False)
class Airfoil:
    """Airfoil defined by name and/or coordinate array.

    If ``name`` is given and ``coordinates`` is None, the catalog is searched
    for a matching ``.dat`` file. NACA 4-digit names (e.g. ``"naca2412"``) are
    generated analytically if not found in the catalog.

    Parameters
    ----------
    name : str
        Airfoil name, e.g. ``"ag35"``, ``"naca2412"``, ``"e387"``.
        Used for catalog lookup and plot labels.
    coordinates : ndarray of shape (N, 2) or None
        Explicit (x, y) coordinate array in Selig format (upper surface first,
        x from 0 to 1 and back to 0). If None, loaded from catalog by name.

    Examples
    --------
    >>> af = Airfoil(name="ag35")         # loads from catalog
    >>> af = Airfoil(name="naca2412")     # generated analytically
    """

    name: str
    coordinates: Optional[np.ndarray] = field(default=None, repr=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Airfoil):
            return NotImplemented
        if self.name != other.name:
            return False
        if self.coordinates is None and other.coordinates is None:
            return True
        if self.coordinates is None or other.coordinates is None:
            return False
        return np.array_equal(self.coordinates, other.coordinates)

    def __hash__(self) -> int:
        return hash(self.name)

    def __post_init__(self) -> None:
        if self.coordinates is None and self.name.lower().startswith("naca"):
            digits = self.name.lower().replace("naca", "").strip()
            if len(digits) == 4 and digits.isdigit():
                self.coordinates = naca4_coordinates(digits)
            elif len(digits) == 5 and digits.isdigit():
                # 5-digit NACA not implemented yet — fall through to catalog lookup
                pass
        if self.coordinates is None:
            self.coordinates = _load_from_catalog(self.name)
        if self.coordinates is not None:
            self.coordinates = np.asarray(self.coordinates, dtype=float)

    @staticmethod
    def from_naca(designation: str, n_points: int = 100) -> Airfoil:
        """Create airfoil from NACA 4-digit designation.

        Parameters
        ----------
        designation : str
            NACA 4-digit string, e.g. "2412" or "0012".
        n_points : int
            Number of points per surface (upper + lower).
        """
        coords = naca4_coordinates(designation, n_points=n_points)
        return Airfoil(name=f"naca{designation}", coordinates=coords)

    @staticmethod
    def from_file(path: str | Path) -> Airfoil:
        """Load airfoil from a Selig-format .dat file.

        The first line is the airfoil name. Subsequent lines are x y coordinate pairs.
        """
        path = Path(path)
        lines = path.read_text().strip().splitlines()
        name = lines[0].strip()
        coords = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
        return Airfoil(name=name, coordinates=np.array(coords))

    def thickness(self) -> float:
        """Maximum thickness as fraction of chord.

        Returns 0.0 if coordinates are not available.
        """
        if self.coordinates is None:
            return 0.0
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]

        # Find the leading edge index (minimum x)
        le_idx = np.argmin(x)

        # Upper surface: from TE to LE (indices 0..le_idx)
        x_upper = x[: le_idx + 1]
        y_upper = y[: le_idx + 1]

        # Lower surface: from LE to TE (indices le_idx..end)
        x_lower = x[le_idx:]
        y_lower = y[le_idx:]

        # Interpolate both surfaces onto common x stations
        x_stations = np.linspace(0.0, 1.0, 101)
        y_up = np.interp(x_stations, np.sort(x_upper), y_upper[np.argsort(x_upper)])
        y_lo = np.interp(x_stations, np.sort(x_lower), y_lower[np.argsort(x_lower)])

        return float(np.max(y_up - y_lo))

    def upper_coordinates(self) -> np.ndarray:
        """Upper surface coordinates, ordered from LE to TE.

        Returns (M, 2) array of [x, y] points.
        Returns a flat plate [0,0] → [1,0] if no coordinates are available.
        """
        if self.coordinates is None:
            return np.array([[0.0, 0.0], [1.0, 0.0]])
        le_idx = int(np.argmin(self.coordinates[:, 0]))
        # In Selig format: indices 0..le_idx are upper surface (TE→LE), reverse for LE→TE
        return self.coordinates[: le_idx + 1][::-1]

    def lower_coordinates(self) -> np.ndarray:
        """Lower surface coordinates, ordered from LE to TE.

        Returns (M, 2) array of [x, y] points.
        Returns a flat plate [0,0] → [1,0] if no coordinates are available.
        """
        if self.coordinates is None:
            return np.array([[0.0, 0.0], [1.0, 0.0]])
        le_idx = int(np.argmin(self.coordinates[:, 0]))
        return self.coordinates[le_idx:]

    def local_camber(self, x_over_c: float | np.ndarray = 0.5) -> float | np.ndarray:
        """Local camber (y/c) at given chordwise position(s).

        Parameters
        ----------
        x_over_c : float or array
            Chordwise positions as fraction of chord [0..1].
        """
        upper = self.upper_coordinates()
        lower = self.lower_coordinates()
        scalar = np.ndim(x_over_c) == 0
        x_arr = np.atleast_1d(np.asarray(x_over_c, dtype=float))
        y_up = np.interp(x_arr, upper[:, 0], upper[:, 1])
        y_lo = np.interp(x_arr, lower[:, 0], lower[:, 1])
        result = (y_up + y_lo) / 2.0
        return float(result[0]) if scalar else result

    def repanel(self, n_points_per_side: int = 100) -> Airfoil:
        """Return a new Airfoil with coordinates re-interpolated to uniform cosine spacing.

        Parameters
        ----------
        n_points_per_side : int
            Number of points on each of upper and lower surfaces.
        """
        upper = self.upper_coordinates()  # LE→TE
        lower = self.lower_coordinates()  # LE→TE
        x_new = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_points_per_side)))
        y_up = np.interp(x_new, upper[:, 0], upper[:, 1])
        y_lo = np.interp(x_new, lower[:, 0], lower[:, 1])
        # Selig format: upper TE→LE then lower LE→TE
        coords = np.column_stack([
            np.concatenate([x_new[::-1], x_new[1:]]),
            np.concatenate([y_up[::-1], y_lo[1:]]),
        ])
        return Airfoil(name=self.name, coordinates=coords)

    def blend_with_another_airfoil(
        self,
        airfoil: Airfoil,
        blend_fraction: float = 0.5,
        n_points_per_side: int = 100,
    ) -> Airfoil:
        """Blend this airfoil with another by linearly interpolating coordinates.

        Parameters
        ----------
        airfoil : Airfoil
            The other airfoil to blend with.
        blend_fraction : float
            0.0 = this airfoil, 1.0 = the other airfoil.
        n_points_per_side : int
            Points per surface for re-panelling before blending.
        """
        a = self.repanel(n_points_per_side)
        b = airfoil.repanel(n_points_per_side)
        coords = (1 - blend_fraction) * a.coordinates + blend_fraction * b.coordinates
        name = f"{(1 - blend_fraction) * 100:.0f}% {self.name}, {blend_fraction * 100:.0f}% {airfoil.name}"
        return Airfoil(name=name, coordinates=coords)

    def max_camber(self) -> float:
        """Maximum camber as fraction of chord.

        Returns 0.0 if coordinates are not available.
        """
        if self.coordinates is None:
            return 0.0
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]

        le_idx = np.argmin(x)

        x_upper = x[: le_idx + 1]
        y_upper = y[: le_idx + 1]
        x_lower = x[le_idx:]
        y_lower = y[le_idx:]

        x_stations = np.linspace(0.0, 1.0, 101)
        y_up = np.interp(x_stations, np.sort(x_upper), y_upper[np.argsort(x_upper)])
        y_lo = np.interp(x_stations, np.sort(x_lower), y_lower[np.argsort(x_lower)])

        camber_line = (y_up + y_lo) / 2.0
        return float(np.max(np.abs(camber_line)))


    def nondim_perimeter(self) -> float:
        """Normalized perimeter (arc length around full profile) as fraction of chord.

        For a zero-thickness flat plate this equals 2.0 (upper + lower = 1 + 1).
        For NACA 0012 it is approximately 2.03.
        Returns 2.0 if no coordinates are available.
        """
        if self.coordinates is None or len(self.coordinates) < 2:
            return 2.0
        diffs = np.diff(self.coordinates, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

    def nondim_area(self) -> float:
        """Normalized cross-sectional area as fraction of chord^2.

        Computed via the shoelace formula on the coordinate polygon.
        For NACA 0012 this is approximately 0.086.
        Returns 0.0 if no coordinates are available.
        """
        if self.coordinates is None or len(self.coordinates) < 3:
            return 0.0
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def get_aero_from_neuralfoil(
        self,
        alpha: float,
        Re: float,
        mach: float = 0.0,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        model_size: str = "large",
        control_surfaces=None,
        **_ignored,
    ) -> dict:
        """2-D aerodynamic coefficients from NeuralFoil.

        Calls the NeuralFoil neural-network-based airfoil model to obtain CL,
        CD, and CM at the given operating condition. NeuralFoil must be
        installed (``pip install neuralfoil``).

        Parameters
        ----------
        alpha : float or array
            Angle of attack [deg].
        Re : float or array
            Reynolds number.
        n_crit : float
            Transition criterion (default 9.0).
        model_size : str
            NeuralFoil model size: "xxsmall", "xsmall", "small", "medium",
            "large" (default), "xlarge", "xxlarge".
        control_surfaces : list or None
            Ignored for now (Phase 4 will wire in deflections).

        Returns
        -------
        dict with keys "CL", "CD", "CM" (and others from NeuralFoil).
        """
        try:
            import neuralfoil as nf
        except ImportError as exc:
            raise ImportError(
                "NeuralFoil is required for AeroBuildup wing analysis.\n"
                "Install with:  pip install neuralfoil"
            ) from exc

        coords = self.coordinates
        if coords is None:
            # No coordinates: generate from name if NACA 4-digit, else error
            if self.name.lower().startswith("naca"):
                digits = self.name.lower().replace("naca", "").strip()
                if len(digits) == 4 and digits.isdigit():
                    coords = naca4_coordinates(digits)
            if coords is None:
                raise ValueError(
                    f"Airfoil '{self.name}' has no coordinates and cannot be "
                    "converted to Kulfan parameters for NeuralFoil."
                )

        result = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=alpha,
            Re=Re,
            model_size=model_size,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
        )

        # Prandtl-Glauert compressibility correction.
        # Scales CL, CD, CM by 1/β where β = sqrt(1 - M²).
        # Valid for subsonic flow (M < ~0.8).  Above that, wave drag dominates
        # and a dedicated transonic model is needed.
        mach_pg = float(np.clip(mach, 0.0, 0.99))
        if mach_pg > 0.01:
            beta = (1.0 - mach_pg**2) ** 0.5
            result = dict(result)   # shallow copy — do not mutate NeuralFoil's output
            result["CL"] = result["CL"] / beta
            result["CD"] = result["CD"] / beta
            result["CM"] = result["CM"] / beta

        return result


def naca4_coordinates(designation: str, n_points: int = 100) -> np.ndarray:
    """Generate NACA 4-digit airfoil coordinates.

    Parameters
    ----------
    designation : str
        4-digit string, e.g. "2412". Digits: max camber %, camber position /10, thickness %.
    n_points : int
        Number of points per surface.

    Returns
    -------
    coordinates : ndarray
        (2*n_points - 1, 2) array in Selig format (upper TE → LE → lower TE).
    """
    m = int(designation[0]) / 100.0       # max camber
    p = int(designation[1]) / 10.0        # position of max camber
    t = int(designation[2:4]) / 100.0     # thickness

    # Cosine spacing for better resolution at LE and TE
    beta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))

    # Thickness distribution (NACA formula, finite TE)
    yt = (
        t
        / 0.2
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )

    # Camber line and its gradient
    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x <= p,
            m / p**2 * (2.0 * p * x - x**2),
            m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
        )
        dyc_dx = np.where(
            x <= p,
            2.0 * m / p**2 * (p - x),
            2.0 * m / (1.0 - p) ** 2 * (p - x),
        )

    theta = np.arctan(dyc_dx)

    # Upper and lower surfaces
    x_upper = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)

    # Selig format: upper surface from TE to LE, then lower surface from LE to TE
    x_coords = np.concatenate([x_upper[::-1], x_lower[1:]])
    y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])

    return np.column_stack([x_coords, y_coords])
