"""Flight condition definition for aerodynamic and mission analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerisplane.utils.atmosphere import isa, speed_of_sound


@dataclass
class FlightCondition:
    """Aerodynamic state for a single analysis point.

    Parameters
    ----------
    velocity : float
        True airspeed [m/s].
    altitude : float
        Geometric altitude above MSL [m].
    alpha : float
        Angle of attack [deg].
    beta : float
        Sideslip angle [deg].
    deflections : dict[str, float]
        Control surface deflections keyed by surface name [deg].
        Example: ``{"elevator": -3.5, "aileron": 10.0}``
    """

    velocity: float
    altitude: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    p: float = 0.0  # Roll rate [rad/s] in body axes
    q: float = 0.0  # Pitch rate [rad/s] in body axes
    r: float = 0.0  # Yaw rate [rad/s] in body axes
    deflections: dict[str, float] = field(default_factory=dict)

    def atmosphere(self) -> tuple[float, float, float, float]:
        """Return (temperature, pressure, density, viscosity) at altitude."""
        return isa(self.altitude)

    def density(self) -> float:
        """Air density at altitude [kg/m^3]."""
        _, _, rho, _ = self.atmosphere()
        return rho

    def reynolds(self, reference_length: float) -> float:
        """Reynolds number (alias for reynolds_number)."""
        return self.reynolds_number(reference_length)

    def copy(self) -> "FlightCondition":
        """Return a shallow copy of this FlightCondition."""
        import dataclasses
        return dataclasses.replace(self)

    def dynamic_pressure(self) -> float:
        """Dynamic pressure q = 0.5 * rho * V^2 [Pa]."""
        _, _, rho, _ = self.atmosphere()
        return 0.5 * rho * self.velocity**2

    def reynolds_number(self, reference_length: float) -> float:
        """Reynolds number Re = rho * V * L / mu.

        Parameters
        ----------
        reference_length : float
            Reference length, typically mean aerodynamic chord [m].
        """
        _, _, rho, mu = self.atmosphere()
        return rho * self.velocity * reference_length / mu

    def mach(self) -> float:
        """Mach number at altitude."""
        a = speed_of_sound(self.altitude)
        return self.velocity / a

    def alpha_rad(self) -> float:
        """Angle of attack in radians."""
        return np.radians(self.alpha)

    def beta_rad(self) -> float:
        """Sideslip angle in radians."""
        return np.radians(self.beta)

    # ------------------------------------------------------------------ #
    # Velocity / axis helpers (used by aero solvers)
    # ------------------------------------------------------------------ #

    def freestream_velocity_geometry_axes(self) -> np.ndarray:
        """Freestream velocity vector in geometry axes [m/s].

        Geometry axes: x downstream, y right, z up.
        Wind axes:     x upstream (into the wind), y right, z up.

        The rotation from wind to geometry is: flip x/z (pi about y),
        then rotate by -alpha about y, then by +beta about z.
        """
        a = np.radians(self.alpha)
        b = np.radians(self.beta)
        # Direction the wind is going to, in geometry axes
        # Derived from: flip_y @ Ry(-alpha) @ Rz(beta) @ [-1, 0, 0]
        direction = np.array([
            np.cos(a) * np.cos(b),
            -np.sin(b),
            np.sin(a) * np.cos(b),
        ])
        return direction * self.velocity

    def rotation_velocity_geometry_axes(self, points: np.ndarray,
                                        p: float = 0.0,
                                        q: float = 0.0,
                                        r: float = 0.0) -> np.ndarray:
        """Velocity at *points* due to aircraft rotation [m/s].

        Parameters
        ----------
        points : (N, 3) array
            Points in geometry axes.
        p, q, r : float
            Roll, pitch, yaw rates [rad/s] in body axes.

        Returns
        -------
        (N, 3) array
            Velocity the wing *sees* at each point (opposite of point velocity).
        """
        # Body-to-geometry sign convention: x_geom = -x_body, z_geom = -z_body
        omega = np.array([-p, q, -r])
        # Cross product omega x r, then negate (velocity wing sees, not wing velocity)
        vel = np.stack([
            omega[1] * points[:, 2] - omega[2] * points[:, 1],
            omega[2] * points[:, 0] - omega[0] * points[:, 2],
            omega[0] * points[:, 1] - omega[1] * points[:, 0],
        ], axis=1)
        return -vel

    def convert_axes(
        self,
        x_from: float | np.ndarray,
        y_from: float | np.ndarray,
        z_from: float | np.ndarray,
        from_axes: str,
        to_axes: str,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Convert a vector between geometry / body / wind / stability axes.

        Follows the conventions in Drela, *Flight Vehicle Aerodynamics*, §6.2.2.

        Geometry axes: x downstream, y right, z up.
        Body axes:     x forward, y right, z down.
        Wind axes:     x into wind, y right, z "up" (lift direction).
        Stability axes: body axes rotated by alpha about y.
        """
        if from_axes == to_axes:
            return x_from, y_from, z_from

        sa = np.sin(np.radians(self.alpha))
        ca = np.cos(np.radians(self.alpha))
        sb = np.sin(np.radians(self.beta))
        cb = np.cos(np.radians(self.beta))

        # --- Step 1: convert from_axes → body ---
        if from_axes == "geometry":
            x_b, y_b, z_b = -x_from, y_from, -z_from
        elif from_axes == "body":
            x_b, y_b, z_b = x_from, y_from, z_from
        elif from_axes == "wind":
            x_b = (cb * ca) * x_from + (-sb * ca) * y_from + (-sa) * z_from
            y_b = sb * x_from + cb * y_from
            z_b = (cb * sa) * x_from + (-sb * sa) * y_from + ca * z_from
        elif from_axes == "stability":
            x_b = ca * x_from - sa * z_from
            y_b = y_from
            z_b = sa * x_from + ca * z_from
        else:
            raise ValueError(f"Unknown from_axes '{from_axes}'")

        # --- Step 2: convert body → to_axes ---
        if to_axes == "geometry":
            return -x_b, y_b, -z_b
        elif to_axes == "body":
            return x_b, y_b, z_b
        elif to_axes == "wind":
            x_to = (cb * ca) * x_b + sb * y_b + (cb * sa) * z_b
            y_to = (-sb * ca) * x_b + cb * y_b + (-sb * sa) * z_b
            z_to = -sa * x_b + ca * z_b
            return x_to, y_to, z_to
        elif to_axes == "stability":
            x_to = ca * x_b + sa * z_b
            y_to = y_b
            z_to = -sa * x_b + ca * z_b
            return x_to, y_to, z_to
        else:
            raise ValueError(f"Unknown to_axes '{to_axes}'")
