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
    deflections: dict[str, float] = field(default_factory=dict)

    def atmosphere(self) -> tuple[float, float, float, float]:
        """Return (temperature, pressure, density, viscosity) at altitude."""
        return isa(self.altitude)

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
