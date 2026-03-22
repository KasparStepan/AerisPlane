"""International Standard Atmosphere (ISA) model.

Valid for the troposphere (0-11,000 m). Returns temperature, pressure,
density, and dynamic viscosity at a given geometric altitude.
"""

import numpy as np


# ISA sea-level constants
T0 = 288.15        # temperature [K]
P0 = 101325.0       # pressure [Pa]
RHO0 = 1.225        # density [kg/m^3]
LAPSE = -0.0065     # temperature lapse rate [K/m]
G = 9.80665         # gravitational acceleration [m/s^2]
R = 287.058         # specific gas constant for dry air [J/(kg*K)]
GAMMA = 1.4         # ratio of specific heats for air

# Sutherland's law constants for dynamic viscosity
MU_REF = 1.716e-5   # reference viscosity [Pa*s]
T_REF = 273.15      # reference temperature [K]
S_SUTH = 110.4      # Sutherland constant [K]


def isa(altitude: float) -> tuple[float, float, float, float]:
    """Compute ISA atmosphere properties at a given altitude.

    Parameters
    ----------
    altitude : float
        Geometric altitude above mean sea level [m]. Valid range: 0 to 11000.

    Returns
    -------
    temperature : float
        Static temperature [K].
    pressure : float
        Static pressure [Pa].
    density : float
        Air density [kg/m^3].
    dynamic_viscosity : float
        Dynamic viscosity [Pa*s].
    """
    altitude = np.clip(altitude, 0.0, 11000.0)

    temperature = T0 + LAPSE * altitude
    pressure = P0 * (temperature / T0) ** (-G / (LAPSE * R))
    density = pressure / (R * temperature)

    # Sutherland's law
    dynamic_viscosity = MU_REF * (temperature / T_REF) ** 1.5 * (T_REF + S_SUTH) / (
        temperature + S_SUTH
    )

    return temperature, pressure, density, dynamic_viscosity


def speed_of_sound(altitude: float) -> float:
    """Speed of sound at altitude [m/s]."""
    temperature, _, _, _ = isa(altitude)
    return float(np.sqrt(GAMMA * R * temperature))
