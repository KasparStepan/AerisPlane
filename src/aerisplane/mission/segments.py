"""Mission segment definitions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass
class Climb:
    """Climb segment at constant airspeed and climb rate.

    Parameters
    ----------
    name : str
        Segment identifier for reporting.
    velocity : float
        True airspeed during climb [m/s].
    climb_rate : float
        Rate of climb [m/s].
    to_altitude : float
        Target altitude at end of segment [m].
    """
    to_altitude: float      # target altitude [m MSL]
    climb_rate: float       # rate of climb [m/s]
    velocity: float         # true airspeed during climb [m/s]
    name: str = "climb"


@dataclass
class Cruise:
    """Cruise segment at constant velocity, altitude, and distance.

    Parameters
    ----------
    name : str
    velocity : float
        True airspeed [m/s].
    altitude : float
        Cruise altitude [m].
    distance : float
        Horizontal distance [m].
    """
    distance: float         # horizontal distance [m]
    velocity: float         # cruise airspeed [m/s]
    altitude: float = 100.0 # cruise altitude [m MSL]
    name: str = "cruise"


@dataclass
class Loiter:
    """Loiter (hold) segment at constant velocity, altitude, and duration.

    Parameters
    ----------
    name : str
    velocity : float
        Loiter airspeed [m/s].
    altitude : float
        Loiter altitude [m].
    duration : float
        Loiter time [s].
    """
    duration: float         # loiter time [s]
    velocity: float         # loiter airspeed [m/s]
    altitude: float = 100.0 # loiter altitude [m MSL]
    name: str = "loiter"


@dataclass
class Return:
    """Return (cruise back) segment — identical physics to Cruise.

    Parameters
    ----------
    name : str
    velocity : float
        True airspeed [m/s].
    altitude : float
        Return altitude [m].
    distance : float
        Horizontal distance [m].
    """
    distance: float
    velocity: float
    altitude: float = 100.0
    name: str = "return"


@dataclass
class Descent:
    """Descent segment at constant airspeed and descent rate.

    Parameters
    ----------
    name : str
    velocity : float
        True airspeed during descent [m/s].
    descent_rate : float
        Rate of descent (positive value) [m/s].
    to_altitude : float
        Target altitude at end of descent [m].
    """
    to_altitude: float          # target altitude [m MSL]
    descent_rate: float = 2.0   # descent rate [m/s] (positive = descending)
    velocity: float = 15.0      # airspeed [m/s]
    name: str = "descent"


@dataclass
class Mission:
    """Mission profile composed of sequential flight segments.

    Parameters
    ----------
    segments : list
        Ordered list of mission segments (Climb, Cruise, Loiter, Descent, Return).
    start_altitude : float, optional
        Starting altitude above MSL [m]. Default 0.0.
    """
    segments: list[Union[Climb, Cruise, Loiter, Return, Descent]]
    start_altitude: float = 0.0
