"""Mission segment definitions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass
class Climb:
    """Climbing segment."""
    to_altitude: float      # target altitude [m MSL]
    climb_rate: float       # rate of climb [m/s]
    velocity: float         # true airspeed during climb [m/s]
    name: str = "climb"


@dataclass
class Cruise:
    """Steady level cruise segment."""
    distance: float         # horizontal distance [m]
    velocity: float         # cruise airspeed [m/s]
    altitude: float = 100.0 # cruise altitude [m MSL]
    name: str = "cruise"


@dataclass
class Loiter:
    """Loitering (circling/holding) segment at constant altitude."""
    duration: float         # loiter time [s]
    velocity: float         # loiter airspeed [m/s]
    altitude: float = 100.0 # loiter altitude [m MSL]
    name: str = "loiter"


@dataclass
class Return:
    """Return leg — functionally identical to cruise, semantically distinct."""
    distance: float
    velocity: float
    altitude: float = 100.0
    name: str = "return"


@dataclass
class Descent:
    """Descending segment (partial power or glide)."""
    to_altitude: float          # target altitude [m MSL]
    descent_rate: float = 2.0   # descent rate [m/s] (positive = descending)
    velocity: float = 15.0      # airspeed [m/s]
    name: str = "descent"


@dataclass
class Mission:
    """Ordered sequence of mission segments."""
    segments: list[Union[Climb, Cruise, Loiter, Return, Descent]]
    start_altitude: float = 0.0
