"""Core data model — geometry, components, flight conditions.

All classes in this module depend only on numpy. No solver or plotting imports.
"""

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.control_surface import ControlSurface, Servo
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.fuselage import Fuselage, FuselageXSec
from aerisplane.core.payload import Payload
from aerisplane.core.propulsion import (
    ESC,
    Battery,
    Motor,
    Propeller,
    PropellerPerfData,
    PropulsionSystem,
)
from aerisplane.core.structures import Material, Skin, Spar, TubeSection
from aerisplane.core.wing import Wing, WingXSec

__all__ = [
    "Aircraft",
    "Airfoil",
    "Battery",
    "ControlSurface",
    "ESC",
    "FlightCondition",
    "Fuselage",
    "FuselageXSec",
    "Material",
    "Motor",
    "Payload",
    "Propeller",
    "PropellerPerfData",
    "PropulsionSystem",
    "Servo",
    "Skin",
    "Spar",
    "TubeSection",
    "Wing",
    "WingXSec",
]
