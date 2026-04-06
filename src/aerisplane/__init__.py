"""AerisPlane — Conceptual MDO toolkit for RC/UAV aircraft design."""

# Discipline modules
from aerisplane import propulsion
from aerisplane.propulsion import PropulsionResult

# Core geometry and components
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.control_surface import ControlSurface, Servo
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.core.fuselage import Fuselage, FuselageXSec
from aerisplane.core.payload import Payload
from aerisplane.core.placement import (
    Collision,
    ComponentBox,
    PlacementResult,
    validate_placement,
)
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
    # Core geometry
    "Aircraft",
    "Airfoil",
    "Wing",
    "WingXSec",
    "Fuselage",
    "FuselageXSec",
    "ControlSurface",
    "Servo",
    "FlightCondition",
    # Structures
    "Material",
    "TubeSection",
    "Spar",
    "Skin",
    # Propulsion
    "Motor",
    "Propeller",
    "PropellerPerfData",
    "Battery",
    "ESC",
    "PropulsionSystem",
    "PropulsionResult",
    "propulsion",
    # Placement
    "ComponentBox",
    "Collision",
    "PlacementResult",
    "validate_placement",
    # Other
    "Payload",
]
