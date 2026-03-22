"""Top-level aircraft configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from aerisplane.core.fuselage import Fuselage
from aerisplane.core.payload import Payload
from aerisplane.core.propulsion import PropulsionSystem
from aerisplane.core.wing import Wing


@dataclass
class Aircraft:
    """Complete aircraft configuration.

    This is the top-level container that holds all aircraft components.
    All discipline modules accept this as their primary input.

    Parameters
    ----------
    name : str
        Aircraft name / configuration identifier.
    wings : list of Wing
        All lifting surfaces (main wing, horizontal tail, vertical tail).
    fuselages : list of Fuselage
        Fuselage bodies.
    propulsion : PropulsionSystem or None
        Propulsion system.
    payload : Payload or None
        Payload definition.
    """

    name: str
    wings: list[Wing] = field(default_factory=list)
    fuselages: list[Fuselage] = field(default_factory=list)
    propulsion: PropulsionSystem | None = None
    payload: Payload | None = None

    def get_wing(self, name: str) -> Wing | None:
        """Find a wing by name. Returns None if not found."""
        for w in self.wings:
            if w.name == name:
                return w
        return None

    def main_wing(self) -> Wing | None:
        """Return the first wing that looks like a main wing (largest area)."""
        if not self.wings:
            return None
        return max(self.wings, key=lambda w: w.area())

    def reference_area(self) -> float:
        """Reference area: area of the main (largest) wing [m^2]."""
        mw = self.main_wing()
        return mw.area() if mw else 0.0

    def reference_span(self) -> float:
        """Reference span: span of the main (largest) wing [m]."""
        mw = self.main_wing()
        return mw.span() if mw else 0.0

    def reference_chord(self) -> float:
        """Reference chord: MAC of the main (largest) wing [m]."""
        mw = self.main_wing()
        return mw.mean_aerodynamic_chord() if mw else 0.0
