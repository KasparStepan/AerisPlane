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
    xyz_ref: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Moment reference point [x, y, z] in aircraft frame [m].

    Defaults to the origin. Set to the CG location for stability analysis.
    """

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

    def aerodynamic_center(self) -> "np.ndarray":
        """Approximate aerodynamic center of the full aircraft [x, y, z] in metres.

        Computed as the planform-area-weighted average of each wing's aerodynamic center.
        Returns the origin [0, 0, 0] if no wings are present.
        """
        import numpy as np

        if not self.wings:
            return np.array([0.0, 0.0, 0.0])

        areas = [w.area() for w in self.wings]
        acs = [w.aerodynamic_center() for w in self.wings]
        total_area = sum(areas)

        if total_area == 0.0:
            return np.array([0.0, 0.0, 0.0])

        return sum(ac * a for ac, a in zip(acs, areas)) / total_area

    def is_entirely_symmetric(self) -> bool:
        """True if every wing on this aircraft is geometrically symmetric.

        Returns True for an aircraft with no wings.
        """
        return all(wing.is_entirely_symmetric() for wing in self.wings)

    def draw(self, backend: str = "plotly", show: bool = True, **kwargs):
        """Visualize this aircraft. See ``aerisplane.viz.draw`` for full docs."""
        from aerisplane.viz import draw
        return draw(self, backend=backend, show=show, **kwargs)
