"""Structural material and cross-section definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Material:
    """Intrinsic material properties — reusable across different sections.

    Parameters
    ----------
    name : str
        Human-readable name (e.g., "Carbon Fiber").
    density : float
        Density [kg/m^3].
    E : float
        Young's modulus [Pa].
    yield_strength : float
        Tensile yield strength [Pa].
    poisson_ratio : float
        Poisson's ratio [-].
    shear_modulus : float or None
        Shear modulus [Pa]. Computed from E and poisson_ratio if not provided.
    """

    name: str
    density: float
    E: float
    yield_strength: float
    poisson_ratio: float = 0.3
    shear_modulus: Optional[float] = None

    def __post_init__(self) -> None:
        if self.shear_modulus is None:
            self.shear_modulus = self.E / (2.0 * (1.0 + self.poisson_ratio))


@dataclass
class TubeSection:
    """Cross-section geometry for a tubular spar.

    Parameters
    ----------
    outer_diameter : float
        Outer diameter [m].
    wall_thickness : float
        Wall thickness [m].
    """

    outer_diameter: float
    wall_thickness: float

    def inner_diameter(self) -> float:
        """Inner diameter [m]."""
        return self.outer_diameter - 2.0 * self.wall_thickness

    def area(self) -> float:
        """Cross-sectional area of the tube wall [m^2]."""
        od = self.outer_diameter
        id_ = self.inner_diameter()
        return np.pi / 4.0 * (od**2 - id_**2)

    def second_moment_of_area(self) -> float:
        """Second moment of area I = pi/64 * (OD^4 - ID^4) [m^4]."""
        od = self.outer_diameter
        id_ = self.inner_diameter()
        return np.pi / 64.0 * (od**4 - id_**4)

    def section_modulus(self) -> float:
        """Section modulus S = I / (OD/2) [m^3]."""
        return self.second_moment_of_area() / (self.outer_diameter / 2.0)


@dataclass
class Spar:
    """Wing spar definition.

    Parameters
    ----------
    position : float
        Chordwise position as fraction of chord (0 = LE, 1 = TE).
    material : Material
        Spar material.
    section : TubeSection
        Cross-section geometry.
    """

    position: float
    material: Material
    section: TubeSection

    def mass_per_length(self) -> float:
        """Linear mass density [kg/m]."""
        return self.section.area() * self.material.density

    def max_bending_stress(self, bending_moment: float) -> float:
        """Maximum bending stress for a given bending moment [Pa].

        Parameters
        ----------
        bending_moment : float
            Applied bending moment [N*m].
        """
        return abs(bending_moment) / self.section.section_modulus()

    def margin_of_safety(self, bending_moment: float) -> float:
        """Margin of safety: (yield / stress) - 1. Positive means safe."""
        stress = self.max_bending_stress(bending_moment)
        if stress == 0.0:
            return float("inf")
        return self.material.yield_strength / stress - 1.0


@dataclass
class Skin:
    """Surface skin definition.

    Parameters
    ----------
    material : Material
        Skin material.
    thickness : float
        Skin thickness [m].
    """

    material: Material
    thickness: float

    def mass_per_area(self) -> float:
        """Areal mass density [kg/m^2]."""
        return self.thickness * self.material.density
