# src/aerisplane/structures/beam.py
"""Euler-Bernoulli cantilever beam model for wing structural sizing.

The wing spar is modelled as a cantilever fixed at root (y=0), free at
tip (y=b).  The net distributed load is:

    q_net(y) = n · q_aero(y)  −  n · g · m'(y)

where q_aero is the aerodynamic lift per unit span (elliptic approximation),
n is the load factor, and m'(y) is the structural mass per unit span
(inertia relief).

Integration proceeds from tip to root for V and M, then root to tip
for slope θ and deflection δ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerisplane.core.wing import Wing, WingXSec
from aerisplane.structures.section import effective_EI

_G = 9.81  # m/s²


@dataclass
class BeamResult:
    """Spanwise structural solution for one wing semi-span.

    Parameters
    ----------
    y : ndarray
        Spanwise stations [m], from root (y[0]) to tip (y[-1]).
    V : ndarray
        Shear force [N] at each station.
    M : ndarray
        Bending moment [N·m] at each station.
    theta : ndarray
        Slope dδ/dy [rad] at each station.
    delta : ndarray
        Deflection [m] at each station (positive = upward).
    EI : ndarray
        Bending stiffness [N·m²] at each station.
    GJ : ndarray
        Torsional stiffness [N·m²/rad] at each station.
    """

    y: np.ndarray
    V: np.ndarray
    M: np.ndarray
    theta: np.ndarray
    delta: np.ndarray
    EI: np.ndarray
    GJ: np.ndarray

    @property
    def tip_deflection(self) -> float:
        """Tip deflection δ(tip) [m]."""
        return float(self.delta[-1])

    @property
    def root_bending_moment(self) -> float:
        """Bending moment at root M(0) [N·m]."""
        return float(self.M[0])

    @property
    def root_shear_force(self) -> float:
        """Shear force at root V(0) [N]."""
        return float(self.V[0])


class WingBeam:
    """Euler-Bernoulli beam model for a wing spar.

    Interpolates chord, EI(y), GJ(y), and mass-per-unit-span from the
    Wing's cross-section definitions.  All cross-sections must have a
    Spar defined; cross-sections without a Spar are skipped when
    computing EI (treated as zero stiffness at that station).

    Parameters
    ----------
    wing : Wing
    n_stations : int
        Number of uniformly spaced spanwise integration stations (default 50).
    """

    def __init__(self, wing: Wing, n_stations: int = 50) -> None:
        self.wing = wing
        self.n_stations = n_stations
        self._build_stations()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _EI_at_xsec(self, xsec: WingXSec) -> float:
        if xsec.spar is None:
            return 0.0
        if xsec.airfoil is not None:
            return effective_EI(xsec.airfoil, xsec.chord, xsec.spar, xsec.skin)
        return xsec.spar.material.E * xsec.spar.section.second_moment_of_area()

    def _GJ_at_xsec(self, xsec: WingXSec) -> float:
        if xsec.spar is None:
            return 0.0
        spar = xsec.spar
        G = spar.material.shear_modulus
        # Polar moment for circular tube: J = 2·I
        J = 2.0 * spar.section.second_moment_of_area()
        return G * J

    def _mass_per_length_at_xsec(self, xsec: WingXSec) -> float:
        """Structural mass per unit span [kg/m] from spar + skin."""
        m = 0.0
        if xsec.spar is not None:
            m += xsec.spar.mass_per_length()
        if xsec.skin is not None:
            # Approximate skin perimeter as 2 × chord (flat-wrap estimate)
            perimeter = 2.0 * xsec.chord
            m += xsec.skin.thickness * xsec.skin.material.density * perimeter
        return m

    def _build_stations(self) -> None:
        """Interpolate all spanwise properties to uniform y stations."""
        xsecs = self.wing.xsecs
        y_xsecs = np.array([xs.xyz_le[1] for xs in xsecs])
        y_root = float(y_xsecs[0])
        y_tip = float(y_xsecs[-1])

        self.y = np.linspace(y_root, y_tip, self.n_stations)

        chords = np.array([xs.chord for xs in xsecs])
        self.chord = np.interp(self.y, y_xsecs, chords)

        EI_xsecs = np.array([self._EI_at_xsec(xs) for xs in xsecs])
        self.EI = np.maximum(np.interp(self.y, y_xsecs, EI_xsecs), 1e-30)

        GJ_xsecs = np.array([self._GJ_at_xsec(xs) for xs in xsecs])
        self.GJ = np.maximum(np.interp(self.y, y_xsecs, GJ_xsecs), 1e-30)

        m_xsecs = np.array([self._mass_per_length_at_xsec(xs) for xs in xsecs])
        self.m_prime = np.interp(self.y, y_xsecs, m_xsecs)

    # ------------------------------------------------------------------
    # Public solver
    # ------------------------------------------------------------------

    def solve(
        self,
        total_lift: float,
        load_factor: float = 1.0,
        inertia_relief: bool = True,
    ) -> BeamResult:
        """Integrate beam equations for a given total lift and load factor.

        Uses an elliptic spanwise lift distribution for the aerodynamic
        load and optionally subtracts the wing structural weight (inertia
        relief).

        Parameters
        ----------
        total_lift : float
            Total aerodynamic lift on the complete aircraft [N].
            The semi-span load is total_lift / 2 for symmetric wings.
        load_factor : float
            Multiplier applied to both the aerodynamic load and the
            inertia relief term (default 1.0 = 1g).
        inertia_relief : bool
            If True, subtract n·g·m'(y) from the applied load
            (default True).

        Returns
        -------
        BeamResult
        """
        y = self.y
        n = load_factor
        b = float(y[-1] - y[0])  # semispan [m]

        # Elliptic distribution: q_aero(y) = q_0 · sqrt(1 − η²)
        # where η = (y − y_root) / b
        if b > 0.0:
            L_semi = total_lift / 2.0  # one semi-span
            q_0 = 4.0 * L_semi / (np.pi * b)
            eta = (y - y[0]) / b
            q_aero = q_0 * np.sqrt(np.maximum(1.0 - eta ** 2, 0.0))
        else:
            q_aero = np.zeros_like(y)

        # Net distributed load [N/m]
        q = n * q_aero
        if inertia_relief:
            q = q - n * _G * self.m_prime

        # Shear V(y) = integral_y^tip q(eta) d_eta  (tip to root)
        V = np.zeros_like(y)
        for i in range(len(y) - 2, -1, -1):
            dy = y[i + 1] - y[i]
            V[i] = V[i + 1] + 0.5 * (q[i] + q[i + 1]) * dy

        # Bending moment M(y) = integral_y^tip V(eta) d_eta  (tip to root)
        M = np.zeros_like(y)
        for i in range(len(y) - 2, -1, -1):
            dy = y[i + 1] - y[i]
            M[i] = M[i + 1] + 0.5 * (V[i] + V[i + 1]) * dy

        # Slope theta(y) = integral_0^y M/EI dy  (root to tip)
        M_over_EI = M / self.EI
        theta = np.zeros_like(y)
        for i in range(1, len(y)):
            dy = y[i] - y[i - 1]
            theta[i] = theta[i - 1] + 0.5 * (M_over_EI[i - 1] + M_over_EI[i]) * dy

        # Deflection delta(y) = integral_0^y theta dy  (root to tip)
        delta = np.zeros_like(y)
        for i in range(1, len(y)):
            dy = y[i] - y[i - 1]
            delta[i] = delta[i - 1] + 0.5 * (theta[i - 1] + theta[i]) * dy

        return BeamResult(y=y, V=V, M=M, theta=theta, delta=delta,
                          EI=self.EI.copy(), GJ=self.GJ.copy())
