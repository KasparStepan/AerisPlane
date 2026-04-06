"""Propulsion system components: motor, propeller, battery, ESC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Motor:
    """Brushless DC motor model.

    Parameters
    ----------
    name : str
        Motor name for display and catalog lookup.
    kv : float
        Motor velocity constant [RPM/V].
    resistance : float
        Winding resistance [Ω].
    no_load_current : float
        No-load current [A].
    max_current : float
        Maximum continuous current [A].
    mass : float
        Motor mass [kg].

    Examples
    --------
    >>> from aerisplane.catalog.motors import sunnysky_x2216_1250
    >>> motor = sunnysky_x2216_1250   # 1250 KV, 28 A max
    """

    name: str
    kv: float
    resistance: float
    no_load_current: float
    max_current: float
    mass: float

    def rpm(self, voltage: float, current: float) -> float:
        """Motor RPM under load.

        RPM = kv * (voltage - current * resistance)
        """
        return self.kv * (voltage - current * self.resistance)

    def torque(self, current: float) -> float:
        """Motor torque [N*m].

        Q = (current - I0) / kv * (30/pi)
        """
        kt = 30.0 / (np.pi * self.kv)  # torque constant [N*m/A]
        return kt * (current - self.no_load_current)

    def efficiency(self, voltage: float, current: float) -> float:
        """Motor electrical-to-mechanical efficiency [-]."""
        if current <= self.no_load_current or voltage == 0.0:
            return 0.0
        power_in = voltage * current
        power_loss_resistive = current**2 * self.resistance
        power_loss_noload = voltage * self.no_load_current
        power_out = power_in - power_loss_resistive - power_loss_noload
        return max(0.0, power_out / power_in)


@dataclass
class PropellerPerfData:
    """Lookup table for propeller performance from test data.

    Parameters
    ----------
    J : ndarray
        Advance ratio array [-].
    CT : ndarray
        Thrust coefficient array [-].
    CP : ndarray
        Power coefficient array [-].
    source : str
        Data source description.
    """

    J: np.ndarray
    CT: np.ndarray
    CP: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        self.J = np.asarray(self.J, dtype=float)
        self.CT = np.asarray(self.CT, dtype=float)
        self.CP = np.asarray(self.CP, dtype=float)


@dataclass
class Propeller:
    """Fixed-pitch propeller model.

    Parameters
    ----------
    diameter : float
        Propeller diameter [m].
    pitch : float
        Geometric pitch [m].
    mass : float
        Propeller mass [kg].
    num_blades : int
        Number of blades. Default 2.

    Examples
    --------
    >>> from aerisplane.catalog.propellers import apc_10x4_7sf
    >>> prop = apc_10x4_7sf   # 10×4.7 in, 18 g
    """

    diameter: float
    pitch: float
    mass: float = 0.03
    num_blades: int = 2
    performance_data: Optional[PropellerPerfData] = None

    def advance_ratio(self, velocity: float, rpm: float) -> float:
        """Advance ratio J = V / (n * D) where n = rpm/60."""
        n = rpm / 60.0
        if n == 0.0:
            return 0.0
        return velocity / (n * self.diameter)

    def _parametric_ct(self, J: float) -> float:
        """Simplified thrust coefficient model based on pitch/diameter ratio.

        Linear CT model fitted to typical RC fixed-pitch propellers:
            CT = CT0 * (1 - J / J0)

        j0 = 1.5 * pitch_ratio based on typical RC propeller behaviour
        (zero-thrust J ≈ 1.4–1.6 × p/D). Parametric approximation; use
        PropellerPerfData for validated performance.
        """
        pitch_ratio = self.pitch / self.diameter
        ct0 = 0.075 * pitch_ratio   # static thrust coefficient estimate
        j0 = 1.5 * pitch_ratio      # zero-thrust advance ratio (~1.4–1.6 × p/D)
        if j0 == 0.0:
            return ct0
        ct = ct0 * (1.0 - J / j0)
        return max(0.0, ct)

    def _parametric_cp(self, J: float) -> float:
        """Simplified power coefficient model."""
        pitch_ratio = self.pitch / self.diameter
        # CP model fitted to typical RC propellers
        cp0 = 0.045 * pitch_ratio
        return max(1e-6, cp0 * (1.0 + 0.2 * J))

    def thrust(self, rpm: float, velocity: float, rho: float) -> float:
        """Thrust [N].

        T = CT * rho * n^2 * D^4
        """
        n = rpm / 60.0
        J = self.advance_ratio(velocity, rpm)

        if self.performance_data is not None:
            ct = float(np.interp(J, self.performance_data.J, self.performance_data.CT))
        else:
            ct = self._parametric_ct(J)

        return ct * rho * n**2 * self.diameter**4

    def power(self, rpm: float, velocity: float, rho: float) -> float:
        """Shaft power required [W].

        P = CP * rho * n^3 * D^5
        """
        n = rpm / 60.0
        J = self.advance_ratio(velocity, rpm)

        if self.performance_data is not None:
            cp = float(np.interp(J, self.performance_data.J, self.performance_data.CP))
        else:
            cp = self._parametric_cp(J)

        return cp * rho * n**3 * self.diameter**5

    def efficiency(self, rpm: float, velocity: float, rho: float) -> float:
        """Propulsive efficiency eta = T * V / P."""
        p = self.power(rpm, velocity, rho)
        if p <= 0.0 or velocity <= 0.0:
            return 0.0
        t = self.thrust(rpm, velocity, rho)
        return t * velocity / p


@dataclass
class Battery:
    """LiPo battery model.

    Parameters
    ----------
    name : str
        Battery name for display.
    capacity_ah : float
        Capacity [Ah].
    nominal_voltage : float
        Nominal voltage [V] (3.7 V × n_cells).
    cell_count : int
        Number of cells in series.
    c_rating : float
        Continuous discharge C-rating.
    mass : float
        Battery mass [kg].
    internal_resistance : float
        Internal resistance [Ω].

    Examples
    --------
    >>> from aerisplane.catalog.batteries import tattu_4s_5200
    >>> batt = tattu_4s_5200   # 5.2 Ah, 14.8 V, 45C, 470 g
    """

    name: str
    capacity_ah: float
    nominal_voltage: float
    cell_count: int
    c_rating: float
    mass: float
    internal_resistance: float = 0.0

    def energy(self) -> float:
        """Total energy [J]."""
        return self.capacity_ah * self.nominal_voltage * 3600.0

    def max_current(self) -> float:
        """Maximum continuous discharge current [A]."""
        return self.capacity_ah * self.c_rating

    def voltage_under_load(self, current: float) -> float:
        """Terminal voltage under load [V], accounting for IR drop."""
        return self.nominal_voltage - current * self.internal_resistance


@dataclass
class ESC:
    """Electronic speed controller.

    Parameters
    ----------
    name : str
        ESC name for display.
    max_current : float
        Maximum continuous current [A].
    mass : float
        ESC mass [kg]. Default 0.03.
    has_telemetry : bool
        Whether the ESC provides telemetry data. Default False.
    """

    name: str
    max_current: float
    mass: float
    has_telemetry: bool = False


@dataclass
class PropulsionSystem:
    """Complete electric propulsion system.

    Combines motor, propeller, battery, and ESC into a single component
    that can be analysed by ``aerisplane.propulsion.analyze()``.

    Parameters
    ----------
    motor : Motor
    propeller : Propeller
    battery : Battery
    esc : ESC

    Examples
    --------
    >>> from aerisplane.catalog.motors import sunnysky_x2216_1250
    >>> from aerisplane.catalog.batteries import tattu_4s_5200
    >>> from aerisplane.catalog.propellers import apc_10x4_7sf
    >>> from aerisplane.core.propulsion import ESC, PropulsionSystem
    >>> prop_sys = PropulsionSystem(
    ...     motor=sunnysky_x2216_1250,
    ...     propeller=apc_10x4_7sf,
    ...     battery=tattu_4s_5200,
    ...     esc=ESC(name="generic_60A", max_current=60.0, mass=0.025),
    ... )
    """

    motor: Motor
    propeller: Propeller
    battery: Battery
    esc: ESC
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    direction: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.direction = np.asarray(self.direction, dtype=float)

    def total_mass(self) -> float:
        """Total propulsion system mass [kg]."""
        return (
            self.motor.mass + self.propeller.mass + self.battery.mass + self.esc.mass
        )

    def thrust_available(self, velocity: float, rho: float) -> float:
        """Maximum thrust at given airspeed [N].

        Finds equilibrium RPM at max current and computes thrust.
        """
        voltage = self.battery.voltage_under_load(self.motor.max_current)
        rpm = self.motor.rpm(voltage, self.motor.max_current)
        if rpm <= 0:
            return 0.0
        return self.propeller.thrust(rpm, velocity, rho)

    def power_required(self, thrust: float, velocity: float, rho: float) -> float:
        """Electrical power for given thrust [W].

        Simplified: finds approximate RPM for requested thrust, then computes
        electrical power including motor losses.
        """
        # Iterative search for RPM that gives requested thrust
        # Simple bisection over RPM range
        voltage = self.battery.nominal_voltage
        rpm_max = self.motor.kv * voltage
        rpm_low, rpm_high = 0.0, rpm_max

        for _ in range(50):
            rpm_mid = (rpm_low + rpm_high) / 2.0
            t = self.propeller.thrust(rpm_mid, velocity, rho)
            if t < thrust:
                rpm_low = rpm_mid
            else:
                rpm_high = rpm_mid

        rpm = (rpm_low + rpm_high) / 2.0
        shaft_power = self.propeller.power(rpm, velocity, rho)

        # Electrical power = shaft power / motor efficiency
        # Estimate current from shaft power
        if voltage <= 0:
            return float("inf")
        current_est = shaft_power / voltage + self.motor.no_load_current
        eff = self.motor.efficiency(voltage, current_est)
        if eff <= 0:
            return float("inf")
        return shaft_power / eff

    def endurance_at_power(self, power_watts: float) -> float:
        """Time in seconds until battery depleted at constant power draw."""
        if power_watts <= 0:
            return float("inf")
        energy_joules = self.battery.energy()
        return energy_joules / power_watts
