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
        Model name.
    kv : float
        Motor velocity constant [RPM/V].
    resistance : float
        Winding resistance [ohm].
    no_load_current : float
        No-load current [A].
    max_current : float
        Maximum continuous current [A].
    mass : float
        Motor mass [kg].
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
    """Propeller model with parametric or lookup-table performance.

    Parameters
    ----------
    diameter : float
        Propeller diameter [m].
    pitch : float
        Geometric pitch [m].
    mass : float
        Propeller mass [kg].
    num_blades : int
        Number of blades.
    performance_data : PropellerPerfData or None
        Optional lookup table. If provided, thrust/power use interpolation.
        Otherwise, a simplified parametric model is used.
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
        """Simplified thrust coefficient model based on pitch/diameter ratio."""
        pitch_ratio = self.pitch / self.diameter
        # Linear CT model: CT = CT0 * (1 - J / J0)
        ct0 = 0.075 * pitch_ratio  # static thrust coefficient estimate
        j0 = 0.8 * pitch_ratio      # zero-thrust advance ratio
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
    """LiPo battery pack.

    Parameters
    ----------
    name : str
        Model name.
    capacity_ah : float
        Capacity [Ah].
    nominal_voltage : float
        Nominal voltage [V] (e.g., 14.8 for 4S).
    cell_count : int
        Number of series cells.
    c_rating : float
        Maximum continuous discharge rate [C].
    mass : float
        Pack mass [kg].
    internal_resistance : float
        Total pack internal resistance [ohm].
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
        Model name.
    max_current : float
        Maximum continuous current [A].
    mass : float
        ESC mass [kg].
    has_telemetry : bool
        Whether ESC supports telemetry feedback.
    """

    name: str
    max_current: float
    mass: float
    has_telemetry: bool = False


@dataclass
class PropulsionSystem:
    """Complete propulsion system: motor + prop + battery + ESC.

    Parameters
    ----------
    motor : Motor
    propeller : Propeller
    battery : Battery
    esc : ESC
    position : array-like
        Installation position [x, y, z] in aircraft frame [m].
    direction : array-like
        Thrust direction unit vector. Default [-1, 0, 0] = forward thrust.
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
