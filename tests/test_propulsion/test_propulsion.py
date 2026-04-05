# tests/test_propulsion/test_propulsion.py
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerisplane.core.propulsion import Motor, Propeller, Battery, ESC, PropulsionSystem
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
import aerisplane.propulsion as prop_module
from aerisplane.propulsion.result import PropulsionResult
from aerisplane.propulsion.solver import solve_operating_point


@pytest.fixture
def propulsion_system():
    motor = Motor(name="test", kv=900, resistance=0.1, no_load_current=0.5,
                  max_current=40.0, mass=0.120)
    propeller = Propeller(diameter=0.254, pitch=0.127, mass=0.025)
    battery = Battery(name="test_bat", capacity_ah=3.0, nominal_voltage=14.8,
                      cell_count=4, c_rating=25.0, mass=0.280)
    esc = ESC(name="test_esc", max_current=40.0, mass=0.030)
    return PropulsionSystem(motor=motor, propeller=propeller, battery=battery,
                            esc=esc, position=np.array([0.0, 0.0, 0.0]))


@pytest.fixture
def aircraft_with_propulsion(propulsion_system):
    return Aircraft(name="test", wings=[], propulsion=propulsion_system)


@pytest.fixture
def condition():
    return FlightCondition(velocity=14.0, altitude=100.0, alpha=3.0)


# ---- solver tests ----

def test_solver_returns_positive_rpm(propulsion_system):
    rpm, current = solve_operating_point(propulsion_system, throttle=1.0,
                                          velocity=0.0, rho=1.225)
    assert rpm > 0
    assert current > 0

def test_solver_throttle_zero(propulsion_system):
    rpm, current = solve_operating_point(propulsion_system, throttle=0.0,
                                          velocity=0.0, rho=1.225)
    assert rpm == pytest.approx(0.0, abs=1.0)

def test_solver_thrust_increases_with_throttle(propulsion_system):
    rho = 1.225
    rpm_low, _ = solve_operating_point(propulsion_system, 0.5, 10.0, rho)
    rpm_high, _ = solve_operating_point(propulsion_system, 1.0, 10.0, rho)
    thrust_low = propulsion_system.propeller.thrust(rpm_low, 10.0, rho)
    thrust_high = propulsion_system.propeller.thrust(rpm_high, 10.0, rho)
    assert thrust_high > thrust_low

# ---- result tests ----

def test_result_fields():
    r = PropulsionResult(
        thrust_n=12.5, current_a=18.3, rpm=7500.0, motor_efficiency=0.82,
        propulsive_efficiency=0.65, electrical_power_w=270.0, shaft_power_w=221.4,
        battery_endurance_s=1200.0, c_rate=5.0, over_current=False,
        throttle=0.75, velocity_ms=14.0,
    )
    assert r.thrust_n == pytest.approx(12.5)
    assert r.over_current is False

def test_result_report():
    r = PropulsionResult(
        thrust_n=10.0, current_a=15.0, rpm=6000.0, motor_efficiency=0.80,
        propulsive_efficiency=0.60, electrical_power_w=200.0, shaft_power_w=160.0,
        battery_endurance_s=900.0, c_rate=4.0, over_current=False,
        throttle=0.70, velocity_ms=12.0,
    )
    report = r.report()
    assert isinstance(report, str)
    assert "Thrust" in report
    assert "RPM" in report

def test_result_plot():
    r = PropulsionResult(
        thrust_n=10.0, current_a=15.0, rpm=6000.0, motor_efficiency=0.80,
        propulsive_efficiency=0.60, electrical_power_w=200.0, shaft_power_w=160.0,
        battery_endurance_s=900.0, c_rate=4.0, over_current=False,
        throttle=0.70, velocity_ms=12.0,
    )
    fig = r.plot()
    assert fig is not None
    plt.close("all")

# ---- analyze() tests ----

def test_analyze_returns_result(aircraft_with_propulsion, condition):
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.8)
    assert isinstance(r, PropulsionResult)

def test_analyze_thrust_positive(aircraft_with_propulsion, condition):
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.8)
    assert r.thrust_n > 0

def test_analyze_rpm_positive(aircraft_with_propulsion, condition):
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.8)
    assert r.rpm > 0

def test_analyze_endurance_positive(aircraft_with_propulsion, condition):
    r = prop_module.analyze(aircraft_with_propulsion, condition, throttle=0.5)
    assert r.battery_endurance_s > 0

def test_analyze_over_current_is_bool(aircraft_with_propulsion):
    cond = FlightCondition(velocity=0.0, altitude=0.0, alpha=0.0)
    r = prop_module.analyze(aircraft_with_propulsion, cond, throttle=1.0)
    assert isinstance(r.over_current, bool)

def test_analyze_no_propulsion_raises(condition):
    ac = Aircraft(name="glider", wings=[])
    with pytest.raises(ValueError, match="no PropulsionSystem"):
        prop_module.analyze(ac, condition, throttle=0.8)

# ---- namespace test ----

def test_aerisplane_namespace(aircraft_with_propulsion, condition):
    import aerisplane as ap
    r = ap.propulsion.analyze(aircraft_with_propulsion, condition, throttle=0.7)
    assert isinstance(r, ap.PropulsionResult)
