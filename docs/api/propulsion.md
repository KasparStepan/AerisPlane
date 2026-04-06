# Propulsion — `aerisplane.propulsion`

Computes the motor/propeller/battery operating point at a given throttle setting
and flight velocity. Returns thrust, current draw, RPM, efficiency, C-rate, and
estimated battery endurance.

---

## Quick start

```python
import aerisplane as ap
from aerisplane.catalog.motors import sunnysky_x2216_1250
from aerisplane.catalog.batteries import tattu_4s_5200
from aerisplane.catalog.propellers import apc_10x4_7sf
from aerisplane.core.propulsion import ESC, PropulsionSystem
from aerisplane.propulsion import analyze

propulsion = PropulsionSystem(
    motor=sunnysky_x2216_1250,
    propeller=apc_10x4_7sf,
    battery=tattu_4s_5200,
    esc=ESC(name="generic_60A", max_current=60.0, mass=0.025),
)
aircraft = ap.Aircraft(name="MyPlane", wings=[wing], propulsion=propulsion)
cond = ap.FlightCondition(velocity=16.0, altitude=0.0)

result = analyze(aircraft, cond, throttle=0.75)
print(result.report())
```

---

## `analyze()`

::: aerisplane.propulsion.analyze

---

## `PropulsionResult`

::: aerisplane.propulsion.result.PropulsionResult

---

## Hardware catalog

See the [Catalog](catalog.md) page for available motors, batteries, and propellers.
To use catalog hardware:

```python
from aerisplane.catalog import list_motors

motors = list_motors()
motor = next(m for m in motors if "SunnySky" in m.name and m.kv == 1250)
```
