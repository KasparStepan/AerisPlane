# Core — `aerisplane.core`

The `core` module contains the geometry and component dataclasses used by every
discipline module. It depends only on `numpy` — no solvers or backends.

---

## Coordinate system

All positions use the aircraft body frame:

- **x** — positive aft (toward the tail)
- **y** — positive right (starboard wing)
- **z** — positive up

`xyz_le` on `WingXSec` and `Wing` is the leading-edge position in this frame.
For symmetric wings, define the right side (y ≥ 0) only; the left side is
mirrored automatically.

---

## `Aircraft`

::: aerisplane.core.aircraft.Aircraft

---

## `Wing` and `WingXSec`

::: aerisplane.core.wing.Wing

::: aerisplane.core.wing.WingXSec

---

## `Fuselage` and `FuselageXSec`

::: aerisplane.core.fuselage.Fuselage

::: aerisplane.core.fuselage.FuselageXSec

---

## `Airfoil`

::: aerisplane.core.airfoil.Airfoil

---

## `ControlSurface` and `Servo`

::: aerisplane.core.control_surface.ControlSurface

::: aerisplane.core.control_surface.Servo

### Sign convention

| Surface | Positive deflection |
|---|---|
| elevator | trailing edge down (nose-up moment) |
| flap | trailing edge down (lift increase) |
| aileron (`symmetric=False`) | right side TE-down / left side TE-up |
| rudder | trailing edge left (nose-right moment) |

---

## `FlightCondition`

::: aerisplane.core.flight_condition.FlightCondition

### Control surface deflections

Deflections are passed through `FlightCondition.deflections` — a dict mapping surface
name to angle in degrees. The surface name must match `ControlSurface.name` on the wing.

```python
cond = ap.FlightCondition(
    velocity=18.0,
    altitude=100.0,
    alpha=3.5,
    deflections={"elevator": -5.0, "aileron": 10.0},
)
```

---

## Structural components

::: aerisplane.core.structures.Spar

::: aerisplane.core.structures.TubeSection

::: aerisplane.core.structures.Material

---

## Propulsion components

::: aerisplane.core.propulsion.Motor

::: aerisplane.core.propulsion.Propeller

::: aerisplane.core.propulsion.Battery

::: aerisplane.core.propulsion.ESC

::: aerisplane.core.propulsion.PropulsionSystem
