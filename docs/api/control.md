# Control — `aerisplane.control`

Control authority analysis: roll rate, pitch authority, rudder authority, and servo
loads at a given flight condition. Computes control derivatives by forward
finite-differencing the aerodynamic solver.

Requires `ControlSurface` objects on the wings and a completed `stability.analyze()`.

---

## Quick start

```python
import aerisplane as ap
from aerisplane.weights import analyze as weight_analyze
from aerisplane.stability import analyze as stab_analyze
from aerisplane.control import analyze

wr = weight_analyze(aircraft)
aircraft.xyz_ref = wr.cg.tolist()

cond = ap.FlightCondition(velocity=18.0, altitude=0.0, alpha=4.0)
sr = stab_analyze(aircraft, cond, wr)
result = analyze(aircraft, cond, wr, sr, aero_method="vlm")
print(result.report())
```

---

## `analyze()`

::: aerisplane.control.analyze

---

## `ControlResult`

::: aerisplane.control.result.ControlResult

### Interpreting authority values

`aileron_authority`, `elevator_authority`, `rudder_authority` are normalised [0, 1]:

- **1.0** — surface meets or exceeds the reference requirement
- **< 1.0** — surface is undersized relative to the reference requirement

Reference values used internally:
- Roll: 60 deg/s minimum roll rate
- Crosswind: 5 m/s crosswind capability
- Elevator: ability to trim over a ±10° alpha range
