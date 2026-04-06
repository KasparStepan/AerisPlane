# AerisPlane

Conceptual MDO toolkit for fixed-wing RC/UAV aircraft design (1–20 kg class).

AerisPlane is a Python library for analysing and optimising small unmanned aircraft
at the conceptual design stage. It covers aerodynamics, weights, structures, stability,
control authority, and mission performance — all wired together through a lightweight
MDO layer.

## Module overview

| Module | Description |
|---|---|
| `aerisplane.core` | Geometry dataclasses: `Aircraft`, `Wing`, `Fuselage`, `FlightCondition`, ... |
| `aerisplane.aero` | Aerodynamic analysis (VLM, LiftingLine, AeroBuildup, flow viz) |
| `aerisplane.weights` | Component mass buildup, CG analysis |
| `aerisplane.structures` | Euler–Bernoulli wing beam solver |
| `aerisplane.stability` | Numerical stability derivatives |
| `aerisplane.control` | Control authority and servo loads |
| `aerisplane.mission` | Point-performance energy budget |
| `aerisplane.propulsion` | Motor/battery/propeller operating point |
| `aerisplane.mdo` | Optimisation: design variables, constraints, SciPy/pygmo drivers |
| `aerisplane.catalog` | Hardware database — motors, batteries, propellers, servos |

## Quick start

```python
import aerisplane as ap
from aerisplane.aero import analyze

wing = ap.Wing(
    name="main_wing",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28, airfoil=ap.Airfoil(name="ag35")),
        ap.WingXSec(xyz_le=[0.07, 1.20, 0.05], chord=0.14, twist=-2.0, airfoil=ap.Airfoil(name="ag35")),
    ],
)
aircraft = ap.Aircraft(name="MyPlane", wings=[wing])
cond = ap.FlightCondition(velocity=16.0, altitude=0.0, alpha=4.0)
result = analyze(aircraft, cond, method="vlm")
result.report()
```
