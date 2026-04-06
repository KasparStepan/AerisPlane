# Structures — `aerisplane.structures`

Wing spar structural analysis using an Euler–Bernoulli beam model.
Checks bending yield, shear yield, buckling, and tip deflection
against ultimate design loads. Requires `Spar` definitions on wing cross-sections.

---

## Quick start

```python
import aerisplane as ap
from aerisplane.aero import analyze as aero_analyze
from aerisplane.weights import analyze as weight_analyze
from aerisplane.structures import analyze
from aerisplane.core.structures import Spar, TubeSection
from aerisplane.catalog.materials import CFRP_UD

# Add a spar to the wing root and tip cross-sections
spar = Spar(
    position=0.25,
    tube=TubeSection(outer_diameter=0.018, wall_thickness=0.0015),
    material=CFRP_UD,
)
wing.xsecs[0].spar = spar
wing.xsecs[1].spar = spar

cond = ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=6.0)
aero_r = aero_analyze(aircraft, cond, method="vlm")
weight_r = weight_analyze(aircraft)
result = analyze(aircraft, aero_r, weight_r, n_limit=4.0)
result.report()
```

---

## `analyze()`

::: aerisplane.structures.analyze

---

## `StructureResult` and `WingStructureResult`

::: aerisplane.structures.result.StructureResult

::: aerisplane.structures.result.WingStructureResult

### Margin of safety interpretation

| Field | Safe if … |
|---|---|
| `bending_margin` | > 0 |
| `shear_margin` | > 0 |
| `buckling_margin` | > 0 |
| `tip_deflection_ratio` | < 0.10 (10% of semi-span) |
| `spar_fits` | True |

---

## Spar definition

See [Core — Structural components](core.md#structural-components) for the
`Spar`, `TubeSection`, and `Material` dataclass references.
