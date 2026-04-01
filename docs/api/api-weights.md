# Weights API

Module: `aerisplane.weights`

---

## Quick start

```python
import aerisplane as ap

# Build aircraft
aircraft = ap.Aircraft(name="my_plane", wings=[main_wing], fuselages=[fuse],
                       propulsion=propulsion)

# Run weight buildup — all masses estimated from geometry + materials
result = ap.weights.analyze(aircraft)
result.report()
```

---

## `analyze()`

```python
aerisplane.weights.analyze(
    aircraft,
    overrides=None,
) -> WeightResult
```

Runs a component-by-component mass buildup from the aircraft geometry and hardware
catalog entries (motors, batteries, servos, materials). Returns a `WeightResult`
with total mass, CG, inertia tensor, and per-component breakdown.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `aircraft` | `Aircraft` | — | Aircraft to analyze |
| `overrides` | `dict[str, ComponentOverride]` | `None` | Replace computed estimates with measured values |

### Overrides

Use `ComponentOverride` to substitute a computed estimate with a measured mass.
The key must match the component name as it appears in `WeightResult.components`.

```python
from aerisplane.weights import analyze
from aerisplane.weights.result import ComponentOverride
import numpy as np

result = analyze(
    aircraft,
    overrides={
        "main_wing_skin": ComponentOverride(mass=0.185),
        "fuselage_skin":  ComponentOverride(
            mass=0.320,
            cg=np.array([0.42, 0.0, 0.0]),   # CG position [x, y, z] [m]
        ),
    },
)
```

If the override key matches an existing component, the computed mass (and optionally
CG) is replaced. If the key is new, a fresh component is added to the breakdown.
This allows progressive replacement of estimates with real measurements as a build
proceeds.

---

## `WeightResult`

### Fields

| Field | Type | Unit | Description |
|---|---|---|---|
| `total_mass` | `float` | kg | Total aircraft mass |
| `cg` | `ndarray (3,)` | m | Centre of gravity [x, y, z] in aircraft frame |
| `inertia_tensor` | `ndarray (3,3)` | kg·m² | Inertia tensor about CG |
| `components` | `dict[str, ComponentMass]` | — | Per-component breakdown |
| `wing_loading` | `float` | g/dm² | Total mass / reference area |

### `ComponentMass` fields

Each entry in `result.components` is a `ComponentMass`:

| Field | Type | Unit | Description |
|---|---|---|---|
| `name` | `str` | — | Component identifier |
| `mass` | `float` | kg | Component mass |
| `cg` | `ndarray (3,)` | m | Component CG position |
| `source` | `str` | — | `"computed"` or `"override"` |

```python
for name, comp in result.components.items():
    print(f"{name:30s}  {comp.mass*1000:7.1f} g  source={comp.source}")
```

### `report()`

```python
result.report()
```

Prints a formatted table listing every component with its mass and CG, followed
by totals for mass, CG location, and wing loading.

---

## Plotting

All plot methods return a `matplotlib.figure.Figure` and call `plt.show()`
internally.

### `plot()`

```python
result.plot()
```

Two-panel figure: mass bar chart (horizontal, sorted by mass) on the left,
CG side-view (waterfall bar chart along the x-axis) on the right.

### `plot_cg()`

```python
result.plot_cg()
```

CG bubble diagram: scatter plot of component CG positions (y vs x), with bubble
size proportional to mass. Labels are drawn with alternating above/below offsets
to reduce overlap.

### `plot_cg_bars()`

```python
result.plot_cg_bars(bin_width_mm=100)
```

Mass histogram along the aircraft x-axis. Components are summed into bins of
`bin_width_mm` mm width. Useful for identifying fore/aft balance trends.

| Parameter | Default | Description |
|---|---|---|
| `bin_width_mm` | `100` | Histogram bin width [mm] |

### `plot_distribution()`

```python
result.plot_distribution()
```

Donut/ring chart showing the fractional mass contribution of each component
group, with callout labels for all slices.

---

## `ComponentOverride`

```python
from aerisplane.weights.result import ComponentOverride

ComponentOverride(
    mass,       # Measured mass [kg]
    cg=None,    # Measured CG position ndarray([x, y, z]) [m], or None to keep computed
)
```

---

## Examples

### Basic buildup

```python
import aerisplane as ap

result = ap.weights.analyze(aircraft)
print(f"Total mass : {result.total_mass*1000:.0f} g")
print(f"CG         : x={result.cg[0]:.3f} m")
print(f"Wing load  : {result.wing_loading:.1f} g/dm²")
result.report()
```

### Progressive override workflow

Start with all-estimated masses, then replace with measured values as components
are built and weighed:

```python
from aerisplane.weights.result import ComponentOverride

# Week 1: estimate everything
r0 = ap.weights.analyze(aircraft)

# Week 2: weighed the wing structure
r1 = ap.weights.analyze(aircraft, overrides={
    "main_wing_spar": ComponentOverride(mass=0.062),
    "main_wing_skin": ComponentOverride(mass=0.174),
})

# Week 3: full airframe weighed, only electronics estimated
r2 = ap.weights.analyze(aircraft, overrides={
    "main_wing_spar":  ComponentOverride(mass=0.062),
    "main_wing_skin":  ComponentOverride(mass=0.174),
    "fuselage_skin":   ComponentOverride(mass=0.298, cg=np.array([0.38, 0.0, 0.0])),
})
```

### Reading the inertia tensor

```python
Ixx, Iyy, Izz = result.inertia_tensor[0,0], result.inertia_tensor[1,1], result.inertia_tensor[2,2]
print(f"Ixx={Ixx:.4f}  Iyy={Iyy:.4f}  Izz={Izz:.4f}  kg·m²")
```

### Visualising the weight breakdown

```python
result.plot()               # bar chart + CG side view
result.plot_cg()            # CG bubble diagram
result.plot_cg_bars()       # mass histogram along x-axis
result.plot_distribution()  # donut chart
```
