# Aerodynamics API

Module: `aerisplane.aero`

---

## Quick start

```python
import aerisplane as ap

# Build aircraft (see core API for full geometry definition)
aircraft = ap.Aircraft(name="my_plane", wings=[main_wing], fuselages=[fuse])

# Define flight condition
condition = ap.FlightCondition(velocity=20.0, altitude=500.0, alpha=4.0)

# Run analysis
result = ap.aero.analyze(aircraft, condition, method="vlm")
result.report()
```

---

## `analyze()`

```python
aerisplane.aero.analyze(
    aircraft,
    condition,
    method="vlm",
    backend="native",
    spanwise_resolution=8,
    chordwise_resolution=4,
    model_size="medium",
    verbose=False,
) -> AeroResult
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `aircraft` | `Aircraft` | — | Aircraft geometry |
| `condition` | `FlightCondition` | — | Operating point |
| `method` | `str` | `"vlm"` | Solver — see table below |
| `backend` | `str` | `"native"` | `"native"` or `"aerosandbox"` |
| `spanwise_resolution` | `int` | `8` | Spanwise panels per wing section (VLM, LL) |
| `chordwise_resolution` | `int` | `4` | Chordwise panels per section (VLM only) |
| `model_size` | `str` | `"medium"` | NeuralFoil model size (LL, NLL, AeroBuildup) |
| `verbose` | `bool` | `False` | Print solver progress |

### Solver methods

| Method string | Physics | Stall | Viscous | Notes |
|---|---|---|---|---|
| `"vlm"` | Inviscid, linear | ✗ | ✗ | Fast. Correct CL, Cm, Cl, Cn. All drag is CDi. |
| `"lifting_line"` | Viscous, linearised CL | partial | ✅ | Requires `neuralfoil` |
| `"nonlinear_lifting_line"` | Viscous, nonlinear CL | ✅ | ✅ | Fixed-point iteration. Requires `neuralfoil` |
| `"aero_buildup"` | Semi-empirical | ✅ | ✅ | Jorgensen fuselage + NeuralFoil wings. Requires `neuralfoil` |

**`model_size` options** (NeuralFoil): `"xxsmall"`, `"xsmall"`, `"small"`, `"medium"`, `"large"`, `"xlarge"`, `"xxlarge"`. Larger models are more accurate but slower.

---

## `FlightCondition`

```python
aerisplane.FlightCondition(
    velocity,           # True airspeed [m/s]
    altitude=0.0,       # Geometric altitude above MSL [m]
    alpha=0.0,          # Angle of attack [deg]
    beta=0.0,           # Sideslip angle [deg]
    p=0.0,              # Roll rate [rad/s]
    q=0.0,              # Pitch rate [rad/s]
    r=0.0,              # Yaw rate [rad/s]
    deflections={},     # Control surface deflections [deg], keyed by surface name
)
```

### Control surface deflections

Deflections are passed through `FlightCondition.deflections` — a plain dict mapping
surface name to angle in degrees. The surface name must match the `ControlSurface.name`
field on the wing.

```python
# Elevator down 5°, aileron right side down 10°
condition = ap.FlightCondition(
    velocity=20.0,
    altitude=500.0,
    alpha=3.0,
    deflections={"elevator": -5.0, "aileron": 10.0},
)
```

Sign conventions:
- Positive = trailing edge down (elevator nose-up, flap lift-up)
- For `ControlSurface(symmetric=False)` (aileron): positive deflects the right panel
  trailing-edge down and the left panel trailing-edge up

### Useful methods

```python
cond.dynamic_pressure()           # q = 0.5 * rho * V^2  [Pa]
cond.reynolds_number(chord)       # Re = rho * V * c / mu
cond.mach()                       # Mach number at altitude
cond.density()                    # Air density [kg/m³]
```

---

## `AeroResult`

Returned by `analyze()`. All quantities are in SI units.

### Forces and coefficients

| Field | Unit | Description |
|---|---|---|
| `CL` | — | Lift coefficient |
| `CD` | — | Total drag coefficient |
| `CY` | — | Side force coefficient |
| `CDi` | — | Induced drag coefficient (`None` for LL/NLL/AeroBuildup) |
| `CDp` | — | Profile drag coefficient (`None` for VLM/LL/NLL) |
| `L` | N | Lift force (wind axes) |
| `D` | N | Drag force (wind axes) |
| `Y` | N | Side force (wind axes) |

### Moments and coefficients

| Field | Unit | Description |
|---|---|---|
| `Cl` | — | Rolling moment coefficient (body axes) |
| `Cm` | — | Pitching moment coefficient (body axes) |
| `Cn` | — | Yawing moment coefficient (body axes) |
| `l_b` | N·m | Rolling moment (body axes) |
| `m_b` | N·m | Pitching moment (body axes) |
| `n_b` | N·m | Yawing moment (body axes) |

Moments are taken about `aircraft.xyz_ref` (default: origin). Set `xyz_ref` to the
CG location for stability analysis.

### Force/moment vectors

| Field | Description |
|---|---|
| `F_g` | Force vector in geometry axes [Fx, Fy, Fz] [N] |
| `F_b` | Force vector in body axes [N] |
| `F_w` | Force vector in wind axes [N] |
| `M_g` | Moment vector in geometry axes [Mx, My, Mz] [N·m] |
| `M_b` | Moment vector in body axes [N·m] |
| `M_w` | Moment vector in wind axes [N·m] |

### Operating condition (echoed)

| Field | Unit | Description |
|---|---|---|
| `alpha` | deg | Angle of attack |
| `beta` | deg | Sideslip angle |
| `velocity` | m/s | True airspeed |
| `altitude` | m | Geometric altitude |
| `dynamic_pressure` | Pa | q = 0.5 ρ V² |
| `reynolds_number` | — | Re based on reference chord |

### Reference geometry (echoed)

| Field | Unit | Description |
|---|---|---|
| `s_ref` | m² | Reference area (main wing area) |
| `c_ref` | m | Reference chord (MAC of main wing) |
| `b_ref` | m | Reference span (main wing span) |

### Methods

```python
result.report()
```
Prints a formatted summary table to stdout.

```python
AeroResult.plot_polar(results, show=True, save_path=None)
```
Static method. Plot CL, CD, and Cm polars from a list of `AeroResult` objects
(typically an alpha sweep).

```python
result.plot_spanwise_loading(show=True, save_path=None)
```
Plot spanwise section-CL distribution. VLM results only (requires `_solver`
to be a `VortexLatticeMethod` instance).

---

## `plot_geometry()`

```python
aerisplane.aero.plot_geometry(
    aircraft,
    style="three_view",   # "three_view" | "wireframe"
    show=True,
    save_path=None,
)
```

Visualise aircraft geometry. `"three_view"` produces a 4-panel engineering drawing
(top, front, side, isometric). `"wireframe"` produces a single 3-D matplotlib view.

---

## Examples

### Alpha sweep with VLM

```python
import numpy as np
import aerisplane as ap
from aerisplane.aero import analyze

condition_base = ap.FlightCondition(velocity=20.0, altitude=500.0)
alphas = np.linspace(-4, 14, 10)
results = [
    analyze(aircraft, ap.FlightCondition(velocity=20.0, altitude=500.0, alpha=a))
    for a in alphas
]

ap.aero.AeroResult.plot_polar(results)
```

### Control surface deflection sweep

```python
deflections = np.linspace(-20, 20, 9)
results = [
    analyze(
        aircraft,
        ap.FlightCondition(velocity=20.0, altitude=500.0, alpha=4.0,
                           deflections={"elevator": d}),
    )
    for d in deflections
]
cms = [r.Cm for r in results]
```

### Accessing raw solver data (VLM)

```python
result = analyze(aircraft, condition, method="vlm")
vlm = result._solver          # VortexLatticeMethod instance
gammas = vlm.vortex_strengths # (N,) vortex strength array
panels = vlm.collocation_points  # (N, 3) panel collocation points
```
