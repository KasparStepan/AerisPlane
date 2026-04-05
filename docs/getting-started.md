# Getting Started

A five-minute introduction to the AerisPlane workflow.

## 1. Define geometry

An aircraft is built from `Wing` and `Fuselage` objects composed of cross-sections.
The main wing, horizontal tail, and vertical tail are all `Wing` objects — there is
no separate tail class.

```python
import aerisplane as ap

# Main wing with two cross-sections (root and tip)
wing = ap.Wing(
    name="main_wing",
    symmetric=True,   # mirror across y=0 plane
    xsecs=[
        ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28, airfoil=ap.Airfoil(name="ag35")),
        ap.WingXSec(xyz_le=[0.07, 1.20, 0.05], chord=0.14, twist=-2.0, airfoil=ap.Airfoil(name="ag35")),
    ],
)

# Horizontal tail with elevator
htail = ap.Wing(
    name="htail",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.95, 0.00, 0.00], chord=0.15, airfoil=ap.Airfoil(name="naca0012")),
        ap.WingXSec(xyz_le=[0.98, 0.40, 0.00], chord=0.10, airfoil=ap.Airfoil(name="naca0012")),
    ],
    control_surfaces=[
        ap.ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
                          chord_fraction=0.35, symmetric=True),
    ],
)

aircraft = ap.Aircraft(name="MyPlane", wings=[wing, htail])
```

Coordinate convention: `xyz_le = [x, y, z]` is the leading-edge position in aircraft
frame. `x` points aft, `y` points right, `z` points up. For symmetric wings, define
the right side (y ≥ 0) only.

## 2. Run an aerodynamic analysis

```python
from aerisplane.aero import analyze, plot_geometry

# Plot the geometry to verify it looks right
plot_geometry(aircraft, style="three_view")

# Run VLM at a single operating point
cond = ap.FlightCondition(velocity=16.0, altitude=0.0, alpha=4.0)
result = analyze(aircraft, cond, method="vlm")
result.report()
```

Available methods: `"vlm"` (fast, inviscid), `"lifting_line"` (viscous with NeuralFoil),
`"nonlinear_lifting_line"` (captures stall), `"aero_buildup"` (semi-empirical).

## 3. Check weights

```python
from aerisplane.weights import analyze as weight_analysis

wr = weight_analysis(aircraft)
wr.report()
wr.plot_mass_breakdown()
```

## 4. Run the full discipline chain

```python
from aerisplane import weights, stability

wr = weights.analyze(aircraft)
aircraft.xyz_ref = wr.cg.tolist()   # set moment reference to CG

sr = stability.analyze(aircraft, cond, wr)
print(f"Static margin: {sr.static_margin:.1%} MAC")
print(f"Cm_alpha:      {sr.Cm_alpha:.4f} 1/deg  ({'stable' if sr.Cm_alpha < 0 else 'UNSTABLE'})")
```

See the [Tutorials](tutorials/01_getting_started.ipynb) for deeper worked examples
covering every discipline module.
