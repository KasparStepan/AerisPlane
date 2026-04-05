# AerisPlane

Conceptual MDO toolkit for fixed-wing RC/UAV aircraft design (1–20 kg class).

AerisPlane is a Python library for analysing and optimising small unmanned aircraft at the conceptual design stage. It covers aerodynamics, weights, structures, stability, control authority, and mission performance — all wired together through a lightweight MDO layer.

## Installation

```bash
pip install -e ".[dev]"          # editable install + pytest/ruff
pip install -e ".[interactive]"  # adds Plotly for interactive flow visualisation
pip install -e ".[oas]"          # adds OpenAeroStruct for detailed structural analysis
pip install -e ".[all]"          # everything
```

**Requirements:** Python >= 3.10, numpy, scipy, matplotlib, neuralfoil.

## Quick start

```python
import aerisplane as ap
from aerisplane.aero import analyze

# Define geometry
wing = ap.Wing(
    name="main_wing",
    symmetric=True,
    xsecs=[
        ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.28, airfoil=ap.Airfoil(name="ag35")),
        ap.WingXSec(xyz_le=[0.07, 1.20, 0.05], chord=0.14, twist=-2.0, airfoil=ap.Airfoil(name="ag35")),
    ],
)
aircraft = ap.Aircraft(name="MyPlane", wings=[wing])

# Run VLM analysis
cond = ap.FlightCondition(velocity=16.0, altitude=0.0, alpha=4.0)
result = analyze(aircraft, cond, method="vlm")
result.report()
```

## Aerodynamics

Four solver methods, all operating on native `aerisplane` geometry (no AeroSandbox dependency):

| Method | Description | Speed |
|---|---|---|
| `"vlm"` | Vortex Lattice Method — inviscid, arbitrary 3-D geometry | fast |
| `"lifting_line"` | Prandtl lifting line with NeuralFoil section polars | fast |
| `"nonlinear_lifting_line"` | Fixed-point iteration with NeuralFoil — captures stall | moderate |
| `"aero_buildup"` | Workbook-style: NeuralFoil wings + Jorgensen fuselage model | fast |

```python
result = analyze(aircraft, condition, method="vlm")

result.CL, result.CD, result.Cm   # force and moment coefficients
result.L, result.D                # forces [N]
result.report()                   # formatted summary table
```

### Flow visualisation (VLM)

```python
result.plot_surface_pressure()    # delta-Cp on panels — 3-D + top-down
result.plot_streamlines("xz")     # upwash/downwash in symmetry plane
result.plot_streamlines("yz")     # tip vortices behind trailing edge
result.plot_flow()                # all four in one dark-background figure
result.plot_interactive()         # interactive Plotly 3-D: Cp surface + streamlines
result.plot_interactive(save_path="flow.html")  # standalone HTML file
```

### Control surface deflections

Deflections are passed through `FlightCondition` — the aircraft object is never mutated:

```python
wing = ap.Wing(
    name="main_wing",
    ...,
    control_surfaces=[
        ap.ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
                          chord_fraction=0.35, symmetric=True),
        ap.ControlSurface(name="aileron",  span_start=0.5, span_end=0.9,
                          chord_fraction=0.25, symmetric=False),
    ],
)

for delta_e in [-20, -10, 0, 10, 20]:
    cond = ap.FlightCondition(velocity=16.0, alpha=4.0,
                              deflections={"elevator": delta_e})
    r = analyze(aircraft, cond, method="vlm")
    print(f"de={delta_e:+3d}  Cm={r.Cm:.4f}")
```

Sign convention: positive deflection = trailing edge down. For `symmetric=False` (ailerons), positive = right TE-down / left TE-up.

### Geometry plotting

```python
from aerisplane.aero import plot_geometry

plot_geometry(aircraft, style="three_view")   # top / front / side / isometric
plot_geometry(aircraft, style="wireframe")    # single 3-D view
```

## Weights

Component-based mass buildup from geometry and the hardware catalog:

```python
from aerisplane.weights import analyze as weight_analysis

result = weight_analysis(aircraft)
result.report()
result.plot_mass_breakdown()
result.plot_cg_envelope()
```

## Module overview

```
aerisplane/
├── core/          Pure geometry dataclasses: Wing, WingXSec, Fuselage, Airfoil,
│                  FlightCondition, ControlSurface, Motor, Battery, ...
├── aero/          Aerodynamic analysis (VLM, LiftingLine, AeroBuildup, flow viz)
├── weights/       Component mass buildup, CG analysis
├── structures/    Beam solver, OpenAeroStruct adapter
├── stability/     Numerical stability derivatives (finite-difference over aero calls)
├── control/       Control authority (roll rate, pitch, rudder, servo loads)
├── mission/       Point-performance energy budget per mission segment
├── mdo/           Optimisation: design variables, evaluation caching, SciPy/pygmo
├── catalog/       Hardware database — motors, batteries, servos, materials,
│                  2175 airfoil coordinate files
└── utils/         ISA atmosphere, unit conversions, cosine spacing, plot style
```

## Tutorials

| Notebook | Content |
|---|---|
| `docs/tutorials/01_getting_started.ipynb` | Aircraft definition, geometry plotting |
| `docs/tutorials/02_aerodynamics_executed.ipynb` | Alpha/velocity/altitude sweeps, polar plots, spanwise loading |
| `docs/tutorials/02_weight_buildup.ipynb` | Mass breakdown, CG analysis, hardware overrides |
| `docs/tutorials/03_control_surfaces.ipynb` | Defining control surfaces, elevator/aileron sweeps, mesh visualisation |
| `docs/tutorials/04_flow_visualisation.ipynb` | Surface Cp, XZ/YZ streamlines, interactive 3-D Plotly flow field |

## Tests

```bash
pytest tests/
pytest tests/test_aero/   # aerodynamics only
```

50 tests covering all four solvers: CL/CD/Cm ranges, sign conventions, L = CL·q·S identity, drag split, control surface effects (elevator Cm monotonicity, aileron antisymmetry, flap CL increase).

## License

MIT. The vendored aerodynamic solvers in `aero/solvers/` and `aero/singularities.py` are adapted from [AeroSandbox v4.2.9](https://github.com/peterdsharpe/AeroSandbox) by Peter Sharpe (MIT License).
