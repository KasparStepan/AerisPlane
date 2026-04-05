# Changelog

All notable changes to AerisPlane are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.1.0] — 2026-04-05

### Added

**Aerodynamics (`aerisplane.aero`)**
- Vortex Lattice Method (VLM) — inviscid 3-D, arbitrary geometry
- Prandtl Lifting Line with NeuralFoil section polars
- Nonlinear Lifting Line (fixed-point stall iteration)
- AeroBuildup semi-empirical method (NeuralFoil wings + Jorgensen fuselage)
- Surface pressure and streamline flow visualisation
- Interactive Plotly 3-D flow field (`plot_interactive()`)

**Weights (`aerisplane.weights`)**
- Component-based mass buildup from geometry and material properties
- CG computation and inertia tensor estimate
- Override mechanism for replacing estimates with measurements

**Structures (`aerisplane.structures`)**
- Euler–Bernoulli wing beam solver
- Failure checks: bending margin, shear margin, buckling margin
- Tip deflection ratio and torsional divergence speed
- OpenAeroStruct adapter (optional dependency)

**Stability (`aerisplane.stability`)**
- Numerical stability derivatives via central finite differences
- Static margin, neutral point, Cm_alpha, CL_alpha
- Lateral-directional: Cl_beta, Cn_beta
- Rate derivatives: CL_q, Cm_q, Cl_p, Cn_p, CY_p, Cn_r, Cl_r, CY_r
- Dynamic mode estimates: short-period frequency and damping

**Control (`aerisplane.control`)**
- Control authority: roll rate, pitch authority, rudder authority
- Servo hinge moment estimation and servo load margin
- Finite-difference control derivatives

**Mission (`aerisplane.mission`)**
- Segment-based energy budget: Climb, Cruise, Loiter, Descent, Return
- Flight envelope: Vs, Vmin_power, Vmax_range, Vy, ceiling
- Range–endurance–speed performance curves

**Propulsion (`aerisplane.propulsion`)**
- Motor/propeller/battery/ESC operating-point solver
- Outputs: thrust, RPM, current, motor efficiency, propulsive efficiency,
  battery endurance, C-rate, over-current flag

**MDO (`aerisplane.mdo`)**
- `MDOProblem`: design variables, constraints, objectives
- String-path design variable syntax (`"wings[0].xsecs[1].chord"`)
- Evaluation caching keyed on design vector
- SciPy drivers: `scipy_de`, `scipy_minimize`, `scipy_shgo`
- pygmo drivers: `pygmo_de`, `pygmo_sade`, `pygmo_nsga2`
- Checkpoint/resume
- Sensitivity analysis: finite-difference gradients, normalized ranking

**Catalog (`aerisplane.catalog`)**
- 20 brushless motors, 15 LiPo batteries, 10 propellers, 10 servos
- 2175 airfoil `.dat` files
- `list_motors()`, `list_batteries()`, `list_propellers()`, `list_servos()`
