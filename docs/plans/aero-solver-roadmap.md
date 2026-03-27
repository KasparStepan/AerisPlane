# Aerodynamics Solver Roadmap

Current status (2026-03-27): VLM, LiftingLine, NonlinearLiftingLine, and AeroBuildup
are all implemented and working.  This document maps out the next steps to improve
accuracy, fidelity, and capability.

---

## Where we are now

| Method | Physics | Speed | Stall | Viscous | Moments |
|---|---|---|---|---|---|
| VLM | Inviscid, linear | ⚡ fast | ✗ | ✗ | ✅ |
| LiftingLine | Viscous, linearised CL | ⚡ fast | partial | ✅ | ✅ |
| NonlinearLiftingLine | Viscous, nonlinear CL | moderate | ✅ | ✅ | ✅ |
| AeroBuildup | Semi-empirical | ⚡ fast | ✅ | ✅ | ✅ |

---

## Tier 1 — Easy wins (no new solver, just improvements)

### 1a. Stall-damping in NLL (1–2 days)

At high alpha NLL can oscillate near stall because the NeuralFoil polar
has a sharp CL peak.  Two fixes:

- **Adaptive relaxation**: reduce relaxation factor automatically when the
  residual stops decreasing monotonically.
- **Aitken acceleration**: compute the optimal relaxation factor each
  iteration from the residual history.  Cuts iterations by ~30%.

### 1b. Projected reference area (1 day)

Our `Wing.reference_area()` uses trapezoidal area; ASB uses projected
(horizontal) area.  This causes ~1% coefficient difference.  Add
`Wing.projected_area()` that projects each panel onto the XY plane.

### 1c. Wake alignment (1 day)

`align_trailing_vortices_with_wind=True` already exists.  The induced
drag is ~2–5% more accurate with the wake aligned to the freestream.
Make this the default.

### 1d. Compressibility correction (Prandtl-Glauert) (1 day)

Multiply NeuralFoil CL by `1/sqrt(1 - M²)` and CD by `1/(1-M²)` for
subsonic compressible flow.  Already have `condition.mach()`.  Matters
above M~0.3.  See `aero/library/transonic.py` for the existing model.

---

## Tier 2 — Significant improvements (1–2 weeks each)

### 2a. Vortex Lattice with thickness (3D source panels)

The current VLM only models the camber surface (no thickness).  Adding
source panels on the body surface gives:
- Correct pressure distribution (not just forces)
- Better fuselage-wing interference
- Thickness drag contribution

**Approach:** Add a doublet panel layer on top of the camber-surface
horseshoes.  ASB's `VLM3D` does this; it is a natural extension of our
existing `singularities.py`.

### 2b. Control surface deflections in VLM (Phase 4, already planned)

Rotate the trailing-edge panels by the deflection angle when building
the mesh.  This is straightforward with our existing `mesh_thin_surface()`
since the panel quad vertices are explicit.

**Impact:** Enables elevator authority, roll rate, and hinge moment
predictions — the original reason this branch exists.

### 2c. Improved fuselage model

The current Jorgensen model in `AeroBuildup` gives good drag but mediocre
lift and moments for non-axisymmetric fuselages (pods, bulges).

**Option A:** Slender body theory (Munk's formula) — 10 lines, much
better for elongated fuselages.

**Option B:** Source-panel method for arbitrary fuselage cross-sections —
more work (~200 lines) but gives pressure distribution.

---

## Tier 3 — Panel method (full 3D, 2–4 weeks)

A panel method replaces the thin-surface assumption entirely.  Every
surface (wing, fuselage, nacelle) is meshed with surface panels carrying
doublet + source singularities.

### What it gives you

- Correct fuselage aerodynamics at any angle of attack
- Fuselage–wing interference (upwash/downwash at root)
- Pressure distribution on the entire airframe (good for structures)
- Works for any geometry, not just wings

### How to build it

**Step 1 — Geometry:** We already have `mesh_thin_surface()` on Wing.
Need `mesh_surface()` that returns the full thick surface (both sides).
For fuselages, revolve the cross-section profile.

**Step 2 — Singularities:** Add `calculate_induced_velocity_doublet()`
and `calculate_induced_velocity_source_panel()` to `aero/singularities.py`.
These are simple extensions of the horseshoe functions already there.

**Step 3 — AIC system:** Assemble (N_doublet × N_doublet) + (N_source ×
N_doublet) matrices, apply Neumann (zero normal flow) boundary condition,
solve.  Same `np.linalg.solve` as VLM.

**Step 4 — Post-processing:** Compute pressure from Bernoulli.  Integrate
over surface for forces and moments.

**Step 5 — Viscous coupling (optional):** Couple the inviscid panel
solution to a 2D boundary layer solver (e.g. Drela's Xfoil approach) for
skin friction.

### AD/differentiation for panel methods

If you ever want gradients through the panel solve (for aerostructural
MDO), you have two options:

**JAX** (`jax.numpy` drop-in for NumPy):
- Works with our existing code — replace `import numpy as np` → `import jax.numpy as np`
- Forward-mode or reverse-mode AD through the entire AIC solve
- Best for relatively small panel counts (<10k panels)
- No new solver required; gradients are free

**CasADi** (symbolic graph + IPOPT):
- Designed for large-scale NLP optimisation
- More setup: must write your panel method symbolically using `casadi.MX`
- Provides sparsity-exploiting exact Jacobians — big advantage when the
  optimiser needs many design variables
- Best if you want to do single-shot MDO (structures + aero simultaneously
  via IPOPT), like ASB's original architecture

For a panel method at UAV scale (<500 panels), **JAX is the better
starting point** — less infrastructure, easier to debug, and gradient
quality is identical.  CasADi becomes worthwhile only if the panel count
grows past ~1000 or you need tight MDO coupling.

---

## Tier 4 — Aerostructural optimization

### OpenMDAO + OpenAeroStruct (recommended path)

OpenAeroStruct (OAS) couples a vortex-lattice method with a finite-element
beam model and runs inside OpenMDAO's derivative framework.

**What it gives:**
- Coupled aero-structural analysis: wing deflection changes the aero loads,
  which changes deflection
- Total derivative (∂L/∂t_skin, ∂D/∂sweep, etc.) through the coupled system
- Works with SLSQP, SNOPT, IPOPT

**Integration with AerisPlane:**
- OAS lives in `aero/` as an optional backend (`backend="openaerostruct"`)
- Our `Aircraft` → OAS translator already partially exists in the repo
- OAS uses its own Wing/Mesh objects; the translation is straightforward

**Install:** `pip install openaerostruct`

### DIY coupled solver

If you want to own the whole stack:
1. Write a thin finite-element beam on each spar (`structures/` module)
2. At each aero iteration: deform the wing mesh by the beam solution
3. Iterate aero ↔ structures until forces and deflections converge
4. Use JAX to get gradients through the coupled iteration

This is ~2 weeks of work but gives complete control and no external
solver dependency.

---

## Summary: recommended sequence

1. **Phase 4** (already planned): control surfaces in VLM — unlocks the
   core use case (elevator, aileron authority)
2. **1d** Prandtl-Glauert compressibility — easy, improves accuracy above
   M=0.3
3. **2b** Wake alignment default — minimal change, better induced drag
4. **Panel method** — when you need pressure distributions or accurate
   fuselage coupling
5. **JAX integration** — when you need gradients through the aero solve
6. **OAS or DIY aerostructural** — when wing deformation under load matters
