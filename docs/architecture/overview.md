# Overview

AerisPlane is a Python MDO toolkit for fixed-wing RC/UAV conceptual design (1–20 kg class).

## What AerisPlane is

- A conceptual design MDO pipeline for RC/UAV aircraft
- A discipline integration layer: aerodynamics, weights, structures, stability,
  control, and mission analysis wired into a single optimization loop
- A Python library with a clean, readable API for engineers and students

## What AerisPlane is not

- Not a CFD/FEA solver — it delegates to backend solvers (AeroSandbox, OpenAeroStruct)
- Not a GUI application
- Not a replacement for AeroSandbox — it builds on top of it and adds disciplines
  AeroSandbox does not cover
- Not a flight simulator or 6-DOF dynamics tool

## Relationship to AeroSandbox

AerisPlane uses AeroSandbox as a backend aero solver via an adapter layer.
It does **not** inherit from or tightly couple to AeroSandbox's class hierarchy.
The `core/` module depends only on `numpy` and has no AeroSandbox imports.
Translation to AeroSandbox objects happens exclusively in `aero/aerosandbox_backend.py`.

## Intended workflow

1. **Define** — build an `Aircraft` from `Wing`, `Fuselage`, and hardware catalog items
2. **Analyse** — run discipline modules individually or in sequence
3. **Optimise** — define an `MDOProblem`, run it, inspect `OptimizationResult`
