"""Weights discipline module — component-based mass buildup.

Entry point: ``analyze(aircraft, overrides=None) -> WeightResult``
"""

from __future__ import annotations

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.weights.buildup import aggregate, compute_buildup
from aerisplane.weights.result import ComponentMass, ComponentOverride, WeightResult


def analyze(
    aircraft: Aircraft,
    overrides: dict[str, ComponentOverride] | None = None,
) -> WeightResult:
    """Run a complete weight buildup for the aircraft.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft configuration to analyze.
    overrides : dict mapping component name to ComponentOverride, optional
        User-provided measured masses. If a key matches an existing computed
        component, the computed value is replaced. If the key is new, it is
        added as an additional component. This allows progressive replacement
        of estimates with real measurements.

    Returns
    -------
    WeightResult
        Complete mass breakdown with total mass, CG, inertia, and wing loading.
    """
    # Step 1: compute all masses from geometry + materials
    components = compute_buildup(aircraft)

    # Step 2: apply overrides
    if overrides:
        comp_dict = {c.name: c for c in components}

        for name, override in overrides.items():
            if name in comp_dict:
                # Replace existing component
                existing = comp_dict[name]
                existing.mass = override.mass
                existing.source = "override"
                if override.cg is not None:
                    existing.cg = override.cg
            else:
                # Add new component
                cg = override.cg if override.cg is not None else np.zeros(3)
                comp_dict[name] = ComponentMass(
                    name=name,
                    mass=override.mass,
                    cg=cg,
                    source="override",
                )

        components = list(comp_dict.values())

    # Step 3: aggregate
    ref_area = aircraft.reference_area()
    total_mass, cg, inertia, wing_loading = aggregate(components, ref_area)

    return WeightResult(
        total_mass=total_mass,
        cg=cg,
        inertia_tensor=inertia,
        components={c.name: c for c in components},
        wing_loading=wing_loading,
    )
