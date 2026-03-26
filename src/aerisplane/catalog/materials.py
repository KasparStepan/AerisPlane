"""Common materials for 3D-printed RC aircraft construction.

All values are for solid material. For 3D prints with infill, the effective
density should be adjusted by the user based on wall count and infill %.
"""

from aerisplane.core.structures import Material

# ---------------------------------------------------------------------------
# 3D printing filaments (solid density)
# ---------------------------------------------------------------------------

petg = Material(
    name="PETG",
    density=1270.0,       # kg/m^3
    E=2.1e9,              # Pa
    yield_strength=50e6,  # Pa
    poisson_ratio=0.36,
)

pla = Material(
    name="PLA",
    density=1240.0,
    E=3.5e9,
    yield_strength=60e6,
    poisson_ratio=0.36,
)

asa = Material(
    name="ASA",
    density=1070.0,
    E=2.0e9,
    yield_strength=40e6,
    poisson_ratio=0.35,
)

pla_lw = Material(
    name="PLA-LW",
    density=800.0,
    E=2636e6,
    yield_strength=23.2e6,
    poisson_ratio=0.36,
)


# ---------------------------------------------------------------------------
# Structural tubes and rods
# ---------------------------------------------------------------------------

carbon_fiber_tube = Material(
    name="Carbon Fiber Tube",
    density=1600.0,
    E=135e9,
    yield_strength=1500e6,
    poisson_ratio=0.3,
)

carbon_fiber_rod = Material(
    name="Carbon Fiber Rod",
    density=1600.0,
    E=135e9,
    yield_strength=1500e6,
    poisson_ratio=0.3,
)

fiberglass = Material(
    name="Fiberglass",
    density=1800.0,
    E=35e9,
    yield_strength=300e6,
    poisson_ratio=0.25,
)

aluminum_6061 = Material(
    name="Aluminum 6061-T6",
    density=2700.0,
    E=68.9e9,
    yield_strength=276e6,
    poisson_ratio=0.33,
)

# ---------------------------------------------------------------------------
# Covering films
# ---------------------------------------------------------------------------

monokote = Material(
    name="MonoKote Film",
    density=900.0,
    E=2.5e9,
    yield_strength=50e6,
    poisson_ratio=0.4,
)
