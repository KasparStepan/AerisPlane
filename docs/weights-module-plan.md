# Weights Module — Implementation Plan

**Date:** 2026-03-26
**Status:** Steps 1–6 complete (core module, tests, report, plot, catalog materials)
**Authoritative spec:** `docs/superpowers/specs/2026-03-22-aerisplane-framework-design.md` (Section 4.2)

---

## 1. Goal

Implement the `weights/` discipline module: a component-based mass buildup that estimates structural and hardware masses from geometry + materials, computes CG and inertia, and supports overriding any component with measured values from CAD or a physical build.

The target use case is **3D-printed RC aircraft** (like CoreFly): printed shells/skins, carbon fiber tube spars, off-the-shelf electronics. The user progressively replaces computed estimates with real measurements as the design matures.

---

## 2. Design Principles

1. **Computed by default, overridable always.** Every component mass starts as a geometry-based estimate. The user can override any component with a measured mass and CG. Overrides take priority unconditionally.

2. **Source tracking.** Every component in the result is tagged as `"computed"` or `"override"` so the user always knows what's estimated vs. measured.

3. **No new dependencies.** The weights module depends only on `core/` and `numpy`. No AeroSandbox, no matplotlib at import time (matplotlib only inside `.plot()`).

4. **Follows the discipline module pattern.** Entry point is `weights.analyze(aircraft, ...) → WeightResult`. Result dataclass has `.report()` and `.plot()`.

---

## 3. Implementation Steps

### Step 1 — `weights/result.py`: Data structures

**File:** `src/aerisplane/weights/result.py`

```python
@dataclass
class ComponentMass:
    """One component's mass contribution."""
    name: str                    # e.g., "main_wing_spar", "battery"
    mass: float                  # kg
    cg: np.ndarray               # [x, y, z] center of gravity in aircraft frame [m]
    source: str = "computed"     # "computed" or "override"

@dataclass
class WeightResult:
    """Complete mass buildup result."""
    total_mass: float            # kg
    cg: np.ndarray               # [x, y, z] overall CG [m]
    inertia_tensor: np.ndarray   # 3x3 inertia about CG [kg*m^2]
    components: dict[str, ComponentMass]  # keyed by component name
    wing_loading: float          # g/dm^2 (based on main wing area)

    def report(self) -> str:
        """Formatted table: component | mass | CG | source | % of total."""

    def plot(self):
        """Pie chart of mass breakdown + side-view CG diagram."""
```

**What it does:** Pure data containers. `WeightResult` aggregates all component masses and derived quantities. `.report()` returns a human-readable string table. `.plot()` produces a matplotlib figure (lazy import).

**Depends on:** `numpy` only.

---

### Step 2 — `weights/buildup.py`: Mass estimation engine

**File:** `src/aerisplane/weights/buildup.py`

This is the core computation. It walks the `Aircraft` tree and estimates mass for each component.

#### 2a. Wing structure mass

```python
def _wing_mass(wing: Wing) -> list[ComponentMass]:
```

For each adjacent pair of `WingXSec` (a "panel"):

- **Spar mass:** If sections have spars, compute tube length as the Euclidean distance between section LE positions (adjusted for spar chordwise position). Mass = `spar.mass_per_length() * panel_length`. If both sections have different spars, average the mass_per_length.
- **Skin mass:** Panel wetted area ≈ `(chord_i + chord_j) / 2 * panel_span * 2` (top + bottom surfaces). For a 3D-printed skin: mass = `skin.mass_per_area() * wetted_area`. If sections have different skins, average the mass_per_area.
- **Rib estimate:** Add an allowance for ribs. Default: one rib per section (at each WingXSec position), rib mass ≈ `chord * max_thickness_fraction * chord * rib_thickness * density`. Simplified to a fraction of panel skin mass as a starting default (e.g., 15%).

CG for each sub-component is at the panel midpoint (average of the two section LE positions, offset by 40% chord for the overall wing structure CG).

For symmetric wings, the spar/skin mass is doubled (left + right), and the CG y-component is 0 (symmetric).

Returns separate `ComponentMass` entries for `"{wing.name}_spar"` and `"{wing.name}_skin"`.

#### 2b. Fuselage structure mass

```python
def _fuselage_mass(fuselage: Fuselage) -> list[ComponentMass]:
```

- **Shell mass:** `fuselage.wetted_area() * fuselage.wall_thickness * fuselage.material.density`
- **CG position:** Centroid of the fuselage shell. Computed as area-weighted average of axial station positions: for each panel between adjacent cross-sections, the local CG_x is the panel midpoint, weighted by the panel's wetted area contribution.

Returns a `ComponentMass` entry for `"{fuselage.name}_shell"`.

If `fuselage.material` is `None`, skip (no structural mass can be computed — this component must be overridden by the user).

#### 2c. Hardware masses

```python
def _hardware_masses(aircraft: Aircraft) -> list[ComponentMass]:
```

Walks the aircraft and extracts mass from hardware components:

| Component | Mass source | CG position source |
|-----------|------------|-------------------|
| Motor | `propulsion.motor.mass` | `propulsion.position` |
| Propeller | `propulsion.propeller.mass` | `propulsion.position` |
| Battery | `propulsion.battery.mass` | `propulsion.position` (offset by half battery length aft, but initially same as propulsion position) |
| ESC | `propulsion.esc.mass` | `propulsion.position` (initially same) |
| Servos | `servo.mass` for each control surface | Estimated at the wing LE position interpolated at the surface's midspan fraction |

Note on CG positions: In the initial implementation, motor/prop/ESC/battery all use `propulsion.position` as their CG. This is a simplification — in reality the battery is often far from the motor. This is exactly where **overrides** become important: the user specifies the actual battery position. In Phase 2 (component boxes), each hardware item gets its own position.

#### 2d. Payload mass

```python
def _payload_mass(aircraft: Aircraft) -> list[ComponentMass]:
```

Direct from `aircraft.payload.mass` and `aircraft.payload.cg`. Returns one `ComponentMass` entry for `"{payload.name}"`.

#### 2e. Aggregation

```python
def compute_buildup(aircraft: Aircraft) -> tuple[list[ComponentMass], float]:
```

1. Collect all `ComponentMass` entries from 2a–2d.
2. Total mass = sum of all component masses.
3. CG = mass-weighted average of component CG positions.
4. Inertia tensor = parallel axis theorem:
   - For each component, treated as a point mass at its CG position
   - `I_total = Σ m_i * (|r_i|² * I_3x3 - r_i ⊗ r_i)` where `r_i = cg_i - cg_total`
5. Wing loading = `total_mass * 1000 / (main_wing.area() * 100)` → g/dm²

---

### Step 3 — `weights/__init__.py`: Entry point with overrides

**File:** `src/aerisplane/weights/__init__.py`

```python
def analyze(
    aircraft: Aircraft,
    overrides: dict[str, ComponentOverride] | None = None,
) -> WeightResult:
```

**`ComponentOverride` dataclass:**

```python
@dataclass
class ComponentOverride:
    """User-provided measured mass and CG for a component."""
    mass: float                  # kg
    cg: np.ndarray | None = None # [x, y, z] — if None, keep the computed CG
```

**Logic:**

1. Run `compute_buildup(aircraft)` to get all computed components.
2. For each key in `overrides`:
   - If the key matches an existing computed component → replace mass (and CG if provided), set source to `"override"`.
   - If the key does NOT match any computed component → add it as a new component with source `"override"`. This allows adding components that the buildup doesn't know about (e.g., "gps_module", "receiver", "fpv_camera").
3. Recompute total mass, CG, inertia, wing loading from the final component list.
4. Return `WeightResult`.

This is the key feature: **overrides can both replace and extend** the buildup.

---

### Step 4 — `catalog/materials.py`: 3D print materials

**File:** `src/aerisplane/catalog/materials.py`

Common materials for 3D-printed RC aircraft:

```python
from aerisplane.core.structures import Material

# 3D printing filaments
petg = Material(name="PETG", density=1270, E=2.1e9, yield_strength=50e6, poisson_ratio=0.36)
pla = Material(name="PLA", density=1240, E=3.5e9, yield_strength=60e6, poisson_ratio=0.36)
asa = Material(name="ASA", density=1070, E=2.0e9, yield_strength=40e6, poisson_ratio=0.35)

# Structural tubes and rods
carbon_fiber_tube = Material(name="Carbon Fiber Tube", density=1600, E=135e9, yield_strength=1500e6, poisson_ratio=0.3)
carbon_fiber_rod = Material(name="Carbon Fiber Rod", density=1600, E=135e9, yield_strength=1500e6, poisson_ratio=0.3)
aluminum_6061 = Material(name="Aluminum 6061-T6", density=2700, E=68.9e9, yield_strength=276e6, poisson_ratio=0.33)

# Covering
monokote = Material(name="MonoKote Film", density=900, E=2.5e9, yield_strength=50e6, poisson_ratio=0.4)
```

Note on 3D print density: The `density` values above are for solid material. For infilled prints, effective density = `density * (wall_fraction + infill_fraction * infill_percentage)`. This correction is applied in the buildup, not in the material definition — Material holds intrinsic properties only (per CLAUDE.md).

---

### Step 5 — Tests

**File:** `tests/test_weights/test_buildup.py`

Test cases:

1. **Wing spar mass** — rectangular wing with known spar (20mm OD, 2mm wall CF tube, 0.75m semispan). Hand-calculate expected mass, compare.
2. **Wing skin mass** — rectangular wing with PETG skin (0.8mm thick, known area). Compare to hand calc.
3. **Fuselage shell mass** — simple fuselage with known geometry and PETG material. Compare.
4. **Hardware masses** — aircraft with propulsion system. Verify motor + prop + battery + ESC masses appear in result.
5. **Servo masses** — wing with control surfaces that have servos. Verify servo masses are counted.
6. **Payload** — verify payload mass and CG are included.
7. **Total mass** — verify total = sum of all components.
8. **CG computation** — aircraft with known component positions. Hand-calculate expected CG, compare.
9. **Inertia tensor** — simple case (two point masses on x-axis). Verify Ixx=0, Iyy and Izz correct.
10. **Wing loading** — verify g/dm² calculation.
11. **Override replaces computed** — override battery mass, verify it replaces the computed value and total updates.
12. **Override adds new component** — add "receiver" via override, verify it appears and total includes it.
13. **Override CG** — override battery with different CG, verify overall CG shifts.
14. **Partial override (mass only)** — override mass but not CG, verify CG from computed is preserved.
15. **Report string** — verify `.report()` returns a string with expected column headers and all components listed.

**File:** `tests/test_weights/conftest.py`

Fixtures:
- `petg` — PETG material
- `carbon_fiber_tube` — CF tube material (reuse from existing conftest or catalog)
- `wing_with_structure` — simple wing with spar and skin assigned to sections
- `aircraft_with_structure` — full aircraft with wing, fuselage (material + wall thickness), propulsion, payload
- `simple_overrides` — dict with a battery override for testing

---

### Step 6 — `.report()` and `.plot()` implementation

**`.report()`** — returns a formatted string:

```
AerisPlane Weight Buildup
═══════════════════════════════════════════════════════════════════
Component              Mass [g]    CG_x [mm]   CG_z [mm]   Source      %
─────────────────────────────────────────────────────────────────────────
main_wing_spar            84.2       210.0         0.0     computed    3.1
main_wing_skin           182.0       215.0         0.0     override    6.7
htail_spar                12.3       780.0         0.0     computed    0.5
htail_skin                28.1       785.0         0.0     computed    1.0
fuselage_shell           145.3       420.0        -5.0     computed    5.4
motor                    152.0       920.0         0.0     computed    5.6
propeller                 30.0       950.0         0.0     computed    1.1
battery                  420.0       180.0       -10.0     override   15.5
esc                       35.0       850.0         0.0     computed    1.3
aileron_servo_L           24.0       220.0         0.0     computed    0.9
aileron_servo_R           24.0       220.0         0.0     computed    0.9
elevator_servo            18.0       770.0         0.0     computed    0.7
payload                  500.0       300.0         0.0     computed   18.4
─────────────────────────────────────────────────────────────────────────
TOTAL                   2712.4
CG position:  [245.2, 0.0, -3.2] mm
Wing loading: 28.3 g/dm²
```

**`.plot()`** — matplotlib figure with two subplots:
1. **Pie chart** — mass breakdown grouped by category (structure, propulsion, avionics, payload)
2. **Side-view CG diagram** — x-z plane, component CGs as circles (size proportional to mass), overall CG as a cross, main wing MAC and quarter-chord marked

`.plot()` uses lazy matplotlib import (not imported at module level).

---

## 4. What Exists Now (Starting State)

| File | Status |
|------|--------|
| `src/aerisplane/weights/__init__.py` | Empty (0 lines) |
| `src/aerisplane/weights/result.py` | Empty (0 lines) |
| `src/aerisplane/weights/buildup.py` | Empty (0 lines) |
| `src/aerisplane/catalog/materials.py` | Empty (0 lines) |
| `tests/test_weights/` | Does not exist |
| `core/` classes | Fully implemented — `Spar.mass_per_length()`, `Skin.mass_per_area()`, `Wing` geometry, `Fuselage.wetted_area()`, `PropulsionSystem.total_mass()`, all `.mass` attributes |

---

## 5. Future Extensions (Not In This Implementation)

These are planned but will be done in separate follow-up work:

- **Component boxes + intersection/containment checks** — give each hardware component physical dimensions (bounding box), check that components fit inside fuselage/wing volumes, check no two components overlap. Enables placement optimization in MDO.
- **Rib mass from geometry** — compute rib count from spacing rule, rib area from airfoil profile, instead of percentage estimate.
- **Fastener/joiner mass** — wing-to-fuselage bolts, tail boom joiners.
- **Wire harness estimate** — mass scales with fuselage length + wingspan.
- **3D print infill correction** — adjust effective density for infill percentage in structural sections.
- **CG envelope** — CG range across loading configurations (with/without payload, different batteries).
- **Ballast calculation** — compute required ballast mass to hit target CG.
- **Side-view CG diagram** — aircraft outline with component boxes and CG position visualization.

---

## 6. File Dependency Map

```
weights/__init__.py          ← analyze() entry point
    ├── weights/result.py    ← ComponentMass, ComponentOverride, WeightResult
    └── weights/buildup.py   ← estimation functions
            ├── core/aircraft.py
            ├── core/wing.py        (geometry: span, area, section positions)
            ├── core/fuselage.py    (geometry: wetted_area, wall_thickness)
            ├── core/structures.py  (Spar.mass_per_length, Skin.mass_per_area)
            ├── core/propulsion.py  (Motor.mass, Battery.mass, etc.)
            ├── core/control_surface.py (Servo.mass)
            └── core/payload.py     (Payload.mass, .cg)

catalog/materials.py         ← standalone, no dependency on weights/
tests/test_weights/          ← depends on weights/ and core/
```

---

## 7. Acceptance Criteria

The weights module is done when:

1. `weights.analyze(aircraft)` returns a correct `WeightResult` for an aircraft with wings (spar + skin), fuselage, propulsion, servos, and payload.
2. Overrides replace computed values and can add new components.
3. Total mass = sum of all component masses (verified by test).
4. CG is the mass-weighted centroid (verified by hand calculation in test).
5. Inertia tensor is correct for a simple case (verified by hand calculation).
6. Wing loading is correct: `total_mass_grams / wing_area_dm2`.
7. `.report()` produces a readable table with all components.
8. All tests pass.
9. No imports of AeroSandbox, matplotlib (except inside `.plot()`), or any solver library.
