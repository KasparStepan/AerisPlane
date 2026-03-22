# AerisPlane Framework Design Specification

**Date:** 2026-03-22
**Version:** 1.0
**Status:** Draft

---

## 1. Purpose and Scope

AerisPlane (`aerisplane`) is a Python multidisciplinary design and optimization (MDO) toolkit for fixed-wing RC-scale UAVs. It targets early-stage conceptual design: configuration sizing, aerodynamic analysis, structural checks, stability verification, control authority assessment, mission performance, and constrained optimization across all disciplines simultaneously.

### 1.1 What AerisPlane IS

- A conceptual design MDO pipeline for RC/UAV aircraft (1-20 kg)
- A discipline integration layer that connects aerodynamics, weights, structures, stability, control, and mission analysis into a single optimization loop
- An engineering workflow tool with built-in plotting and reporting
- A Python library with a clean, readable API aimed at engineers and students

### 1.2 What AerisPlane IS NOT

- Not a CFD/FEA solver — it delegates to backend solvers (AeroSandbox, OpenAeroStruct)
- Not a GUI application
- Not a replacement for AeroSandbox — it builds on top of it and adds disciplines AeroSandbox doesn't cover
- Not a flight simulator or 6-DOF dynamics tool (v1)

### 1.3 Relationship to AeroSandbox

AerisPlane uses AeroSandbox as a backend aero solver via an adapter. It does NOT inherit from, subclass, or tightly couple to AeroSandbox's class hierarchy. AeroSandbox is a dependency called through a translation layer — the same way one might call XFOIL or OpenVSP.

AerisPlane's geometry classes are **independent of AeroSandbox**. The `core/` module depends only on `numpy` — it has no AeroSandbox imports. Translation to AeroSandbox objects happens exclusively in the adapter layer (`aero/aerosandbox_backend.py`), where functions like `aircraft_to_asb(aircraft) -> asb.Airplane` handle the conversion. Geometry methods on core classes (`span()`, `area()`, `mean_aerodynamic_chord()`) are implemented directly using numpy, not delegated to AeroSandbox.

This means:
- AeroSandbox version updates only affect the adapter code in `aero/aerosandbox_backend.py`
- The `core/` module can be used without AeroSandbox installed
- The v3 migration to a custom panel method requires no changes to `core/`
- The public API is stable and owned by AerisPlane

### 1.4 Long-Term Evolution

- **v1:** AeroSandbox + OpenAeroStruct adapters, beam structures, point performance mission, MDO with external optimizer
- **v2:** Time-stepping mission simulation, expanded hardware catalog, more reporting
- **v3:** Own panel method solver, own meshing, full-aircraft aerodynamics with fuselage interaction and control surface analysis

---

## 2. Architecture Overview

### 2.1 Layered Design

```
┌──────────────────────────────────────────────────────┐
│                     mdo/                              │  Optimization orchestration
│  MDOProblem, DesignVar, Constraint, Objective         │  Drives the discipline chain
├──────────────────────────────────────────────────────┤
│  aero/  │ weights/ │ structures/ │ stability/ │ control/ │ mission/  │  Discipline modules
│         │          │             │            │          │           │  Each: analyze() → Result
├──────────────────────────────────────────────────────┤
│                     core/                             │  Data model (geometry + components)
│  Aircraft, Wing, WingXSec, Fuselage, Airfoil, ...    │  Zero dependencies on disciplines
├──────────────────────────────────────────────────────┤
│                   catalog/                            │  Hardware database
│  Motors, Batteries, Servos, Propellers, Materials     │  Creates core/ objects
├──────────────────────────────────────────────────────┤
│                    utils/                             │  Shared utilities
│  Atmosphere, units, plotting style                    │  No domain logic
└──────────────────────────────────────────────────────┘
```

**Dependency rules:**
- `core/` depends on: `numpy` only — no AeroSandbox, no backend imports
- `catalog/` depends on: `core/`
- `utils/` depends on: `numpy`, `scipy`, `matplotlib`
- Each discipline module depends on: `core/`, `utils/`, and its backend libraries (lazy-imported)
- `mdo/` depends on: `core/`, all discipline modules
- Nothing depends on `mdo/` — it's the top of the dependency tree
- AeroSandbox → `asb.Airplane` translation lives in `aero/aerosandbox_backend.py`, not in `core/`

### 2.2 Discipline Pipeline

For a given aircraft configuration and flight condition, the analysis chain runs:

```
Aircraft + FlightCondition
        │
        ▼
   ┌─────────┐
   │  aero/  │ ──→ AeroResult (CL, CD, Cm, distributions)
   └─────────┘
        │
        ├──────────────────────┐
        ▼                      ▼
   ┌──────────┐         ┌──────────────┐
   │ weights/ │         │ structures/  │ ──→ StructResult (stresses, margins)
   └──────────┘         └──────────────┘
        │                      │
        ▼                      │
   WeightResult                │
   (mass, CG, inertia)        │
        │                      │
        ├──────────────────────┘
        ▼
   ┌─────────────┐
   │ stability/  │ ──→ StabilityResult (static margin, NP, derivatives)
   └─────────────┘
        │
        ▼
   ┌───────────┐
   │ control/  │ ──→ ControlResult (roll rate, pitch authority, rudder authority)
   └───────────┘
        │
        ▼
   ┌───────────┐
   │ mission/  │ ──→ MissionResult (endurance, range, energy budget, feasibility)
   └───────────┘
```

Data flows explicitly between modules via result objects. There are no hidden side effects or global state.

### 2.3 Backend Adapter Pattern

Disciplines that use external solvers (aero, structures) follow the adapter pattern:

```python
# In aero/__init__.py
def analyze(aircraft, condition, backend="aerosandbox", method="vlm"):
    if backend == "aerosandbox":
        from .aerosandbox_backend import analyze as _analyze
        return _analyze(aircraft, condition, method=method)
    elif backend == "openaerostruct":
        from .oas_backend import analyze as _analyze
        return _analyze(aircraft, condition)
    else:
        raise ValueError(f"Unknown aero backend: {backend}")
```

Each backend adapter:
1. Translates `ap.Aircraft` → backend-specific objects
2. Calls the backend
3. Translates results → `AeroResult` (or `StructResult`)

This keeps backend dependencies lazy-loaded and isolated.

---

## 3. Core Data Model (`core/`)

All classes in `core/` are plain Python dataclasses or simple classes. They carry data only — no analysis logic, no backend dependencies.

### 3.1 `Aircraft`

```python
@dataclass
class Aircraft:
    name: str
    wings: List[Wing]
    fuselages: List[Fuselage]
    propulsion: PropulsionSystem
    payload: Payload
```

`Aircraft` is the top-level container. It holds all components. It has no AeroSandbox or matplotlib dependency — it is pure data + numpy geometry methods.

Plotting is provided as standalone functions in `utils/plotting.py` to keep `core/` numpy-only:

```python
# In utils/plotting.py
def plot_aircraft(aircraft: Aircraft) -> Figure:
    """3-view or 3D wireframe of the aircraft."""

def plot_planform(aircraft: Aircraft) -> Figure:
    """Top-down planform with dimensions annotated."""
```

### 3.2 `Wing` and `WingXSec`

```python
@dataclass
class WingXSec:
    xyz_le: np.ndarray          # [x, y, z] leading edge position, meters
    chord: float                # meters
    twist: float = 0.0          # degrees (nose-up positive)
    airfoil: Airfoil = None     # defaults to NACA 0012 if not set

    # Structural properties — AerisPlane-specific, not in AeroSandbox
    spar: Optional[Spar] = None
    skin: Optional[Skin] = None
```

```python
@dataclass
class Wing:
    name: str = "wing"
    xsecs: List[WingXSec] = field(default_factory=list)
    symmetric: bool = True

    # Control surfaces defined on this wing
    control_surfaces: List[ControlSurface] = field(default_factory=list)

    # Geometry methods — implemented with numpy, no AeroSandbox dependency
    def span(self) -> float:
        """Total span (tip-to-tip if symmetric, root-to-tip if not)."""
    def area(self) -> float:
        """Planform area by trapezoidal integration of xsecs."""
    def aspect_ratio(self) -> float:
        """b^2 / S."""
    def mean_aerodynamic_chord(self) -> float:
        """MAC by integration of c^2 dy / integral of c dy."""
    def aerodynamic_center(self) -> np.ndarray:
        """Approximate AC at quarter-chord of MAC."""
```

**Design note:** `Wing` is used for both main wings and tail surfaces. There is no separate `Tail` class — a horizontal stabilizer is just a `Wing` with a different name, position, and control surface (elevator). This avoids duplicating code.

### 3.3 `Fuselage` and `FuselageXSec`

```python
@dataclass
class FuselageXSec:
    x: float                        # axial position from nose, meters
    radius: float                   # cross-section radius, meters
    shape: str = "circle"           # "circle", "ellipse", "rectangle" (future)
    width: Optional[float] = None   # for non-circular sections
    height: Optional[float] = None  # for non-circular sections

    def area(self) -> float:
        """Cross-sectional area."""
```

```python
@dataclass
class Fuselage:
    name: str = "fuselage"
    xsecs: List[FuselageXSec] = field(default_factory=list)
    x_le: float = 0.0              # nose position x
    y_le: float = 0.0              # nose position y
    z_le: float = 0.0              # nose position z

    # Structural properties
    material: Optional[Material] = None
    wall_thickness: float = 0.001   # meters

    def length(self) -> float: ...
    def volume(self) -> float: ...
    def wetted_area(self) -> float: ...
```

### 3.4 `Airfoil`

```python
@dataclass
class Airfoil:
    name: str                           # e.g., "naca2412", "ag35"
    coordinates: Optional[np.ndarray] = None  # (N, 2) if custom

    def thickness(self) -> float: ...
    def max_camber(self) -> float: ...

    @staticmethod
    def from_naca(designation: str) -> "Airfoil":
        """Generate NACA 4-digit or 5-digit airfoil coordinates.
        Implemented with numpy — no AeroSandbox dependency."""

    @staticmethod
    def from_file(path: str) -> "Airfoil":
        """Load airfoil coordinates from Selig-format .dat file."""
```

NACA coordinate generation is implemented in `core/airfoil.py` using standard NACA equations (pure numpy). Custom airfoils are loaded from `.dat` files. The aero adapter in `aerosandbox_backend.py` handles any coordinate manipulation needed by AeroSandbox (e.g., repaneling). `core/` has no AeroSandbox import.

### 3.5 `ControlSurface`

```python
@dataclass
class ControlSurface:
    name: str                       # "aileron", "elevator", "rudder", "flap"
    span_start: float               # fraction of wing spanwise extent (0=root, 1=tip)
    span_end: float                 # fraction of wing spanwise extent (0=root, 1=tip)
    chord_fraction: float           # fraction of local chord (0-1)
    max_deflection: float = 25.0    # degrees
    min_deflection: float = -25.0   # degrees (negative = trailing edge up for elevator)

    # Hardware
    servo: Optional[Servo] = None

    # Computed properties (set after parent Wing is known)
    hinge_line: Optional[np.ndarray] = None  # 3D line in aircraft frame

    def deflection_area(self, wing: "Wing") -> float:
        """Compute the planform area of the deflected surface."""
```

### 3.6 Structural Components

```python
@dataclass
class Material:
    """Pure material properties — reusable across different sections and components."""
    name: str
    density: float              # kg/m^3
    E: float                    # Young's modulus, Pa
    yield_strength: float       # Pa
    shear_modulus: Optional[float] = None  # Pa (computed from E and Poisson if not set)
    poisson_ratio: float = 0.3
```

```python
@dataclass
class TubeSection:
    """Cross-section geometry for a tubular spar."""
    outer_diameter: float       # meters
    wall_thickness: float       # meters

    def inner_diameter(self) -> float:
        return self.outer_diameter - 2 * self.wall_thickness

    def area(self) -> float:
        """Cross-sectional area of the tube wall."""

    def second_moment_of_area(self) -> float:
        """I = pi/64 * (OD^4 - ID^4)."""

    def section_modulus(self) -> float:
        """S = I / (OD/2)."""
```

```python
@dataclass
class Spar:
    position: float             # fraction of chord (0 = LE, 1 = TE)
    material: Material
    section: TubeSection        # cross-section geometry

@dataclass
class Skin:
    material: Material
    thickness: float            # meters
```

**Design note:** `Material` holds only intrinsic material properties (density, stiffness, strength). Cross-section geometry (`TubeSection`) is separate so the same material (e.g., carbon fiber) can be reused across different spar diameters without duplication. The catalog stores `Material` instances; `TubeSection` is defined per-spar in the aircraft definition.

### 3.7 Propulsion Components

```python
@dataclass
class Motor:
    name: str
    kv: float                   # RPM per volt
    resistance: float           # ohms
    no_load_current: float      # amps
    max_current: float          # amps
    mass: float                 # kg

    def torque(self, current: float) -> float: ...
    def rpm(self, voltage: float, torque: float) -> float: ...
    def efficiency(self, current: float, voltage: float) -> float: ...

@dataclass
class Propeller:
    diameter: float             # meters
    pitch: float                # meters
    mass: float = 0.03          # kg
    num_blades: int = 2

    # Optional: performance lookup table from UIUC or APC data
    # If provided, thrust/power use interpolation. Otherwise, parametric model.
    performance_data: Optional[PropellerPerfData] = None

    def advance_ratio(self, velocity: float, rpm: float) -> float:
        """J = V / (n * D) where n = rpm/60."""

    def thrust(self, rpm: float, velocity: float, rho: float) -> float:
        """Thrust. If performance_data available, interpolate CT(J).
        Otherwise, use parametric model: CT = CT0 * (1 - J/J0)
        fitted from diameter and pitch ratio."""

    def power(self, rpm: float, velocity: float, rho: float) -> float:
        """Shaft power. If performance_data available, interpolate CP(J).
        Otherwise, parametric model."""

    def efficiency(self, rpm: float, velocity: float, rho: float) -> float:
        """Propulsive efficiency eta = J * CT / CP."""

@dataclass
class PropellerPerfData:
    """Lookup table for propeller performance, e.g., from UIUC or APC data files."""
    J: np.ndarray               # advance ratio array
    CT: np.ndarray              # thrust coefficient array
    CP: np.ndarray              # power coefficient array
    source: str = ""            # e.g., "APC 10x5E static + wind tunnel"

@dataclass
class Battery:
    name: str
    capacity_ah: float          # amp-hours
    nominal_voltage: float      # volts (e.g., 14.8 for 4S)
    cell_count: int             # number of series cells
    c_rating: float             # max continuous discharge rate
    mass: float                 # kg
    internal_resistance: float = 0.0  # ohms (total pack)

    def energy(self) -> float:
        """Total energy in joules."""

    def max_current(self) -> float:
        """Max continuous current from C rating."""

    def voltage_under_load(self, current: float) -> float:
        """Voltage accounting for internal resistance sag."""

@dataclass
class ESC:
    name: str
    max_current: float          # amps
    mass: float                 # kg
    has_telemetry: bool = False

@dataclass
class PropulsionSystem:
    motor: Motor
    propeller: Propeller
    battery: Battery
    esc: ESC
    position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    direction: np.ndarray = field(default_factory=lambda: np.array([-1, 0, 0]))  # thrust vector

    def thrust_available(self, velocity: float, rho: float) -> float:
        """Max thrust at given airspeed."""

    def power_required(self, thrust: float, velocity: float, rho: float) -> float:
        """Electrical power for given thrust."""

    def endurance_at_power(self, power_watts: float) -> float:
        """Time in seconds until battery depleted at constant power."""

    def total_mass(self) -> float:
        """Sum of motor + propeller + battery + ESC masses."""
```

### 3.8 `Payload` and `FlightCondition`

```python
@dataclass
class Payload:
    mass: float                 # kg
    cg: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # meters
    name: str = "payload"

@dataclass
class FlightCondition:
    velocity: float             # m/s (true airspeed)
    altitude: float = 0.0       # meters MSL
    alpha: float = 0.0          # degrees angle of attack
    beta: float = 0.0           # degrees sideslip

    # Control surface deflections (degrees), keyed by surface name
    # e.g., {"elevator": -3.5, "aileron": 10.0, "rudder": 0.0}
    deflections: Dict[str, float] = field(default_factory=dict)

    def dynamic_pressure(self) -> float:
        """0.5 * rho * V^2 using ISA atmosphere at altitude."""

    def reynolds_number(self, reference_length: float) -> float:
        """Re = rho * V * L / mu."""

    def mach(self) -> float:
        """Mach number at altitude."""
```

`FlightCondition` carries the aerodynamic state: velocity, altitude, orientation angles, and control surface deflections. Deflections are passed as a dictionary keyed by surface name — this is how the stability and control modules communicate deflection angles to the aero backend without mutating the `Aircraft` object. Body rates (p, q, r) are not included — those belong in the flight dynamics module (future).

### 3.9 `Servo` (in `core/control_surface.py`)

```python
@dataclass
class Servo:
    name: str
    torque: float               # N*m at rated voltage
    speed: float                # degrees/second at rated voltage
    voltage: float              # rated voltage
    mass: float                 # kg
```

---

## 4. Discipline Modules

### 4.1 Aerodynamics (`aero/`)

**Files:**
- `__init__.py` — `analyze()` entry point with backend dispatch
- `result.py` — `AeroResult` dataclass with `.plot()` and `.report()`
- `aerosandbox_backend.py` — adapter to AeroSandbox VLM/LLT
- `oas_backend.py` — adapter to OpenAeroStruct VLM

**`AeroResult` dataclass:**

```python
@dataclass
class AeroResult:
    # Overall coefficients
    CL: float                       # lift coefficient
    CD: float                       # drag coefficient (total)
    CDi: float                      # induced drag coefficient
    CDp: float                      # parasitic/profile drag coefficient
    Cm: float                       # pitching moment coefficient (about CG or ref point)
    CY: float                       # side force coefficient

    # Reference values used
    S_ref: float                    # reference area (m^2)
    c_ref: float                    # reference chord (m)
    b_ref: float                    # reference span (m)

    # Distributions (arrays along semispan)
    y_stations: np.ndarray          # spanwise positions
    cl_distribution: np.ndarray     # local lift coefficient
    cdi_distribution: np.ndarray    # local induced drag coefficient
    chord_distribution: np.ndarray  # local chord

    # Dimensional forces
    lift: float                     # Newtons
    drag: float                     # Newtons

    # Per-surface breakdown (dict keyed by wing name)
    per_surface: Dict[str, dict]    # {"wing": {"CL": ..., "CD": ...}, "htail": {...}}

    def plot(self) -> Figure:
        """Plot spanwise CL distribution, drag polar, Cm curve."""

    def report(self) -> str:
        """Print summary table: CL, CD, L/D, Cm, per-surface breakdown."""
```

**AeroSandbox adapter (`aerosandbox_backend.py`):**

```python
def aircraft_to_asb(aircraft: Aircraft) -> asb.Airplane:
    """Translate ap.Aircraft → asb.Airplane.
    Converts Wings, WingXSecs, Fuselages, Airfoils to AeroSandbox equivalents.
    Control surface deflections are set from the FlightCondition.deflections dict.
    This is the ONLY place in the codebase that imports aerosandbox."""

def condition_to_asb(condition: FlightCondition) -> asb.OperatingPoint:
    """Translate ap.FlightCondition → asb.OperatingPoint."""

def analyze(aircraft: Aircraft, condition: FlightCondition,
            method: str = "vlm") -> AeroResult:
    """
    1. Build asb.Airplane via aircraft_to_asb(aircraft)
    2. Apply control surface deflections from condition.deflections
    3. Build asb.OperatingPoint via condition_to_asb(condition)
    4. Run asb.AeroBuildup or asb.VortexLatticeMethod depending on method
    5. Extract results into AeroResult
    """
```

Supported methods via AeroSandbox:
- `"vlm"` — Vortex Lattice Method (3D, lifting surfaces only)
- `"llt"` — Lifting Line Theory
- `"aerobuildup"` — AeroSandbox's semi-empirical buildup (includes fuselage drag)
- `"neuralfoil"` — 2D section polars only (used inside VLM/LLT)
- `"xfoil"` — 2D section polars via XFOIL wrapper (validation)

**OpenAeroStruct adapter (`oas_backend.py`):**

```python
def analyze(aircraft: Aircraft, condition: FlightCondition) -> AeroResult:
    """
    1. Build OAS mesh from aircraft wing geometry
    2. Set up OpenMDAO problem with OAS AeroPoint component
    3. Run the analysis
    4. Extract results into AeroResult
    """
```

The OAS adapter in aero/ runs OAS in **aero-only mode** (no structures). The coupled aerostructural mode lives in `structures/oas_backend.py`.

### 4.2 Weights (`weights/`)

**Files:**
- `__init__.py` — `analyze()` entry point
- `result.py` — `WeightResult` dataclass with `.plot()` and `.report()`
- `buildup.py` — component mass buildup logic

**`WeightResult` dataclass:**

```python
@dataclass
class WeightResult:
    total_mass: float               # kg
    cg: np.ndarray                  # [x, y, z] center of gravity, meters
    inertia_tensor: np.ndarray      # 3x3 inertia tensor about CG, kg*m^2

    # Component breakdown
    components: Dict[str, ComponentMass]
    # e.g. {"wing_structure": ComponentMass(mass=0.35, cg=[0.2, 0.3, 0]),
    #        "motor": ComponentMass(mass=0.152, cg=[0.9, 0, 0]),
    #        "battery": ComponentMass(mass=0.42, cg=[0.25, 0, 0]), ...}

    wing_loading: float             # g/dm^2

    def plot(self) -> Figure:
        """Pie chart of masses, side-view CG diagram on aircraft silhouette."""

    def report(self) -> str:
        """Component table: name, mass, CG_x, CG_y, CG_z, % of total."""
```

**Buildup logic (`buildup.py`):**

Weight is computed by summing:

1. **Structural masses** — computed from geometry + material:
   - Wing structure: spar mass (tube length * cross-section area * density) + skin mass (wetted area * thickness * density) + rib estimates
   - Fuselage structure: shell mass (wetted area * wall thickness * density) + internal reinforcement estimate
   - Tail structure: same as wing

2. **Hardware masses** — from catalog or user-defined:
   - Motor, propeller, ESC, battery: `.mass` attribute
   - Servos: sum of all servo masses from control surfaces
   - Avionics: from payload definition (flight controller, GPS, receiver, companion computer)

3. **Payload mass** — user-defined

CG is computed as mass-weighted average of component CG positions. Inertia tensor uses parallel axis theorem from component masses and positions.

### 4.3 Structures (`structures/`)

**Files:**
- `__init__.py` — `analyze()` entry point with backend dispatch
- `result.py` — `StructResult` dataclass with `.plot()` and `.report()`
- `beam.py` — simple 1D Euler-Bernoulli beam solver
- `oas_backend.py` — OpenAeroStruct coupled aerostructural adapter

**`StructResult` dataclass:**

```python
@dataclass
class StructResult:
    # Spanwise distributions (arrays)
    y_stations: np.ndarray
    bending_moment: np.ndarray      # N*m
    shear_force: np.ndarray         # N
    torsion: np.ndarray             # N*m (if computed)
    deflection: np.ndarray          # meters
    twist: np.ndarray               # degrees (aeroelastic twist, if computed)

    # Critical values
    max_bending_moment: float       # N*m (at root)
    max_stress: float               # Pa
    max_deflection: float           # meters (at tip)
    max_twist: float                # degrees (at tip)

    # Margins
    spar_margin_of_safety: float    # (yield_strength / max_stress) - 1
    feasible: bool                  # all margins > 0

    def plot(self) -> Figure:
        """Spanwise bending moment, shear, deflection diagrams."""

    def report(self) -> str:
        """Max stress, margin of safety, tip deflection, feasibility."""
```

**Beam solver (`beam.py`):**

A 1D Euler-Bernoulli beam model for the wing spar:

1. Discretize the semispan into stations matching `WingXSec` positions
2. Load distribution from `AeroResult.cl_distribution` scaled by load factor and weight
3. Compute shear force and bending moment by integration from tip to root
4. Compute deflection by double integration of M/(E*I) with boundary conditions (zero deflection and slope at root)
5. Compute stress from bending moment and section modulus: sigma = M / S
6. Compare to material yield strength for margin of safety

This is a fast, analytical solver — suitable for the optimizer's outer loop.

**OAS backend (`oas_backend.py`):**

Runs OpenAeroStruct in coupled aerostructural mode:

1. Translate `ap.Aircraft` wing geometry + structural properties to OAS mesh + structural definition
2. Set up OpenMDAO problem with `AerostructPoint`
3. Run the coupled analysis (or optimization of structural sizing)
4. Return `StructResult` with aeroelastically-corrected deflections and stresses

This is the high-fidelity path — used for refinement of promising designs, not inside the main optimization loop.

### 4.4 Stability (`stability/`)

**Files:**
- `__init__.py` — `analyze()` entry point
- `result.py` — `StabilityResult` dataclass with `.plot()` and `.report()`
- `derivatives.py` — numerical stability derivative computation

**`StabilityResult` dataclass:**

```python
@dataclass
class StabilityResult:
    # Longitudinal
    static_margin: float            # fraction of MAC (positive = stable)
    neutral_point: float            # x-position, meters from nose
    Cm_alpha: float                 # dCm/dalpha (1/deg) — should be negative
    CL_alpha: float                 # dCL/dalpha (1/deg)

    # Lateral-directional
    Cl_beta: float                  # dCl/dbeta (roll due to sideslip) — should be negative
    Cn_beta: float                  # dCn/dbeta (yaw due to sideslip) — should be positive

    # Trim
    trim_alpha: float               # degrees — alpha for level flight at given speed
    trim_elevator: float            # degrees — elevator deflection for trim

    # Tail volume coefficients
    Vh: float                       # horizontal tail volume coefficient
    Vv: float                       # vertical tail volume coefficient

    # CG envelope
    cg_forward_limit: float         # fraction of MAC
    cg_aft_limit: float             # fraction of MAC

    def plot(self) -> Figure:
        """Cm vs alpha, CG envelope on planform, NP location."""

    def report(self) -> str:
        """Static margin, derivatives, trim conditions, volume coefficients."""
```

**Derivative computation (`derivatives.py`):**

All stability derivatives computed by numerical finite differences using the aero backend:

```python
def compute_derivatives(aircraft, condition, aero_backend, weight_result):
    """
    Central differences — 5 aero evaluations total:

    1. Run aero at (alpha - d_alpha, beta)       → for alpha derivatives
    2. Run aero at (alpha + d_alpha, beta)       → for alpha derivatives
    3. Run aero at (alpha, beta - d_beta)        → for beta derivatives
    4. Run aero at (alpha, beta + d_beta)        → for beta derivatives
    5. Run aero at (alpha, beta) — baseline      → for reference values

    From these:
    6. dCL/dalpha = (CL(+da) - CL(-da)) / (2 * d_alpha)
    7. dCm/dalpha = (Cm(+da) - Cm(-da)) / (2 * d_alpha)
    8. dCl/dbeta  = (Cl(+db) - Cl(-db)) / (2 * d_beta)
    9. dCn/dbeta  = (Cn(+db) - Cn(-db)) / (2 * d_beta)
    10. Compute neutral point from dCm/dCL
    11. Compute static margin from NP and CG (from weight_result)
    12. Compute trim by finding alpha where Cm = 0 (bisection or Newton)
    13. Compute tail volume coefficients from geometry
    """
```

Step sizes: `d_alpha = 0.5 deg`, `d_beta = 1.0 deg`. Central differences for all derivatives. Cost: 5 aero evaluations per stability analysis — important to account for in MDO performance budgeting.

### 4.5 Control Authority (`control/`)

**Files:**
- `__init__.py` — `analyze()` entry point
- `result.py` — `ControlResult` dataclass with `.plot()` and `.report()`
- `authority.py` — control authority calculations

**`ControlResult` dataclass:**

```python
@dataclass
class ControlResult:
    # Roll
    max_roll_rate: float            # deg/s at given flight condition
    aileron_authority: float        # 0-1 normalized (1 = meets requirement)
    Cl_delta_a: float               # roll moment per degree aileron

    # Pitch
    elevator_authority: float       # 0-1 normalized
    Cm_delta_e: float               # pitch moment per degree elevator
    max_pitch_acceleration: float   # deg/s^2

    # Yaw
    rudder_authority: float         # 0-1 normalized
    Cn_delta_r: float               # yaw moment per degree rudder
    max_crosswind: float            # m/s — max crosswind for coordinated flight

    # Servo loads (if servo data available)
    aileron_hinge_moment: Optional[float]   # N*m
    elevator_hinge_moment: Optional[float]  # N*m
    rudder_hinge_moment: Optional[float]    # N*m

    def plot(self) -> Figure:
        """Authority bar chart, roll rate vs aileron deflection, servo load margins."""

    def report(self) -> str:
        """Authority summary, derivative values, servo adequacy check."""
```

**Authority computation (`authority.py`):**

```python
def compute_authority(aircraft, condition, aero_result, weight_result):
    """
    For each control surface:
    1. Run aero with surface deflected at max deflection
    2. Run aero with surface at zero deflection
    3. Compute delta_CL, delta_Cm, delta_Cl, delta_Cn from difference
    4. Compute control derivatives (per degree)

    Roll rate:
    5. Steady-state roll rate: p_ss = -Cl_delta_a * delta_a_max / Cl_p
       where Cl_p (roll damping) estimated from strip theory

    Pitch authority:
    6. Max pitch acceleration from elevator = Cm_delta_e * q * S * c / I_yy

    Rudder / crosswind:
    7. Max beta for zero yaw = Cn_delta_r * delta_r_max / Cn_beta

    Hinge moments (simplified):
    8. Estimated from surface area, deflection, and dynamic pressure
       using thin-airfoil hinge moment coefficients
    """
```

### 4.6 Mission (`mission/`)

**Files:**
- `__init__.py` — `analyze()` entry point
- `result.py` — `MissionResult` dataclass with `.plot()` and `.report()`
- `segments.py` — mission segment definitions
- `performance.py` — point performance equations

**Mission segments:**

```python
@dataclass
class Climb:
    to_altitude: float          # meters
    climb_rate: float           # m/s
    velocity: float             # m/s (airspeed during climb)
    name: str = "climb"

@dataclass
class Cruise:
    distance: float             # meters
    velocity: float             # m/s (cruise airspeed)
    altitude: float = 100.0     # meters MSL
    name: str = "cruise"

@dataclass
class Loiter:
    duration: float             # seconds
    velocity: float             # m/s (loiter airspeed — typically best endurance speed)
    altitude: float = 100.0     # meters MSL
    name: str = "loiter"

@dataclass
class Return:
    distance: float             # meters
    velocity: float             # m/s (return airspeed)
    altitude: float = 100.0     # meters MSL
    name: str = "return"

@dataclass
class Descent:
    to_altitude: float          # meters
    descent_rate: float = 2.0   # m/s
    velocity: float = 15.0      # m/s (airspeed during descent)
    name: str = "descent"

@dataclass
class Mission:
    segments: List[Union[Climb, Cruise, Loiter, Return, Descent]]
    start_altitude: float = 0.0     # meters MSL — ground level for first segment
```

**Segment sequencing:** Segments are executed in order. Each segment's starting altitude is inherited from the previous segment's end altitude (or `Mission.start_altitude` for the first segment). This means `Descent.to_altitude` is unambiguous — it always descends from wherever the previous segment ended.

**`MissionResult` dataclass:**

```python
@dataclass
class MissionResult:
    total_energy: float             # Joules
    total_time: float               # seconds
    endurance: float                # seconds (max possible with remaining energy)
    range_total: float              # meters

    battery_energy_available: float # Joules
    energy_margin: float            # fraction remaining (0 = empty, 1 = full)
    feasible: bool                  # enough battery energy for the mission

    # Per-segment breakdown
    segments: List[SegmentResult]
    # SegmentResult: name, duration, distance, energy, avg_power, avg_speed

    def plot(self) -> Figure:
        """Energy budget bar chart, altitude profile, power vs time."""

    def report(self) -> str:
        """Segment table: name, duration, distance, energy, power. Total summary."""
```

**Point performance equations (`performance.py`):**

For each segment, compute power required at equilibrium flight:

```python
def power_required_level(aircraft, condition, aero_result, weight_result):
    """
    P_req = D * V = (CD / CL) * W * V
    where W = total_mass * g, and CL/CD from aero_result at trim alpha
    """

def power_required_climb(aircraft, condition, aero_result, weight_result, climb_rate):
    """
    P_req = D * V + W * climb_rate
    """

def energy_for_segment(segment, aircraft, aero_backend):
    """
    1. Determine flight condition for the segment (speed, altitude)
    2. Run aero at trim alpha
    3. Compute power required
    4. Account for propulsive efficiency (propulsion system model)
    5. Energy = electrical_power * duration
    """
```

Cruise speed is determined by finding the speed that minimizes power required (max endurance) or the speed that the user specifies. For v1, use a fixed cruise speed derived from the design condition.

---

## 5. MDO Module (`mdo/`)

### 5.1 Files

- `__init__.py`
- `problem.py` — `MDOProblem`, `DesignVar`, `Constraint`, `Objective`
- `result.py` — `OptimizationResult` with `.plot()` and `.report()`
- `drivers.py` — thin wrappers around SciPy, pygmo, etc.

### 5.2 `MDOProblem`

```python
@dataclass
class DesignVar:
    path: str                   # e.g., "wings[0].xsecs[1].chord"
    lower: float                # lower bound
    upper: float                # upper bound
    scale: float = 1.0          # scaling for optimizer (value / scale)

@dataclass
class Constraint:
    path: str                   # e.g., "stability.static_margin"
    lower: Optional[float] = None
    upper: Optional[float] = None
    equals: Optional[Any] = None
    scale: float = 1.0

@dataclass
class Objective:
    path: str                   # e.g., "mission.endurance"
    maximize: bool = True
    scale: float = 1.0
```

```python
class MDOProblem:
    def __init__(self,
                 aircraft: Aircraft,
                 mission: Mission,
                 design_variables: List[DesignVar],
                 constraints: List[Constraint],
                 objective: Objective,
                 aero_backend: str = "aerosandbox",
                 aero_method: str = "vlm",
                 structures_backend: str = "beam",
                 load_factor: float = 4.0):
        ...

    def validate(self):
        """
        Called at construction time. Resolves all design variable and
        constraint paths against the aircraft object. Raises clear errors
        for typos or invalid paths BEFORE the optimizer starts.
        """

    def evaluate(self, x: np.ndarray) -> dict:
        """
        Core evaluation function. Called by the optimizer.
        Results are cached keyed on x — if called again with the same x
        (e.g., by objective_function then constraint_functions in the same
        optimizer iteration), returns the cached result without re-running.

        1. Check cache: if np.array_equal(x, self._cached_x), return cached
        2. Unpack x into aircraft parameters via design variable paths
        3. Create modified Aircraft deep copy
        4. Determine trim alpha at cruise condition
        5. Run aero at trim
        6. Run weights
        7. Run structures (with aero loads * load_factor)
        8. Run stability (5 aero calls for central differences)
        9. Run control authority
        10. Run mission
        11. Collect objective and all constraint values
        12. Cache result, log iteration via Python logging module
        13. Return dict with all results

        Returns:
            {
                "objective": float,
                "constraints": {"stability.static_margin": 0.08, ...},
                "results": {
                    "aero": AeroResult,
                    "weights": WeightResult,
                    "structures": StructResult,
                    "stability": StabilityResult,
                    "control": ControlResult,
                    "mission": MissionResult,
                },
                "aircraft": Aircraft,  # the modified aircraft for this x
            }
        """

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds arrays for the optimizer."""

    def objective_function(self, x: np.ndarray) -> float:
        """Calls evaluate(x) (cached) and returns scalar objective."""

    def constraint_functions(self, x: np.ndarray) -> np.ndarray:
        """Calls evaluate(x) (cached) and returns constraint violation vector."""

    def optimize(self, method: str = "scipy_de", options: dict = None,
                 callback: Optional[Callable] = None) -> "OptimizationResult":
        """
        Convenience wrapper. Supported methods:
        - "scipy_de": scipy.optimize.differential_evolution
        - "scipy_minimize": scipy.optimize.minimize (gradient-free, e.g., Nelder-Mead)
        - "scipy_shgo": scipy.optimize.shgo (global)
        - "pygmo_de": pygmo differential evolution
        - "pygmo_nsga2": pygmo multi-objective (future)

        Logging: each evaluate() call logs iteration number, objective value,
        constraint satisfaction, and wall time via Python's logging module.
        callback: optional function called after each iteration with
        (iteration, x, objective, constraints_satisfied) for custom progress reporting.
        """
```

### 5.3 Design Variable Path Resolution

The string path system (`"wings[0].xsecs[1].chord"`) maps between a flat numpy vector and nested aircraft attributes:

```python
def _resolve_path(aircraft, path):
    """
    Parse "wings[0].xsecs[1].chord" and return (parent_object, attribute_name).
    Supports:
    - Attribute access: "name.subname"
    - List indexing: "list[0]"
    - Nested: "wings[0].control_surfaces[0].chord_fraction"
    """

def _pack(aircraft, design_vars) -> np.ndarray:
    """Extract current values from aircraft into flat vector."""

def _unpack(aircraft, design_vars, x) -> Aircraft:
    """Create a deep copy of aircraft with values from x applied."""
```

### 5.4 `OptimizationResult`

```python
@dataclass
class OptimizationResult:
    aircraft: Aircraft              # optimized aircraft
    x_initial: np.ndarray           # starting design vector
    x_optimal: np.ndarray           # optimal design vector
    objective_initial: float
    objective_optimal: float
    constraints_satisfied: bool
    n_evaluations: int
    convergence_history: List[float] # objective value per iteration

    # Full discipline results at optimum
    aero: AeroResult
    weights: WeightResult
    structures: StructResult
    stability: StabilityResult
    control: ControlResult
    mission: MissionResult

    # Design variable comparison
    variables: Dict[str, Tuple[float, float]]  # {"path": (initial, optimal)}

    def plot(self) -> Figure:
        """Convergence history, design variable bar chart (initial vs optimal)."""

    def report(self) -> str:
        """
        Table: variable name, initial, optimal, lower bound, upper bound
        Objective: initial → optimal
        Constraints: all values and pass/fail
        """
```

---

## 6. Hardware Catalog (`catalog/`)

### 6.1 Structure

```
catalog/
├── __init__.py         # list_motors(), list_batteries(), search(), etc.
├── motors.py           # Motor instances
├── batteries.py        # Battery instances
├── servos.py           # Servo instances
├── propellers.py       # Propeller instances
└── materials.py        # Material instances
```

### 6.2 Catalog Entry Format

Each file contains module-level instances:

```python
# catalog/motors.py
from aerisplane.core.propulsion import Motor

turnigy_d3548_1100kv = Motor(
    name="Turnigy D3548/4 1100kV",
    kv=1100,
    resistance=0.028,
    no_load_current=1.2,
    max_current=40.0,
    mass=0.152,
)

sunnysky_x2216_1250kv = Motor(
    name="SunnySky X2216 1250kV",
    kv=1250,
    resistance=0.035,
    no_load_current=0.8,
    max_current=35.0,
    mass=0.095,
)

# ... more motors
```

### 6.3 Catalog Discovery

```python
# catalog/__init__.py

def list_motors() -> None:
    """Print formatted table of all motors with key specs."""

def list_batteries() -> None:
    """Print formatted table of all batteries."""

def list_all() -> None:
    """Print all catalog entries by category."""

def search(query: str) -> List:
    """Search by name substring across all categories."""
```

Users add new hardware by editing or adding to these Python files. IDE autocomplete works: `from aerisplane.catalog.motors import turnigy_d3548_1100kv`.

### 6.4 Starter Catalog

v1 ships with a small but real catalog:
- **Motors:** 5-8 common RC outrunners (Turnigy, SunnySky, T-Motor) in the 800-1500 kV range
- **Batteries:** 5-8 common 3S/4S/6S LiPo packs (Tattu, CNHL, Turnigy)
- **Servos:** 5-8 common digital servos (Hitec, Savox, Emax)
- **Propellers:** 5-8 common sizes (APC, Gemfan) in 8x6 to 12x6 range
- **Materials:** carbon fiber, PETG, ASA, PLA, balsa, plywood, fiberglass, EPP foam (pure material properties — tube sections defined per-spar)

---

## 7. Utilities (`utils/`)

### 7.1 `atmosphere.py`

ISA (International Standard Atmosphere) model:

```python
def isa(altitude: float) -> Tuple[float, float, float, float]:
    """
    Returns (temperature, pressure, density, dynamic_viscosity)
    at given altitude in meters MSL.
    Valid 0-11000 m (troposphere).
    """
```

### 7.2 `units.py`

Convenience conversions. AerisPlane uses SI internally (meters, kg, seconds, Pascals). Conversion helpers for common RC/aero units:

```python
def ft_to_m(ft): ...
def m_to_ft(m): ...
def mph_to_ms(mph): ...
def ms_to_mph(ms): ...
def in_to_m(inches): ...          # propeller diameters often in inches
def oz_to_kg(oz): ...             # RC weights often in ounces
def g_per_dm2(mass_kg, area_m2): ...  # wing loading
def deg_to_rad(deg): ...
def rad_to_deg(rad): ...
```

### 7.3 `plotting.py`

Shared matplotlib style for consistent output:

```python
def apply_style():
    """Set AerisPlane matplotlib style: font sizes, grid, colors."""

COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "accent": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "neutral": "#6b7280",
}

def new_figure(nrows=1, ncols=1, figsize=None) -> Tuple[Figure, Axes]:
    """Create a styled figure with sensible defaults."""
```

---

## 8. Repository Layout (Final)

```
AerisPlane/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   └── aerisplane/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── aircraft.py
│       │   ├── wing.py
│       │   ├── fuselage.py
│       │   ├── airfoil.py
│       │   ├── control_surface.py
│       │   ├── propulsion.py
│       │   ├── payload.py
│       │   ├── structures.py
│       │   └── flight_condition.py
│       │
│       ├── aero/
│       │   ├── __init__.py
│       │   ├── result.py
│       │   ├── aerosandbox_backend.py
│       │   └── oas_backend.py
│       │
│       ├── weights/
│       │   ├── __init__.py
│       │   ├── result.py
│       │   └── buildup.py
│       │
│       ├── structures/
│       │   ├── __init__.py
│       │   ├── result.py
│       │   ├── beam.py
│       │   └── oas_backend.py
│       │
│       ├── stability/
│       │   ├── __init__.py
│       │   ├── result.py
│       │   └── derivatives.py
│       │
│       ├── control/
│       │   ├── __init__.py
│       │   ├── result.py
│       │   └── authority.py
│       │
│       ├── mission/
│       │   ├── __init__.py
│       │   ├── result.py
│       │   ├── segments.py
│       │   └── performance.py
│       │
│       ├── mdo/
│       │   ├── __init__.py
│       │   ├── problem.py
│       │   ├── result.py
│       │   └── drivers.py
│       │
│       ├── catalog/
│       │   ├── __init__.py
│       │   ├── motors.py
│       │   ├── batteries.py
│       │   ├── servos.py
│       │   ├── propellers.py
│       │   └── materials.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── atmosphere.py
│           ├── units.py
│           └── plotting.py
│
├── docs/
│   ├── overview.md
│   └── architecture.md
│
├── tutorials/
│   ├── 01_define_aircraft.ipynb
│   ├── 02_aero_analysis.ipynb
│   ├── 03_stability_check.ipynb
│   └── 04_optimize_wing.ipynb
│
├── planes/
│   └── CoreFly/
│       ├── requirements.md
│       ├── corefly_baseline.py
│       ├── configs/
│       │   ├── wing_endurance.py
│       │   ├── wing_general.py
│       │   └── wing_sport.py
│       ├── optimization/
│       │   ├── optimize_endurance.py
│       │   └── results/
│       └── reports/
│
└── tests/
    ├── conftest.py
    ├── test_core/
    ├── test_aero/
    ├── test_weights/
    ├── test_structures/
    ├── test_stability/
    ├── test_control/
    ├── test_mission/
    └── test_mdo/
```

---

## 9. Dependencies

### 9.1 Required

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24 | Array operations everywhere |
| `scipy` | >= 1.10 | Optimization, interpolation, integration |
| `matplotlib` | >= 3.7 | All plotting |
| `aerosandbox` | >= 4.0 | Primary aero backend, geometry math, airfoil database |

### 9.2 Optional

| Package | Version | Purpose |
|---------|---------|---------|
| `openaerostruct` | >= 2.7 | Aerostructural analysis backend |
| `openmdao` | >= 3.30 | Required by OpenAeroStruct |
| `pygmo` | >= 2.19 | Advanced optimization algorithms |

Optional dependencies are imported lazily — only when the corresponding backend is requested.

### 9.3 `pyproject.toml`

```toml
[project]
name = "aerisplane"
version = "0.1.0"
description = "Conceptual MDO toolkit for RC/UAV aircraft design"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "aerosandbox>=4.0",
]

[project.optional-dependencies]
oas = ["openaerostruct>=2.7", "openmdao>=3.30"]
optimize = ["pygmo>=2.19"]
dev = ["pytest>=7.0", "ruff>=0.1.0"]
all = ["aerisplane[oas,optimize,dev]"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 10. Public API (`__init__.py`)

```python
# src/aerisplane/__init__.py

# Core geometry
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.wing import Wing, WingXSec
from aerisplane.core.fuselage import Fuselage, FuselageXSec
from aerisplane.core.airfoil import Airfoil
from aerisplane.core.control_surface import ControlSurface
from aerisplane.core.structures import Spar, Skin, Material, TubeSection
from aerisplane.core.propulsion import (
    Motor, Propeller, PropellerPerfData, Battery, ESC, PropulsionSystem
)
from aerisplane.core.payload import Payload
from aerisplane.core.flight_condition import FlightCondition

# Discipline modules
from aerisplane import aero
from aerisplane import weights
from aerisplane import structures
from aerisplane import stability
from aerisplane import control
from aerisplane import mission

# Mission segments
from aerisplane.mission.segments import (
    Mission, Climb, Cruise, Loiter, Return, Descent
)

# MDO
from aerisplane.mdo.problem import MDOProblem, DesignVar, Constraint, Objective

# Hardware catalog
from aerisplane import catalog

# Servo (lives alongside ControlSurface, not in propulsion)
from aerisplane.core.control_surface import Servo
```

Usage:

```python
import aerisplane as ap

wing = ap.Wing(
    xsecs=[
        ap.WingXSec(xyz_le=[0, 0, 0], chord=0.25, airfoil=ap.Airfoil("naca2412")),
        ap.WingXSec(xyz_le=[0.02, 0.7, 0.05], chord=0.12, airfoil=ap.Airfoil("naca2412")),
    ],
    symmetric=True,
    control_surfaces=[
        ap.ControlSurface(name="aileron", span_start=0.6, span_end=0.95, chord_fraction=0.25),
    ],
)

aircraft = ap.Aircraft(name="my_uav", wings=[wing], fuselages=[fus], propulsion=prop)
condition = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=5.0)

aero_result = ap.aero.analyze(aircraft, condition, backend="aerosandbox", method="vlm")
aero_result.plot()
aero_result.report()
```

---

## 11. Design Decisions Log

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Geometry classes | Own classes, ASB-independent core, adapter translates | Inherit asb, composition with cached asb objects, from scratch | Clean dependency boundary: core/ has zero ASB imports, adapters own the translation |
| Aero backend | AeroSandbox primary, OAS secondary | Own VLM, OpenVSP | AeroSandbox is mature, fast, Python-native. OAS adds aerostructural. |
| Optimizer | External (SciPy/pygmo), not asb.opti | asb.opti, OpenMDAO | asb.opti requires full vectorization; breaks with non-smooth disciplines |
| Weight model | Component buildup + hardware catalog | Semi-empirical regression, hybrid | Accurate for known hardware, grows with catalog. Matches RC builder workflow. |
| Structures v1 | Simple beam + OAS adapter | OAS only, full FEA, skip | Beam is fast for optimizer loop, OAS available for refinement |
| Stability derivatives | Numerical finite differences from aero | Analytical formulas, AVL | Consistent with aero solver, no additional assumptions |
| Mission v1 | Point performance (energy budget) | Time-stepping, full trajectory | Fast enough for optimizer. Time-stepping added later for analysis. |
| Config format | Pure Python API, no config files | YAML, TOML, JSON | IDE autocomplete, version control, readable. Config I/O is future work. |
| Catalog format | Python module-level instances | SQLite, JSON, YAML | Autocomplete, git-friendly, zero parsing code |
| Control surfaces | First-class with hinge geometry + servo | Part of WingXSec, separate module | Needed as design variables for MDO. Servo loads needed for hardware selection. |
