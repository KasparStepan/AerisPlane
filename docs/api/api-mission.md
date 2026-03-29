# Mission & Flight Performance API

Module: `aerisplane.mission`

---

## Quick start

```python
import aerisplane as ap
from aerisplane import weights, mission
from aerisplane.mission.segments import Mission, Climb, Cruise, Loiter, Descent, Return

# 1. Build aircraft with propulsion system
aircraft = ap.Aircraft(name="my_uav", wings=[main_wing], fuselages=[fuse],
                       propulsion=propulsion)

# 2. Weight analysis (needed for CG and mass)
weight_result = weights.analyze(aircraft)

# 3. Flight envelope (power curves, speeds, climb, glide, endurance)
env = mission.envelope(aircraft, weight_result)
print(env.report())
env.plot()

# 4. Mission energy budget
my_mission = Mission(segments=[
    Climb(to_altitude=100, climb_rate=2.0, velocity=12.0),
    Cruise(distance=5000, velocity=15.0, altitude=100.0),
    Descent(to_altitude=0.0),
])
result = mission.analyze(aircraft, weight_result, my_mission)
print(result.report())
result.plot()
```

---

## Entry points

### `mission.performance()`

```python
mission.performance(aircraft, weight_result, altitude=0.0, aero_method="vlm") -> DragPolar
```

Fits a parabolic drag polar CD = CD0 + k·CL² from VLM evaluations at 3 speeds.

### `mission.envelope()`

```python
mission.envelope(aircraft, weight_result, CL_max=1.4, aero_method="vlm") -> EnvelopeResult
```

Computes the full flight performance envelope across altitudes (0–3000 m by default).

### `mission.analyze()`

```python
mission.analyze(aircraft, weight_result, mission, aero_method="vlm") -> MissionResult
```

Runs a mission energy budget analysis. For each segment, computes duration, distance, and energy consumed.

---

## Performance functions

Module: `aerisplane.mission.performance`

### Drag polar

```python
fit_drag_polar(aircraft, weight_result, altitude=0.0, aero_method="vlm",
               speeds=(10.0, 15.0, 20.0)) -> DragPolar
```

Fits CD = CD0 + k·CL² by running the aero solver at several speeds and doing a least-squares fit. Since VLM is inviscid, parasitic drag (CD0) is estimated from turbulent flat-plate skin friction when the fitted value is too small.

#### `DragPolar` dataclass

| Field | Type | Description |
|---|---|---|
| `CD0` | `float` | Zero-lift drag coefficient |
| `k` | `float` | Induced drag factor (k = 1/(π·AR·e)) |
| `S_ref` | `float` | Reference wing area [m²] |

| Method | Returns | Description |
|---|---|---|
| `cd(cl)` | `float` | Total CD at given CL |
| `ld_max()` | `float` | Maximum lift-to-drag ratio |
| `cl_for_ld_max()` | `float` | CL at max L/D |
| `cl_for_min_power()` | `float` | CL at minimum power required |

### Power curves

```python
power_required(velocity, polar, mass, altitude=0.0) -> float    # [W]
power_available(propulsion, velocity, altitude=0.0, throttle=1.0) -> float  # [W]
```

- `power_required`: P_R = 0.5·ρ·S·CD0·V³ + 2·k·W²/(ρ·S·V)
- `power_available`: thrust × velocity at the motor-prop operating point

### Characteristic speeds

```python
stall_speed(mass, S, CL_max=1.4, altitude=0.0) -> float              # [m/s]
best_range_speed(polar, mass, altitude=0.0) -> float                  # [m/s]
best_endurance_speed(polar, mass, altitude=0.0) -> float              # [m/s]
max_level_speed(polar, mass, propulsion, altitude=0.0) -> float|None  # [m/s]
```

| Speed | Formula | Meaning |
|---|---|---|
| V_stall | √(2W / (ρ·S·CL_max)) | Minimum flight speed |
| V* (best range) | V at CL = √(CD0/k) | Maximum L/D, longest range |
| V_mp (best endurance) | 0.76 × V* | Minimum power, longest time aloft |
| V_max | Largest V where P_A ≥ P_R | Maximum level flight speed |

### Climb performance

```python
rate_of_climb(velocity, polar, mass, propulsion, altitude=0.0) -> float       # [m/s]
max_rate_of_climb(polar, mass, propulsion, altitude=0.0) -> (ROC_max, V_y)    # [m/s]
```

ROC = (P_A − P_R) / W. Returns the excess-power climb rate.

### Glide performance

```python
glide_range(polar, from_altitude) -> float                    # [m]
glide_performance(polar, mass, altitude=0.0) -> GlidePerformance
```

#### `GlidePerformance` dataclass

| Field | Type | Description |
|---|---|---|
| `best_glide_ratio` | `float` | L/D max |
| `best_glide_speed` | `float` | Speed for best glide [m/s] |
| `min_sink_speed` | `float` | Speed for minimum sink rate [m/s] |
| `min_sink_rate` | `float` | Minimum vertical descent rate [m/s] |

### Endurance and range

```python
max_endurance(polar, mass, propulsion, altitude=0.0, eta_total=None) -> float  # [seconds]
max_range(polar, mass, propulsion, altitude=0.0, eta_total=None) -> float      # [meters]
```

- Endurance: E_battery × η / P_R_min (at best endurance speed)
- Range: E_battery × η × (L/D)_max / W (at best range speed)

If `eta_total` is None, overall efficiency is estimated from the motor and propeller models.

---

## Envelope

Module: `aerisplane.mission.envelope`

```python
compute_envelope(aircraft, weight_result, CL_max=1.4, altitudes=None,
                 aero_method="vlm") -> EnvelopeResult
```

Sweeps performance across altitudes (default: 0–3000 m in 200 m steps). Fits the drag polar once at sea level and evaluates all performance metrics at each altitude.

#### `EnvelopeResult` dataclass

| Field | Type | Description |
|---|---|---|
| `altitudes` | `ndarray` | Altitude array [m] |
| `stall_speeds` | `ndarray` | Stall speed at each altitude [m/s] |
| `best_endurance_speeds` | `ndarray` | Best endurance speed [m/s] |
| `best_range_speeds` | `ndarray` | Best range speed [m/s] |
| `max_speeds` | `ndarray` | Max level speed [m/s] (NaN if cannot fly) |
| `max_rocs` | `ndarray` | Maximum rate of climb [m/s] |
| `best_climb_speeds` | `ndarray` | V_y at each altitude [m/s] |
| `ld_max` | `float` | Maximum L/D at sea level |
| `endurance_s` | `float` | Max endurance [s] |
| `range_m` | `float` | Max range [m] |
| `service_ceiling` | `float` | Altitude where ROC = 0.5 m/s [m] |
| `absolute_ceiling` | `float` | Altitude where ROC = 0 [m] |
| `best_glide_ratio` | `float` | L/D max |
| `best_glide_speed` | `float` | Best glide speed at SL [m/s] |
| `min_sink_rate` | `float` | Min sink rate [m/s] |
| `polar` | `DragPolar` | Fitted drag polar |

| Method | Description |
|---|---|
| `report()` | Formatted text summary of all performance metrics |
| `plot()` | 4-panel figure: power curves, speed envelope, ROC vs altitude, summary |

---

## Mission segments

Module: `aerisplane.mission.segments`

| Segment | Required fields | Optional (defaults) | Description |
|---|---|---|---|
| `Climb` | `to_altitude`, `climb_rate`, `velocity` | `name="climb"` | Constant-rate climb |
| `Cruise` | `distance`, `velocity` | `altitude=100.0` | Steady level cruise |
| `Loiter` | `duration`, `velocity` | `altitude=100.0` | Circling/holding at altitude |
| `Return` | `distance`, `velocity` | `altitude=100.0` | Return leg (like Cruise) |
| `Descent` | `to_altitude` | `descent_rate=2.0`, `velocity=15.0` | Partial-power descent |

```python
Mission(segments=[...], start_altitude=0.0)
```

Ordered sequence of segments. Each segment's start altitude is the previous segment's end altitude.

---

## Mission result

Module: `aerisplane.mission.result`

#### `MissionResult` dataclass

| Field | Type | Description |
|---|---|---|
| `total_energy` | `float` | Total energy consumed [J] |
| `total_time` | `float` | Total mission duration [s] |
| `total_distance` | `float` | Total horizontal distance [m] |
| `battery_energy_available` | `float` | Battery capacity [J] |
| `energy_margin` | `float` | Fraction remaining (0=empty, 1=full) |
| `feasible` | `bool` | True if enough battery for the mission |
| `segments` | `list[SegmentResult]` | Per-segment breakdown |

#### `SegmentResult` dataclass

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Segment name |
| `duration` | `float` | Duration [s] |
| `distance` | `float` | Horizontal distance [m] |
| `energy` | `float` | Energy consumed [J] |
| `avg_power` | `float` | Average electrical power [W] |
| `avg_speed` | `float` | Average airspeed [m/s] |
| `altitude_start` | `float` | Start altitude [m] |
| `altitude_end` | `float` | End altitude [m] |

| Method | Description |
|---|---|
| `report()` | Tabulated energy budget with totals and feasibility |
| `plot()` | 2-panel figure: energy bars + altitude profile |

---

## Dependencies

```
weights.analyze() → WeightResult (mass, CG)
       ↓
mission.performance() → DragPolar (3 aero calls)
       ↓
mission.envelope() → EnvelopeResult (no extra aero calls)
mission.analyze() → MissionResult (no extra aero calls)
```

The user must run `weights.analyze()` first. The drag polar is fitted once (3 VLM evaluations), then all subsequent performance calculations are purely analytic — no additional aero calls needed.
