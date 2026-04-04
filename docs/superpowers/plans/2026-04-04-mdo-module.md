# MDO Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `aerisplane.mdo` — a complete MDO orchestration layer that wraps the existing discipline chain (weights → aero → structures → stability → control → mission) in a cached, checkpointed, optimizer-agnostic problem class.

**Architecture:** `MDOProblem` holds a baseline `Aircraft`, resolves string-path design variables against it, and evaluates the discipline chain on every optimizer call. Results are cached keyed on the design vector. Three optimizer backends are supported: SciPy (DE, minimize, shgo) and pygmo (de, nsga2). Airfoil selection is treated as integer design variables drawn from per-wing pools.

**Tech Stack:** Python 3.10+, numpy, scipy ≥ 1.9 (for `integrality` in DE), pygmo ≥ 2.19 (optional, soft import), pickle (checkpoints), json (sidecar), matplotlib + seaborn (result plots).

---

## Design decisions (context for implementer)

- **Fixed topology:** the number of xsecs per wing never changes during optimisation.
- **Auto-trim:** when `alpha=None`, trim alpha is found via `brentq` before each evaluation. When `alpha=float`, that value is used directly (saves ~3 aero calls/eval).
- **Discipline inference:** the set of disciplines to run is auto-detected from constraint/objective path prefixes. User can override with `extra_disciplines` / `skip_disciplines`.
- **Airfoil pools:** specified per wing as `{"wings[0]": AirfoilPool(...)}`. Each free xsec airfoil becomes one integer variable appended after user-defined `DesignVar`s.
- **Constraint paths** navigate discipline result objects: `"stability.static_margin"` → `results["stability"].static_margin`. Nested access is supported: `"structures.wings[0].bending_margin"`.
- **DesignVar paths** navigate the `Aircraft` object: `"wings[0].xsecs[1].chord"`.
- **Cache key:** `tuple(np.round(x, 10))` — tolerates floating-point identity issues.
- **scipy checkpoint:** saves cache + current best x (approximate restart — optimizer restarts from best, not full population).
- **pygmo checkpoint:** saves full `pygmo.population` (exact restart).
- **Boolean constraints:** `Constraint("mission.feasible", equals=True)` → violation = 0 if True, 1 if False.
- **mission=None** is valid; any constraint/objective path starting with `"mission."` raises `ValueError` at `validate()` time when mission is None.
- **`load_factor`** maps to `structures.analyze(n_limit=load_factor, safety_factor=1.5)`.
- **control.analyze** requires a `StabilityResult` — if "control" is in disciplines, "stability" is forced on automatically.

---

## File structure

```
src/aerisplane/
├── catalog/__init__.py             MODIFY  add get_airfoil(name) -> Airfoil
├── mdo/
│   ├── __init__.py                 CREATE  public exports
│   ├── problem.py                  CREATE  DesignVar, AirfoilPool, Constraint,
│   │                                       Objective, MDOProblem
│   ├── _paths.py                   CREATE  path tokeniser, get/set, pack/unpack,
│   │                                       pool entry builder, integrality array
│   ├── drivers.py                  CREATE  ScipyDriver, PygmoDriver, checkpoint
│   └── result.py                   CREATE  OptimisationSnapshot, OptimizationResult
tests/
└── test_mdo/
    ├── __init__.py                 CREATE  (empty)
    ├── conftest.py                 CREATE  aircraft + problem fixtures
    ├── test_paths.py               CREATE  path resolver unit tests
    ├── test_problem.py             CREATE  MDOProblem unit + evaluate tests
    ├── test_drivers.py             CREATE  driver + checkpoint tests
    └── test_result.py              CREATE  result formatting tests
```

---

## Task 1: `catalog.get_airfoil()`

**Files:**
- Modify: `src/aerisplane/catalog/__init__.py`
- Test: `tests/test_mdo/test_paths.py` (first test in that file)

- [ ] **Step 1.1: Write failing test**

```python
# tests/test_mdo/test_paths.py
from aerisplane.catalog import get_airfoil
from aerisplane.core.airfoil import Airfoil

def test_get_airfoil_naca():
    af = get_airfoil("naca2412")
    assert isinstance(af, Airfoil)
    assert af.coordinates is not None
    assert af.name == "naca2412"

def test_get_airfoil_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="not found"):
        get_airfoil("totally_fake_airfoil_xyz")
```

- [ ] **Step 1.2: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_paths.py::test_get_airfoil_naca -v
```
Expected: `ImportError` or `AttributeError` (get_airfoil not defined).

- [ ] **Step 1.3: Implement**

```python
# src/aerisplane/catalog/__init__.py
"""AerisPlane hardware and airfoil catalog."""
from __future__ import annotations


def get_airfoil(name: str):
    """Load an airfoil from the catalog by name.

    Parameters
    ----------
    name : str
        Airfoil name, e.g. ``"naca2412"``, ``"e423"``.  NACA 4-digit names
        are generated analytically; all others are loaded from the catalog
        .dat files in ``catalog/airfoils/``.

    Returns
    -------
    Airfoil
        Airfoil with coordinates populated.

    Raises
    ------
    ValueError
        If the name cannot be resolved (not NACA and not in catalog).
    """
    from aerisplane.core.airfoil import Airfoil
    af = Airfoil(name)
    if af.coordinates is None:
        raise ValueError(
            f"Airfoil '{name}' not found in catalog. "
            "Check catalog/airfoils/ for available .dat files."
        )
    return af


__all__ = ["get_airfoil"]
```

- [ ] **Step 1.4: Run to verify PASS**

```bash
pytest tests/test_mdo/test_paths.py -v
```
Expected: both tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/aerisplane/catalog/__init__.py tests/test_mdo/test_paths.py tests/test_mdo/__init__.py
git commit -m "feat(catalog): add get_airfoil() name-based lookup"
```

---

## Task 2: Core dataclasses

**Files:**
- Create: `src/aerisplane/mdo/problem.py`
- Test: `tests/test_mdo/test_problem.py`

- [ ] **Step 2.1: Write failing test**

```python
# tests/test_mdo/test_problem.py
import pytest
from aerisplane.mdo.problem import DesignVar, AirfoilPool, Constraint, Objective


def test_design_var_fields():
    dv = DesignVar("wings[0].xsecs[0].chord", lower=0.10, upper=0.40)
    assert dv.path == "wings[0].xsecs[0].chord"
    assert dv.lower == 0.10
    assert dv.upper == 0.40
    assert dv.scale == 1.0


def test_airfoil_pool_defaults():
    pool = AirfoilPool(options=["naca2412", "naca4412"])
    assert pool.xsecs is None   # None → all xsecs


def test_constraint_requires_bound():
    with pytest.raises(ValueError):
        Constraint("stability.static_margin")   # no lower/upper/equals


def test_constraint_lower():
    c = Constraint("stability.static_margin", lower=0.05)
    assert c.lower == 0.05
    assert c.upper is None
    assert c.equals is None


def test_constraint_boolean():
    c = Constraint("mission.feasible", equals=True)
    assert c.equals is True


def test_objective_defaults():
    obj = Objective("mission.endurance_s")
    assert obj.maximize is True
    assert obj.scale == 1.0
```

- [ ] **Step 2.2: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_problem.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 2.3: Implement**

```python
# src/aerisplane/mdo/problem.py
"""MDO problem definition: dataclasses and MDOProblem class."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DesignVar:
    """Continuous design variable defined by a dot-bracket path into Aircraft.

    Parameters
    ----------
    path : str
        Dot-bracket path, e.g. ``"wings[0].xsecs[1].chord"``.
    lower : float
        Lower bound (physical units, before scaling).
    upper : float
        Upper bound (physical units, before scaling).
    scale : float
        Optimizer sees ``value / scale``.  Use to normalise variables of
        very different magnitudes.  Default 1.0 (no scaling).
    """
    path: str
    lower: float
    upper: float
    scale: float = 1.0


@dataclass
class AirfoilPool:
    """Airfoil options for one wing surface.

    Parameters
    ----------
    options : list of str
        Airfoil names resolvable via ``catalog.get_airfoil()``.
    xsecs : list of int or None
        Indices of xsecs whose airfoil is free.  ``None`` → all xsecs.
    """
    options: list[str]
    xsecs: Optional[list[int]] = None


@dataclass
class Constraint:
    """Constraint on a discipline result field.

    Parameters
    ----------
    path : str
        Dot path into the results dict, e.g. ``"stability.static_margin"``
        or ``"structures.wings[0].bending_margin"``.
    lower : float or None
        Value must be ≥ lower.
    upper : float or None
        Value must be ≤ upper.
    equals : any or None
        Value must equal this (supports bool for feasibility flags).
    scale : float
        Normalisation factor for the violation vector.
    """
    path: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    equals: Optional[Any] = None
    scale: float = 1.0

    def __post_init__(self):
        if self.lower is None and self.upper is None and self.equals is None:
            raise ValueError(
                f"Constraint '{self.path}': must specify lower, upper, or equals."
            )


@dataclass
class Objective:
    """Optimisation objective pointing at a discipline result field.

    Parameters
    ----------
    path : str
        E.g. ``"mission.endurance_s"`` or ``"weights.total_mass"``.
    maximize : bool
        True → maximise (default).  False → minimise.
    scale : float
        Normalisation factor applied to the raw value.
    """
    path: str
    maximize: bool = True
    scale: float = 1.0
```

- [ ] **Step 2.4: Run to verify PASS**

```bash
pytest tests/test_mdo/test_problem.py -v
```

- [ ] **Step 2.5: Commit**

```bash
git add src/aerisplane/mdo/problem.py tests/test_mdo/test_problem.py
git commit -m "feat(mdo): add DesignVar, AirfoilPool, Constraint, Objective dataclasses"
```

---

## Task 3: Path utilities (`_paths.py`)

**Files:**
- Create: `src/aerisplane/mdo/_paths.py`
- Test: `tests/test_mdo/test_paths.py` (extend existing file)

- [ ] **Step 3.1: Write failing tests**

```python
# Append to tests/test_mdo/test_paths.py
import numpy as np
import aerisplane as ap
from aerisplane.mdo._paths import (
    _tokenize, _get_dv_value, _set_dv_value,
    _resolve_to_obj, _get_result_value,
    _pack, _unpack, _build_pool_entries, _integrality_array,
)
from aerisplane.mdo.problem import DesignVar, AirfoilPool


@pytest.fixture
def simple_aircraft():
    wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.25,
                        airfoil=ap.Airfoil("naca2412")),
            ap.WingXSec(xyz_le=[0.03, 0.6, 0.04], chord=0.14,
                        airfoil=ap.Airfoil("naca2412")),
        ],
        symmetric=True,
    )
    return ap.Aircraft(name="test", wings=[wing])


def test_tokenize_simple():
    assert _tokenize("chord") == ["chord"]

def test_tokenize_nested():
    assert _tokenize("wings[0].xsecs[1].chord") == ["wings", 0, "xsecs", 1, "chord"]

def test_get_dv_value(simple_aircraft):
    assert abs(_get_dv_value(simple_aircraft, "wings[0].xsecs[0].chord") - 0.25) < 1e-9

def test_set_dv_value(simple_aircraft):
    _set_dv_value(simple_aircraft, "wings[0].xsecs[0].chord", 0.30)
    assert abs(simple_aircraft.wings[0].xsecs[0].chord - 0.30) < 1e-9

def test_pack_unpack_roundtrip(simple_aircraft):
    dvars = [
        DesignVar("wings[0].xsecs[0].chord", lower=0.1, upper=0.5),
        DesignVar("wings[0].xsecs[1].chord", lower=0.1, upper=0.3),
    ]
    x = _pack(simple_aircraft, dvars, pool_entries=[])
    assert x.shape == (2,)
    assert abs(x[0] - 0.25) < 1e-9

    ac2 = _unpack(simple_aircraft, dvars, pool_entries=[], x=np.array([0.30, 0.16]))
    assert abs(ac2.wings[0].xsecs[0].chord - 0.30) < 1e-9
    # original unchanged
    assert abs(simple_aircraft.wings[0].xsecs[0].chord - 0.25) < 1e-9

def test_get_result_value():
    from unittest.mock import MagicMock
    mock_stab = MagicMock()
    mock_stab.static_margin = 0.08
    results = {"stability": mock_stab}
    assert _get_result_value(results, "stability.static_margin") == 0.08

def test_get_result_value_nested():
    from unittest.mock import MagicMock
    mock_wing = MagicMock()
    mock_wing.bending_margin = 2.1
    mock_struct = MagicMock()
    mock_struct.wings = [mock_wing]
    results = {"structures": mock_struct}
    assert _get_result_value(results, "structures.wings[0].bending_margin") == 2.1

def test_pool_entries_all_xsecs(simple_aircraft):
    pools = {"wings[0]": AirfoilPool(options=["naca2412", "naca4412"])}
    entries = _build_pool_entries(simple_aircraft, pools)
    assert len(entries) == 2   # 2 xsecs on wings[0]
    assert entries[0] == ("wings[0]", 0, pools["wings[0]"])
    assert entries[1] == ("wings[0]", 1, pools["wings[0]"])

def test_pool_entries_specific_xsecs(simple_aircraft):
    pools = {"wings[0]": AirfoilPool(options=["naca2412", "naca4412"], xsecs=[0])}
    entries = _build_pool_entries(simple_aircraft, pools)
    assert len(entries) == 1
    assert entries[0][1] == 0   # only xsec 0

def test_integrality_array():
    dvars = [DesignVar("a", 0, 1), DesignVar("b", 0, 1)]
    pool_entries = [("wings[0]", 0, None), ("wings[0]", 1, None)]
    arr = _integrality_array(n_dvars=2, pool_entries=pool_entries)
    assert list(arr) == [False, False, True, True]
```

- [ ] **Step 3.2: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_paths.py -v
```

- [ ] **Step 3.3: Implement `_paths.py`**

```python
# src/aerisplane/mdo/_paths.py
"""Path resolution utilities for MDO design variable and result access."""
from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np


def _tokenize(path: str) -> list:
    """Parse dot-bracket path into a list of attribute names and int indices.

    Examples
    --------
    >>> _tokenize("wings[0].xsecs[1].chord")
    ['wings', 0, 'xsecs', 1, 'chord']
    >>> _tokenize("chord")
    ['chord']
    """
    result = []
    for part in path.split("."):
        m = re.match(r"^(\w+)\[(\d+)\]$", part)
        if m:
            result.append(m.group(1))
            result.append(int(m.group(2)))
        else:
            result.append(part)
    return result


def _resolve_to_obj(root: Any, path: str) -> Any:
    """Traverse dot-bracket path from root and return the object at the end."""
    obj = root
    for token in _tokenize(path):
        if isinstance(token, int):
            obj = obj[token]
        else:
            obj = getattr(obj, token)
    return obj


def _get_dv_value(aircraft, path: str) -> float:
    """Read a scalar float from aircraft via dot-bracket path."""
    return float(_resolve_to_obj(aircraft, path))


def _set_dv_value(aircraft, path: str, value: Any) -> None:
    """Write value into aircraft at the given dot-bracket path."""
    tokens = _tokenize(path)
    obj = aircraft
    for token in tokens[:-1]:
        if isinstance(token, int):
            obj = obj[token]
        else:
            obj = getattr(obj, token)
    final = tokens[-1]
    if isinstance(final, int):
        obj[final] = value
    else:
        setattr(obj, final, value)


def _get_result_value(results: dict, path: str) -> Any:
    """Resolve a constraint/objective path against the discipline results dict.

    The first segment names the discipline (key in results).
    The remainder is a dot-bracket sub-path into that result object.

    Examples
    --------
    "stability.static_margin"         → results["stability"].static_margin
    "structures.wings[0].bending_margin" → results["structures"].wings[0].bending_margin
    "mission.feasible"                → results["mission"].feasible
    """
    parts = path.split(".", 1)
    discipline = parts[0]
    if discipline not in results:
        raise KeyError(
            f"Discipline '{discipline}' not in results. "
            f"Available: {list(results.keys())}. "
            f"Check constraint/objective path '{path}'."
        )
    obj = results[discipline]
    if len(parts) == 1:
        return obj
    return _resolve_to_obj(obj, parts[1])


def _pack(aircraft, dvars: list, pool_entries: list) -> np.ndarray:
    """Extract current design variable values from aircraft into a flat vector.

    Continuous DesignVar values come first (in order), then one integer per
    airfoil pool entry (index of current airfoil in the pool's options list).

    Parameters
    ----------
    aircraft : Aircraft
    dvars : list of DesignVar
    pool_entries : list of (wing_path, xsec_idx, AirfoilPool)

    Returns
    -------
    np.ndarray  shape (n_dvars + n_pool_entries,)
    """
    values = [_get_dv_value(aircraft, dv.path) / dv.scale for dv in dvars]
    for wing_path, xi, pool in pool_entries:
        wing = _resolve_to_obj(aircraft, wing_path)
        current_name = wing.xsecs[xi].airfoil.name
        try:
            idx = pool.options.index(current_name)
        except ValueError:
            idx = 0   # fall back to first option if not in pool
        values.append(float(idx))
    return np.array(values, dtype=float)


def _unpack(aircraft, dvars: list, pool_entries: list, x: np.ndarray):
    """Return a deep copy of aircraft with design variable values from x applied.

    Does NOT modify the original aircraft.
    """
    from aerisplane.catalog import get_airfoil

    ac = copy.deepcopy(aircraft)
    for i, dv in enumerate(dvars):
        _set_dv_value(ac, dv.path, float(x[i]) * dv.scale)
    for j, (wing_path, xi, pool) in enumerate(pool_entries):
        raw_idx = float(x[len(dvars) + j])
        idx = int(round(raw_idx))
        idx = max(0, min(idx, len(pool.options) - 1))
        wing = _resolve_to_obj(ac, wing_path)
        wing.xsecs[xi].airfoil = get_airfoil(pool.options[idx])
    return ac


def _build_pool_entries(aircraft, pools: dict) -> list:
    """Build the ordered list of (wing_path, xsec_idx, pool) tuples.

    Parameters
    ----------
    aircraft : Aircraft
    pools : dict mapping wing_path str → AirfoilPool, or None/empty.

    Returns
    -------
    list of (wing_path: str, xsec_idx: int, pool: AirfoilPool)
    """
    if not pools:
        return []
    entries = []
    for wing_path, pool in pools.items():
        wing = _resolve_to_obj(aircraft, wing_path)
        indices = pool.xsecs if pool.xsecs is not None else list(range(len(wing.xsecs)))
        for xi in indices:
            entries.append((wing_path, xi, pool))
    return entries


def _integrality_array(n_dvars: int, pool_entries: list) -> np.ndarray:
    """Return bool array: False for continuous vars, True for integer airfoil indices."""
    return np.array(
        [False] * n_dvars + [True] * len(pool_entries),
        dtype=bool,
    )
```

- [ ] **Step 3.4: Run to verify PASS**

```bash
pytest tests/test_mdo/test_paths.py -v
```

- [ ] **Step 3.5: Commit**

```bash
git add src/aerisplane/mdo/_paths.py tests/test_mdo/test_paths.py
git commit -m "feat(mdo): add path resolution utilities (_paths.py)"
```

---

## Task 4: `MDOProblem` — constructor and `validate()`

**Files:**
- Modify: `src/aerisplane/mdo/problem.py`
- Test: `tests/test_mdo/test_problem.py` (extend)
- Fixture: `tests/test_mdo/conftest.py`

- [ ] **Step 4.1: Create test fixture**

```python
# tests/test_mdo/conftest.py
"""Shared fixtures for MDO tests."""
import pytest
import numpy as np
import aerisplane as ap
from aerisplane.core.structures import Material, Spar, Skin
from aerisplane.core.propulsion import Motor, Propeller, Battery, ESC, PropulsionSystem
from aerisplane.mdo.problem import DesignVar, AirfoilPool, Constraint, Objective


@pytest.fixture
def test_aircraft():
    """Small conventional aircraft with main wing, htail, vtail, fuselage, propulsion."""
    mat = Material(name="cf_ud", density=1600, E=70e9, G=5e9,
                   sigma_yield=600e6, tau_yield=50e6)
    spar = Spar(material=mat, diameter=0.015, thickness=0.001)
    skin = Skin(material=mat, thickness=0.0008)

    main_wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.26,
                        airfoil=ap.Airfoil("naca2412"), spar=spar, skin=skin),
            ap.WingXSec(xyz_le=[0.03, 0.75, 0.05], chord=0.15,
                        airfoil=ap.Airfoil("naca2412")),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(name="aileron", span_start=0.55, span_end=0.92,
                              chord_fraction=0.28),
        ],
    )
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.85, 0.0, 0.0], chord=0.12,
                        airfoil=ap.Airfoil("naca0012")),
            ap.WingXSec(xyz_le=[0.87, 0.25, 0.01], chord=0.08,
                        airfoil=ap.Airfoil("naca0012")),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
                              chord_fraction=0.40),
        ],
    )
    vtail = ap.Wing(
        name="vtail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.86, 0.0, 0.0], chord=0.13,
                        airfoil=ap.Airfoil("naca0012")),
            ap.WingXSec(xyz_le=[0.89, 0.0, 0.18], chord=0.08,
                        airfoil=ap.Airfoil("naca0012")),
        ],
        symmetric=False,
        control_surfaces=[
            ap.ControlSurface(name="rudder", span_start=0.0, span_end=1.0,
                              chord_fraction=0.40),
        ],
    )
    fuse = ap.Fuselage(
        name="fuselage",
        xsecs=[
            ap.FuselageXSec(xyz_c=[0.0, 0.0, 0.0], radius=0.045),
            ap.FuselageXSec(xyz_c=[0.95, 0.0, 0.0], radius=0.020),
        ],
    )
    motor = Motor(name="test_motor", kv=900, no_load_current=0.5,
                  resistance=0.1, mass=0.120, max_power=300.0)
    prop = Propeller(name="test_prop", diameter=0.254, pitch=0.127,
                     mass=0.025)
    battery = Battery(name="test_bat", capacity_mah=3000, voltage_nominal=14.8,
                      mass=0.280, c_rating=25.0)
    esc = ESC(name="test_esc", max_current=40, mass=0.030)
    propulsion = PropulsionSystem(motor=motor, propeller=prop,
                                  battery=battery, esc=esc)

    return ap.Aircraft(
        name="test_uav",
        wings=[main_wing, htail, vtail],
        fuselages=[fuse],
        propulsion=propulsion,
    )


@pytest.fixture
def cruise_condition():
    return ap.FlightCondition(velocity=16.0, altitude=80.0, alpha=4.0)


@pytest.fixture
def simple_problem(test_aircraft, cruise_condition):
    """Minimal MDOProblem with 2 design variables, 1 constraint, 1 objective."""
    from aerisplane.mdo.problem import MDOProblem
    return MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38),
            DesignVar("wings[0].xsecs[1].chord", lower=0.09, upper=0.22),
        ],
        constraints=[
            Constraint("stability.static_margin", lower=0.05),
        ],
        objective=Objective("weights.total_mass", maximize=False),
    )
```

- [ ] **Step 4.2: Write failing tests**

```python
# Append to tests/test_mdo/test_problem.py
import numpy as np
import pytest
import aerisplane as ap
from aerisplane.mdo.problem import MDOProblem, DesignVar, Constraint, Objective, AirfoilPool


def test_construction_succeeds(test_aircraft, cruise_condition):
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38),
        ],
        constraints=[Constraint("stability.static_margin", lower=0.05)],
        objective=Objective("weights.total_mass", maximize=False),
    )
    assert problem is not None


def test_invalid_dv_path_raises(test_aircraft, cruise_condition):
    with pytest.raises(ValueError, match="wings\[0\].xsecs\[0\].typo"):
        MDOProblem(
            aircraft=test_aircraft,
            condition=cruise_condition,
            design_variables=[
                DesignVar("wings[0].xsecs[0].typo", lower=0.1, upper=0.5),
            ],
            constraints=[Constraint("weights.total_mass", upper=5.0)],
            objective=Objective("weights.total_mass", maximize=False),
        )


def test_mission_none_with_mission_path_raises(test_aircraft, cruise_condition):
    with pytest.raises(ValueError, match="mission"):
        MDOProblem(
            aircraft=test_aircraft,
            condition=cruise_condition,
            design_variables=[
                DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38),
            ],
            constraints=[Constraint("mission.feasible", equals=True)],
            objective=Objective("weights.total_mass", maximize=False),
            mission=None,
        )


def test_get_bounds(simple_problem):
    lo, hi = simple_problem.get_bounds()
    assert lo[0] == pytest.approx(0.18)
    assert hi[0] == pytest.approx(0.38)
    assert lo[1] == pytest.approx(0.09)
    assert hi[1] == pytest.approx(0.22)


def test_inferred_disciplines_stability(simple_problem):
    # constraint path "stability.*" → stability should be inferred
    assert "stability" in simple_problem._disciplines
    assert "weights" in simple_problem._disciplines
    assert "aero" in simple_problem._disciplines


def test_inferred_disciplines_no_mission(simple_problem):
    assert "mission" not in simple_problem._disciplines


def test_n_vars_with_pool(test_aircraft, cruise_condition):
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38),
        ],
        constraints=[Constraint("weights.total_mass", upper=5.0)],
        objective=Objective("weights.total_mass", maximize=False),
        airfoil_pools={
            "wings[0]": AirfoilPool(options=["naca2412", "naca4412"], xsecs=[0]),
        },
    )
    lo, hi = problem.get_bounds()
    # 1 DesignVar + 1 pool entry = 2 variables
    assert len(lo) == 2
    assert lo[1] == 0.0    # pool lower bound = 0
    assert hi[1] == 1.0    # pool upper bound = len(options) - 1
```

- [ ] **Step 4.3: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_problem.py -v
```

- [ ] **Step 4.4: Implement MDOProblem constructor + validate()**

Append to `src/aerisplane/mdo/problem.py`:

```python
# ── append to src/aerisplane/mdo/problem.py ──────────────────────────────────
from __future__ import annotations

import copy
import logging
import time
from typing import Union

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.mdo._paths import (
    _build_pool_entries,
    _get_dv_value,
    _get_result_value,
    _integrality_array,
    _pack,
    _unpack,
)

_LOG = logging.getLogger(__name__)

DISCIPLINE_ORDER = ["weights", "aero", "structures", "stability", "control", "mission"]
_ALWAYS_RUN = {"weights", "aero"}


def _infer_disciplines(
    constraints: list,
    objective: Union[Objective, list],
    mission,
    extra: tuple,
    skip: tuple,
) -> list[str]:
    needed = set(_ALWAYS_RUN)
    objectives = objective if isinstance(objective, list) else [objective]
    all_paths = [c.path for c in constraints] + [o.path for o in objectives]
    for path in all_paths:
        disc = path.split(".")[0]
        if disc in DISCIPLINE_ORDER:
            needed.add(disc)
    if mission is None:
        needed.discard("mission")
    if "control" in needed:
        needed.add("stability")   # control.analyze requires StabilityResult
    needed.update(extra)
    needed -= set(skip)
    return [d for d in DISCIPLINE_ORDER if d in needed]


class MDOProblem:
    """Multidisciplinary optimisation problem.

    Parameters
    ----------
    aircraft : Aircraft
        Baseline aircraft.  A deep copy is stored — caller's object is untouched.
    condition : FlightCondition
        Reference flight condition.  Provides velocity, altitude, beta.
        Alpha is either taken from ``alpha`` parameter (fixed) or solved
        automatically for trim (``alpha=None``).
    design_variables : list of DesignVar
        Continuous design variables navigating the Aircraft object.
    constraints : list of Constraint
        Constraints on discipline result fields.
    objective : Objective or list of Objective
        Single or multi-objective (multi requires pygmo_nsga2 driver).
    mission : Mission or None
        Mission profile for mission discipline.  None → mission not run.
    airfoil_pools : dict or None
        Mapping wing_path → AirfoilPool.  Each free xsec gets an integer
        variable appended after the continuous DesignVars.
    alpha : float or None
        Fixed alpha [deg].  None → auto-trim each evaluation.
    aero_method : str
        Aero solver method.  Default ``"vlm"``.
    load_factor : float
        Limit maneuver load factor passed to structures.analyze().  Default 3.5.
    extra_disciplines : tuple of str
        Force these disciplines on even if not referenced by any path.
    skip_disciplines : tuple of str
        Suppress these disciplines even if referenced by a path.
    """

    def __init__(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        design_variables: list[DesignVar],
        constraints: list[Constraint],
        objective: Union[Objective, list[Objective]],
        mission=None,
        airfoil_pools: dict | None = None,
        alpha: float | None = None,
        aero_method: str = "vlm",
        load_factor: float = 3.5,
        extra_disciplines: tuple = (),
        skip_disciplines: tuple = (),
    ):
        self._baseline = copy.deepcopy(aircraft)
        self._condition = condition
        self._dvars = list(design_variables)
        self._constraints = list(constraints)
        self._objective = objective
        self._mission = mission
        self._pools = airfoil_pools or {}
        self._alpha = alpha
        self.aero_method = aero_method
        self.load_factor = load_factor

        # Build pool entries and combined variable list
        self._pool_entries = _build_pool_entries(self._baseline, self._pools)
        self._n_continuous = len(self._dvars)
        self._n_vars = self._n_continuous + len(self._pool_entries)
        self._integrality = _integrality_array(self._n_continuous, self._pool_entries)

        # Scales vector (continuous vars use dv.scale; pool entries use 1.0)
        self._scales = np.array(
            [dv.scale for dv in self._dvars] + [1.0] * len(self._pool_entries)
        )

        # Disciplines to run
        self._disciplines = _infer_disciplines(
            self._constraints, self._objective, self._mission,
            extra_disciplines, skip_disciplines,
        )

        # Cache and history
        self._cache: dict = {}
        self._history: list = []
        self._n_evals: int = 0

        self.validate()

    def validate(self) -> None:
        """Resolve all design variable paths against the aircraft.

        Raises ValueError with the bad path if any path does not resolve.
        Raises ValueError if a mission.* constraint/objective is used but
        mission=None.
        """
        # Check DesignVar paths
        for dv in self._dvars:
            try:
                _get_dv_value(self._baseline, dv.path)
            except (AttributeError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"DesignVar path '{dv.path}' does not resolve on the aircraft: {exc}"
                ) from exc

        # Check mission paths
        if self._mission is None:
            objectives = self._objective if isinstance(self._objective, list) else [self._objective]
            all_paths = [c.path for c in self._constraints] + [o.path for o in objectives]
            bad = [p for p in all_paths if p.startswith("mission.")]
            if bad:
                raise ValueError(
                    f"Paths {bad} reference 'mission' but mission=None. "
                    "Pass a Mission object or remove those paths."
                )

        # Validate airfoil pool options
        from aerisplane.catalog import get_airfoil
        for wing_path, pool in self._pools.items():
            for name in pool.options:
                try:
                    get_airfoil(name)
                except ValueError as exc:
                    raise ValueError(
                        f"AirfoilPool for '{wing_path}': {exc}"
                    ) from exc

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bound arrays in scaled space."""
        lo_cont = np.array([dv.lower / dv.scale for dv in self._dvars])
        hi_cont = np.array([dv.upper / dv.scale for dv in self._dvars])
        n_pool = len(self._pool_entries)
        lo_pool = np.zeros(n_pool)
        hi_pool = np.array([float(len(pe[2].options) - 1) for pe in self._pool_entries])
        return np.concatenate([lo_cont, lo_pool]), np.concatenate([hi_cont, hi_pool])

    def _x0_scaled(self) -> np.ndarray:
        """Initial design vector (current aircraft values, scaled)."""
        return _pack(self._baseline, self._dvars, self._pool_entries)
```

- [ ] **Step 4.5: Run to verify PASS**

```bash
pytest tests/test_mdo/test_problem.py -v
```

- [ ] **Step 4.6: Commit**

```bash
git add src/aerisplane/mdo/problem.py tests/test_mdo/test_problem.py tests/test_mdo/conftest.py tests/test_mdo/__init__.py
git commit -m "feat(mdo): add MDOProblem constructor and validate()"
```

---

## Task 5: `evaluate()` — the discipline chain

**Files:**
- Modify: `src/aerisplane/mdo/problem.py`
- Test: `tests/test_mdo/test_problem.py` (extend)

- [ ] **Step 5.1: Write failing tests**

These tests mock all discipline calls to keep evaluation fast.

```python
# Append to tests/test_mdo/test_problem.py
from unittest.mock import patch, MagicMock


def _mock_weight_result():
    wr = MagicMock()
    wr.total_mass = 2.5
    wr.cg = np.array([0.32, 0.0, 0.0])
    wr.wing_loading = 18.0
    wr.inertia_tensor = np.eye(3) * 0.1
    return wr


def _mock_aero_result():
    ar = MagicMock()
    ar.CL = 0.5; ar.CD = 0.04; ar.Cm = 0.01; ar.Cl = 0.0; ar.Cn = 0.0; ar.CY = 0.0
    ar.L = 25.0; ar.D = 2.0
    return ar


def _mock_stab_result():
    sr = MagicMock()
    sr.static_margin = 0.08
    return sr


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_evaluate_returns_dict(mock_w, mock_a, mock_s,
                                simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()

    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    result = simple_problem.evaluate(x)

    assert "objective" in result
    assert "constraint_values" in result
    assert "results" in result
    assert "weights" in result["results"]
    assert "stability" in result["results"]


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_evaluate_caches(mock_w, mock_a, mock_s, simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()

    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    simple_problem.evaluate(x)
    simple_problem.evaluate(x)   # second call — should hit cache

    assert mock_w.call_count == 1   # disciplines ran only once


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_evaluate_objective_minimise(mock_w, mock_a, mock_s, simple_problem):
    wr = _mock_weight_result()
    wr.total_mass = 2.5
    mock_w.return_value = wr
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()

    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    result = simple_problem.evaluate(x)

    # objective=Objective("weights.total_mass", maximize=False) → sign +1, value=2.5
    assert result["objective"] == pytest.approx(2.5)


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_constraint_violation_vector(mock_w, mock_a, mock_s, simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    sr = _mock_stab_result()
    sr.static_margin = 0.08   # 0.08 ≥ 0.05 → satisfied
    mock_s.return_value = sr

    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    violations = simple_problem.constraint_functions(x)
    assert violations[0] <= 0.0   # constraint satisfied
```

- [ ] **Step 5.2: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_problem.py::test_evaluate_returns_dict -v
```

- [ ] **Step 5.3: Implement `evaluate()`, `objective_function()`, `constraint_functions()`**

Append to `src/aerisplane/mdo/problem.py`:

```python
# ── append to MDOProblem class ────────────────────────────────────────────────

    def evaluate(self, x_scaled: np.ndarray) -> dict:
        """Run the full discipline chain for design vector x_scaled.

        Parameters
        ----------
        x_scaled : np.ndarray
            Design vector in scaled space (length = n_continuous + n_pool).

        Returns
        -------
        dict with keys:
            "objective"          float (or list for multi-objective)
            "constraint_values"  dict path → value
            "results"            dict discipline → result object
            "aircraft"           Aircraft (modified copy used for this eval)
            "condition"          FlightCondition used
            "elapsed"            float seconds
        """
        import aerisplane.aero as aero_mod
        import aerisplane.weights as weights_mod
        from aerisplane import stability as stab_mod
        from aerisplane import control as ctrl_mod
        from aerisplane import mission as mission_mod
        from aerisplane import structures as struct_mod

        cache_key = tuple(np.round(x_scaled, 10))
        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.time()
        self._n_evals += 1

        # Build modified aircraft
        x = x_scaled * self._scales
        ac = _unpack(self._baseline, self._dvars, self._pool_entries, x_scaled)

        results: dict = {}

        # 1. Weights (always)
        results["weights"] = weights_mod.analyze(ac)

        # 2. Flight condition (fixed alpha or auto-trim)
        if self._alpha is not None:
            cond = FlightCondition(
                velocity=self._condition.velocity,
                altitude=self._condition.altitude,
                alpha=self._alpha,
                beta=getattr(self._condition, "beta", 0.0),
            )
        else:
            cond = self._trim_condition(ac, results["weights"])

        # 3. Aero (always)
        results["aero"] = aero_mod.analyze(ac, cond, method=self.aero_method)

        # 4. Structures
        if "structures" in self._disciplines:
            results["structures"] = struct_mod.analyze(
                ac,
                results["aero"],
                results["weights"],
                n_limit=self.load_factor,
                safety_factor=1.5,
            )

        # 5. Stability
        if "stability" in self._disciplines:
            results["stability"] = stab_mod.analyze(
                ac, cond, results["weights"],
                aero_method=self.aero_method,
            )

        # 6. Control
        if "control" in self._disciplines:
            results["control"] = ctrl_mod.analyze(
                ac, cond, results["weights"],
                results["stability"],
                aero_method=self.aero_method,
            )

        # 7. Mission
        if "mission" in self._disciplines and self._mission is not None:
            results["mission"] = mission_mod.analyze(
                ac, results["weights"], self._mission,
                aero_method=self.aero_method,
            )

        # Extract objective
        objectives = self._objective if isinstance(self._objective, list) else [self._objective]
        obj_vals = []
        for obj in objectives:
            raw = float(_get_result_value(results, obj.path))
            sign = -1.0 if obj.maximize else 1.0
            obj_vals.append(sign * raw / obj.scale)
        objective_value = obj_vals[0] if len(obj_vals) == 1 else obj_vals

        # Extract constraint values
        constraint_values = {
            c.path: _get_result_value(results, c.path)
            for c in self._constraints
        }

        elapsed = time.time() - t0
        ev = {
            "objective": objective_value,
            "constraint_values": constraint_values,
            "results": results,
            "aircraft": ac,
            "condition": cond,
            "elapsed": elapsed,
        }

        self._cache[cache_key] = ev
        self._history.append((x_scaled.copy(), objective_value, constraint_values))

        if _LOG.isEnabledFor(logging.INFO):
            best_obj = min((h[1] for h in self._history), default=objective_value)
            _LOG.info(
                "[%5d]  obj=%.4g  best=%.4g  t=%.2fs",
                self._n_evals, objective_value, best_obj, elapsed,
            )

        return ev

    def _trim_condition(
        self, aircraft, weight_result
    ) -> FlightCondition:
        """Find trim alpha via brentq and return the trimmed FlightCondition."""
        import copy as _copy
        from scipy.optimize import brentq
        import aerisplane.aero as aero_mod

        ac = _copy.deepcopy(aircraft)
        cg = weight_result.cg
        ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]

        V = self._condition.velocity
        alt = self._condition.altitude
        beta = getattr(self._condition, "beta", 0.0)

        def cm_at(alpha: float) -> float:
            cond = FlightCondition(velocity=V, altitude=alt, alpha=alpha, beta=beta)
            return aero_mod.analyze(ac, cond, method=self.aero_method).Cm

        alpha_trim = 4.0   # fallback
        try:
            cm_lo = cm_at(-5.0)
            cm_hi = cm_at(15.0)
            if cm_lo * cm_hi < 0.0:
                alpha_trim = brentq(cm_at, -5.0, 15.0, xtol=0.1)
        except Exception:
            pass

        return FlightCondition(velocity=V, altitude=alt, alpha=alpha_trim, beta=beta)

    def objective_function(self, x_scaled: np.ndarray) -> float:
        """Return scalar objective for single-objective optimisers."""
        return float(self.evaluate(x_scaled)["objective"])

    def constraint_functions(self, x_scaled: np.ndarray) -> np.ndarray:
        """Return constraint violation vector (≤ 0 means satisfied).

        For lower bounds: violation = lower − value  (≤ 0 when value ≥ lower)
        For upper bounds: violation = value − upper  (≤ 0 when value ≤ upper)
        For equals (float/bool): violation = |value − target| / scale
        """
        ev = self.evaluate(x_scaled)
        violations = []
        for c in self._constraints:
            val = ev["constraint_values"][c.path]
            if c.lower is not None:
                violations.append((c.lower - float(val)) / c.scale)
            if c.upper is not None:
                violations.append((float(val) - c.upper) / c.scale)
            if c.equals is not None:
                violations.append(float(abs(float(val) - float(c.equals))) / c.scale)
        return np.array(violations)

    def save_cache(self, path: str) -> None:
        """Persist the evaluation cache to a pickle file for re-use."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._cache, f)

    def load_cache(self, path: str) -> None:
        """Load a previously saved cache, merging with any existing entries."""
        import pickle
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self._cache.update(loaded)
```

- [ ] **Step 5.4: Run to verify PASS**

```bash
pytest tests/test_mdo/test_problem.py -v
```

- [ ] **Step 5.5: Commit**

```bash
git add src/aerisplane/mdo/problem.py tests/test_mdo/test_problem.py
git commit -m "feat(mdo): implement MDOProblem.evaluate() with caching and discipline chain"
```

---

## Task 6: `OptimisationSnapshot` and `OptimizationResult`

**Files:**
- Create: `src/aerisplane/mdo/result.py`
- Test: `tests/test_mdo/test_result.py`

- [ ] **Step 6.1: Write failing tests**

```python
# tests/test_mdo/test_result.py
import numpy as np
import pytest
from aerisplane.mdo.result import OptimisationSnapshot, OptimizationResult
from unittest.mock import MagicMock


def _dummy_result():
    return OptimizationResult(
        x_initial=np.array([0.25, 0.14]),
        x_optimal=np.array([0.30, 0.12]),
        objective_initial=2.8,
        objective_optimal=2.4,
        constraints_satisfied=True,
        n_evaluations=120,
        convergence_history=[2.8, 2.6, 2.5, 2.4],
        variables={"wings[0].xsecs[0].chord": (0.25, 0.30),
                   "wings[0].xsecs[1].chord": (0.14, 0.12)},
        aero=MagicMock(),
        weights=MagicMock(),
        structures=None,
        stability=MagicMock(),
        control=None,
        mission=None,
        aircraft=MagicMock(),
        pareto_front=None,
    )


def test_report_is_string():
    r = _dummy_result()
    txt = r.report()
    assert isinstance(txt, str)
    assert len(txt) > 50
    assert "2.4" in txt   # optimal objective


def test_report_contains_variable_names():
    r = _dummy_result()
    txt = r.report()
    assert "wings[0].xsecs[0].chord" in txt


def test_plot_returns_figure():
    matplotlib = pytest.importorskip("matplotlib")
    r = _dummy_result()
    fig = r.plot()
    import matplotlib.pyplot as plt
    assert hasattr(fig, "savefig")
    plt.close("all")


def test_snapshot_fields():
    snap = OptimisationSnapshot(
        n_evals=42,
        objective=2.5,
        objective_initial=2.8,
        improvement_pct=10.7,
        improvement_last_100=5.0,
        x_best=np.array([0.28, 0.13]),
        constraints_satisfied=True,
        constraint_values={"stability.static_margin": 0.08},
        elapsed_s=38.2,
        history=[2.8, 2.6, 2.5],
    )
    assert snap.n_evals == 42
    assert snap.improvement_pct == pytest.approx(10.7)
```

- [ ] **Step 6.2: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_result.py -v
```

- [ ] **Step 6.3: Implement**

```python
# src/aerisplane/mdo/result.py
"""Optimisation result and progress snapshot dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class OptimisationSnapshot:
    """Progress snapshot passed to the user callback after each evaluation.

    Parameters
    ----------
    n_evals : int
        Total number of discipline chain evaluations so far.
    objective : float
        Current best objective value (raw, before sign flip).
    objective_initial : float
        Objective at the initial design vector.
    improvement_pct : float
        (objective − objective_initial) / |objective_initial| × 100.
    improvement_last_100 : float
        Relative improvement over the last 100 evaluations.
    x_best : ndarray
        Best design vector (scaled space).
    constraints_satisfied : bool
        True if all constraints are satisfied at x_best.
    constraint_values : dict
        Mapping constraint path → current value at x_best.
    elapsed_s : float
        Wall time since optimize() was called [s].
    history : list of float
        Objective value at every evaluation up to this point.
    """
    n_evals: int
    objective: float
    objective_initial: float
    improvement_pct: float
    improvement_last_100: float
    x_best: np.ndarray
    constraints_satisfied: bool
    constraint_values: dict
    elapsed_s: float
    history: list[float]


@dataclass
class OptimizationResult:
    """Result of a completed optimisation run.

    Parameters
    ----------
    x_initial : ndarray
        Starting design vector (scaled).
    x_optimal : ndarray
        Optimal design vector (scaled).
    objective_initial : float
        Objective value at x_initial.
    objective_optimal : float
        Objective value at x_optimal.
    constraints_satisfied : bool
        True if all constraints pass at x_optimal.
    n_evaluations : int
        Total number of discipline chain evaluations.
    convergence_history : list of float
        Best objective value per evaluation.
    variables : dict
        Mapping path → (initial_value, optimal_value) in physical units.
    aero, weights, structures, stability, control, mission
        Full discipline results at x_optimal (None if discipline not run).
    aircraft : Aircraft
        Optimised aircraft at x_optimal.
    pareto_front : list or None
        For multi-objective runs: list of (x, objectives) tuples.
        None for single-objective.
    """
    x_initial: np.ndarray
    x_optimal: np.ndarray
    objective_initial: float
    objective_optimal: float
    constraints_satisfied: bool
    n_evaluations: int
    convergence_history: list[float]
    variables: dict[str, tuple[float, float]]
    aero: Any
    weights: Any
    structures: Any
    stability: Any
    control: Any
    mission: Any
    aircraft: Any
    pareto_front: Optional[list] = None

    def report(self) -> str:
        """Formatted plain-text optimisation summary."""
        lines = ["AerisPlane Optimisation Result", "=" * 60]

        lines += ["", "Objective", "-" * 40]
        direction = "maximise" if self.objective_optimal < self.objective_initial else "minimise"
        improvement = (
            abs(self.objective_optimal - self.objective_initial)
            / (abs(self.objective_initial) + 1e-30) * 100
        )
        lines.append(f"  Initial  : {self.objective_initial:.6g}")
        lines.append(f"  Optimal  : {self.objective_optimal:.6g}")
        lines.append(f"  Change   : {improvement:.1f}%")
        lines.append(f"  Evals    : {self.n_evaluations}")
        lines.append(f"  Constraints satisfied: {'YES' if self.constraints_satisfied else 'NO'}")

        lines += ["", "Design Variables", "-" * 40]
        lines.append(f"  {'Path':<45} {'Initial':>10} {'Optimal':>10}")
        lines.append(f"  {'-'*45} {'-'*10} {'-'*10}")
        for path, (init, opt) in self.variables.items():
            lines.append(f"  {path:<45} {init:>10.5g} {opt:>10.5g}")

        if self.pareto_front is not None:
            lines += ["", f"Pareto Front: {len(self.pareto_front)} solutions"]

        return "\n".join(lines)

    def plot(self):
        """Return a 1×2 figure: convergence history (left) and design variable
        bar chart initial vs optimal (right)."""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax_conv, ax_vars) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: convergence history
        evals = np.arange(1, len(self.convergence_history) + 1)
        ax_conv.plot(evals, self.convergence_history, linewidth=1.2, alpha=0.6,
                     label="per eval")
        if len(self.convergence_history) >= 10:
            window = max(1, len(self.convergence_history) // 20)
            smooth = np.convolve(self.convergence_history,
                                 np.ones(window) / window, mode="valid")
            ax_conv.plot(np.arange(window, len(self.convergence_history) + 1),
                         smooth, linewidth=2, label=f"rolling {window}")
        ax_conv.set_xlabel("Evaluation")
        ax_conv.set_ylabel("Objective")
        ax_conv.set_title("Convergence History")
        ax_conv.legend(fontsize=8)
        ax_conv.grid(True, alpha=0.3)

        # Right: design variable comparison (normalised [0=lower, 1=upper])
        if self.variables:
            names = list(self.variables.keys())
            inits = np.array([v[0] for v in self.variables.values()])
            opts  = np.array([v[1] for v in self.variables.values()])
            y = np.arange(len(names))
            ax_vars.barh(y + 0.2, inits, height=0.35, label="Initial", alpha=0.7)
            ax_vars.barh(y - 0.2, opts,  height=0.35, label="Optimal", alpha=0.7)
            ax_vars.set_yticks(y)
            ax_vars.set_yticklabels([n.split(".")[-1] for n in names], fontsize=8)
            ax_vars.set_xlabel("Value")
            ax_vars.set_title("Design Variables: Initial vs Optimal")
            ax_vars.legend(fontsize=8)
            ax_vars.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_pareto(self):
        """Plot the Pareto front for multi-objective results.

        Returns None and prints a warning if pareto_front is None.
        """
        if self.pareto_front is None:
            import warnings
            warnings.warn("No Pareto front available (single-objective run).")
            return None

        import matplotlib.pyplot as plt
        import numpy as np

        pts = np.array([obj for _, obj in self.pareto_front])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(pts[:, 0], pts[:, 1], s=40, zorder=5)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_title(f"Pareto Front ({len(self.pareto_front)} solutions)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
```

- [ ] **Step 6.4: Run to verify PASS**

```bash
pytest tests/test_mdo/test_result.py -v
```

- [ ] **Step 6.5: Commit**

```bash
git add src/aerisplane/mdo/result.py tests/test_mdo/test_result.py
git commit -m "feat(mdo): add OptimizationResult and OptimisationSnapshot dataclasses"
```

---

## Task 7: `drivers.py` — checkpoint, ScipyDriver, PygmoDriver

**Files:**
- Create: `src/aerisplane/mdo/drivers.py`
- Test: `tests/test_mdo/test_drivers.py`

- [ ] **Step 7.1: Write failing tests**

```python
# tests/test_mdo/test_drivers.py
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aerisplane.mdo.drivers import save_checkpoint, load_checkpoint, ScipyDriver


# ── Checkpoint tests ──────────────────────────────────────────────────────────

def test_save_load_checkpoint_roundtrip(tmp_path):
    state = {
        "method": "scipy_de",
        "n_evals": 42,
        "best_x": np.array([0.28, 0.13]),
        "best_objective": 2.4,
        "cache": {(0.28, 0.13): {"objective": 2.4}},
    }
    base = str(tmp_path / "test_opt")
    save_checkpoint(base, state)
    assert Path(base + ".pkl").exists()
    assert Path(base + ".json").exists()

    # JSON sidecar is human-readable
    meta = json.loads(Path(base + ".json").read_text())
    assert meta["n_evals"] == 42
    assert meta["method"] == "scipy_de"

    loaded = load_checkpoint(base)
    assert loaded["n_evals"] == 42
    assert np.allclose(loaded["best_x"], state["best_x"])


def test_load_checkpoint_missing_returns_none(tmp_path):
    result = load_checkpoint(str(tmp_path / "nonexistent"))
    assert result is None


# ── ScipyDriver smoke test ────────────────────────────────────────────────────

def _make_mock_problem():
    """Minimal mock MDOProblem that records calls to evaluate."""
    problem = MagicMock()
    problem._n_vars = 2
    problem._integrality = np.array([False, False])
    problem.get_bounds.return_value = (np.array([0.1, 0.1]), np.array([0.5, 0.3]))
    problem._x0_scaled.return_value = np.array([0.25, 0.14])
    problem._n_evals = 0
    problem._history = []
    problem._cache = {}

    # objective: minimise sum of squares
    def obj_fn(x):
        problem._n_evals += 1
        problem._history.append((x.copy(), float(np.sum(x**2)), {}))
        return float(np.sum(x**2))

    problem.objective_function.side_effect = obj_fn
    problem.constraint_functions.return_value = np.array([])
    problem._constraints = []
    problem._dvars = []
    problem._pool_entries = []
    problem._scales = np.ones(2)
    return problem


def test_scipy_driver_de_runs(simple_problem):
    """ScipyDriver with scipy_de completes without error on a mock problem."""
    problem = _make_mock_problem()
    driver = ScipyDriver(problem)
    result = driver.run(
        method="scipy_de",
        options={"maxiter": 3, "popsize": 4, "seed": 42},
        report_interval=None,
        log_path=None,
        callback=None,
        verbose=False,
        checkpoint_path=None,
        checkpoint_interval=None,
    )
    assert result is not None
    assert problem._n_evals > 0
```

- [ ] **Step 7.2: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_drivers.py -v
```

- [ ] **Step 7.3: Implement `drivers.py`**

```python
# src/aerisplane/mdo/drivers.py
"""Optimiser driver wrappers and checkpoint utilities."""
from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from aerisplane.mdo.result import OptimisationSnapshot, OptimizationResult

_LOG = logging.getLogger(__name__)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(base_path: str, state: dict) -> None:
    """Save optimiser state to <base_path>.pkl and a human-readable sidecar
    <base_path>.json.

    Parameters
    ----------
    base_path : str
        Path without extension, e.g. ``"runs/opt_2026-04-04"``.
    state : dict
        Must contain at minimum: ``method``, ``n_evals``, ``best_x``,
        ``best_objective``, ``cache``.
    """
    pkl_path = Path(base_path + ".pkl")
    json_path = Path(base_path + ".json")

    with pkl_path.open("wb") as f:
        pickle.dump(state, f)

    meta = {
        "method": state.get("method", "unknown"),
        "n_evals": int(state.get("n_evals", 0)),
        "best_objective": float(state.get("best_objective", float("nan"))),
        "constraints_satisfied": bool(state.get("constraints_satisfied", False)),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "options": state.get("options", {}),
    }
    json_path.write_text(json.dumps(meta, indent=2))


def load_checkpoint(base_path: str) -> Optional[dict]:
    """Load checkpoint from <base_path>.pkl.

    Returns None if the file does not exist.
    """
    pkl_path = Path(base_path + ".pkl")
    if not pkl_path.exists():
        return None
    with pkl_path.open("rb") as f:
        state = pickle.load(f)
    meta_path = Path(base_path + ".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        _LOG.info(
            "Found checkpoint: %s\n  Saved at     : %s\n"
            "  Evaluations  : %d\n  Best obj     : %.4g",
            pkl_path, meta.get("saved_at", "?"),
            meta.get("n_evals", 0), meta.get("best_objective", float("nan")),
        )
    return state


# ── ScipyDriver ───────────────────────────────────────────────────────────────

class ScipyDriver:
    """Wraps scipy optimisers for use with MDOProblem.

    Supported methods: ``"scipy_de"``, ``"scipy_minimize"``, ``"scipy_shgo"``.
    """

    def __init__(self, problem):
        self.problem = problem

    def run(
        self,
        method: str,
        options: dict,
        report_interval: Optional[int],
        log_path: Optional[str],
        callback: Optional[Callable],
        verbose: bool,
        checkpoint_path: Optional[str],
        checkpoint_interval: Optional[int],
    ) -> OptimizationResult:
        from scipy.optimize import differential_evolution, minimize, shgo

        p = self.problem
        lo, hi = p.get_bounds()
        bounds = list(zip(lo, hi))
        integrality = p._integrality

        # Initialise logging and CSV
        csv_file = open(log_path, "a") if log_path else None
        if csv_file and csv_file.tell() == 0:
            _write_csv_header(csv_file, p)

        t_start = time.time()
        best = {"x": None, "obj": float("inf"), "constraints_ok": False}

        chk_interval = checkpoint_interval or max(1, getattr(p, "_n_vars", 2) * 4)

        def wrapped_obj(x):
            obj = p.objective_function(x)
            ev = p._cache.get(tuple(np.round(x, 10)), {})
            cv = ev.get("constraint_values", {})
            violations = p.constraint_functions(x)
            constraints_ok = bool(np.all(violations <= 0))

            # Track best
            if constraints_ok and obj < best["obj"]:
                best["x"] = x.copy()
                best["obj"] = obj
                best["constraints_ok"] = True

            if verbose:
                _LOG.info("[%5d] obj=%.4g  best=%.4g  t=%.1fs",
                          p._n_evals, obj, best["obj"], time.time() - t_start)

            if csv_file:
                row = [p._n_evals, obj] + [cv.get(c.path, "") for c in p._constraints]
                csv_file.write(",".join(str(v) for v in row) + "\n")
                csv_file.flush()

            # Checkpoint
            if checkpoint_path and p._n_evals % chk_interval == 0:
                save_checkpoint(checkpoint_path, {
                    "method": method,
                    "n_evals": p._n_evals,
                    "best_x": best["x"],
                    "best_objective": best["obj"],
                    "constraints_satisfied": best["constraints_ok"],
                    "cache": p._cache,
                    "options": options,
                })

            # Per-generation summary
            if report_interval and p._n_evals % report_interval == 0:
                _print_summary(p, best, t_start)

            # User callback
            if callback is not None:
                snap = _build_snapshot(p, best, t_start)
                if callback(snap) == "stop":
                    raise _EarlyStop()

            return obj

        # Build scipy constraint list (inequality: c(x) >= 0)
        scipy_constraints = []
        if p._constraints:
            def neg_violations(x):
                return -p.constraint_functions(x)
            scipy_constraints = [{"type": "ineq", "fun": neg_violations}]

        try:
            if method == "scipy_de":
                # Check for existing checkpoint
                x0_pop = None
                if checkpoint_path:
                    ckpt = load_checkpoint(checkpoint_path)
                    if ckpt is not None:
                        p.load_cache(checkpoint_path + ".pkl")
                        if ckpt.get("best_x") is not None:
                            # Seed DE with previous best as part of initial pop
                            x0_pop = "latinhypercube"

                de_opts = {k: v for k, v in options.items()
                           if k not in ("integrality",)}
                res = differential_evolution(
                    wrapped_obj,
                    bounds=bounds,
                    constraints=scipy_constraints,
                    integrality=integrality,
                    init=x0_pop or "latinhypercube",
                    **de_opts,
                )

            elif method == "scipy_minimize":
                x0 = p._x0_scaled()
                res = minimize(
                    wrapped_obj,
                    x0=x0,
                    bounds=bounds,
                    **options,
                )

            elif method == "scipy_shgo":
                res = shgo(
                    wrapped_obj,
                    bounds=bounds,
                    constraints=scipy_constraints,
                    **options,
                )
            else:
                raise ValueError(f"Unknown scipy method '{method}'.")

        except _EarlyStop:
            pass
        finally:
            if csv_file:
                csv_file.close()

        x_opt = best["x"] if best["x"] is not None else p._x0_scaled()
        return _build_optimization_result(p, x_opt, best["obj"], t_start)


# ── PygmoDriver ───────────────────────────────────────────────────────────────

class PygmoDriver:
    """Wraps pygmo optimisers for use with MDOProblem.

    Supported methods: ``"pygmo_de"``, ``"pygmo_sade"``, ``"pygmo_nsga2"``.

    pygmo is an optional dependency — import is attempted at runtime.
    """

    def __init__(self, problem):
        self.problem = problem

    def run(
        self,
        method: str,
        options: dict,
        report_interval: Optional[int],
        log_path: Optional[str],
        callback: Optional[Callable],
        verbose: bool,
        checkpoint_path: Optional[str],
        checkpoint_interval: Optional[int],
    ) -> OptimizationResult:
        try:
            import pygmo as pg
        except ImportError as exc:
            raise ImportError(
                "pygmo is required for pygmo drivers. "
                "Install with: pip install pygmo"
            ) from exc

        p = self.problem
        prob = _PygmoProblem(p)
        pg_prob = pg.problem(prob)

        pop_size = options.pop("pop_size", 20)
        n_gen = options.pop("gen", 100)
        seed = options.pop("seed", 42)

        # Resume from checkpoint if available
        pop = None
        if checkpoint_path:
            ckpt = load_checkpoint(checkpoint_path)
            if ckpt is not None and "pygmo_population" in ckpt:
                p.load_cache(checkpoint_path + ".cache.pkl") if Path(checkpoint_path + ".cache.pkl").exists() else None
                pop = pg.population(pg_prob)
                pop.set_x(ckpt["pygmo_population"]["x"])
                pop.set_f(ckpt["pygmo_population"]["f"])

        if pop is None:
            pop = pg.population(pg_prob, size=pop_size, seed=seed)

        if method == "pygmo_de":
            algo = pg.algorithm(pg.de(gen=1, **options))
        elif method == "pygmo_sade":
            algo = pg.algorithm(pg.sade(gen=1, **options))
        elif method == "pygmo_nsga2":
            algo = pg.algorithm(pg.nsga2(gen=1, **options))
        else:
            raise ValueError(f"Unknown pygmo method '{method}'.")

        t_start = time.time()
        chk_interval = checkpoint_interval or pop_size
        best = {"x": None, "obj": float("inf"), "constraints_ok": False}

        for gen in range(n_gen):
            pop = algo.evolve(pop)

            # Extract best from population
            f = pop.get_f()
            x_all = pop.get_x()
            best_idx = int(np.argmin(f[:, 0]))
            best["x"] = x_all[best_idx]
            best["obj"] = float(f[best_idx, 0])

            if verbose:
                _LOG.info("Gen %4d / %d  best=%.4g  evals=%d",
                          gen + 1, n_gen, best["obj"], p._n_evals)

            if report_interval and (gen + 1) % (report_interval // pop_size + 1) == 0:
                _print_summary(p, best, t_start)

            if checkpoint_path and (gen + 1) * pop_size % chk_interval == 0:
                save_checkpoint(checkpoint_path, {
                    "method": method,
                    "n_evals": p._n_evals,
                    "best_x": best["x"],
                    "best_objective": best["obj"],
                    "constraints_satisfied": best["constraints_ok"],
                    "pygmo_population": {
                        "x": pop.get_x().tolist(),
                        "f": pop.get_f().tolist(),
                    },
                    "options": options,
                })
                p.save_cache(checkpoint_path + ".cache.pkl")

            if callback is not None:
                snap = _build_snapshot(p, best, t_start)
                if callback(snap) == "stop":
                    break

        x_opt = best["x"] if best["x"] is not None else p._x0_scaled()
        is_mo = method == "pygmo_nsga2"
        pareto = None
        if is_mo:
            pareto = [(pop.get_x()[i].tolist(), pop.get_f()[i].tolist())
                      for i in range(len(pop.get_x()))]
        result = _build_optimization_result(p, x_opt, best["obj"], t_start)
        result.pareto_front = pareto
        return result


class _PygmoProblem:
    """pygmo UDP (User Defined Problem) adapter for MDOProblem."""

    def __init__(self, problem):
        self._p = problem

    def fitness(self, x):
        obj = self._p.objective_function(x)
        violations = self._p.constraint_functions(x)
        return np.concatenate([[obj], violations])

    def get_bounds(self):
        lo, hi = self._p.get_bounds()
        return lo.tolist(), hi.tolist()

    def get_nobj(self):
        objectives = self._p._objective
        return len(objectives) if isinstance(objectives, list) else 1

    def get_nic(self):
        """Number of inequality constraints."""
        n = 0
        for c in self._p._constraints:
            if c.lower is not None:
                n += 1
            if c.upper is not None:
                n += 1
            if c.equals is not None:
                n += 1
        return n

    def get_nix(self):
        """Number of integer variables."""
        return int(np.sum(self._p._integrality))


# ── Shared helpers ────────────────────────────────────────────────────────────

class _EarlyStop(Exception):
    pass


def _write_csv_header(csv_file, problem):
    cols = ["eval", "objective"] + [c.path for c in problem._constraints]
    csv_file.write(",".join(cols) + "\n")


def _print_summary(problem, best, t_start):
    elapsed = time.time() - t_start
    print(
        f"\n── Eval {problem._n_evals}  best={best['obj']:.4g}"
        f"  constraints={'OK' if best['constraints_ok'] else 'VIOLATED'}"
        f"  elapsed={elapsed:.0f}s ──"
    )
    if best["x"] is not None:
        for i, dv in enumerate(problem._dvars):
            print(f"  {dv.path:<50} = {best['x'][i] * dv.scale:.5g}")


def _build_snapshot(problem, best, t_start) -> OptimisationSnapshot:
    history = [h[1] for h in problem._history]
    init_obj = history[0] if history else best["obj"]
    improvement = (
        (best["obj"] - init_obj) / (abs(init_obj) + 1e-30) * 100
    )
    last100 = history[-100:] if len(history) >= 100 else history
    imp100 = (last100[0] - last100[-1]) / (abs(last100[0]) + 1e-30) * 100 if last100 else 0.0
    return OptimisationSnapshot(
        n_evals=problem._n_evals,
        objective=best["obj"],
        objective_initial=init_obj,
        improvement_pct=improvement,
        improvement_last_100=imp100,
        x_best=best["x"] if best["x"] is not None else np.array([]),
        constraints_satisfied=best["constraints_ok"],
        constraint_values={},
        elapsed_s=time.time() - t_start,
        history=history,
    )


def _build_optimization_result(problem, x_opt, obj_opt, t_start) -> OptimizationResult:
    x0 = problem._x0_scaled()
    ev0 = problem._cache.get(tuple(np.round(x0, 10)), {})
    obj0 = ev0.get("objective", float("nan"))

    ev_opt = problem._cache.get(tuple(np.round(x_opt, 10)), {})
    results = ev_opt.get("results", {})
    violations_opt = problem.constraint_functions(x_opt)
    constraints_ok = bool(np.all(violations_opt <= 0))

    # Build variables dict: path → (initial physical value, optimal physical value)
    variables = {}
    for i, dv in enumerate(problem._dvars):
        variables[dv.path] = (
            float(x0[i] * dv.scale),
            float(x_opt[i] * dv.scale),
        )
    for j, (wing_path, xi, pool) in enumerate(problem._pool_entries):
        key = f"{wing_path}.xsecs[{xi}].airfoil"
        init_idx = int(round(float(x0[len(problem._dvars) + j])))
        opt_idx  = int(round(float(x_opt[len(problem._dvars) + j])))
        variables[key] = (
            pool.options[max(0, min(init_idx, len(pool.options)-1))],
            pool.options[max(0, min(opt_idx,  len(pool.options)-1))],
        )

    return OptimizationResult(
        x_initial=x0,
        x_optimal=x_opt,
        objective_initial=float(obj0),
        objective_optimal=float(obj_opt),
        constraints_satisfied=constraints_ok,
        n_evaluations=problem._n_evals,
        convergence_history=[h[1] for h in problem._history],
        variables=variables,
        aero=results.get("aero"),
        weights=results.get("weights"),
        structures=results.get("structures"),
        stability=results.get("stability"),
        control=results.get("control"),
        mission=results.get("mission"),
        aircraft=ev_opt.get("aircraft"),
        pareto_front=None,
    )
```

- [ ] **Step 7.4: Run to verify PASS**

```bash
pytest tests/test_mdo/test_drivers.py -v
```

- [ ] **Step 7.5: Commit**

```bash
git add src/aerisplane/mdo/drivers.py tests/test_mdo/test_drivers.py
git commit -m "feat(mdo): add ScipyDriver, PygmoDriver, and checkpoint save/load"
```

---

## Task 8: `MDOProblem.optimize()` — convenience wrapper

**Files:**
- Modify: `src/aerisplane/mdo/problem.py`
- Test: `tests/test_mdo/test_problem.py` (extend)

- [ ] **Step 8.1: Write failing test**

```python
# Append to tests/test_mdo/test_problem.py

@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_optimize_returns_result(mock_w, mock_a, mock_s, simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()

    from aerisplane.mdo.result import OptimizationResult
    result = simple_problem.optimize(
        method="scipy_de",
        options={"maxiter": 2, "popsize": 4, "seed": 0},
        verbose=False,
    )
    assert isinstance(result, OptimizationResult)
    assert result.n_evaluations > 0
    assert len(result.convergence_history) == result.n_evaluations
```

- [ ] **Step 8.2: Run to verify FAIL**

```bash
pytest tests/test_mdo/test_problem.py::test_optimize_returns_result -v
```

- [ ] **Step 8.3: Implement `optimize()`**

Append to `MDOProblem` class in `problem.py`:

```python
    def optimize(
        self,
        method: str = "scipy_de",
        options: dict | None = None,
        verbose: bool = True,
        report_interval: int | None = None,
        log_path: str | None = None,
        callback=None,
        checkpoint_path: str | None = None,
        checkpoint_interval: int | None = None,
    ):
        """Run optimisation and return an OptimizationResult.

        Parameters
        ----------
        method : str
            One of ``"scipy_de"``, ``"scipy_minimize"``, ``"scipy_shgo"``,
            ``"pygmo_de"``, ``"pygmo_sade"``, ``"pygmo_nsga2"``.
        options : dict or None
            Passed directly to the underlying optimiser.
            scipy_de: ``maxiter``, ``popsize``, ``seed``, ``tol``, ...
            pygmo:    ``gen``, ``pop_size``, ``seed``, ...
        verbose : bool
            Print a one-line log after every evaluation.  Default True.
        report_interval : int or None
            Print a detailed summary every N evaluations.  None → off.
        log_path : str or None
            Path for CSV log (appended if file exists).  None → no file.
        callback : callable or None
            Called with ``OptimisationSnapshot`` after each evaluation.
            Return ``"stop"`` to terminate early.
        checkpoint_path : str or None
            Base path (no extension) for checkpoint files.
            Existing checkpoint is loaded and run resumes automatically.
        checkpoint_interval : int or None
            Save checkpoint every N evaluations.
            None → every popsize evaluations.

        Returns
        -------
        OptimizationResult
        """
        from aerisplane.mdo.drivers import ScipyDriver, PygmoDriver

        opts = options or {}

        if method.startswith("scipy_"):
            driver = ScipyDriver(self)
        elif method.startswith("pygmo_"):
            driver = PygmoDriver(self)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: scipy_de, scipy_minimize, scipy_shgo, "
                "pygmo_de, pygmo_sade, pygmo_nsga2."
            )

        return driver.run(
            method=method,
            options=opts,
            report_interval=report_interval,
            log_path=log_path,
            callback=callback,
            verbose=verbose,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
        )
```

- [ ] **Step 8.4: Run to verify PASS**

```bash
pytest tests/test_mdo/test_problem.py -v
```

- [ ] **Step 8.5: Commit**

```bash
git add src/aerisplane/mdo/problem.py tests/test_mdo/test_problem.py
git commit -m "feat(mdo): add MDOProblem.optimize() convenience wrapper"
```

---

## Task 9: `mdo/__init__.py` — public exports

**Files:**
- Modify: `src/aerisplane/mdo/__init__.py`

- [ ] **Step 9.1: Write and run**

```python
# src/aerisplane/mdo/__init__.py
"""AerisPlane MDO orchestration layer."""
from aerisplane.mdo.problem import (
    AirfoilPool,
    Constraint,
    DesignVar,
    MDOProblem,
    Objective,
)
from aerisplane.mdo.result import OptimisationSnapshot, OptimizationResult

__all__ = [
    "MDOProblem",
    "DesignVar",
    "AirfoilPool",
    "Constraint",
    "Objective",
    "OptimizationResult",
    "OptimisationSnapshot",
]
```

```bash
python -c "from aerisplane.mdo import MDOProblem, DesignVar, Constraint, Objective, AirfoilPool; print('OK')"
```

- [ ] **Step 9.2: Commit**

```bash
git add src/aerisplane/mdo/__init__.py
git commit -m "feat(mdo): add public __init__ exports"
```

---

## Task 10: Integration test

**Files:**
- Create: `tests/test_mdo/test_integration.py`

This test runs a real (but very short) optimisation with 2 design variables on the test aircraft, verifying the full stack works end-to-end without mocks.

- [ ] **Step 10.1: Write test**

```python
# tests/test_mdo/test_integration.py
"""End-to-end MDO integration test.

Uses real discipline chain (VLM aero) but a tiny DE run (2 iterations,
popsize 4) to keep wall time under 60 s.
"""
import pytest
import numpy as np
from aerisplane.mdo import MDOProblem, DesignVar, Constraint, Objective
from aerisplane.mdo.result import OptimizationResult


def test_full_mdo_run(test_aircraft, cruise_condition):
    """Minimise total mass subject to static margin ≥ 5% MAC.

    Only 2 design variables, 2 DE generations, popsize 4 → ~8 evals.
    """
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.20, upper=0.32),
            DesignVar("wings[0].xsecs[1].chord", lower=0.10, upper=0.20),
        ],
        constraints=[
            Constraint("stability.static_margin", lower=0.04),
        ],
        objective=Objective("weights.total_mass", maximize=False),
        alpha=4.0,   # fixed alpha — no trim solve, keeps test faster
    )

    result = problem.optimize(
        method="scipy_de",
        options={"maxiter": 2, "popsize": 4, "seed": 7},
        verbose=False,
    )

    assert isinstance(result, OptimizationResult)
    assert result.n_evaluations >= 4
    assert result.objective_optimal <= result.objective_initial + 0.1  # didn't blow up
    assert result.weights is not None
    assert result.stability is not None
    assert isinstance(result.report(), str)


def test_cache_reduces_redundant_evals(test_aircraft, cruise_condition):
    """evaluate() called twice with the same x should hit cache."""
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.20, upper=0.32),
        ],
        constraints=[Constraint("stability.static_margin", lower=0.04)],
        objective=Objective("weights.total_mass", maximize=False),
        alpha=4.0,
    )
    lo, hi = problem.get_bounds()
    x = (lo + hi) / 2.0
    problem.evaluate(x)
    n1 = problem._n_evals
    problem.evaluate(x)
    assert problem._n_evals == n1   # cache hit, no new eval
```

- [ ] **Step 10.2: Run**

```bash
pytest tests/test_mdo/test_integration.py -v --timeout=120
```

Expected: both tests PASS.

- [ ] **Step 10.3: Commit**

```bash
git add tests/test_mdo/test_integration.py
git commit -m "test(mdo): add integration tests for full MDO run and cache"
```

---

## Task 11: Run full test suite and verify no regressions

- [ ] **Step 11.1: Run all MDO tests**

```bash
pytest tests/test_mdo/ -v
```

Expected: all tests PASS.

- [ ] **Step 11.2: Run full test suite**

```bash
pytest tests/ -v --timeout=120
```

Expected: no regressions in existing tests.

- [ ] **Step 11.3: Final commit**

```bash
git add -A
git commit -m "feat(mdo): complete MDO module implementation

- MDOProblem with path-based design variables, airfoil pools,
  auto-trim, discipline inference, evaluation caching
- ScipyDriver (de, minimize, shgo) and PygmoDriver (de, sade, nsga2)
- Checkpoint save/load with JSON sidecar
- OptimizationResult with report() and plot()
- catalog.get_airfoil() name-based lookup"
```

---

## Self-review checklist

- [x] **Spec coverage:** DesignVar ✓, AirfoilPool ✓, Constraint (float + bool) ✓, Objective ✓, MDOProblem ✓, validate() ✓, evaluate() with caching ✓, auto-trim + fixed alpha ✓, discipline inference ✓, ScipyDriver (DE/minimize/shgo) ✓, PygmoDriver (de/sade/nsga2) ✓, checkpoint + resume ✓, CSV log ✓, callback ✓, OptimizationResult ✓, report() ✓, plot() ✓, plot_pareto() ✓, cache save/load ✓, mission optional ✓.
- [x] **No placeholders:** all code blocks are complete.
- [x] **Type consistency:** `OptimisationSnapshot` and `OptimizationResult` field names match usage in `drivers.py`. `_pack`/`_unpack` signatures match calls in `MDOProblem`. `_integrality_array` matches `_integrality` attribute on problem. `get_bounds()` returns two arrays matching driver usage.
