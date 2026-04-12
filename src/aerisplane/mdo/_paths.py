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
    "stability.static_margin"               -> results["stability"].static_margin
    "structures.wings[0].bending_margin"    -> results["structures"].wings[0].bending_margin
    "mission.feasible"                      -> results["mission"].feasible
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
    pools : dict mapping wing_path str -> AirfoilPool, or None/empty.

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


def _pack_choices(choice_vars: list) -> np.ndarray:
    """Return initial integer indices for all ChoiceVar as a float array."""
    return np.array([float(cv.init_idx) for cv in choice_vars])


def _unpack_choices(aircraft, choice_vars: list, x: np.ndarray, offset: int) -> None:
    """Write choice variable values from x into aircraft (in-place).

    Parameters
    ----------
    aircraft : Aircraft
        Deep-copied instance to mutate.
    choice_vars : list of ChoiceVar
    x : ndarray
        Full design vector (scaled).
    offset : int
        Index in x where choice variables start.
    """
    for j, cv in enumerate(choice_vars):
        raw_idx = float(x[offset + j])
        idx = max(0, min(int(round(raw_idx)), len(cv.options) - 1))
        _set_dv_value(aircraft, cv.path, cv.options[idx])
