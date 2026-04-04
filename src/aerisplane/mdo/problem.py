"""MDO problem definition: dataclasses and MDOProblem class."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np


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
        Optimizer sees ``value / scale``.  Default 1.0 (no scaling).
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
        Value must be >= lower.
    upper : float or None
        Value must be <= upper.
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


# ── MDOProblem ─────────────────────────────────────────────────────────────────

from aerisplane.mdo._paths import (  # noqa: E402
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
    objective,
    mission,
    extra: tuple,
    skip: tuple,
) -> list:
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
        needed.add("stability")
    needed.update(extra)
    needed -= set(skip)
    return [d for d in DISCIPLINE_ORDER if d in needed]


class MDOProblem:
    """Multidisciplinary optimisation problem.

    Parameters
    ----------
    aircraft : Aircraft
        Baseline aircraft. A deep copy is stored.
    condition : FlightCondition
        Reference flight condition. Provides velocity, altitude, beta.
        Alpha is taken from the ``alpha`` parameter (fixed) or solved
        for trim (``alpha=None``).
    design_variables : list of DesignVar
    constraints : list of Constraint
    objective : Objective or list of Objective
    mission : Mission or None
    airfoil_pools : dict or None
        Mapping wing_path -> AirfoilPool.
    alpha : float or None
        Fixed alpha [deg]. None -> auto-trim each evaluation.
    aero_method : str
    load_factor : float
        Limit maneuver load factor for structures.analyze(). Default 3.5.
    extra_disciplines : tuple of str
    skip_disciplines : tuple of str
    """

    def __init__(
        self,
        aircraft,
        condition,
        design_variables: list,
        constraints: list,
        objective,
        mission=None,
        airfoil_pools: dict = None,
        alpha: float = None,
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

        self._pool_entries = _build_pool_entries(self._baseline, self._pools)
        self._n_continuous = len(self._dvars)
        self._n_vars = self._n_continuous + len(self._pool_entries)
        self._integrality = _integrality_array(self._n_continuous, self._pool_entries)

        self._scales = np.array(
            [dv.scale for dv in self._dvars] + [1.0] * len(self._pool_entries)
        )

        self._disciplines = _infer_disciplines(
            self._constraints, self._objective, self._mission,
            extra_disciplines, skip_disciplines,
        )

        self._cache: dict = {}
        self._history: list = []
        self._n_evals: int = 0

        self.validate()

    def validate(self) -> None:
        """Resolve all DesignVar paths against the aircraft. Raise ValueError on bad paths."""
        for dv in self._dvars:
            try:
                _get_dv_value(self._baseline, dv.path)
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"DesignVar path '{dv.path}' does not resolve on the aircraft: {exc}"
                ) from exc

        if self._mission is None:
            objectives = self._objective if isinstance(self._objective, list) else [self._objective]
            all_paths = [c.path for c in self._constraints] + [o.path for o in objectives]
            bad = [p for p in all_paths if p.startswith("mission.")]
            if bad:
                raise ValueError(
                    f"Paths {bad} reference 'mission' but mission=None."
                )

        # Check constraint/objective path prefixes are recognised disciplines
        objectives = self._objective if isinstance(self._objective, list) else [self._objective]
        all_paths = [c.path for c in self._constraints] + [o.path for o in objectives]
        for path in all_paths:
            prefix = path.split(".")[0]
            if prefix not in DISCIPLINE_ORDER:
                raise ValueError(
                    f"Path '{path}' has unknown discipline prefix '{prefix}'. "
                    f"Valid prefixes: {DISCIPLINE_ORDER}."
                )
            if prefix not in self._disciplines:
                raise ValueError(
                    f"Path '{path}' references discipline '{prefix}' which is not in the "
                    f"active discipline set {self._disciplines}. "
                    "Use extra_disciplines to enable it or remove the path."
                )

        from aerisplane.catalog import get_airfoil
        for wing_path, pool in self._pools.items():
            for name in pool.options:
                try:
                    get_airfoil(name)
                except ValueError as exc:
                    raise ValueError(f"AirfoilPool for '{wing_path}': {exc}") from exc

    def get_bounds(self):
        """Return (lower, upper) bound arrays."""
        lo_cont = np.array([dv.lower / dv.scale for dv in self._dvars])
        hi_cont = np.array([dv.upper / dv.scale for dv in self._dvars])
        n_pool = len(self._pool_entries)
        lo_pool = np.zeros(n_pool)
        hi_pool = np.array([float(len(pe[2].options) - 1) for pe in self._pool_entries])
        return np.concatenate([lo_cont, lo_pool]), np.concatenate([hi_cont, hi_pool])

    def _x0_scaled(self):
        """Initial design vector from current aircraft values."""
        return _pack(self._baseline, self._dvars, self._pool_entries)

    def evaluate(self, x_scaled: np.ndarray) -> dict:
        """Run the full discipline chain for design vector x_scaled.

        Returns dict with keys: objective, constraint_values, results,
        aircraft, condition, elapsed.
        """
        import aerisplane.aero as aero_mod
        import aerisplane.weights as weights_mod
        import aerisplane.stability as stab_mod
        import aerisplane.control as ctrl_mod
        import aerisplane.mission as mission_mod
        import aerisplane.structures as struct_mod

        cache_key = tuple(np.round(x_scaled, 10))
        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.time()
        self._n_evals += 1

        ac = _unpack(self._baseline, self._dvars, self._pool_entries, x_scaled)
        results: dict = {}

        # 1. Weights (always)
        results["weights"] = weights_mod.analyze(ac)

        # Set moment reference to CG for all subsequent aero/stability calls
        cg = results["weights"].cg
        ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]

        # 2. Flight condition
        from aerisplane.core.flight_condition import FlightCondition
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
                ac, results["aero"], results["weights"],
                n_limit=self.load_factor, safety_factor=1.5,
            )

        # 5. Stability
        if "stability" in self._disciplines:
            results["stability"] = stab_mod.analyze(
                ac, cond, results["weights"], aero_method=self.aero_method,
            )

        # 6. Control
        if "control" in self._disciplines:
            results["control"] = ctrl_mod.analyze(
                ac, cond, results["weights"], results["stability"],
                aero_method=self.aero_method,
            )

        # 7. Mission
        if "mission" in self._disciplines and self._mission is not None:
            results["mission"] = mission_mod.analyze(
                ac, results["weights"], self._mission, aero_method=self.aero_method,
            )

        # Extract objective
        objectives = self._objective if isinstance(self._objective, list) else [self._objective]
        obj_vals = []
        for obj in objectives:
            raw = float(_get_result_value(results, obj.path))
            sign = -1.0 if obj.maximize else 1.0
            obj_vals.append(sign * raw / obj.scale)
        objective_value = obj_vals[0] if len(obj_vals) == 1 else obj_vals

        constraint_values = {
            c.path: _get_result_value(results, c.path) for c in self._constraints
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
            log_obj = objective_value[0] if isinstance(objective_value, list) else objective_value
            history_scalars = [h[1][0] if isinstance(h[1], list) else h[1] for h in self._history]
            best_obj = min(history_scalars, default=log_obj)
            _LOG.info("[%5d]  obj=%.4g  best=%.4g  t=%.2fs",
                      self._n_evals, log_obj, best_obj, elapsed)
        return ev

    def _trim_condition(self, aircraft, weight_result):
        """Find trim alpha via brentq, return trimmed FlightCondition."""
        from aerisplane.core.flight_condition import FlightCondition
        from scipy.optimize import brentq
        import aerisplane.aero as aero_mod

        ac = copy.deepcopy(aircraft)
        cg = weight_result.cg
        ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]
        V = self._condition.velocity
        alt = self._condition.altitude
        beta = getattr(self._condition, "beta", 0.0)

        def cm_at(alpha: float) -> float:
            cond = FlightCondition(velocity=V, altitude=alt, alpha=alpha, beta=beta)
            return aero_mod.analyze(ac, cond, method=self.aero_method).Cm

        alpha_trim = 4.0
        try:
            cm_lo = cm_at(-5.0)
            cm_hi = cm_at(15.0)
            if cm_lo * cm_hi < 0.0:
                alpha_trim = brentq(cm_at, -5.0, 15.0, xtol=0.1)
            else:
                _LOG.warning(
                    "Trim: Cm does not change sign in [-5, 15] deg "
                    "(Cm_lo=%.3f, Cm_hi=%.3f). Falling back to alpha=%.1f deg.",
                    cm_lo, cm_hi, alpha_trim,
                )
        except Exception as exc:
            _LOG.warning("Trim solver failed (%s). Falling back to alpha=%.1f deg.", exc, alpha_trim)
        return FlightCondition(velocity=V, altitude=alt, alpha=alpha_trim, beta=beta)

    def objective_function(self, x_scaled: np.ndarray) -> float:
        """Return scalar objective for single-objective optimisers."""
        return float(self.evaluate(x_scaled)["objective"])

    def constraint_functions(self, x_scaled: np.ndarray) -> np.ndarray:
        """Return violation vector (<= 0 means satisfied)."""
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
        """Persist evaluation cache to a pickle file."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._cache, f)

    def load_cache(self, path: str) -> None:
        """Load a previously saved cache, merging with existing entries."""
        import pickle
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Cache file '{path}' does not contain a dict (got {type(loaded).__name__}). "
                "File may be corrupted or from an incompatible version."
            )
        self._cache.update(loaded)
