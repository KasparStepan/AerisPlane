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
    integrality: bool = False  # True → optimizer treats this as integer


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
class ChoiceVar:
    """Discrete choice variable — selects one item from an ordered list.

    Works with any field type (Airfoil, Motor, Propeller, str, etc.).
    The optimizer sees an integer index 0..len(options)-1.
    Use ``opti.choice()`` or ``opti.ranked_choice()`` to create these;
    they are auto-discovered by ``Opti.problem()``.

    Parameters
    ----------
    path : str
        Dot-bracket path into Aircraft, e.g. ``"wings[0].xsecs[0].airfoil"``.
    options : list
        Ordered list of possible values.
    init_idx : int
        Index of the initial/baseline option. Default 0.
    """
    path: str
    options: list
    init_idx: int = 0


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
    _pack_choices,
    _unpack,
    _unpack_choices,
)

_LOG = logging.getLogger(__name__)

DISCIPLINE_ORDER = ["weights", "aero", "structures", "stability", "control", "propulsion", "mission"]
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
        condition=None,            # single FlightCondition (backward compat)
        conditions: dict = None,   # dict[str, FlightCondition] for multi-condition
        design_variables: list = None,
        constraints: list = None,
        objective=None,
        mission=None,
        airfoil_pools: dict = None,
        alpha: float = None,
        aero_method: str = "vlm",
        load_factor: float = 3.5,
        throttle: float = 1.0,
        extra_disciplines: tuple = (),
        skip_disciplines: tuple = (),
        disciplines: list = None,
        aero_result=None,
        choice_variables: list = None,
        xyz_ref: list = None,
    ):
        self._baseline = copy.deepcopy(aircraft)

        if condition is not None and conditions is not None:
            raise ValueError("Provide either 'condition' or 'conditions', not both.")
        if condition is None and conditions is None:
            raise ValueError("Provide either 'condition' or 'conditions'.")
        self._condition = condition
        self._conditions = conditions  # None when single-condition mode

        design_variables = list(design_variables or [])
        constraints = list(constraints or [])
        self._dvars = list(design_variables)
        self._constraints = list(constraints)
        self._objective = objective
        self._mission = mission
        self._pools = airfoil_pools or {}
        self._alpha = alpha
        self.aero_method = aero_method
        self.load_factor = load_factor
        self._throttle = throttle
        self._aero_result = aero_result
        self._xyz_ref = list(xyz_ref) if xyz_ref is not None else None

        self._pool_entries = _build_pool_entries(self._baseline, self._pools)
        self._choice_vars = list(choice_variables or [])
        self._n_continuous = len(self._dvars)
        self._n_pool = len(self._pool_entries)
        self._n_choices = len(self._choice_vars)
        self._n_vars = self._n_continuous + self._n_pool + self._n_choices
        self._integrality = np.array(
            [dv.integrality for dv in self._dvars]
            + [True] * self._n_pool
            + [True] * self._n_choices,
            dtype=bool,
        )
        self._scales = np.array(
            [dv.scale for dv in self._dvars]
            + [1.0] * self._n_pool
            + [1.0] * self._n_choices
        )

        if disciplines is not None:
            needed = set(disciplines)
            # Force weights only when we need CG for xyz_ref (i.e. no fixed ref given)
            if self._xyz_ref is None:
                needed.add("weights")
            if aero_result is None:
                needed.add("aero")
            self._disciplines = [d for d in DISCIPLINE_ORDER if d in needed]
        else:
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

        if self._conditions is not None:
            # Multi-condition: paths must start with a known condition name
            for path in all_paths:
                cond_name = path.split(".")[0]
                if cond_name not in self._conditions:
                    raise ValueError(
                        f"Unknown condition '{cond_name}' in path '{path}'. "
                        f"Available conditions: {list(self._conditions.keys())}. "
                        f"Multi-condition paths must be 'condition_name.discipline.field'."
                    )

        for path in all_paths:
            if self._conditions is not None:
                # path = "cond_name.discipline.field"
                parts = path.split(".")
                if len(parts) < 2:
                    raise ValueError(
                        f"Multi-condition path '{path}' must be 'cond.discipline.field'."
                    )
                prefix = parts[1]   # discipline is second segment
            else:
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

        for cv in self._choice_vars:
            if not cv.options:
                raise ValueError(f"ChoiceVar '{cv.path}': options list is empty.")
            if not (0 <= cv.init_idx < len(cv.options)):
                raise ValueError(
                    f"ChoiceVar '{cv.path}': init_idx {cv.init_idx} out of range "
                    f"[0, {len(cv.options) - 1}]."
                )

        from aerisplane.catalog import get_airfoil
        for wing_path, pool in self._pools.items():
            for name in pool.options:
                try:
                    get_airfoil(name)
                except ValueError as exc:
                    raise ValueError(f"AirfoilPool for '{wing_path}': {exc}") from exc

    def simulate(self) -> dict:
        """Run the discipline chain once at the baseline design. No optimization.

        Useful for initial sizing: inspect discipline results before launching
        an optimizer. Results are cached — calling ``simulate()`` twice is free.

        Returns
        -------
        dict
            Mapping discipline name → result object, e.g.::

                results = problem.simulate()
                print(results["aero"].CL_over_CD)
                print(results["weights"].total_mass)
                results["aero"].report()
        """
        return self.evaluate(self._x0_scaled())["results"]

    def get_bounds(self):
        """Return (lower, upper) bound arrays."""
        lo_cont = np.array([dv.lower / dv.scale for dv in self._dvars])
        hi_cont = np.array([dv.upper / dv.scale for dv in self._dvars])
        n_pool = len(self._pool_entries)
        lo_pool = np.zeros(n_pool)
        hi_pool = np.array([float(len(pe[2].options) - 1) for pe in self._pool_entries])
        lo_ch = np.zeros(self._n_choices)
        hi_ch = np.array([float(len(cv.options) - 1) for cv in self._choice_vars])
        return (
            np.concatenate([lo_cont, lo_pool, lo_ch]),
            np.concatenate([hi_cont, hi_pool, hi_ch]),
        )

    def _x0_scaled(self):
        """Initial design vector from current aircraft values."""
        x_cont_pool = _pack(self._baseline, self._dvars, self._pool_entries)
        x_choices = _pack_choices(self._choice_vars)
        return np.concatenate([x_cont_pool, x_choices])

    def evaluate(self, x_scaled: np.ndarray) -> dict:
        """Run the full discipline chain for design vector x_scaled.

        Returns dict with keys: objective, constraint_values, results,
        aircraft, condition, elapsed.
        """
        from aerisplane.mdo.registry import default_registry

        cache_key = tuple(np.round(x_scaled, 10))
        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.time()
        self._n_evals += 1

        ac = _unpack(self._baseline, self._dvars, self._pool_entries, x_scaled)

        if self._choice_vars:
            offset = self._n_continuous + self._n_pool
            _unpack_choices(ac, self._choice_vars, x_scaled, offset)

        # Set moment reference point and compute weights if needed
        import aerisplane.weights as weights_mod
        if self._xyz_ref is not None:
            # Fixed reference supplied — use it as-is, skip the weights run
            ac.xyz_ref = list(self._xyz_ref)
            if "weights" in self._disciplines:
                weights_result = weights_mod.analyze(ac)
                initial_results = {"weights": weights_result}
            else:
                weights_result = None
                initial_results = {}
        else:
            # Derive reference from CG (default behaviour)
            weights_result = weights_mod.analyze(ac)
            cg = weights_result.cg
            ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]
            initial_results = {"weights": weights_result}

        if self._conditions is not None:
            # Multi-condition: run the chain for each named condition
            results = {}
            for cond_name, cond in self._conditions.items():
                results[cond_name] = default_registry.run_chain(
                    disciplines=self._disciplines,
                    aircraft=ac,
                    condition=cond,
                    aero_result=self._aero_result,
                    initial_results=initial_results,
                    aero_method=self.aero_method,
                    load_factor=self.load_factor,
                    safety_factor=1.5,
                    throttle=self._throttle,
                    mission=self._mission,
                )
        else:
            # Single-condition: determine flight condition (trim or fixed alpha)
            from aerisplane.core.flight_condition import FlightCondition
            if self._alpha is not None:
                cond = FlightCondition(
                    velocity=self._condition.velocity,
                    altitude=self._condition.altitude,
                    alpha=self._alpha,
                    beta=getattr(self._condition, "beta", 0.0),
                )
            elif self._aero_result is None:
                cond = self._trim_condition(ac, weights_result)
            else:
                cond = self._condition

            results = default_registry.run_chain(
                disciplines=self._disciplines,
                aircraft=ac,
                condition=cond,
                aero_result=self._aero_result,
                initial_results=initial_results,
                aero_method=self.aero_method,
                load_factor=self.load_factor,
                safety_factor=1.5,
                throttle=self._throttle,
                mission=self._mission,
            )

        # Extract objective
        from aerisplane.mdo._paths import _get_result_value_multicond
        objectives = self._objective if isinstance(self._objective, list) else [self._objective]
        obj_vals = []
        for obj in objectives:
            raw = float(_get_result_value_multicond(results, obj.path, self._conditions))
            sign = -1.0 if obj.maximize else 1.0
            obj_vals.append(sign * raw / obj.scale)
        objective_value = obj_vals[0] if len(obj_vals) == 1 else obj_vals

        constraint_values = {
            c.path: _get_result_value_multicond(results, c.path, self._conditions)
            for c in self._constraints
        }

        elapsed = time.time() - t0
        ev = {
            "objective": objective_value,
            "constraint_values": constraint_values,
            "results": results,
            "aircraft": ac,
            "condition": self._conditions if self._conditions is not None else cond,
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

    def optimize(
        self,
        method: str = "scipy_de",
        options: dict = None,
        verbose: bool = True,
        report_interval: int = None,
        log_path: str = None,
        callback=None,
        checkpoint_path: str = None,
        checkpoint_interval: int = None,
    ):
        """Run optimisation and return an OptimizationResult.

        Parameters
        ----------
        method : str
            One of ``"scipy_de"``, ``"scipy_minimize"``, ``"scipy_shgo"``,
            ``"pymoo_de"``, ``"pymoo_nsga2"``, ``"pymoo_nsga3"``, ``"pymoo_pso"``.
        options : dict or None
            Passed directly to the underlying optimiser.
            scipy_de: ``maxiter``, ``popsize``, ``seed``, ``tol``, ...
            pymoo:    ``pop_size``, ``n_gen``, ``seed``, ...
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
        from aerisplane.mdo.drivers import ScipyDriver, PymooDriver

        opts = options or {}

        if method.startswith("scipy_"):
            driver = ScipyDriver(self)
        elif method.startswith("pymoo_"):
            driver = PymooDriver(self)
        elif method.startswith("pygmo_"):
            raise ValueError(
                f"Method '{method}' uses pygmo which requires conda. "
                "Use pymoo equivalents instead: "
                "pygmo_de → pymoo_de, pygmo_sade → pymoo_de, pygmo_nsga2 → pymoo_nsga2."
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: scipy_de, scipy_minimize, scipy_shgo, "
                "pymoo_de, pymoo_nsga2, pymoo_nsga3, pymoo_pso."
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

    def sensitivity(
        self,
        x: "np.ndarray",
        step: float = 1e-4,
    ) -> "SensitivityResult":
        """Compute finite-difference sensitivity at a given design point.

        Parameters
        ----------
        x : ndarray — design vector in scaled optimizer space
        step : float — forward finite-difference step in scaled space

        Returns
        -------
        SensitivityResult
            Gradients and normalized sensitivities ranked by objective influence.
        """
        from aerisplane.mdo.sensitivity import compute_sensitivity
        return compute_sensitivity(self, x, step=step)
