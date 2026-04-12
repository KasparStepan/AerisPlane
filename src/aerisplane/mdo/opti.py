"""Opti — inline design variable API for AerisPlane MDO."""
from __future__ import annotations

import itertools
from dataclasses import fields, is_dataclass
from typing import Any, Optional


_id_counter = itertools.count()


class _Var(float):
    """A float that also carries optimization bounds.

    Behaves exactly like a float in all arithmetic. The Opti context
    discovers these by walking the aircraft dataclass tree.
    """

    def __new__(
        cls,
        value: float,
        lower: float,
        upper: float,
        scale: float = 1.0,
    ) -> "_Var":
        obj = super().__new__(cls, value)
        obj._lower = lower
        obj._upper = upper
        obj._scale = scale
        obj._is_integer = False
        obj._var_id = next(_id_counter)
        if lower > upper:
            raise ValueError(
                f"_Var: lower={lower} > upper={upper}. Bounds are inverted."
            )
        return obj

    def __deepcopy__(self, memo):
        new = _Var(float(self), self._lower, self._upper, self._scale)
        new._var_id = self._var_id
        new._is_integer = self._is_integer
        memo[id(self)] = new
        return new


class _IntVar(_Var):
    """An integer-valued design variable declared inline.

    The optimizer sees an integer in [lower, upper].  Use via
    ``opti.integer_variable()``.

    Unlike ``opti.variable()`` (continuous), the step between adjacent
    values is always 1.  scale is fixed at 1.0.
    """

    def __new__(cls, value: int, lower: int, upper: int) -> "_IntVar":
        obj = float.__new__(cls, value)
        obj._lower = float(lower)
        obj._upper = float(upper)
        obj._scale = 1.0
        obj._is_integer = True
        obj._var_id = next(_id_counter)
        return obj

    def __deepcopy__(self, memo):
        new = _IntVar(int(self), int(self._lower), int(self._upper))
        new._var_id = self._var_id
        memo[id(self)] = new
        return new


class _Choice:
    """A discrete catalog selection sentinel.

    Use via ``opti.choice()`` or ``opti.ranked_choice()``.
    The options list is stored sorted ascending by score (for ranked_choice)
    or in original order (for choice). The optimizer sees integer index 0..N-1.
    """

    def __init__(self, options: list, init_idx: int = 0, scores=None):
        if scores is not None:
            if len(scores) != len(options):
                raise ValueError(
                    f"scores length ({len(scores)}) must match options length ({len(options)})."
                )
            paired = sorted(zip(scores, range(len(options)), options), key=lambda x: x[0])
            self.options = [o for _, _, o in paired]
            orig_indices = [orig for _, orig, _ in paired]
            self.init_idx = orig_indices.index(init_idx)
        else:
            self.options = list(options)
            self.init_idx = init_idx
        self._var_id = next(_id_counter)


def _discover_vars(
    obj: Any,
    path: str = "",
    found_vars: Optional[dict] = None,
    found_choices: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Recursively walk *obj* and return two dicts:
    - found_vars:    path -> _Var  (continuous / integer variables)
    - found_choices: path -> _Choice  (discrete catalog choices)

    Only walks into dataclass instances and lists.
    """
    if found_vars is None:
        found_vars = {}
    if found_choices is None:
        found_choices = {}

    if isinstance(obj, _Var):
        found_vars[path] = obj
        return found_vars, found_choices

    # _Choice is defined later in this file; guard with name check to avoid forward-ref
    if type(obj).__name__ == "_Choice":
        found_choices[path] = obj
        return found_vars, found_choices

    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            val = getattr(obj, f.name)
            child = f"{path}.{f.name}" if path else f.name
            _discover_vars(val, child, found_vars, found_choices)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            child = f"{path}[{i}]"
            _discover_vars(item, child, found_vars, found_choices)

    return found_vars, found_choices


# ── Opti ──────────────────────────────────────────────────────────────────────

class Opti:
    """Context object for inline design variable declaration.

    Create variables inline when building the aircraft::

        opti = Opti()
        wing = Wing(span=opti.variable(1.2, lower=0.8, upper=2.0), ...)
        problem = opti.problem(aircraft=..., disciplines=["aero"], ...)
    """

    def __init__(self) -> None:
        self._vars: dict = {}  # var_id -> _Var or _Choice

    def variable(
        self,
        init: float,
        lower: float,
        upper: float,
        scale: float = 1.0,
    ) -> "_Var":
        """Declare a continuous design variable and return its initial value.

        Parameters
        ----------
        init : float
            Initial (baseline) value.
        lower : float
            Lower bound (physical units).
        upper : float
            Upper bound (physical units).
        scale : float
            Optimizer sees ``value / scale``. Default 1.0.
        """
        v = _Var(init, lower=lower, upper=upper, scale=scale)
        self._vars[v._var_id] = v
        return v

    def integer_variable(self, init: int, lower: int, upper: int) -> "_IntVar":
        """Declare an integer design variable.

        Use for parameters that are naturally integers but not a catalog list:
        number of battery cells, propeller blades, ribs, motor pole pairs, etc.

        Parameters
        ----------
        init : int
            Initial (baseline) value.
        lower : int
            Lower bound (inclusive).
        upper : int
            Upper bound (inclusive).

        Returns
        -------
        _IntVar
            A float subclass — assign it exactly like ``opti.variable()``.
        """
        v = _IntVar(init, lower=lower, upper=upper)
        self._vars[v._var_id] = v
        return v

    def choice(self, options: list, init: int = 0) -> "_Choice":
        """Declare a discrete catalog choice variable.

        The optimizer selects one item from *options* by integer index.
        Use for any field: Airfoil, Motor, Propeller, material, etc.

        Parameters
        ----------
        options : list
            Possible values. Any type supported.
        init : int
            Index of the initial option. Default 0.
        """
        c = _Choice(options=options, init_idx=init)
        self._vars[c._var_id] = c
        return c

    def ranked_choice(self, options: list, scores: list, init: int = 0) -> "_Choice":
        """Declare a scored discrete catalog choice variable.

        Options are sorted ascending by *scores* before being passed to the
        optimizer. This means adjacent integer indices correspond to
        similar-performing options, giving a smoother optimization landscape.

        Parameters
        ----------
        options : list
            Possible values.
        scores : list of float
            One score per option. Lower score = index 0.
            Pre-compute outside the optimization loop.
        init : int
            Index of the initial option in the *original* (unsorted) list.
        """
        c = _Choice(options=options, init_idx=init, scores=scores)
        self._vars[c._var_id] = c
        return c

    def problem(
        self,
        aircraft,
        condition=None,
        conditions: dict = None,
        disciplines: list = None,
        objective=None,
        constraints: list = None,
        mission=None,
        airfoil_pools: dict = None,
        alpha: float = None,
        aero_method: str = "vlm",
        load_factor: float = 3.5,
        throttle: float = 1.0,
        aero_result=None,
        choice_variables: list = None,
    ):
        """Build an MDOProblem from the aircraft, auto-discovering design variables.

        Any field set using ``opti.variable()`` is automatically registered
        as a design variable — no string paths required.

        Parameters
        ----------
        aircraft : Aircraft
        condition : FlightCondition, optional
            Single flight condition (backward compat).
        conditions : dict, optional
            Dict of name -> FlightCondition for multi-condition optimization.
            Not yet supported by MDOProblem; reserved for future use.
        disciplines : list of str
            Disciplines to run, e.g. ``["aero"]`` or ``["aero", "stability"]``.
            Pass ``None`` to auto-infer from objective/constraint paths.
            Note: explicit discipline control is added in a future task.
        objective : Objective or list of Objective
        constraints : list of Constraint, optional
        aero_result : AeroResult or None
            Pre-computed aero result. Reserved for future use.
        """
        from aerisplane.mdo.problem import ChoiceVar, DesignVar, MDOProblem
        from aerisplane.mdo._paths import _set_dv_value

        # Discover _Var (continuous) and _Choice (discrete) fields
        found_vars, found_choices = _discover_vars(aircraft)

        design_variables = [
            DesignVar(
                path=path,
                lower=var._lower,
                upper=var._upper,
                scale=var._scale,
                integrality=var._is_integer,
            )
            for path, var in found_vars.items()
        ]

        # Replace _Choice sentinels with initial values; build ChoiceVar list
        cv_list = []
        for path, choice in found_choices.items():
            _set_dv_value(aircraft, path, choice.options[choice.init_idx])
            cv_list.append(ChoiceVar(
                path=path,
                options=choice.options,
                init_idx=choice.init_idx,
            ))

        return MDOProblem(
            aircraft=aircraft,
            condition=condition,
            design_variables=design_variables,
            constraints=constraints or [],
            objective=objective,
            mission=mission,
            airfoil_pools=airfoil_pools,
            alpha=alpha,
            aero_method=aero_method,
            load_factor=load_factor,
            throttle=throttle,
            choice_variables=cv_list,
        )
