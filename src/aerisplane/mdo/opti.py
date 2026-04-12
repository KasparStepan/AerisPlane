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
        return obj

    def __deepcopy__(self, memo):
        new = _Var(float(self), self._lower, self._upper, self._scale)
        new._var_id = self._var_id
        memo[id(self)] = new
        return new


def _discover_vars(
    obj: Any,
    path: str = "",
    found_vars: Optional[dict] = None,
    found_choices: Optional[dict] = None,
) -> tuple:
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
