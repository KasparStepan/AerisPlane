"""Pluggable discipline registry for MDO evaluation chain.

Adding a new discipline::

    from aerisplane.mdo.registry import default_registry

    def my_aeroelastic_runner(aircraft, condition, results, **kwargs):
        from aerisplane.aeroelastic import analyze
        return analyze(aircraft, results["aero"], results["structures"])

    default_registry.register("aeroelastic", my_aeroelastic_runner, after="structures")

The discipline is then available as ``disciplines=["aeroelastic"]`` in MDOProblem.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

DISCIPLINE_ORDER = [
    "weights",
    "aero",
    "structures",
    "stability",
    "control",
    "propulsion",
    "mission",
]


class DisciplineRegistry:
    """Ordered registry of discipline runner functions.

    Each runner signature::

        def runner(aircraft, condition, results: dict, **kwargs) -> Any

    where *results* contains all results computed so far in the chain.
    The return value is stored in ``results[discipline_name]``.
    """

    def __init__(self) -> None:
        self._runners: dict[str, Callable] = {}
        self._order: list[str] = list(DISCIPLINE_ORDER)
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._runners["weights"] = _run_weights
        self._runners["aero"] = _run_aero
        self._runners["structures"] = _run_structures
        self._runners["stability"] = _run_stability
        self._runners["control"] = _run_control
        self._runners["propulsion"] = _run_propulsion
        self._runners["mission"] = _run_mission

    def has(self, name: str) -> bool:
        return name in self._runners

    def ordered_names(self) -> list[str]:
        """Return all registered discipline names in execution order."""
        return [n for n in self._order if n in self._runners]

    def register(
        self,
        name: str,
        runner: Callable,
        after: Optional[str] = None,
    ) -> None:
        """Register a new discipline runner.

        Parameters
        ----------
        name : str
            Unique discipline name, e.g. ``"aeroelastic"``.
        runner : callable
            ``runner(aircraft, condition, results, **kwargs) -> Any``
        after : str or None
            Insert immediately after *after* in execution order.
            ``None`` → append at the end.

        Raises
        ------
        ValueError
            If *name* is already registered.
        """
        if name in self._runners:
            raise ValueError(
                f"Discipline '{name}' is already registered. "
                "Use replace() to overwrite it."
            )
        self._runners[name] = runner
        if after is not None and after in self._order:
            idx = self._order.index(after)
            self._order.insert(idx + 1, name)
        else:
            self._order.append(name)

    def replace(self, name: str, runner: Callable) -> None:
        """Replace an existing discipline runner."""
        if name not in self._runners:
            raise ValueError(f"Discipline '{name}' is not registered.")
        self._runners[name] = runner

    def run_chain(
        self,
        disciplines: list,
        aircraft,
        condition,
        aero_result=None,
        **kwargs,
    ) -> dict:
        """Run the requested disciplines in order and return the results dict.

        Parameters
        ----------
        disciplines : list of str
            Subset of registered discipline names to run.
        aircraft : Aircraft
        condition : FlightCondition
        aero_result : AeroResult or None
            Pre-computed aero result. When provided, ``"aero"`` is skipped
            and this result is injected into the results dict directly.
        **kwargs
            Forwarded to every runner (e.g. ``aero_method``, ``load_factor``).
        """
        results: dict[str, Any] = {}

        if aero_result is not None:
            results["aero"] = aero_result

        for name in self.ordered_names():
            if name not in disciplines:
                continue
            if name == "aero" and aero_result is not None:
                continue
            runner = self._runners[name]
            results[name] = runner(aircraft, condition, results, **kwargs)

        return results


# ── Default discipline runners ─────────────────────────────────────────────────

def _run_weights(aircraft, condition, results, **kwargs):
    import aerisplane.weights as weights_mod
    return weights_mod.analyze(aircraft)


def _run_aero(aircraft, condition, results, **kwargs):
    import aerisplane.aero as aero_mod
    return aero_mod.analyze(aircraft, condition, method=kwargs.get("aero_method", "vlm"))


def _run_structures(aircraft, condition, results, **kwargs):
    import aerisplane.structures as struct_mod
    return struct_mod.analyze(
        aircraft,
        results["aero"],
        results["weights"],
        n_limit=kwargs.get("load_factor", 3.5),
        safety_factor=kwargs.get("safety_factor", 1.5),
    )


def _run_stability(aircraft, condition, results, **kwargs):
    import aerisplane.stability as stab_mod
    return stab_mod.analyze(
        aircraft, condition, results["weights"],
        aero_method=kwargs.get("aero_method", "vlm"),
    )


def _run_control(aircraft, condition, results, **kwargs):
    import aerisplane.control as ctrl_mod
    return ctrl_mod.analyze(
        aircraft, condition, results["weights"], results["stability"],
        aero_method=kwargs.get("aero_method", "vlm"),
    )


def _run_propulsion(aircraft, condition, results, **kwargs):
    from aerisplane.propulsion import analyze as propulsion_analyze
    return propulsion_analyze(aircraft, condition, throttle=kwargs.get("throttle", 1.0))


def _run_mission(aircraft, condition, results, **kwargs):
    import aerisplane.mission as mission_mod
    mission = kwargs.get("mission")
    if mission is None:
        return None
    return mission_mod.analyze(
        aircraft, results["weights"], mission,
        aero_method=kwargs.get("aero_method", "vlm"),
    )


# ── Module-level default registry and convenience function ─────────────────────

default_registry = DisciplineRegistry()


def register_discipline(name: str, runner: Callable, after: str = None) -> None:
    """Register a new discipline in the module-level default registry.

    Convenience wrapper around ``default_registry.register()``.

    Example::

        from aerisplane.mdo import register_discipline

        def my_aeroelastic(aircraft, condition, results, **kwargs):
            from aerisplane.aeroelastic import analyze
            return analyze(aircraft, results["aero"], results["structures"])

        register_discipline("aeroelastic", my_aeroelastic, after="structures")
    """
    default_registry.register(name, runner, after=after)
