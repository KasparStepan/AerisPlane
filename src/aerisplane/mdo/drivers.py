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

        # Initialise CSV log
        csv_file = open(log_path, "a") if log_path else None
        if csv_file and csv_file.tell() == 0:
            _write_csv_header(csv_file, p)

        t_start = time.time()
        best = {"x": None, "obj": float("inf"), "constraints_ok": False}

        chk_interval = checkpoint_interval or max(1, getattr(p, "_n_vars", 2) * 4)

        def wrapped_obj(x):
            obj = p.objective_function(x)
            violations = p.constraint_functions(x)
            constraints_ok = bool(np.all(violations <= 0))

            # Track best feasible (or best infeasible if nothing feasible yet)
            if constraints_ok and obj < best["obj"]:
                best["x"] = x.copy()
                best["obj"] = obj
                best["constraints_ok"] = True
            elif best["x"] is None:
                best["x"] = x.copy()
                best["obj"] = obj

            if verbose:
                _LOG.info("[%5d] obj=%.4g  best=%.4g  t=%.1fs",
                          p._n_evals, obj, best["obj"], time.time() - t_start)

            if csv_file:
                ev = p._cache.get(tuple(np.round(x, 10)), {})
                cv = ev.get("constraint_values", {})
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

            # Per-eval summary
            if report_interval and p._n_evals % report_interval == 0:
                _print_summary(p, best, t_start)

            # User callback
            if callback is not None:
                snap = _build_snapshot(p, best, t_start)
                if callback(snap) == "stop":
                    raise _EarlyStop()

            return obj

        # scipy_minimize uses dict constraints; differential_evolution and shgo
        # use NonlinearConstraint objects (scipy ≥ 1.7).
        nonlinear_constraints = []
        dict_constraints = []
        if p._constraints:
            from scipy.optimize import NonlinearConstraint

            def _neg_violations(x):
                return -p.constraint_functions(x)

            n_c = len(p.constraint_functions(lo))   # pre-compute shape
            nonlinear_constraints = [
                NonlinearConstraint(_neg_violations, lb=0.0, ub=np.inf)
            ] if n_c > 0 else []
            dict_constraints = [{"type": "ineq", "fun": _neg_violations}] if n_c > 0 else []

        try:
            if method == "scipy_de":
                # Resume from checkpoint if available
                if checkpoint_path:
                    ckpt = load_checkpoint(checkpoint_path)
                    if ckpt is not None and isinstance(ckpt.get("cache"), dict):
                        p._cache.update(ckpt["cache"])

                de_opts = {k: v for k, v in options.items()}
                res = differential_evolution(
                    wrapped_obj,
                    bounds=bounds,
                    constraints=nonlinear_constraints,
                    integrality=integrality,
                    **de_opts,
                )

            elif method == "scipy_minimize":
                x0 = p._x0_scaled()
                res = minimize(
                    wrapped_obj,
                    x0=x0,
                    bounds=bounds,
                    constraints=dict_constraints or None,
                    **options,
                )

            elif method == "scipy_shgo":
                res = shgo(
                    wrapped_obj,
                    bounds=bounds,
                    constraints=nonlinear_constraints,
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

        opts = dict(options)
        pop_size = opts.pop("pop_size", 20)
        n_gen = opts.pop("gen", 100)
        seed = opts.pop("seed", 42)

        # Resume from checkpoint if available
        pop = None
        if checkpoint_path:
            ckpt = load_checkpoint(checkpoint_path)
            if ckpt is not None and "pygmo_population" in ckpt:
                cache_pkl = checkpoint_path + ".cache.pkl"
                if Path(cache_pkl).exists():
                    p.load_cache(cache_pkl)
                pop = pg.population(pg_prob)
                pop.set_x(ckpt["pygmo_population"]["x"])
                pop.set_f(ckpt["pygmo_population"]["f"])

        if pop is None:
            pop = pg.population(pg_prob, size=pop_size, seed=seed)

        if method == "pygmo_de":
            algo = pg.algorithm(pg.de(gen=1, **opts))
        elif method == "pygmo_sade":
            algo = pg.algorithm(pg.sade(gen=1, **opts))
        elif method == "pygmo_nsga2":
            algo = pg.algorithm(pg.nsga2(gen=1, **opts))
        else:
            raise ValueError(f"Unknown pygmo method '{method}'.")

        t_start = time.time()
        chk_interval = checkpoint_interval or pop_size
        best = {"x": None, "obj": float("inf"), "constraints_ok": False}

        for gen in range(n_gen):
            pop = algo.evolve(pop)

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
    improvement = (best["obj"] - init_obj) / (abs(init_obj) + 1e-30) * 100
    last100 = history[-100:] if len(history) >= 100 else history
    imp100 = (
        (last100[0] - last100[-1]) / (abs(last100[0]) + 1e-30) * 100
        if len(last100) >= 2 else 0.0
    )
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
            pool.options[max(0, min(init_idx, len(pool.options) - 1))],
            pool.options[max(0, min(opt_idx,  len(pool.options) - 1))],
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
