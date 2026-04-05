import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock

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
    problem._constraints = []
    problem._dvars = []
    problem._pool_entries = []
    problem._scales = np.ones(2)

    # objective: minimise sum of squares
    def obj_fn(x):
        problem._n_evals += 1
        problem._history.append((x.copy(), float(np.sum(x**2)), {}))
        return float(np.sum(x**2))

    problem.objective_function.side_effect = obj_fn
    problem.constraint_functions.return_value = np.array([])
    return problem


def test_scipy_driver_de_runs():
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
