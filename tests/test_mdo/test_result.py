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
