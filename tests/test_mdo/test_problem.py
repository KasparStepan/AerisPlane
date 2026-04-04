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
