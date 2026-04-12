import pytest
from dataclasses import dataclass, field
from aerisplane.mdo.opti import _Var, _discover_vars


@dataclass
class _Leaf:
    chord: float = 0.3
    twist: float = 0.0


@dataclass
class _Branch:
    leaves: list = field(default_factory=list)
    span: float = 1.2


@dataclass
class _Root:
    branches: list = field(default_factory=list)
    mass: float = 1.0


def test_var_is_float():
    v = _Var(0.30, lower=0.10, upper=0.80)
    assert isinstance(v, float)
    assert float(v) == pytest.approx(0.30)


def test_var_attrs():
    v = _Var(0.30, lower=0.10, upper=0.80, scale=0.1)
    assert v._lower == pytest.approx(0.10)
    assert v._upper == pytest.approx(0.80)
    assert v._scale == pytest.approx(0.1)


def test_var_default_scale():
    v = _Var(1.2, lower=0.5, upper=2.0)
    assert v._scale == pytest.approx(1.0)


def test_var_unique_ids():
    v1 = _Var(0.3, lower=0.1, upper=0.8)
    v2 = _Var(0.3, lower=0.1, upper=0.8)
    assert v1._var_id != v2._var_id


def test_discover_vars_nested():
    leaf0 = _Leaf(chord=_Var(0.26, lower=0.10, upper=0.50))
    leaf1 = _Leaf(chord=_Var(0.15, lower=0.05, upper=0.30))
    branch = _Branch(leaves=[leaf0, leaf1], span=_Var(1.2, lower=0.8, upper=2.0))
    root = _Root(branches=[branch])

    found, _ = _discover_vars(root)
    assert set(found.keys()) == {
        "branches[0].leaves[0].chord",
        "branches[0].leaves[1].chord",
        "branches[0].span",
    }


def test_discover_vars_empty():
    leaf = _Leaf(chord=0.3)
    root = _Root(branches=[_Branch(leaves=[leaf])])
    found, _ = _discover_vars(root)
    assert found == {}


def test_discover_vars_float_field():
    root = _Root(mass=_Var(1.0, lower=0.5, upper=3.0))
    found, _ = _discover_vars(root)
    assert "mass" in found
    assert found["mass"]._lower == pytest.approx(0.5)
