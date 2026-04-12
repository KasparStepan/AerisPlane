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


import aerisplane as ap
from aerisplane.mdo.opti import Opti
from aerisplane.mdo.problem import MDOProblem, Objective, Constraint


@pytest.fixture
def simple_aircraft():
    """Minimal two-variable aircraft for Opti tests."""
    from aerisplane.catalog.materials import carbon_fiber_tube, petg
    spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.015, wall_thickness=0.001),
    )
    skin = ap.Skin(material=petg, thickness=0.0008)
    opti = Opti()
    main_wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(
                xyz_le=[0.0, 0.0, 0.0],
                chord=opti.variable(0.26, lower=0.10, upper=0.50),
                airfoil=ap.Airfoil("naca2412"),
                spar=spar,
                skin=skin,
            ),
            ap.WingXSec(
                xyz_le=[0.03, 0.75, 0.05],
                chord=opti.variable(0.15, lower=0.05, upper=0.30),
                airfoil=ap.Airfoil("naca2412"),
            ),
        ],
        symmetric=True,
    )
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.85, 0.0, 0.0], chord=0.12,
                        airfoil=ap.Airfoil("naca0012")),
            ap.WingXSec(xyz_le=[0.87, 0.25, 0.01], chord=0.08,
                        airfoil=ap.Airfoil("naca0012")),
        ],
        symmetric=True,
    )
    aircraft = ap.Aircraft(
        name="test",
        wings=[main_wing, htail],
        fuselages=[],
        xyz_ref=[0.12, 0.0, 0.0],
    )
    return opti, aircraft


def test_opti_variable_returns_var():
    opti = Opti()
    v = opti.variable(0.30, lower=0.10, upper=0.80)
    assert isinstance(v, _Var)
    assert float(v) == pytest.approx(0.30)


def test_opti_variable_tracked():
    opti = Opti()
    v = opti.variable(0.30, lower=0.10, upper=0.80)
    assert v._var_id in opti._vars


def test_opti_problem_returns_mdo_problem(simple_aircraft):
    opti, aircraft = simple_aircraft
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = opti.problem(
        aircraft=aircraft,
        condition=cond,
        disciplines=["aero"],
        objective=Objective("aero.CL_over_CD", maximize=True),
    )
    assert isinstance(problem, MDOProblem)


def test_opti_problem_discovers_two_vars(simple_aircraft):
    opti, aircraft = simple_aircraft
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = opti.problem(
        aircraft=aircraft,
        condition=cond,
        disciplines=["aero"],
        objective=Objective("aero.CL_over_CD", maximize=True),
    )
    assert len(problem._dvars) == 2


def test_opti_problem_var_bounds(simple_aircraft):
    opti, aircraft = simple_aircraft
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = opti.problem(
        aircraft=aircraft,
        condition=cond,
        disciplines=["aero"],
        objective=Objective("aero.CL_over_CD", maximize=True),
    )
    paths = {dv.path: dv for dv in problem._dvars}
    assert "wings[0].xsecs[0].chord" in paths
    assert paths["wings[0].xsecs[0].chord"].lower == pytest.approx(0.10)
    assert paths["wings[0].xsecs[0].chord"].upper == pytest.approx(0.50)
