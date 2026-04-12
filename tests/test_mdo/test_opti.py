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


from aerisplane.mdo.opti import _Choice


def test_choice_stores_options():
    c = _Choice(options=["a", "b", "c"], init_idx=1)
    assert c.options == ["a", "b", "c"]
    assert c.init_idx == 1


def test_ranked_choice_reorders_by_score():
    # options sorted ascending by score: b(5) < a(10) < c(15)
    c = _Choice(options=["a", "b", "c"], init_idx=0, scores=[10, 5, 15])
    assert c.options == ["b", "a", "c"]
    # original init_idx=0 was "a"; after reorder "a" is at index 1
    assert c.init_idx == 1


def test_ranked_choice_unique_ids():
    c1 = _Choice(options=["a", "b"], init_idx=0)
    c2 = _Choice(options=["a", "b"], init_idx=0)
    assert c1._var_id != c2._var_id


def test_discover_vars_finds_choice():
    from dataclasses import dataclass as dc2, field as f2
    @dc2
    class _Leaf2:
        chord: float = 0.3
        airfoil: object = None

    @dc2
    class _Root2:
        leaf: object = None

    choice = _Choice(options=["naca2412", "e423"], init_idx=0)
    root = _Root2(leaf=_Leaf2(chord=0.3, airfoil=choice))
    found_vars, found_choices = _discover_vars(root)
    assert "leaf.airfoil" in found_choices
    assert found_choices["leaf.airfoil"] is choice


def test_opti_choice_returns_choice():
    opti = Opti()
    c = opti.choice(options=["naca2412", "e423", "s1223"], init=0)
    assert isinstance(c, _Choice)
    assert c.options == ["naca2412", "e423", "s1223"]
    assert c.init_idx == 0


def test_opti_ranked_choice_sorts():
    opti = Opti()
    c = opti.ranked_choice(
        options=["naca2412", "e423", "s1223"],
        scores=[8.0, 11.5, 10.0],
        init=0,  # "naca2412" originally at index 0
    )
    # sorted by score ascending: naca2412(8.0) < s1223(10.0) < e423(11.5)
    assert c.options[0] == "naca2412"
    assert c.options[1] == "s1223"
    assert c.options[2] == "e423"
    # naca2412 was at original idx 0; after sort it's at new idx 0
    assert c.init_idx == 0


def test_opti_problem_with_choice(simple_aircraft):
    """opti.problem() creates ChoiceVar from _Choice fields."""
    from aerisplane.mdo.problem import ChoiceVar
    opti, aircraft = simple_aircraft
    choice = opti.choice(
        options=[ap.Airfoil("naca2412"), ap.Airfoil("e423")],
        init=0,
    )
    aircraft.wings[0].xsecs[0].airfoil = choice

    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = opti.problem(
        aircraft=aircraft,
        condition=cond,
        disciplines=["aero"],
        objective=Objective("aero.CL_over_CD", maximize=True),
    )
    # The field must be replaced with the actual Airfoil object
    assert isinstance(aircraft.wings[0].xsecs[0].airfoil, ap.Airfoil)
    # The problem must have one ChoiceVar
    assert len(problem._choice_vars) == 1
    cv = problem._choice_vars[0]
    assert isinstance(cv, ChoiceVar)
    assert cv.path == "wings[0].xsecs[0].airfoil"
    assert len(cv.options) == 2


from aerisplane.mdo.opti import _IntVar
import numpy as np


def test_int_var_is_float():
    v = _IntVar(4, lower=2, upper=8)
    assert isinstance(v, float)
    assert float(v) == pytest.approx(4.0)


def test_int_var_has_integrality_flag():
    v = _IntVar(4, lower=2, upper=8)
    assert v._is_integer is True


def test_int_var_scale_fixed_at_one():
    v = _IntVar(4, lower=2, upper=8)
    assert v._scale == pytest.approx(1.0)


def test_opti_integer_variable_returns_int_var():
    opti = Opti()
    v = opti.integer_variable(4, lower=2, upper=8)
    assert isinstance(v, _IntVar)
    assert float(v) == pytest.approx(4.0)


def test_opti_integer_variable_creates_integer_desvar(simple_aircraft):
    """MDOProblem from integer variable has integrality=True for that var."""
    opti2 = Opti()
    opti_base, aircraft = simple_aircraft
    # Override one chord with an integer variable
    aircraft.wings[0].xsecs[0].chord = opti2.integer_variable(3, lower=2, upper=6)
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = opti2.problem(
        aircraft=aircraft,
        condition=cond,
        disciplines=["aero"],
        objective=Objective("aero.CL_over_CD", maximize=True),
    )
    int_dvars = [dv for dv in problem._dvars if dv.integrality]
    assert len(int_dvars) == 1
    assert int_dvars[0].path == "wings[0].xsecs[0].chord"
    assert problem._integrality[0] is np.bool_(True)
