import pytest
import aerisplane as ap
from aerisplane.mdo.problem import MDOProblem, DesignVar, Objective, Constraint


@pytest.fixture
def two_condition_aircraft():
    from aerisplane.catalog.materials import carbon_fiber_tube, petg
    spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.015, wall_thickness=0.001),
    )
    skin = ap.Skin(material=petg, thickness=0.0008)
    main_wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.26,
                        airfoil=ap.Airfoil("naca2412"), spar=spar, skin=skin),
            ap.WingXSec(xyz_le=[0.03, 0.75, 0.05], chord=0.15,
                        airfoil=ap.Airfoil("naca2412")),
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
    return ap.Aircraft(
        name="test",
        wings=[main_wing, htail],
        fuselages=[],
        xyz_ref=[0.12, 0.0, 0.0],
    )


def test_multicondition_problem_accepts_conditions_dict(two_condition_aircraft):
    cruise = ap.FlightCondition(velocity=15.0, altitude=500.0, alpha=4.0)
    climb  = ap.FlightCondition(velocity=10.0, altitude=100.0, alpha=8.0)
    problem = MDOProblem(
        aircraft=two_condition_aircraft,
        conditions={"cruise": cruise, "climb": climb},
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.15, upper=0.40),
        ],
        constraints=[],
        objective=Objective("cruise.aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )
    assert problem._conditions is not None
    assert set(problem._conditions.keys()) == {"cruise", "climb"}


def test_multicondition_simulate_has_both_conditions(two_condition_aircraft):
    cruise = ap.FlightCondition(velocity=15.0, altitude=500.0, alpha=4.0)
    climb  = ap.FlightCondition(velocity=10.0, altitude=100.0, alpha=8.0)
    problem = MDOProblem(
        aircraft=two_condition_aircraft,
        conditions={"cruise": cruise, "climb": climb},
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.15, upper=0.40),
        ],
        constraints=[],
        objective=Objective("cruise.aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )
    results = problem.simulate()
    assert "cruise" in results
    assert "climb" in results
    assert "aero" in results["cruise"]
    assert "aero" in results["climb"]


def test_multicondition_constraint_on_climb(two_condition_aircraft):
    cruise = ap.FlightCondition(velocity=15.0, altitude=500.0, alpha=4.0)
    climb  = ap.FlightCondition(velocity=10.0, altitude=100.0, alpha=8.0)
    problem = MDOProblem(
        aircraft=two_condition_aircraft,
        conditions={"cruise": cruise, "climb": climb},
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.15, upper=0.40),
        ],
        constraints=[Constraint("climb.aero.CL", lower=0.5)],
        objective=Objective("cruise.aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )
    x0 = problem._x0_scaled()
    violations = problem.constraint_functions(x0)
    assert violations.shape == (1,)


def test_multicondition_validate_rejects_bad_cond_name(two_condition_aircraft):
    cruise = ap.FlightCondition(velocity=15.0, altitude=500.0, alpha=4.0)
    with pytest.raises(ValueError, match="Unknown condition"):
        MDOProblem(
            aircraft=two_condition_aircraft,
            conditions={"cruise": cruise},
            design_variables=[],
            constraints=[],
            objective=Objective("typo.aero.CL_over_CD", maximize=True),
            disciplines=["aero"],
        )


def test_single_condition_backward_compat(two_condition_aircraft):
    """condition= (singular) still works exactly as before."""
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = MDOProblem(
        aircraft=two_condition_aircraft,
        condition=cond,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.15, upper=0.40),
        ],
        constraints=[],
        objective=Objective("aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )
    results = problem.simulate()
    assert "aero" in results    # flat results dict, not nested by condition name
    assert "cruise" not in results
