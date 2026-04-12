import pytest
import numpy as np
import aerisplane as ap
from aerisplane.mdo.problem import MDOProblem, DesignVar, Objective, Constraint
from aerisplane.mdo.drivers import PymooDriver


@pytest.fixture
def test_aircraft_fixture():
    from aerisplane.catalog.materials import carbon_fiber_tube, petg
    spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.015, wall_thickness=0.001),
    )
    skin = ap.Skin(material=petg, thickness=0.0008)
    return ap.Aircraft(
        name="test",
        wings=[
            ap.Wing(
                name="main_wing",
                xsecs=[
                    ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.26,
                                airfoil=ap.Airfoil("naca2412"), spar=spar, skin=skin),
                    ap.WingXSec(xyz_le=[0.03, 0.75, 0.05], chord=0.15,
                                airfoil=ap.Airfoil("naca2412")),
                ],
                symmetric=True,
            ),
            ap.Wing(
                name="htail",
                xsecs=[
                    ap.WingXSec(xyz_le=[0.85, 0.0, 0.0], chord=0.12,
                                airfoil=ap.Airfoil("naca0012")),
                    ap.WingXSec(xyz_le=[0.87, 0.25, 0.01], chord=0.08,
                                airfoil=ap.Airfoil("naca0012")),
                ],
                symmetric=True,
            ),
        ],
        fuselages=[],
        xyz_ref=[0.12, 0.0, 0.0],
    )


@pytest.fixture
def simple_problem(test_aircraft_fixture):
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    return MDOProblem(
        aircraft=test_aircraft_fixture,
        condition=cond,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.15, upper=0.40),
        ],
        constraints=[],
        objective=Objective("aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )


def test_pymoo_driver_instantiates(simple_problem):
    driver = PymooDriver(simple_problem)
    assert driver.problem is simple_problem


def test_pymoo_de_method_accepted(simple_problem):
    pytest.importorskip("pymoo")
    result = simple_problem.optimize(
        method="pymoo_de",
        options={"pop_size": 5, "n_gen": 3, "seed": 0},
        verbose=False,
    )
    from aerisplane.mdo.result import OptimizationResult
    assert isinstance(result, OptimizationResult)


def test_pymoo_nsga2_method_accepted(simple_problem):
    pytest.importorskip("pymoo")
    result = simple_problem.optimize(
        method="pymoo_nsga2",
        options={"pop_size": 6, "n_gen": 2, "seed": 0},
        verbose=False,
    )
    from aerisplane.mdo.result import OptimizationResult
    assert isinstance(result, OptimizationResult)


def test_unknown_pymoo_method_raises(simple_problem):
    with pytest.raises(ValueError, match="Unknown method"):
        simple_problem.optimize(method="pymoo_unknown")


def test_pygmo_method_raises_with_helpful_message(simple_problem):
    with pytest.raises(ValueError, match="pygmo"):
        simple_problem.optimize(method="pygmo_de")
