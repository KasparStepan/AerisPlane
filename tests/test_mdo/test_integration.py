"""End-to-end MDO integration test.

Uses real discipline chain (VLM aero) but a tiny DE run (2 iterations,
popsize 4) to keep wall time under 60 s.
"""
import numpy as np
import pytest

import aerisplane as ap
from aerisplane.mdo import MDOProblem, DesignVar, Constraint, Objective
from aerisplane.mdo.result import OptimizationResult


@pytest.fixture(scope="module")
def integration_aircraft():
    """Conventional aircraft fixture for integration tests."""
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
        control_surfaces=[
            ap.ControlSurface(name="aileron", span_start=0.55, span_end=0.92,
                              chord_fraction=0.28),
        ],
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
        control_surfaces=[
            ap.ControlSurface(name="elevator", span_start=0.0, span_end=1.0,
                              chord_fraction=0.40),
        ],
    )
    vtail = ap.Wing(
        name="vtail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.86, 0.0, 0.0], chord=0.13,
                        airfoil=ap.Airfoil("naca0012")),
            ap.WingXSec(xyz_le=[0.89, 0.0, 0.18], chord=0.08,
                        airfoil=ap.Airfoil("naca0012")),
        ],
        symmetric=False,
        control_surfaces=[
            ap.ControlSurface(name="rudder", span_start=0.0, span_end=1.0,
                              chord_fraction=0.40),
        ],
    )
    fuse = ap.Fuselage(
        name="fuselage",
        xsecs=[
            ap.FuselageXSec(x=0.0, radius=0.045),
            ap.FuselageXSec(x=0.95, radius=0.020),
        ],
        material=petg,
        wall_thickness=0.001,
    )
    motor = ap.Motor(name="test_motor", kv=900, no_load_current=0.5,
                     resistance=0.1, mass=0.120, max_current=40.0)
    prop = ap.Propeller(diameter=0.254, pitch=0.127, mass=0.025)
    battery = ap.Battery(name="test_bat", capacity_ah=3.0, nominal_voltage=14.8,
                         cell_count=4, c_rating=25.0, mass=0.280)
    esc = ap.ESC(name="test_esc", max_current=40.0, mass=0.030)
    propulsion = ap.PropulsionSystem(motor=motor, propeller=prop, battery=battery, esc=esc,
                                     position=np.array([0.0, 0.0, 0.0]))
    return ap.Aircraft(name="test_uav", wings=[main_wing, htail, vtail],
                       fuselages=[fuse], propulsion=propulsion)


@pytest.fixture(scope="module")
def integration_condition():
    return ap.FlightCondition(velocity=16.0, altitude=80.0, alpha=4.0)


@pytest.mark.slow
def test_full_mdo_run(integration_aircraft, integration_condition):
    """Minimise total mass subject to static margin ≥ 4% MAC.

    Only 2 design variables, 2 DE generations, popsize 4 → ~8 evals.
    Uses alpha=4.0 fixed to avoid trim solve overhead.
    """
    problem = MDOProblem(
        aircraft=integration_aircraft,
        condition=integration_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.20, upper=0.32),
            DesignVar("wings[0].xsecs[1].chord", lower=0.10, upper=0.20),
        ],
        constraints=[
            Constraint("stability.static_margin", lower=0.04),
        ],
        objective=Objective("weights.total_mass", maximize=False),
        alpha=4.0,
    )

    result = problem.optimize(
        method="scipy_de",
        options={"maxiter": 2, "popsize": 4, "seed": 7},
        verbose=False,
    )

    assert isinstance(result, OptimizationResult)
    assert result.n_evaluations >= 4
    assert np.isfinite(result.objective_optimal)
    # objective_optimal should be a plausible aircraft mass (< 10 kg for this test aircraft)
    assert 0.0 < result.objective_optimal < 10.0
    assert result.weights is not None
    assert result.stability is not None
    assert isinstance(result.report(), str)


def test_cache_reduces_redundant_evals(integration_aircraft, integration_condition):
    """evaluate() called twice with the same x should hit cache."""
    problem = MDOProblem(
        aircraft=integration_aircraft,
        condition=integration_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.20, upper=0.32),
        ],
        constraints=[Constraint("stability.static_margin", lower=0.04)],
        objective=Objective("weights.total_mass", maximize=False),
        alpha=4.0,
    )
    lo, hi = problem.get_bounds()
    x = (lo + hi) / 2.0
    problem.evaluate(x)
    n1 = problem._n_evals
    problem.evaluate(x)
    assert problem._n_evals == n1   # cache hit, no new eval


def test_opti_api_aero_only():
    """Full smoke test: Opti inline variables -> aero-only optimization."""
    from aerisplane.mdo import Opti, Objective
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
                chord=opti.variable(0.26, lower=0.15, upper=0.40),
                airfoil=ap.Airfoil("naca2412"),
                spar=spar,
                skin=skin,
            ),
            ap.WingXSec(
                xyz_le=[0.03, 0.75, 0.05],
                chord=opti.variable(0.15, lower=0.08, upper=0.25),
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
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)

    problem = opti.problem(
        aircraft=aircraft,
        condition=cond,
        disciplines=["aero"],
        objective=Objective("aero.CL_over_CD", maximize=True),
    )

    assert len(problem._dvars) == 2
    result = problem.optimize(
        method="scipy_de",
        options={"maxiter": 3, "popsize": 4, "seed": 0},
        verbose=False,
    )
    assert result.aero is not None
    assert isinstance(result.objective_optimal, float)


def test_opti_api_simulate_at_baseline():
    """simulate() runs discipline chain at baseline without optimizing."""
    from aerisplane.mdo import Opti, Objective
    from aerisplane.catalog.materials import carbon_fiber_tube, petg

    spar = ap.Spar(
        position=0.25,
        material=carbon_fiber_tube,
        section=ap.TubeSection(outer_diameter=0.015, wall_thickness=0.001),
    )
    skin = ap.Skin(material=petg, thickness=0.0008)

    opti = Opti()
    aircraft = ap.Aircraft(
        name="test",
        wings=[
            ap.Wing(
                name="main_wing",
                xsecs=[
                    ap.WingXSec(
                        xyz_le=[0.0, 0.0, 0.0],
                        chord=opti.variable(0.26, lower=0.15, upper=0.40),
                        airfoil=ap.Airfoil("naca2412"),
                        spar=spar,
                        skin=skin,
                    ),
                    ap.WingXSec(
                        xyz_le=[0.03, 0.75, 0.05],
                        chord=0.15,
                        airfoil=ap.Airfoil("naca2412"),
                    ),
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
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)

    problem = opti.problem(
        aircraft=aircraft,
        condition=cond,
        disciplines=["aero"],
        objective=Objective("aero.CL_over_CD", maximize=True),
    )

    # simulate() is Task 9 — skip this test if it's not implemented yet
    if not hasattr(problem, "simulate"):
        pytest.skip("simulate() not yet implemented (Task 9)")

    results = problem.simulate()
    assert "aero" in results
    assert "weights" in results
    assert hasattr(results["aero"], "CL_over_CD")
