import pytest
import numpy as np
import aerisplane as ap
from aerisplane.mdo.problem import DesignVar, AirfoilPool, Constraint, Objective, MDOProblem


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


@pytest.fixture
def test_aircraft():
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


@pytest.fixture
def cruise_condition():
    return ap.FlightCondition(velocity=16.0, altitude=80.0, alpha=4.0)


@pytest.fixture
def simple_problem(test_aircraft, cruise_condition):
    return MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38),
            DesignVar("wings[0].xsecs[1].chord", lower=0.09, upper=0.22),
        ],
        constraints=[Constraint("stability.static_margin", lower=0.05)],
        objective=Objective("weights.total_mass", maximize=False),
    )


def test_construction_succeeds(test_aircraft, cruise_condition):
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38)],
        constraints=[Constraint("stability.static_margin", lower=0.05)],
        objective=Objective("weights.total_mass", maximize=False),
    )
    assert problem is not None


def test_invalid_dv_path_raises(test_aircraft, cruise_condition):
    with pytest.raises(ValueError, match="typo"):
        MDOProblem(
            aircraft=test_aircraft,
            condition=cruise_condition,
            design_variables=[DesignVar("wings[0].xsecs[0].typo", lower=0.1, upper=0.5)],
            constraints=[Constraint("weights.total_mass", upper=5.0)],
            objective=Objective("weights.total_mass", maximize=False),
        )


def test_mission_none_with_mission_path_raises(test_aircraft, cruise_condition):
    with pytest.raises(ValueError, match="mission"):
        MDOProblem(
            aircraft=test_aircraft,
            condition=cruise_condition,
            design_variables=[DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38)],
            constraints=[Constraint("mission.feasible", equals=True)],
            objective=Objective("weights.total_mass", maximize=False),
            mission=None,
        )


def test_get_bounds(simple_problem):
    lo, hi = simple_problem.get_bounds()
    assert lo[0] == pytest.approx(0.18)
    assert hi[0] == pytest.approx(0.38)
    assert lo[1] == pytest.approx(0.09)
    assert hi[1] == pytest.approx(0.22)


def test_inferred_disciplines_stability(simple_problem):
    assert "stability" in simple_problem._disciplines
    assert "weights" in simple_problem._disciplines
    assert "aero" in simple_problem._disciplines


def test_inferred_disciplines_no_mission(simple_problem):
    assert "mission" not in simple_problem._disciplines


def test_n_vars_with_pool(test_aircraft, cruise_condition):
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38)],
        constraints=[Constraint("weights.total_mass", upper=5.0)],
        objective=Objective("weights.total_mass", maximize=False),
        airfoil_pools={"wings[0]": AirfoilPool(options=["naca2412", "naca4412"], xsecs=[0])},
    )
    lo, hi = problem.get_bounds()
    assert len(lo) == 2   # 1 DesignVar + 1 pool entry
    assert lo[1] == 0.0
    assert hi[1] == 1.0


def test_invalid_constraint_path_prefix_raises(test_aircraft, cruise_condition):
    with pytest.raises(ValueError, match="unknown discipline prefix"):
        MDOProblem(
            aircraft=test_aircraft,
            condition=cruise_condition,
            design_variables=[DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38)],
            constraints=[Constraint("notadiscipline.value", upper=5.0)],
            objective=Objective("weights.total_mass", maximize=False),
        )


from unittest.mock import patch, MagicMock


def _mock_weight_result():
    wr = MagicMock()
    wr.total_mass = 2.5
    wr.cg = np.array([0.32, 0.0, 0.0])
    wr.wing_loading = 18.0
    wr.inertia_tensor = np.eye(3) * 0.1
    return wr


def _mock_aero_result():
    ar = MagicMock()
    ar.CL = 0.5; ar.CD = 0.04; ar.Cm = 0.01
    ar.Cl = 0.0; ar.Cn = 0.0; ar.CY = 0.0
    ar.L = 25.0; ar.D = 2.0
    return ar


def _mock_stab_result():
    sr = MagicMock()
    sr.static_margin = 0.08
    return sr


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_evaluate_returns_dict(mock_w, mock_a, mock_s, simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()
    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    result = simple_problem.evaluate(x)
    assert "objective" in result
    assert "constraint_values" in result
    assert "results" in result
    assert "weights" in result["results"]
    assert "stability" in result["results"]


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_evaluate_caches(mock_w, mock_a, mock_s, simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()
    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    simple_problem.evaluate(x)
    simple_problem.evaluate(x)
    assert mock_w.call_count == 1   # ran once only
    assert simple_problem._n_evals == 1  # incremented only on first call


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_evaluate_objective_minimise(mock_w, mock_a, mock_s, simple_problem):
    wr = _mock_weight_result()
    wr.total_mass = 2.5
    mock_w.return_value = wr
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()
    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    result = simple_problem.evaluate(x)
    # maximize=False → sign +1 → objective = 2.5
    assert result["objective"] == pytest.approx(2.5)


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_evaluate_objective_maximise(mock_w, mock_a, mock_s, test_aircraft, cruise_condition):
    """maximize=True → sign -1 → objective = -raw_value (optimizer minimises internally)."""
    from aerisplane.mdo.problem import MDOProblem
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cruise_condition,
        design_variables=[DesignVar("wings[0].xsecs[0].chord", lower=0.18, upper=0.38)],
        constraints=[Constraint("stability.static_margin", lower=0.05)],
        objective=Objective("weights.total_mass", maximize=True),
    )
    wr = _mock_weight_result()
    wr.total_mass = 2.5
    mock_w.return_value = wr
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()
    lo, hi = problem.get_bounds()
    x = (lo + hi) / 2.0
    result = problem.evaluate(x)
    # maximize=True → sign -1 → objective = -2.5
    assert result["objective"] == pytest.approx(-2.5)


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_optimize_returns_result(mock_w, mock_a, mock_s, simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    mock_s.return_value = _mock_stab_result()
    from aerisplane.mdo.result import OptimizationResult
    result = simple_problem.optimize(
        method="scipy_de",
        options={"maxiter": 2, "popsize": 4, "seed": 0},
        verbose=False,
    )
    assert isinstance(result, OptimizationResult)
    assert result.n_evaluations > 0
    assert len(result.convergence_history) == result.n_evaluations


@patch("aerisplane.stability.analyze")
@patch("aerisplane.aero.analyze")
@patch("aerisplane.weights.analyze")
def test_constraint_violation_vector(mock_w, mock_a, mock_s, simple_problem):
    mock_w.return_value = _mock_weight_result()
    mock_a.return_value = _mock_aero_result()
    sr = _mock_stab_result()
    sr.static_margin = 0.08   # 0.08 >= 0.05 → satisfied
    mock_s.return_value = sr
    lo, hi = simple_problem.get_bounds()
    x = (lo + hi) / 2.0
    violations = simple_problem.constraint_functions(x)
    assert violations[0] <= 0.0   # satisfied


def test_mdoproblem_explicit_disciplines(test_aircraft):
    from aerisplane.mdo.problem import MDOProblem, DesignVar, Objective
    import aerisplane as ap
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cond,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.10, upper=0.50),
        ],
        constraints=[],
        objective=Objective("aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )
    assert "aero" in problem._disciplines
    assert "structures" not in problem._disciplines
    assert "stability" not in problem._disciplines


def test_mdoproblem_disciplines_none_auto_infers(test_aircraft):
    from aerisplane.mdo.problem import MDOProblem, DesignVar, Objective
    import aerisplane as ap
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cond,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.10, upper=0.50),
        ],
        constraints=[],
        objective=Objective("aero.CL_over_CD", maximize=True),
        disciplines=None,
    )
    assert "aero" in problem._disciplines


def test_mdoproblem_aero_result_param_accepted(test_aircraft):
    from aerisplane.mdo.problem import MDOProblem, DesignVar, Objective
    from unittest.mock import MagicMock
    import aerisplane as ap
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    fake_aero = MagicMock()
    fake_aero.CL_over_CD = 12.0
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cond,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.10, upper=0.50),
        ],
        constraints=[],
        objective=Objective("aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
        aero_result=fake_aero,
    )
    assert problem._aero_result is fake_aero


def test_simulate_returns_results_dict(test_aircraft):
    from aerisplane.mdo.problem import MDOProblem, DesignVar, Objective
    import aerisplane as ap
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cond,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.10, upper=0.50),
        ],
        constraints=[],
        objective=Objective("aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )
    results = problem.simulate()
    assert isinstance(results, dict)
    assert "aero" in results
    assert "weights" in results
    assert hasattr(results["aero"], "CL_over_CD")


def test_simulate_uses_cache(test_aircraft):
    """Second call to simulate() must not increment n_evals (cache hit)."""
    from aerisplane.mdo.problem import MDOProblem, DesignVar, Objective
    import aerisplane as ap
    cond = ap.FlightCondition(velocity=15.0, altitude=100.0, alpha=4.0)
    problem = MDOProblem(
        aircraft=test_aircraft,
        condition=cond,
        design_variables=[
            DesignVar("wings[0].xsecs[0].chord", lower=0.10, upper=0.50),
        ],
        constraints=[],
        objective=Objective("aero.CL_over_CD", maximize=True),
        disciplines=["aero"],
    )
    problem.simulate()
    n_after_first = problem._n_evals
    problem.simulate()
    assert problem._n_evals == n_after_first
