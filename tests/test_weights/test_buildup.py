"""Tests for the weights module: buildup, overrides, aggregation."""

import numpy as np
import numpy.testing as npt
import pytest

import aerisplane as ap
from aerisplane import weights
from aerisplane.weights.buildup import (
    _fastener_mass,
    _fuselage_mass,
    _hardware_masses,
    _payload_mass,
    _wing_mass,
    aggregate,
    compute_buildup,
)
from aerisplane.weights.result import ComponentOverride


# ===================================================================
# Wing structure mass
# ===================================================================

class TestWingMass:
    """Tests for wing spar and skin mass computation."""

    def test_spar_mass_rectangular_wing(self, wing_with_structure, cf_spar):
        """Rectangular wing: spar mass = mass_per_length * semispan * 2."""
        comps = _wing_mass(wing_with_structure)
        spar_comp = next(c for c in comps if "spar" in c.name)

        expected_mpl = cf_spar.mass_per_length()
        semispan = 0.75  # m
        expected_mass = expected_mpl * semispan * 2  # symmetric
        assert spar_comp.mass == pytest.approx(expected_mass, rel=1e-6)

    def test_skin_mass_rectangular_wing(self, wing_with_structure, petg_skin):
        """Rectangular wing: skin mass = mass_per_area * wetted_area (no rib factor)."""
        comps = _wing_mass(wing_with_structure)
        skin_comp = next(c for c in comps if c.name.endswith("_skin"))

        mpa = petg_skin.mass_per_area()
        # Wetted area per semi: chord * semispan * 2 surfaces = 0.2 * 0.75 * 2
        semi_wetted = 0.2 * 0.75 * 2.0
        expected_mass = mpa * semi_wetted * 2  # * 2 for symmetric
        assert skin_comp.mass == pytest.approx(expected_mass, rel=1e-6)

    def test_rib_mass_separate_component(self, wing_with_structure):
        """Ribs are now a separate component, not baked into skin."""
        comps = _wing_mass(wing_with_structure)
        rib_comp = next((c for c in comps if "ribs" in c.name), None)
        assert rib_comp is not None
        assert rib_comp.mass > 0

    def test_wing_spar_cg_symmetric(self, wing_with_structure):
        """Symmetric wing spar CG should have y=0."""
        comps = _wing_mass(wing_with_structure)
        spar_comp = next(c for c in comps if "spar" in c.name)
        assert spar_comp.cg[1] == pytest.approx(0.0)

    def test_wing_skin_cg_symmetric(self, wing_with_structure):
        """Symmetric wing skin CG should have y=0."""
        comps = _wing_mass(wing_with_structure)
        skin_comp = next(c for c in comps if "skin" in c.name)
        assert skin_comp.cg[1] == pytest.approx(0.0)

    def test_wing_no_structure_returns_empty(self):
        """Wing without spar or skin returns no components."""
        bare_wing = ap.Wing(
            name="bare",
            xsecs=[
                ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2),
                ap.WingXSec(xyz_le=[0, 0.5, 0], chord=0.2),
            ],
        )
        comps = _wing_mass(bare_wing)
        assert comps == []

    def test_wing_single_section_returns_empty(self):
        """Wing with only one section has no panels."""
        wing = ap.Wing(
            name="one",
            xsecs=[ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2)],
        )
        comps = _wing_mass(wing)
        assert comps == []

    def test_spar_cg_x_offset(self, wing_with_structure):
        """Spar CG_x should be offset by spar chordwise position * chord."""
        comps = _wing_mass(wing_with_structure)
        spar_comp = next(c for c in comps if "spar" in c.name)
        # Wing LE at x=0.1, spar at 25% of 0.2m chord → x=0.1 + 0.05 = 0.15
        # Midpoint of panel is at y=0.375 (but x stays 0.1 for rectangular)
        expected_x = 0.1 + 0.25 * 0.2
        assert spar_comp.cg[0] == pytest.approx(expected_x, rel=1e-6)

    def test_rib_mass_scales_with_chord(self):
        """Bigger chord → bigger ribs → more rib mass."""
        from aerisplane.catalog.materials import carbon_fiber_tube, petg

        skin = ap.Skin(material=petg, thickness=0.8e-3)
        spar = ap.Spar(position=0.25, material=carbon_fiber_tube,
                        section=ap.TubeSection(outer_diameter=0.012, wall_thickness=0.0015))

        wing_small = ap.Wing(name="small", xsecs=[
            ap.WingXSec(xyz_le=[0, 0, 0], chord=0.15, spar=spar, skin=skin),
            ap.WingXSec(xyz_le=[0, 0.5, 0], chord=0.15, spar=spar, skin=skin),
        ], symmetric=True)
        wing_big = ap.Wing(name="big", xsecs=[
            ap.WingXSec(xyz_le=[0, 0, 0], chord=0.30, spar=spar, skin=skin),
            ap.WingXSec(xyz_le=[0, 0.5, 0], chord=0.30, spar=spar, skin=skin),
        ], symmetric=True)

        ribs_small = next(c for c in _wing_mass(wing_small) if "ribs" in c.name)
        ribs_big = next(c for c in _wing_mass(wing_big) if "ribs" in c.name)
        assert ribs_big.mass > ribs_small.mass


# ===================================================================
# Fastener mass
# ===================================================================

class TestFastenerMass:
    def test_fasteners_present(self, aircraft_with_structure):
        """Fastener mass should be computed for aircraft with wings."""
        comps = _fastener_mass(aircraft_with_structure)
        assert len(comps) == 1
        assert comps[0].name == "fasteners"
        assert comps[0].mass > 0

    def test_no_wings_no_fasteners(self):
        ac = ap.Aircraft(name="empty")
        comps = _fastener_mass(ac)
        assert comps == []

    def test_fastener_mass_scales_with_surfaces(self):
        """More control surfaces → more hinge fasteners."""
        servo = ap.Servo(name="s", torque=1, speed=100, voltage=6, mass=0.01)
        wing_1cs = ap.Wing(name="w", xsecs=[
            ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2),
            ap.WingXSec(xyz_le=[0, 0.5, 0], chord=0.2),
        ], control_surfaces=[
            ap.ControlSurface(name="a", span_start=0.5, span_end=1.0, chord_fraction=0.25, servo=servo),
        ])
        wing_3cs = ap.Wing(name="w", xsecs=[
            ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2),
            ap.WingXSec(xyz_le=[0, 0.5, 0], chord=0.2),
        ], control_surfaces=[
            ap.ControlSurface(name="a", span_start=0.3, span_end=0.5, chord_fraction=0.25, servo=servo),
            ap.ControlSurface(name="b", span_start=0.5, span_end=0.7, chord_fraction=0.25, servo=servo),
            ap.ControlSurface(name="c", span_start=0.7, span_end=0.9, chord_fraction=0.25, servo=servo),
        ])

        ac1 = ap.Aircraft(name="1cs", wings=[wing_1cs])
        ac3 = ap.Aircraft(name="3cs", wings=[wing_3cs])

        m1 = _fastener_mass(ac1)[0].mass
        m3 = _fastener_mass(ac3)[0].mass
        assert m3 > m1


# ===================================================================
# Fuselage structure mass
# ===================================================================

class TestFuselageMass:
    """Tests for fuselage shell mass computation."""

    def test_shell_mass(self, simple_fuselage, petg_material):
        """Shell mass = wetted_area * wall_thickness * density."""
        comps = _fuselage_mass(simple_fuselage)
        assert len(comps) == 1

        expected = (
            simple_fuselage.wetted_area()
            * simple_fuselage.wall_thickness
            * petg_material.density
        )
        assert comps[0].mass == pytest.approx(expected, rel=1e-6)

    def test_shell_cg_in_fuselage_range(self, simple_fuselage):
        """Shell CG_x should be within the fuselage length."""
        comps = _fuselage_mass(simple_fuselage)
        cg_x = comps[0].cg[0]
        assert 0.0 <= cg_x <= 0.95

    def test_no_material_returns_empty(self):
        """Fuselage without material returns empty (must be overridden)."""
        fus = ap.Fuselage(
            name="bare",
            xsecs=[
                ap.FuselageXSec(x=0.0, radius=0.05),
                ap.FuselageXSec(x=1.0, radius=0.05),
            ],
            material=None,
        )
        comps = _fuselage_mass(fus)
        assert comps == []


# ===================================================================
# Hardware masses
# ===================================================================

class TestHardwareMasses:
    """Tests for propulsion and servo mass extraction."""

    def test_propulsion_components(self, aircraft_with_structure):
        """Propulsion system produces motor, propeller, battery, ESC entries."""
        comps = _hardware_masses(aircraft_with_structure)
        names = {c.name for c in comps}
        assert "motor" in names
        assert "propeller" in names
        assert "battery" in names
        assert "esc" in names

    def test_propulsion_masses_correct(self, aircraft_with_structure):
        """Hardware masses match the component .mass attributes."""
        comps = _hardware_masses(aircraft_with_structure)
        comp_dict = {c.name: c for c in comps}
        ps = aircraft_with_structure.propulsion
        assert comp_dict["motor"].mass == ps.motor.mass
        assert comp_dict["propeller"].mass == ps.propeller.mass
        assert comp_dict["battery"].mass == ps.battery.mass
        assert comp_dict["esc"].mass == ps.esc.mass

    def test_servo_masses_symmetric(self, aircraft_with_structure):
        """Symmetric wing aileron produces L and R servo entries."""
        comps = _hardware_masses(aircraft_with_structure)
        names = {c.name for c in comps}
        assert "aileron_servo_R" in names
        assert "aileron_servo_L" in names

    def test_servo_mass_value(self, aircraft_with_structure, test_servo):
        """Each servo entry has the correct mass."""
        comps = _hardware_masses(aircraft_with_structure)
        servo_comps = [c for c in comps if "servo" in c.name]
        for sc in servo_comps:
            assert sc.mass == test_servo.mass

    def test_no_propulsion(self):
        """Aircraft without propulsion produces no propulsion entries."""
        ac = ap.Aircraft(name="glider")
        comps = _hardware_masses(ac)
        prop_names = {"motor", "propeller", "battery", "esc"}
        assert not prop_names.intersection(c.name for c in comps)


# ===================================================================
# Payload
# ===================================================================

class TestPayloadMass:
    def test_payload_included(self, aircraft_with_structure):
        comps = _payload_mass(aircraft_with_structure)
        assert len(comps) == 1
        assert comps[0].mass == 0.100
        npt.assert_array_almost_equal(comps[0].cg, [0.25, 0.0, 0.0])

    def test_no_payload(self):
        ac = ap.Aircraft(name="empty")
        comps = _payload_mass(ac)
        assert comps == []


# ===================================================================
# Aggregation
# ===================================================================

class TestAggregate:
    def test_total_mass_is_sum(self):
        """Total mass equals the sum of component masses."""
        comps = [
            ap.weights.result.ComponentMass("a", 1.0, np.array([0, 0, 0])),
            ap.weights.result.ComponentMass("b", 2.0, np.array([1, 0, 0])),
            ap.weights.result.ComponentMass("c", 3.0, np.array([2, 0, 0])),
        ]
        total, cg, inertia, wl = aggregate(comps, reference_area=1.0)
        assert total == pytest.approx(6.0)

    def test_cg_mass_weighted(self):
        """CG is the mass-weighted average of component CGs."""
        comps = [
            ap.weights.result.ComponentMass("a", 1.0, np.array([0, 0, 0])),
            ap.weights.result.ComponentMass("b", 3.0, np.array([1, 0, 0])),
        ]
        total, cg, inertia, wl = aggregate(comps, reference_area=1.0)
        # CG_x = (1*0 + 3*1) / 4 = 0.75
        assert cg[0] == pytest.approx(0.75)
        assert cg[1] == pytest.approx(0.0)
        assert cg[2] == pytest.approx(0.0)

    def test_inertia_two_point_masses(self):
        """Inertia tensor for two point masses on x-axis."""
        # 1 kg at x=-1 and 1 kg at x=+1, CG at origin
        comps = [
            ap.weights.result.ComponentMass("a", 1.0, np.array([-1, 0, 0])),
            ap.weights.result.ComponentMass("b", 1.0, np.array([1, 0, 0])),
        ]
        total, cg, inertia, wl = aggregate(comps, reference_area=1.0)

        npt.assert_array_almost_equal(cg, [0, 0, 0])
        # Ixx = 0 (masses on x-axis, no y or z offset)
        assert inertia[0, 0] == pytest.approx(0.0)
        # Iyy = m1*r1^2 + m2*r2^2 = 1*1 + 1*1 = 2
        assert inertia[1, 1] == pytest.approx(2.0)
        # Izz = same
        assert inertia[2, 2] == pytest.approx(2.0)

    def test_inertia_symmetric(self):
        """Inertia tensor should be symmetric."""
        comps = [
            ap.weights.result.ComponentMass("a", 1.0, np.array([1, 2, 3])),
            ap.weights.result.ComponentMass("b", 2.0, np.array([-1, 0, 1])),
        ]
        _, _, inertia, _ = aggregate(comps, reference_area=1.0)
        npt.assert_array_almost_equal(inertia, inertia.T)

    def test_wing_loading(self):
        """Wing loading = total_mass_grams / area_dm2."""
        comps = [
            ap.weights.result.ComponentMass("a", 2.5, np.array([0, 0, 0])),
        ]
        _, _, _, wl = aggregate(comps, reference_area=0.30)
        # 2500 g / (0.30 * 100 dm^2) = 2500 / 30 = 83.33
        assert wl == pytest.approx(83.333, rel=1e-3)

    def test_empty_components(self):
        """Empty list returns zeros."""
        total, cg, inertia, wl = aggregate([], reference_area=1.0)
        assert total == 0.0
        npt.assert_array_equal(cg, [0, 0, 0])
        npt.assert_array_equal(inertia, np.zeros((3, 3)))
        assert wl == 0.0


# ===================================================================
# Full analyze() with overrides
# ===================================================================

class TestAnalyze:
    def test_full_buildup(self, aircraft_with_structure):
        """analyze() returns a valid WeightResult for a complete aircraft."""
        result = weights.analyze(aircraft_with_structure)

        assert result.total_mass > 0
        assert len(result.components) > 0
        # Total should equal sum of components
        comp_sum = sum(c.mass for c in result.components.values())
        assert result.total_mass == pytest.approx(comp_sum, rel=1e-10)

    def test_all_sources_computed(self, aircraft_with_structure):
        """Without overrides, all sources are 'computed'."""
        result = weights.analyze(aircraft_with_structure)
        for comp in result.components.values():
            assert comp.source == "computed"

    def test_override_replaces_mass(self, aircraft_with_structure):
        """Override replaces a computed component's mass."""
        result = weights.analyze(
            aircraft_with_structure,
            overrides={"battery": ComponentOverride(mass=0.350)},
        )
        assert result.components["battery"].mass == pytest.approx(0.350)
        assert result.components["battery"].source == "override"

    def test_override_replaces_cg(self, aircraft_with_structure):
        """Override with CG replaces the computed CG."""
        new_cg = np.array([0.20, 0.0, -0.01])
        result = weights.analyze(
            aircraft_with_structure,
            overrides={"battery": ComponentOverride(mass=0.350, cg=new_cg)},
        )
        npt.assert_array_almost_equal(result.components["battery"].cg, new_cg)

    def test_override_preserves_computed_cg(self, aircraft_with_structure):
        """Override without CG keeps the computed CG."""
        result_base = weights.analyze(aircraft_with_structure)
        original_cg = result_base.components["battery"].cg.copy()

        result = weights.analyze(
            aircraft_with_structure,
            overrides={"battery": ComponentOverride(mass=0.999)},
        )
        npt.assert_array_almost_equal(result.components["battery"].cg, original_cg)

    def test_override_adds_new_component(self, aircraft_with_structure):
        """Override with unknown name adds a new component."""
        result = weights.analyze(
            aircraft_with_structure,
            overrides={
                "receiver": ComponentOverride(
                    mass=0.010, cg=np.array([0.3, 0.0, 0.0])
                ),
            },
        )
        assert "receiver" in result.components
        assert result.components["receiver"].mass == pytest.approx(0.010)
        assert result.components["receiver"].source == "override"

    def test_override_affects_total(self, aircraft_with_structure):
        """Overriding battery mass changes the total."""
        result_base = weights.analyze(aircraft_with_structure)
        result_mod = weights.analyze(
            aircraft_with_structure,
            overrides={"battery": ComponentOverride(mass=0.999)},
        )
        delta = 0.999 - aircraft_with_structure.propulsion.battery.mass
        assert result_mod.total_mass == pytest.approx(
            result_base.total_mass + delta, rel=1e-6
        )

    def test_override_shifts_cg(self, aircraft_with_structure):
        """Moving a heavy component shifts the overall CG."""
        result_base = weights.analyze(aircraft_with_structure)
        # Move battery far aft
        result_mod = weights.analyze(
            aircraft_with_structure,
            overrides={
                "battery": ComponentOverride(
                    mass=0.200, cg=np.array([0.9, 0.0, 0.0])
                ),
            },
        )
        # CG should shift aft (larger x)
        assert result_mod.cg[0] > result_base.cg[0]

    def test_wing_loading_positive(self, aircraft_with_structure):
        """Wing loading should be a positive number."""
        result = weights.analyze(aircraft_with_structure)
        assert result.wing_loading > 0


# ===================================================================
# Report output
# ===================================================================

class TestReport:
    def test_report_is_string(self, aircraft_with_structure):
        result = weights.analyze(aircraft_with_structure)
        report = result.report()
        assert isinstance(report, str)

    def test_report_contains_components(self, aircraft_with_structure):
        result = weights.analyze(aircraft_with_structure)
        report = result.report()
        for name in result.components:
            assert name in report

    def test_report_contains_total(self, aircraft_with_structure):
        result = weights.analyze(aircraft_with_structure)
        report = result.report()
        assert "TOTAL" in report
        assert "Wing loading" in report
        assert "CG position" in report

    def test_report_shows_override_source(self, aircraft_with_structure):
        result = weights.analyze(
            aircraft_with_structure,
            overrides={"battery": ComponentOverride(mass=0.350)},
        )
        report = result.report()
        assert "override" in report
