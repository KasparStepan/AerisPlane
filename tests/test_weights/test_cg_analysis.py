"""Tests for CG envelope and ballast calculation."""

import numpy as np
import pytest

from aerisplane.weights.cg_analysis import (
    CGEnvelope,
    compute_ballast,
    compute_cg_envelope,
)
from aerisplane.weights.result import ComponentMass, ComponentOverride, WeightResult


# ===================================================================
# Ballast calculation
# ===================================================================

class TestBallast:
    @pytest.fixture
    def simple_result(self):
        """Aircraft at 1 kg, CG at x=0.3 m."""
        return WeightResult(
            total_mass=1.0,
            cg=np.array([0.3, 0.0, 0.0]),
            inertia_tensor=np.zeros((3, 3)),
            components={},
            wing_loading=30.0,
        )

    def test_nose_ballast_to_move_cg_forward(self, simple_result):
        """CG at 0.3, target 0.25, ballast at nose (x=0.0)."""
        mass = compute_ballast(simple_result, target_cg_x=0.25, ballast_position_x=0.0)
        # m_b = 1.0 * (0.25 - 0.3) / (0.0 - 0.25) = 1.0 * (-0.05) / (-0.25) = 0.2 kg
        assert mass == pytest.approx(0.2)

    def test_tail_ballast_to_move_cg_aft(self, simple_result):
        """CG at 0.3, target 0.35, ballast at tail (x=0.9)."""
        mass = compute_ballast(simple_result, target_cg_x=0.35, ballast_position_x=0.9)
        # m_b = 1.0 * (0.35 - 0.3) / (0.9 - 0.35) = 0.05 / 0.55 ≈ 0.0909 kg
        assert mass == pytest.approx(1.0 * 0.05 / 0.55)

    def test_cg_already_at_target(self, simple_result):
        """No ballast needed if CG is at target."""
        mass = compute_ballast(simple_result, target_cg_x=0.3, ballast_position_x=0.0)
        assert mass == pytest.approx(0.0)

    def test_cg_past_target_returns_zero(self, simple_result):
        """CG at 0.3, target 0.35, ballast at nose — can't push CG aft with nose ballast."""
        mass = compute_ballast(simple_result, target_cg_x=0.35, ballast_position_x=0.0)
        assert mass == 0.0

    def test_ballast_at_target_position_returns_zero(self, simple_result):
        """Ballast at the target position can't shift CG."""
        mass = compute_ballast(simple_result, target_cg_x=0.25, ballast_position_x=0.25)
        assert mass == 0.0

    def test_verify_ballast_achieves_target(self, simple_result):
        """Add computed ballast and verify CG hits target."""
        target = 0.25
        ballast_x = 0.05
        m_b = compute_ballast(simple_result, target_cg_x=target, ballast_position_x=ballast_x)

        # Recompute CG with ballast
        M = simple_result.total_mass
        new_cg_x = (M * simple_result.cg[0] + m_b * ballast_x) / (M + m_b)
        assert new_cg_x == pytest.approx(target, abs=1e-10)


# ===================================================================
# CG Envelope
# ===================================================================

class TestCGEnvelope:
    def test_envelope_with_aircraft(self, aircraft_with_structure):
        """Compute CG envelope for multiple configurations."""
        envelope = compute_cg_envelope(
            aircraft_with_structure,
            configurations={
                "baseline": {},
                "no payload": {
                    "payload": ComponentOverride(mass=0.0, cg=np.array([0.25, 0, 0])),
                },
                "heavy battery": {
                    "battery": ComponentOverride(mass=0.350, cg=np.array([0.0, 0, 0])),
                },
            },
        )
        assert len(envelope.cases) == 3
        assert envelope.cg_x_range >= 0
        assert envelope.mass_min <= envelope.mass_max

    def test_envelope_cg_range(self, aircraft_with_structure):
        """Moving battery forward should shift CG forward."""
        envelope = compute_cg_envelope(
            aircraft_with_structure,
            configurations={
                "battery_aft": {
                    "battery": ComponentOverride(mass=0.200, cg=np.array([0.8, 0, 0])),
                },
                "battery_fwd": {
                    "battery": ComponentOverride(mass=0.200, cg=np.array([0.1, 0, 0])),
                },
            },
        )
        aft_case = next(c for c in envelope.cases if c.name == "battery_aft")
        fwd_case = next(c for c in envelope.cases if c.name == "battery_fwd")
        assert fwd_case.result.cg[0] < aft_case.result.cg[0]

    def test_envelope_with_base_overrides(self, aircraft_with_structure):
        """Base overrides are applied to all configurations."""
        envelope = compute_cg_envelope(
            aircraft_with_structure,
            configurations={
                "config_a": {},
                "config_b": {
                    "battery": ComponentOverride(mass=0.300),
                },
            },
            base_overrides={
                "receiver": ComponentOverride(mass=0.010, cg=np.array([0.3, 0, 0])),
            },
        )
        # Receiver should be in both configs
        for case in envelope.cases:
            assert "receiver" in case.result.components

    def test_report_string(self, aircraft_with_structure):
        envelope = compute_cg_envelope(
            aircraft_with_structure,
            configurations={"baseline": {}, "light": {}},
        )
        report = envelope.report()
        assert "CG Envelope" in report
        assert "baseline" in report
        assert "CG_x range" in report

    def test_report_with_mac(self, aircraft_with_structure):
        envelope = compute_cg_envelope(
            aircraft_with_structure,
            configurations={"baseline": {}},
        )
        report = envelope.report(mac=0.2, mac_le_x=0.1)
        assert "%MAC" in report
