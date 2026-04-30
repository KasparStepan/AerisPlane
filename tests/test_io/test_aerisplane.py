"""Round-trip tests for the native AerisPlane JSON format."""

import json

import numpy as np
import pytest

from aerisplane.io import load_aircraft, save_aircraft
from aerisplane.io.aerisplane import SCHEMA_VERSION


class TestSaveLoadGeometry:
    def test_roundtrip_preserves_wings(self, tmp_path, aircraft_with_controls):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_with_controls, path)
        loaded = load_aircraft(path)

        assert loaded.name == aircraft_with_controls.name
        assert len(loaded.wings) == len(aircraft_with_controls.wings)
        for orig, lw in zip(aircraft_with_controls.wings, loaded.wings):
            assert lw.name == orig.name
            assert lw.symmetric == orig.symmetric
            assert len(lw.xsecs) == len(orig.xsecs)
            for ox, lx in zip(orig.xsecs, lw.xsecs):
                np.testing.assert_allclose(lx.xyz_le, ox.xyz_le)
                assert lx.chord == pytest.approx(ox.chord)
                assert lx.twist == pytest.approx(ox.twist)
                assert lx.airfoil.name == ox.airfoil.name

    def test_roundtrip_preserves_control_surfaces(self, tmp_path, aircraft_with_controls):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_with_controls, path)
        loaded = load_aircraft(path)

        orig_cs = aircraft_with_controls.wings[0].control_surfaces[0]
        new_cs = loaded.wings[0].control_surfaces[0]
        assert new_cs.name == orig_cs.name
        assert new_cs.span_start == pytest.approx(orig_cs.span_start)
        assert new_cs.span_end == pytest.approx(orig_cs.span_end)
        assert new_cs.chord_fraction == pytest.approx(orig_cs.chord_fraction)
        assert new_cs.symmetric == orig_cs.symmetric

    def test_roundtrip_preserves_fuselage(self, tmp_path, aircraft_with_controls):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_with_controls, path)
        loaded = load_aircraft(path)

        orig_fuse = aircraft_with_controls.fuselages[0]
        new_fuse = loaded.fuselages[0]
        assert new_fuse.x_le == pytest.approx(orig_fuse.x_le)
        assert len(new_fuse.xsecs) == len(orig_fuse.xsecs)
        for ox, nx in zip(orig_fuse.xsecs, new_fuse.xsecs):
            assert nx.x == pytest.approx(ox.x)
            assert nx.width == pytest.approx(ox.width)
            assert nx.height == pytest.approx(ox.height)
            assert nx.shape == pytest.approx(ox.shape)

    def test_roundtrip_preserves_xyz_ref(self, tmp_path, aircraft_with_controls):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_with_controls, path)
        loaded = load_aircraft(path)
        np.testing.assert_allclose(loaded.xyz_ref, aircraft_with_controls.xyz_ref)


class TestSaveLoadFullAircraft:
    def test_roundtrip_preserves_propulsion(self, tmp_path, aircraft_full):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_full, path)
        loaded = load_aircraft(path)

        assert loaded.propulsion is not None
        assert loaded.propulsion.motor.kv == aircraft_full.propulsion.motor.kv
        assert loaded.propulsion.propeller.diameter == pytest.approx(
            aircraft_full.propulsion.propeller.diameter
        )
        np.testing.assert_allclose(
            loaded.propulsion.position, aircraft_full.propulsion.position
        )

    def test_roundtrip_preserves_payload(self, tmp_path, aircraft_full):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_full, path)
        loaded = load_aircraft(path)

        assert loaded.payload is not None
        assert loaded.payload.mass == pytest.approx(aircraft_full.payload.mass)
        np.testing.assert_allclose(loaded.payload.cg, aircraft_full.payload.cg)

    def test_roundtrip_preserves_spar(self, tmp_path, aircraft_full):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_full, path)
        loaded = load_aircraft(path)

        orig_spar = aircraft_full.wings[0].xsecs[0].spar
        new_spar = loaded.wings[0].xsecs[0].spar
        assert new_spar is not None
        assert new_spar.material.name == orig_spar.material.name
        assert new_spar.material.E == pytest.approx(orig_spar.material.E)
        assert new_spar.section.outer_diameter == pytest.approx(
            orig_spar.section.outer_diameter
        )

    def test_roundtrip_preserves_servo(self, tmp_path, aircraft_full):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_full, path)
        loaded = load_aircraft(path)

        new_servo = loaded.wings[0].control_surfaces[0].servo
        orig_servo = aircraft_full.wings[0].control_surfaces[0].servo
        assert new_servo.name == orig_servo.name
        assert new_servo.torque == pytest.approx(orig_servo.torque)


class TestEmbeddedDeduplication:
    def test_shared_airfoil_emitted_once(self, tmp_path, aircraft_with_controls):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_with_controls, path)

        doc = json.loads(path.read_text())
        # main wing shares one airfoil across all 3 xsecs; htail shares another.
        assert len(doc["embedded"]["airfoils"]) == 2

    def test_format_and_version_recorded(self, tmp_path, aircraft_with_controls):
        path = tmp_path / "ac.apl.json"
        save_aircraft(aircraft_with_controls, path)
        doc = json.loads(path.read_text())
        assert doc["format"] == "aerisplane"
        assert doc["version"] == SCHEMA_VERSION


class TestSchemaValidation:
    def test_rejects_unknown_format(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"format": "openvsp", "version": 1, "aircraft": {}}))
        with pytest.raises(ValueError, match="not an AerisPlane file"):
            load_aircraft(p)

    def test_rejects_future_version(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({
            "format": "aerisplane", "version": 999,
            "aircraft": {"name": "x"}, "embedded": {},
        }))
        with pytest.raises(ValueError, match="unsupported AerisPlane schema version"):
            load_aircraft(p)
