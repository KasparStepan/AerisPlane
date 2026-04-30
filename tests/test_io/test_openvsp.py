"""Tests for the OpenVSP vspscript exporter.

These tests validate the *content* of the emitted vspscript — running it
inside OpenVSP itself is a manual integration step.
"""

import warnings

import pytest

from aerisplane.io import to_openvsp
from aerisplane.io.openvsp import OPENVSP_VERSION


class TestToOpenVSPStructure:
    def test_writes_file(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.vspscript"
        to_openvsp(aircraft_with_controls, out)
        assert out.exists()
        text = out.read_text()
        assert "void main()" in text
        assert OPENVSP_VERSION in text

    def test_emits_wing_geom_per_wing(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.vspscript"
        to_openvsp(aircraft_with_controls, out)
        text = out.read_text()
        # Two wings → two AddGeom("WING") calls.
        assert text.count('AddGeom("WING")') == 2
        # Wing names appear as comments.
        assert '"main_wing"' in text
        assert '"htail"' in text

    def test_emits_fuselage_geom(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.vspscript"
        to_openvsp(aircraft_with_controls, out)
        text = out.read_text()
        assert text.count('AddGeom("FUSELAGE")') == 1
        assert "Super_Width" in text
        assert "Super_Height" in text

    def test_emits_symmetry_for_symmetric_wing(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.vspscript"
        to_openvsp(aircraft_with_controls, out)
        text = out.read_text()
        # Sym_Planar_Flag = 2.0 means XZ mirror — the value VSP expects for
        # a symmetric wing.
        assert 'SetParmVal( wid, "Sym_Planar_Flag", "Sym", 2.0 )' in text


class TestPerSectionParms:
    def test_root_and_tip_chord_per_section(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.vspscript"
        to_openvsp(aircraft_with_controls, out)
        text = out.read_text()
        # main_wing has 3 xsecs → 2 sections (XSec_1 and XSec_2).
        assert '"Root_Chord", "XSec_1"' in text
        assert '"Tip_Chord", "XSec_1"' in text
        assert '"Root_Chord", "XSec_2"' in text

    def test_dihedral_computed_from_geometry(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.vspscript"
        to_openvsp(aircraft_with_controls, out)
        text = out.read_text()
        # main_wing has positive z gain across span → positive dihedral.
        assert 'Dihedral", "XSec_1"' in text


class TestAirfoilEmission:
    def test_writes_airfoil_points(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.vspscript"
        to_openvsp(aircraft_with_controls, out)
        text = out.read_text()
        # 5 total xsecs across 2 wings → 5 SetAirfoilPnts calls.
        assert text.count("SetAirfoilPnts(") == 5
        assert "vec3d(" in text


class TestDroppedFieldsWarning:
    def test_warns_about_propulsion_and_controls(self, tmp_path, aircraft_full):
        out = tmp_path / "plane.vspscript"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            to_openvsp(aircraft_full, out)
        msg = " ".join(str(w.message) for w in caught)
        assert "propulsion" in msg
        assert "control surfaces" in msg


class TestEdgeCases:
    def test_rejects_single_section_wing(self, tmp_path):
        import aerisplane as ap
        af = ap.Airfoil.from_naca("0012")
        bad = ap.Aircraft(
            name="bad",
            wings=[ap.Wing(name="w", xsecs=[ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2, airfoil=af)])],
        )
        with pytest.raises(ValueError, match="at least 2 cross-sections"):
            to_openvsp(bad, tmp_path / "out.vspscript")

    def test_handles_airfoil_without_coordinates(self, tmp_path):
        import aerisplane as ap
        af = ap.Airfoil(name="unknown_airfoil")          # coordinates=None
        ac = ap.Aircraft(
            name="t",
            wings=[ap.Wing(
                name="w",
                xsecs=[
                    ap.WingXSec(xyz_le=[0, 0, 0], chord=0.2, airfoil=af),
                    ap.WingXSec(xyz_le=[0, 0.5, 0], chord=0.2, airfoil=af),
                ],
            )],
        )
        out = tmp_path / "out.vspscript"
        to_openvsp(ac, out)
        # Falls back to flat-plate; file still emits airfoil pts.
        assert "SetAirfoilPnts(" in out.read_text()
