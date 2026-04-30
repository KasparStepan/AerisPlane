"""Round-trip and parser tests for AVL geometry exchange."""

import warnings

import numpy as np
import pytest

from aerisplane.io import from_avl, to_avl


SIMPLE_AVL = """\
test_plane
#Mach
0.0
#IYsym  IZsym  Zsym
0  0  0.0
#Sref  Cref  Bref
0.45  0.225  2.0
#Xref  Yref  Zref
0.10  0.0   0.0

#-----------------------------------------------------------------------
SURFACE
main_wing
12  1.0  20  -1.5
YDUPLICATE
0.0
SECTION
0.000  0.000  0.000  0.300  0.0
NACA
2412
CONTROL
aileron  1.0  0.75  0 0 0  -1.0
SECTION
0.020  0.500  0.020  0.220  0.0
NACA
2412
CONTROL
aileron  1.0  0.75  0 0 0  -1.0
SECTION
0.050  1.000  0.050  0.150  -1.5
NACA
2412
CONTROL
aileron  1.0  0.75  0 0 0  -1.0

#-----------------------------------------------------------------------
SURFACE
htail
8  1.0  10  -1.5
YDUPLICATE
0.0
SECTION
0.900  0.000  0.050  0.120  0.0
NACA
0012
CONTROL
elevator  1.0  0.70  0 0 0  +1.0
SECTION
0.920  0.250  0.050  0.080  0.0
NACA
0012
CONTROL
elevator  1.0  0.70  0 0 0  +1.0
"""


class TestFromAVL:
    def test_parses_header(self, tmp_path):
        p = tmp_path / "plane.avl"
        p.write_text(SIMPLE_AVL)
        ac = from_avl(p)
        assert ac.name == "test_plane"
        np.testing.assert_allclose(ac.xyz_ref, [0.10, 0.0, 0.0])

    def test_parses_wing_count(self, tmp_path):
        p = tmp_path / "plane.avl"
        p.write_text(SIMPLE_AVL)
        ac = from_avl(p)
        assert [w.name for w in ac.wings] == ["main_wing", "htail"]

    def test_parses_sections(self, tmp_path):
        p = tmp_path / "plane.avl"
        p.write_text(SIMPLE_AVL)
        ac = from_avl(p)
        main = ac.wings[0]
        assert len(main.xsecs) == 3
        assert main.xsecs[0].chord == pytest.approx(0.30)
        assert main.xsecs[-1].chord == pytest.approx(0.15)
        assert main.xsecs[-1].twist == pytest.approx(-1.5)
        assert main.symmetric is True

    def test_aggregates_aileron_control(self, tmp_path):
        p = tmp_path / "plane.avl"
        p.write_text(SIMPLE_AVL)
        ac = from_avl(p)
        cs = ac.wings[0].control_surfaces
        assert len(cs) == 1
        assert cs[0].name == "aileron"
        assert cs[0].symmetric is False
        assert cs[0].chord_fraction == pytest.approx(0.25)

    def test_aggregates_elevator_control(self, tmp_path):
        p = tmp_path / "plane.avl"
        p.write_text(SIMPLE_AVL)
        ac = from_avl(p)
        cs = ac.wings[1].control_surfaces
        assert len(cs) == 1
        assert cs[0].name == "elevator"
        assert cs[0].symmetric is True
        assert cs[0].chord_fraction == pytest.approx(0.30)

    def test_naca_airfoil_resolved(self, tmp_path):
        p = tmp_path / "plane.avl"
        p.write_text(SIMPLE_AVL)
        ac = from_avl(p)
        af = ac.wings[0].xsecs[0].airfoil
        assert af.name == "naca2412"
        assert af.coordinates is not None


class TestToAVL:
    def test_writes_file(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.avl"
        to_avl(aircraft_with_controls, out)
        assert out.exists()
        text = out.read_text()
        assert "SURFACE" in text
        assert "SECTION" in text
        assert "YDUPLICATE" in text

    def test_emits_naca_for_naca_airfoils(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.avl"
        to_avl(aircraft_with_controls, out)
        text = out.read_text()
        assert "NACA\n2412" in text
        assert "NACA\n0012" in text

    def test_emits_control_lines(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.avl"
        to_avl(aircraft_with_controls, out)
        text = out.read_text()
        assert "CONTROL" in text
        assert "aileron" in text
        assert "elevator" in text

    def test_warns_about_dropped_propulsion(self, tmp_path, aircraft_full):
        out = tmp_path / "plane.avl"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            to_avl(aircraft_full, out)
        msgs = [str(w.message) for w in caught]
        assert any("propulsion" in m for m in msgs)


class TestAvlRoundTrip:
    def test_roundtrip_preserves_section_geometry(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.avl"
        to_avl(aircraft_with_controls, out)
        loaded = from_avl(out)

        orig = aircraft_with_controls.wings[0]
        new = loaded.wings[0]
        assert new.symmetric == orig.symmetric
        assert len(new.xsecs) == len(orig.xsecs)
        for ox, nx in zip(orig.xsecs, new.xsecs):
            np.testing.assert_allclose(nx.xyz_le, ox.xyz_le, atol=1e-6)
            assert nx.chord == pytest.approx(ox.chord, abs=1e-6)
            assert nx.twist == pytest.approx(ox.twist, abs=1e-6)

    def test_roundtrip_preserves_control_chord_fraction(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.avl"
        to_avl(aircraft_with_controls, out)
        loaded = from_avl(out)

        orig_cs = aircraft_with_controls.wings[0].control_surfaces[0]
        new_cs = loaded.wings[0].control_surfaces[0]
        assert new_cs.name == orig_cs.name
        assert new_cs.symmetric == orig_cs.symmetric
        assert new_cs.chord_fraction == pytest.approx(orig_cs.chord_fraction, abs=1e-4)

    def test_roundtrip_preserves_xyz_ref(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.avl"
        to_avl(aircraft_with_controls, out)
        loaded = from_avl(out)
        np.testing.assert_allclose(loaded.xyz_ref, aircraft_with_controls.xyz_ref, atol=1e-6)

    def test_roundtrip_preserves_fuselage_radii(self, tmp_path, aircraft_with_controls):
        out = tmp_path / "plane.avl"
        to_avl(aircraft_with_controls, out)
        loaded = from_avl(out)

        assert len(loaded.fuselages) == 1
        orig = aircraft_with_controls.fuselages[0]
        new = loaded.fuselages[0]
        assert new.x_le == pytest.approx(orig.x_le)
        assert len(new.xsecs) == len(orig.xsecs)
        # Width and height are equalized to equivalent_radius * 2 (axisymmetric).
        for ox, nx in zip(orig.xsecs, new.xsecs):
            assert nx.x == pytest.approx(ox.x, abs=1e-6)
            r_orig = ox.equivalent_radius()
            assert nx.width / 2 == pytest.approx(r_orig, abs=1e-6)
