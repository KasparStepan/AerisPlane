"""Tests for viz/_mesh.py — mesh generation without any display."""
import numpy as np
import pytest
import aerisplane as ap
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.wing import Wing, WingXSec
from aerisplane.core.fuselage import Fuselage, FuselageXSec


@pytest.fixture
def simple_aircraft():
    af = ap.Airfoil.from_naca("0012")
    wing = Wing(
        name="main_wing",
        xsecs=[
            WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.3, airfoil=af),
            WingXSec(xyz_le=[0.02, 0.75, 0.0], chord=0.15, airfoil=af),
        ],
        symmetric=True,
    )
    fuse = Fuselage(
        name="fuselage",
        xsecs=[
            FuselageXSec(x=0.0, radius=0.03),
            FuselageXSec(x=0.5, radius=0.06),
            FuselageXSec(x=1.0, radius=0.03),
        ],
    )
    return Aircraft(name="test_ac", wings=[wing], fuselages=[fuse])


class TestAircraftToMeshes:
    def test_returns_list(self, simple_aircraft):
        from aerisplane.viz._mesh import aircraft_to_meshes
        components = aircraft_to_meshes(simple_aircraft)
        assert isinstance(components, list)

    def test_one_component_per_surface(self, simple_aircraft):
        from aerisplane.viz._mesh import aircraft_to_meshes
        components = aircraft_to_meshes(simple_aircraft)
        # 1 wing + 1 fuselage = 2 components
        assert len(components) == 2

    def test_each_component_has_required_keys(self, simple_aircraft):
        from aerisplane.viz._mesh import aircraft_to_meshes
        components = aircraft_to_meshes(simple_aircraft)
        for comp in components:
            assert "name" in comp
            assert "type" in comp
            assert "points" in comp
            assert "faces" in comp

    def test_points_are_3d(self, simple_aircraft):
        from aerisplane.viz._mesh import aircraft_to_meshes
        components = aircraft_to_meshes(simple_aircraft)
        for comp in components:
            assert comp["points"].ndim == 2
            assert comp["points"].shape[1] == 3

    def test_faces_are_index_arrays(self, simple_aircraft):
        from aerisplane.viz._mesh import aircraft_to_meshes
        components = aircraft_to_meshes(simple_aircraft)
        for comp in components:
            faces = comp["faces"]
            assert faces.ndim == 2
            assert faces.min() >= 0
            assert faces.max() < len(comp["points"])

    def test_wing_component_type(self, simple_aircraft):
        from aerisplane.viz._mesh import aircraft_to_meshes
        components = aircraft_to_meshes(simple_aircraft)
        wing_comps = [c for c in components if c["type"] == "wing"]
        assert len(wing_comps) == 1
        assert wing_comps[0]["name"] == "main_wing"

    def test_fuselage_component_type(self, simple_aircraft):
        from aerisplane.viz._mesh import aircraft_to_meshes
        components = aircraft_to_meshes(simple_aircraft)
        fuse_comps = [c for c in components if c["type"] == "fuselage"]
        assert len(fuse_comps) == 1
        assert fuse_comps[0]["name"] == "fuselage"

    def test_wing_only_aircraft(self):
        from aerisplane.viz._mesh import aircraft_to_meshes
        af = ap.Airfoil.from_naca("0012")
        wing = Wing(
            xsecs=[WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af),
                   WingXSec(xyz_le=[0, 0.5, 0], chord=0.15, airfoil=af)],
            symmetric=True,
        )
        ac = Aircraft(name="w", wings=[wing])
        components = aircraft_to_meshes(ac)
        assert len(components) == 1
        assert components[0]["type"] == "wing"


class TestDrawPublicAPI:
    """Tests for the public draw() entry point. Skipped if plotly not installed."""

    def test_draw_aircraft_plotly_no_show(self, simple_aircraft):
        go = pytest.importorskip("plotly.graph_objects")
        from aerisplane.viz import draw
        fig = draw(simple_aircraft, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # 1 wing + 1 fuselage

    def test_draw_wing_plotly_no_show(self):
        go = pytest.importorskip("plotly.graph_objects")
        from aerisplane.viz import draw
        af = ap.Airfoil.from_naca("0012")
        wing = Wing(
            xsecs=[WingXSec(xyz_le=[0, 0, 0], chord=0.3, airfoil=af),
                   WingXSec(xyz_le=[0, 0.5, 0], chord=0.15, airfoil=af)],
            symmetric=True,
        )
        fig = draw(wing, backend="plotly", show=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_draw_bad_backend_raises(self, simple_aircraft):
        from aerisplane.viz import draw
        with pytest.raises(ValueError, match="backend"):
            draw(simple_aircraft, backend="vtk", show=False)
