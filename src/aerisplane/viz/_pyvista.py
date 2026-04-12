"""PyVista rendering backend for AerisPlane aircraft geometry.

Produces an interactive rotatable 3-D window. Requires ``pip install pyvista``.
"""
from __future__ import annotations

import numpy as np

# Default color palette — wing blues, fuselage grays
_WING_COLORS = ["#6B9BD2", "#5A84BB", "#4A6FA0", "#3A5F90"]
_FUSE_COLORS = ["#A8A8A8", "#909090", "#787878"]


def _make_polydata_faces(faces: np.ndarray) -> np.ndarray:
    """Convert (M, n) face index array to PyVista flat format.

    PyVista expects: [n_verts, i0, i1, ..., n_verts, i0, i1, ...]
    """
    n_faces, n_verts = faces.shape
    flat = np.empty(n_faces * (n_verts + 1), dtype=np.intp)
    flat[0 :: n_verts + 1] = n_verts
    for k in range(n_verts):
        flat[k + 1 :: n_verts + 1] = faces[:, k]
    return flat


def render(
    components: list[dict],
    show: bool = True,
    background: str = "white",
    show_edges: bool = True,
    opacity: float = 0.9,
    window_size: tuple[int, int] = (1200, 800),
    title: str = "AerisPlane — Aircraft Geometry",
) -> "pyvista.Plotter":
    """Render aircraft geometry in an interactive PyVista window.

    Parameters
    ----------
    components : list[dict]
        Output of ``aircraft_to_meshes()``.
    show : bool
        Open the interactive window. Set False to build the plotter without displaying.
    background : str
        Background colour (any matplotlib-compatible string).
    show_edges : bool
        Draw mesh edges for a wireframe overlay.
    opacity : float
        Surface opacity (0 = transparent, 1 = opaque).
    window_size : tuple[int, int]
        Window width × height in pixels.
    title : str
        Window title.

    Returns
    -------
    pyvista.Plotter
        The configured plotter. Call ``.show()`` on it again later if needed.

    Examples
    --------
    >>> from aerisplane.viz._mesh import aircraft_to_meshes
    >>> from aerisplane.viz._pyvista import render
    >>> comps = aircraft_to_meshes(my_aircraft)
    >>> render(comps, show=True)
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista is required for the 'pyvista' backend. "
            "Install it with: pip install pyvista"
        ) from exc

    plotter = pv.Plotter(window_size=list(window_size), title=title)
    plotter.set_background(background)

    wing_idx = 0
    fuse_idx = 0

    for comp in components:
        pts = comp["points"].astype(float)
        faces = comp["faces"]
        comp_type = comp.get("type", "wing")

        pv_faces = _make_polydata_faces(faces)
        mesh = pv.PolyData(pts, pv_faces)

        if comp.get("color") is not None:
            color = comp["color"]
        elif comp_type == "wing":
            color = _WING_COLORS[wing_idx % len(_WING_COLORS)]
            wing_idx += 1
        else:
            color = _FUSE_COLORS[fuse_idx % len(_FUSE_COLORS)]
            fuse_idx += 1

        plotter.add_mesh(
            mesh,
            color=color,
            show_edges=show_edges,
            opacity=opacity,
            label=comp["name"],
        )

    plotter.add_axes(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    plotter.camera_position = "iso"

    if show:
        plotter.show()

    return plotter
