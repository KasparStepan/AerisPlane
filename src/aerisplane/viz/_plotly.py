"""Plotly rendering backend for AerisPlane aircraft geometry.

Produces an interactive web/notebook figure. Requires ``pip install plotly``.
"""
from __future__ import annotations

import numpy as np

_WING_COLORS = ["#6B9BD2", "#5A84BB", "#4A6FA0", "#3A5F90"]
_FUSE_COLORS = ["#A8A8A8", "#909090", "#787878"]


def _quads_to_tris(faces: np.ndarray) -> np.ndarray:
    """Split (M, 4) quad face array into (2M, 3) triangle faces.

    Each quad [a, b, c, d] becomes [a, b, c] and [a, c, d].
    """
    return np.vstack([faces[:, :3], faces[:, [0, 2, 3]]])


def render(
    components: list[dict],
    show: bool = True,
    title: str = "AerisPlane — Aircraft Geometry",
    opacity: float = 0.85,
    width: int = 900,
    height: int = 650,
) -> "plotly.graph_objects.Figure":
    """Render aircraft geometry as an interactive Plotly 3-D figure.

    Works in Jupyter notebooks, VS Code notebooks, and standalone browsers.
    Mouse: left-drag to rotate, scroll to zoom, right-drag to pan.

    Parameters
    ----------
    components : list[dict]
        Output of ``aircraft_to_meshes()``.
    show : bool
        Call ``fig.show()`` immediately. Set False to return the figure only.
    title : str
        Plot title shown at the top.
    opacity : float
        Surface opacity (0 = transparent, 1 = opaque).
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> from aerisplane.viz._mesh import aircraft_to_meshes
    >>> from aerisplane.viz._plotly import render
    >>> comps = aircraft_to_meshes(my_aircraft)
    >>> fig = render(comps, show=True)
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for the 'plotly' backend. "
            "Install it with: pip install plotly"
        ) from exc

    traces = []
    wing_idx = 0
    fuse_idx = 0

    for comp in components:
        pts = comp["points"].astype(float)
        faces = comp["faces"]
        comp_type = comp.get("type", "wing")

        # Plotly Mesh3d requires triangles
        if faces.shape[1] == 4:
            tris = _quads_to_tris(faces)
        else:
            tris = faces

        if comp.get("color") is not None:
            color = comp["color"]
        elif comp_type == "wing":
            color = _WING_COLORS[wing_idx % len(_WING_COLORS)]
            wing_idx += 1
        else:
            color = _FUSE_COLORS[fuse_idx % len(_FUSE_COLORS)]
            fuse_idx += 1

        traces.append(go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=tris[:, 0],
            j=tris[:, 1],
            k=tris[:, 2],
            color=color,
            opacity=opacity,
            name=comp["name"],
            showlegend=True,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=1, y=1, z=2),
        ))

    layout = go.Layout(
        title=dict(text=title, font=dict(size=16)),
        width=width,
        height=height,
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.7),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig = go.Figure(data=traces, layout=layout)

    if show:
        fig.show()

    return fig
