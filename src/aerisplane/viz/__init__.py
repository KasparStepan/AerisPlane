"""AerisPlane geometry visualization — PyVista and Plotly backends.

Quick start
-----------
>>> from aerisplane.viz import draw
>>> draw(aircraft, backend="plotly")   # interactive browser/notebook
>>> draw(aircraft, backend="pyvista")  # interactive desktop window

You can also visualize individual components:
>>> draw(wing, backend="plotly")
>>> draw(fuselage, backend="pyvista")

Colors
------
Set a ``color`` attribute on any Wing or Fuselage to override the default palette:
>>> wing.color = "#FF4444"
>>> draw(aircraft, backend="plotly")
"""
from __future__ import annotations


def draw(
    obj,
    backend: str = "plotly",
    show: bool = True,
    chordwise_resolution: int = 20,
    tangential_resolution: int = 36,
    use_wing_body: bool = False,
    **kwargs,
):
    """Visualize an Aircraft, Wing, or Fuselage interactively.

    Parameters
    ----------
    obj : Aircraft, Wing, or Fuselage
        The geometry to visualize.
    backend : str
        "plotly"  — interactive browser/notebook (default). Requires ``pip install plotly``.
        "pyvista" — interactive desktop window. Requires ``pip install pyvista``.
    show : bool
        Display immediately. Set False to get the figure/plotter object without opening it.
    chordwise_resolution : int
        Chordwise panels per wing (more = smoother, slower).
    tangential_resolution : int
        Circumferential panels per fuselage cross-section.
    use_wing_body : bool
        If True, mesh the full 3-D wing OML instead of the faster camber surface.
    **kwargs
        Passed directly to the backend renderer (e.g., ``opacity``, ``title``).

    Returns
    -------
    plotly.graph_objects.Figure  or  pyvista.Plotter

    Examples
    --------
    >>> from aerisplane.viz import draw
    >>> draw(aircraft)                          # Plotly, opens in browser
    >>> draw(aircraft, backend="pyvista")       # PyVista, opens a window
    >>> fig = draw(aircraft, show=False)        # get figure without displaying
    >>> draw(aircraft, opacity=0.7, title="My Aircraft")  # extra kwargs
    """
    from aerisplane.core.aircraft import Aircraft
    from aerisplane.core.wing import Wing
    from aerisplane.core.fuselage import Fuselage
    from aerisplane.viz._mesh import aircraft_to_meshes

    # Wrap single components into a temporary Aircraft
    if isinstance(obj, Wing):
        target = Aircraft(name=obj.name, wings=[obj])
    elif isinstance(obj, Fuselage):
        target = Aircraft(name=obj.name, fuselages=[obj])
    elif isinstance(obj, Aircraft):
        target = obj
    else:
        raise TypeError(
            f"obj must be Aircraft, Wing, or Fuselage, got {type(obj).__name__!r}"
        )

    components = aircraft_to_meshes(
        target,
        chordwise_resolution=chordwise_resolution,
        tangential_resolution=tangential_resolution,
        use_wing_body=use_wing_body,
    )

    # Apply per-component color overrides from geometry objects
    all_surfs = list(target.wings) + list(target.fuselages)
    for comp, surf in zip(components, all_surfs):
        color = getattr(surf, "color", None)
        if color is not None:
            comp["color"] = color

    if backend == "plotly":
        from aerisplane.viz._plotly import render
    elif backend == "pyvista":
        from aerisplane.viz._pyvista import render
    else:
        raise ValueError(
            f"backend must be 'plotly' or 'pyvista', got '{backend}'"
        )

    return render(components, show=show, **kwargs)
