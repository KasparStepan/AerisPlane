"""Mesh assembly: converts AerisPlane geometry objects to (points, faces) dicts.

Each dict has:
    name   : str   — component name
    type   : str   — "wing" or "fuselage"
    points : (N, 3) float ndarray
    faces  : (M, k) int ndarray  — k=3 triangles, k=4 quads
    color  : None  — caller may set to override the default palette
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aerisplane.core.aircraft import Aircraft
    from aerisplane.core.wing import Wing
    from aerisplane.core.fuselage import Fuselage


def wing_to_mesh(
    wing: "Wing",
    chordwise_resolution: int = 20,
    use_body: bool = False,
) -> tuple:
    """Return (points, faces) for a Wing.

    Parameters
    ----------
    use_body : bool
        If True, use the full 3-D OML mesh (both surfaces).
        If False (default), use the camber-surface thin mesh (faster, still looks good).
    """
    if use_body:
        return wing.mesh_body(chordwise_resolution=chordwise_resolution)
    return wing.mesh_thin_surface(chordwise_resolution=chordwise_resolution)


def fuselage_to_mesh(
    fuselage: "Fuselage",
    tangential_resolution: int = 36,
) -> tuple:
    """Return (points, faces) for a Fuselage."""
    return fuselage.mesh_body(tangential_resolution=tangential_resolution)


def aircraft_to_meshes(
    aircraft: "Aircraft",
    chordwise_resolution: int = 20,
    tangential_resolution: int = 36,
    use_wing_body: bool = False,
) -> list[dict]:
    """Convert an Aircraft to a list of per-component mesh dicts.

    Parameters
    ----------
    aircraft : Aircraft
    chordwise_resolution : int
        Chordwise panels per wing chord.
    tangential_resolution : int
        Circumferential panels around each fuselage cross-section.
    use_wing_body : bool
        If True, use full OML mesh for wings. Default: camber surface.

    Returns
    -------
    list of dicts with keys "name", "type", "points", "faces", "color".
    """
    components = []

    for wing in aircraft.wings:
        pts, faces = wing_to_mesh(
            wing,
            chordwise_resolution=chordwise_resolution,
            use_body=use_wing_body,
        )
        components.append({
            "name": wing.name,
            "type": "wing",
            "points": pts,
            "faces": faces,
            "color": getattr(wing, "color", None),
        })

    for fuse in aircraft.fuselages:
        pts, faces = fuselage_to_mesh(
            fuse,
            tangential_resolution=tangential_resolution,
        )
        components.append({
            "name": fuse.name,
            "type": "fuselage",
            "points": pts,
            "faces": faces,
            "color": getattr(fuse, "color", None),
        })

    return components
