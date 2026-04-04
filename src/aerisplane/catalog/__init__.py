"""AerisPlane hardware and airfoil catalog."""
from __future__ import annotations


def get_airfoil(name: str):
    """Load an airfoil from the catalog by name.

    Parameters
    ----------
    name : str
        Airfoil name, e.g. ``"naca2412"``, ``"e423"``.  NACA 4-digit names
        are generated analytically; all others are loaded from the catalog
        .dat files in ``catalog/airfoils/``.

    Returns
    -------
    Airfoil
        Airfoil with coordinates populated.

    Raises
    ------
    ValueError
        If the name cannot be resolved (not NACA and not in catalog).
    """
    from aerisplane.core.airfoil import Airfoil
    af = Airfoil(name)
    if af.coordinates is None:
        raise ValueError(
            f"Airfoil '{name}' not found in catalog. "
            "Check catalog/airfoils/ for available .dat files."
        )
    return af


__all__ = ["get_airfoil"]
