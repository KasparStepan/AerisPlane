"""Shared helpers for io modules: array conversion and airfoil resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aerisplane.core.airfoil import Airfoil


def to_list(arr: Any) -> list:
    """Convert a numpy array (or array-like) to a plain nested Python list.

    Used for JSON serialization of dataclass fields that hold numpy arrays.
    """
    return np.asarray(arr, dtype=float).tolist()


def resolve_airfoil(name: str, search_dir: Path | None = None) -> Airfoil:
    """Build an :class:`Airfoil` from a name, searching the catalog and a directory.

    Resolution order:

    1. NACA 4-digit (analytic, via :class:`Airfoil` ``__post_init__``).
    2. Catalog ``.dat`` file (handled inside ``Airfoil``).
    3. ``<search_dir>/<name>.dat`` if *search_dir* is given.

    The returned airfoil may have ``coordinates=None`` if no source was found.
    """
    af = Airfoil(name=name)
    if af.coordinates is not None:
        return af

    if search_dir is not None:
        candidate = Path(search_dir) / f"{name}.dat"
        if not candidate.exists():
            candidate = Path(search_dir) / f"{name.lower()}.dat"
        if candidate.exists():
            return Airfoil.from_file(candidate)

    return af
