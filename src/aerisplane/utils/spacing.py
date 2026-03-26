"""Spacing functions for mesh generation and panel distributions."""

from __future__ import annotations

import numpy as np


def cosspace(start: float, stop: float, num: int) -> np.ndarray:
    """Cosine-spaced points — bunched near both ends.

    Equivalent to Chebyshev node distribution on [start, stop].
    """
    mean = (start + stop) / 2
    amp = (stop - start) / 2
    return mean + amp * np.cos(np.linspace(np.pi, 0, num))


def sinspace(start: float, stop: float, num: int, reverse: bool = False) -> np.ndarray:
    """Sine-spaced points — bunched near the start (or end if reverse=True)."""
    s = np.sin(np.linspace(0, np.pi / 2, num))
    if reverse:
        s = 1 - s[::-1]
    return start + (stop - start) * s
