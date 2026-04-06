"""Native aircraft geometry plotter.

Pure matplotlib — no AeroSandbox required.

Public API
----------
plot_geometry(aircraft, style="three_view", show=True, save_path=None)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from aerisplane.core.aircraft import Aircraft


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _wing_outline(wing) -> list[np.ndarray]:
    """Return a list of (N, 3) polylines outlining the wing surface.

    Returns one leading-edge polyline, one trailing-edge polyline, and one
    spanwise tip-cap line.  Accounts for symmetric wings by mirroring.
    """
    xsecs = wing.xsecs
    le = np.array([xs.xyz_le for xs in xsecs])           # (n, 3)
    te = le.copy()
    te[:, 0] += np.array([xs.chord for xs in xsecs])     # TE = LE + chord along x

    lines = []
    # Leading edge
    lines.append(le.copy())
    # Trailing edge
    lines.append(te.copy())
    # Root cap
    lines.append(np.array([le[0], te[0]]))
    # Tip cap
    lines.append(np.array([le[-1], te[-1]]))
    # Spanwise chords at each xsec
    for i in range(len(xsecs)):
        lines.append(np.array([le[i], te[i]]))

    if wing.symmetric:
        mirror = []
        for poly in lines:
            m = poly.copy()
            m[:, 1] = -m[:, 1]
            mirror.append(m)
        lines += mirror

    return lines


def _fuselage_outline(fuselage) -> dict[str, np.ndarray]:
    """Return top, side, and centerline polylines for a fuselage.

    Returns dict with keys "top", "side", "centerline".
    """
    xsecs = fuselage.xsecs
    xs_x = np.array([fuselage.x_le + xs.x for xs in xsecs])

    def _radius(xs):
        if xs.shape in ("ellipse", "rectangle") and xs.width and xs.height:
            return xs.width / 2.0, xs.height / 2.0
        r = xs.radius or 0.0
        return r, r

    half_w = np.array([_radius(xs)[0] for xs in xsecs])
    half_h = np.array([_radius(xs)[1] for xs in xsecs])
    z_c = np.full(len(xsecs), fuselage.z_le)

    top_upper = np.column_stack([xs_x, np.zeros(len(xsecs)), z_c + half_h])
    top_lower = np.column_stack([xs_x, np.zeros(len(xsecs)), z_c - half_h])
    side_right = np.column_stack([xs_x, half_w, z_c])
    side_left  = np.column_stack([xs_x, -half_w, z_c])

    return {
        "top":    [top_upper, top_lower],
        "side":   [side_right, side_left],
        "center": [np.column_stack([xs_x, np.zeros(len(xsecs)), z_c])],
    }


def _collect_geometry(aircraft: Aircraft):
    """Collect all wing outlines and fuselage outlines for the aircraft."""
    wing_lines = []
    for wing in aircraft.wings:
        wing_lines.extend(_wing_outline(wing))

    fuse_lines = []
    for fuse in aircraft.fuselages:
        outlines = _fuselage_outline(fuse)
        for lines in outlines.values():
            fuse_lines.extend(lines)

    return wing_lines, fuse_lines


def _plot_lines_2d(ax, lines, color, lw=1.2):
    for poly in lines:
        ax.plot(poly[:, 0], poly[:, 1], color=color, lw=lw)


def _plot_lines_3d(ax, lines, color, lw=1.0):
    for poly in lines:
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color=color, lw=lw)


def _equal_aspect_2d(ax):
    ax.set_aspect("equal", adjustable="datalim")


def _equal_aspect_3d(ax):
    """Force equal aspect ratio on 3-D axes."""
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = lims.mean(axis=1)
    half = (lims[:, 1] - lims[:, 0]).max() / 2.0
    ax.set_xlim3d(center[0] - half, center[0] + half)
    ax.set_ylim3d(center[1] - half, center[1] + half)
    ax.set_zlim3d(center[2] - half, center[2] + half)


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

def plot_geometry(
    aircraft: Aircraft,
    style: str = "three_view",
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot aircraft geometry.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft to visualise.
    style : str
        ``"three_view"`` — 4-panel engineering drawing (top, front, side, iso).
        ``"wireframe"``  — single 3-D view.
    show : bool
        Call ``plt.show()`` after drawing.
    save_path : str or None
        Save figure to path if given.
    """
    if style not in ("three_view", "wireframe"):
        raise ValueError(f"Unknown style '{style}'. Choose 'three_view' or 'wireframe'.")

    wing_lines, fuse_lines = _collect_geometry(aircraft)
    all_lines = wing_lines + fuse_lines

    # ── colours ──────────────────────────────────────────────────────
    wc = "#1565C0"   # wing: deep blue
    fc = "#546E7A"   # fuselage: blue-grey

    if style == "wireframe":
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        _plot_lines_3d(ax, wing_lines, wc, lw=1.2)
        _plot_lines_3d(ax, fuse_lines, fc, lw=1.0)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_title(aircraft.name, fontweight="bold")
        _equal_aspect_3d(ax)

    else:  # three_view
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(aircraft.name, fontsize=13, fontweight="bold")
        fig.subplots_adjust(hspace=0.35, wspace=0.28,
                             left=0.07, right=0.97, top=0.93, bottom=0.07)

        ax_top  = axes[0, 0]   # XY plane: x→right, y→up  (top view)
        ax_front= axes[0, 1]   # YZ plane: y→right, z→up  (front view)
        ax_side = axes[1, 0]   # XZ plane: x→right, z→up  (side view)
        ax_3d   = fig.add_subplot(2, 2, 4, projection="3d")
        axes[1, 1].set_visible(False)

        # Top view: x (chord) vs y (span)
        for poly in wing_lines:
            ax_top.plot(poly[:, 1], -poly[:, 0], color=wc, lw=1.2)
        for poly in fuse_lines:
            ax_top.plot(poly[:, 1], -poly[:, 0], color=fc, lw=1.0)
        ax_top.set_xlabel("y [m]"); ax_top.set_ylabel("−x [m]")
        ax_top.set_title("Top view"); _equal_aspect_2d(ax_top)

        # Front view: y (span) vs z (height)
        for poly in wing_lines:
            ax_front.plot(poly[:, 1], poly[:, 2], color=wc, lw=1.2)
        for poly in fuse_lines:
            ax_front.plot(poly[:, 1], poly[:, 2], color=fc, lw=1.0)
        ax_front.set_xlabel("y [m]"); ax_front.set_ylabel("z [m]")
        ax_front.set_title("Front view"); _equal_aspect_2d(ax_front)

        # Side view: x (chord) vs z (height)
        for poly in wing_lines:
            ax_side.plot(poly[:, 0], poly[:, 2], color=wc, lw=1.2)
        for poly in fuse_lines:
            ax_side.plot(poly[:, 0], poly[:, 2], color=fc, lw=1.0)
        ax_side.set_xlabel("x [m]"); ax_side.set_ylabel("z [m]")
        ax_side.set_title("Side view"); _equal_aspect_2d(ax_side)

        # Isometric 3-D
        _plot_lines_3d(ax_3d, wing_lines, wc, lw=1.0)
        _plot_lines_3d(ax_3d, fuse_lines, fc, lw=0.9)
        ax_3d.set_xlabel("x"); ax_3d.set_ylabel("y"); ax_3d.set_zlabel("z")
        ax_3d.set_title("Isometric")
        _equal_aspect_3d(ax_3d)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
