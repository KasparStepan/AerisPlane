"""VLM flow visualisation — pressure distribution and streamlines.

Public API
----------
plot_surface_pressure(result, show=True, save_path=None)
    Cp on wing panels: 3-D view + top-down view.

plot_streamlines(result, plane="xz", x_slice=None, show=True, save_path=None)
    Streamlines on a 2-D cross-section of the flow field.
    plane="xz"  — side view (y = 0): shows upwash/downwash
    plane="yz"  — rear view (x = x_slice): shows tip vortices

plot_flow(result, show=True, save_path=None)
    Combined 4-panel figure: Cp 3D, Cp top-down, XZ streamlines, YZ crossflow.

plot_interactive(result, n_streamlines=40, show=True, save_path=None)
    Interactive Plotly 3-D figure: wing surface coloured by ΔCp + 3-D
    streamlines traced through the full VLM velocity field.  Requires plotly.

All functions require the result to have been produced by method="vlm".
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _require_vlm(result) -> object:
    """Return the VLM solver or raise a clear error."""
    vlm = result._solver
    if vlm is None or result.method != "vlm":
        raise RuntimeError(
            "Flow visualisation requires method='vlm' and an AeroResult "
            "produced by aerisplane.aero.analyze()."
        )
    return vlm


def _panel_cp(vlm, q: float) -> np.ndarray:
    """Pressure coefficient at each panel from the VLM normal force.

    ΔCp = F_normal / (q * A)  — positive is suction (upper surface).
    """
    F_g = np.asarray(vlm.forces_geometry)        # (N, 3) in geometry axes
    normals = np.asarray(vlm.normal_directions)  # (N, 3) unit normals
    areas = np.asarray(vlm.areas)                # (N,)
    F_n = np.sum(F_g * normals, axis=1)          # normal component
    return F_n / (q * areas + 1e-30)


def _wing_bounds(vlm):
    """Return (x_min, x_max, y_min, y_max, z_min, z_max) of all panel vertices."""
    pts = np.vstack([
        vlm.front_left_vertices,
        vlm.back_left_vertices,
        vlm.back_right_vertices,
        vlm.front_right_vertices,
    ])
    return pts.min(axis=0), pts.max(axis=0)


def _velocity_grid_xz(vlm, n_x=70, n_z=50, y=0.0):
    """Compute **induced** velocity on an XZ grid at a given y.

    Subtracts the freestream so the perturbation (upwash/downwash) is clearly
    visible instead of being swamped by the large freestream x-component.

    Returns (x_grid, z_grid, Vx_ind, Vz_ind, speed_ind).
    """
    mn, mx = _wing_bounds(vlm)
    chord_ref = mx[0] - mn[0]
    span_ref  = mx[1] - mn[1]

    x_min = mn[0] - 1.0 * chord_ref
    x_max = mx[0] + 5.0 * chord_ref
    z_half = max(chord_ref * 1.5, span_ref * 0.12)
    z_min = mn[2] - z_half
    z_max = mx[2] + z_half

    xg = np.linspace(x_min, x_max, n_x)
    zg = np.linspace(z_min, z_max, n_z)
    X, Z = np.meshgrid(xg, zg)
    Y = np.full_like(X, y)

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    V_total = vlm.get_velocity_at_points(points)

    # Subtract freestream to get induced-only perturbation field
    V_fs = vlm.steady_freestream_velocity   # (3,)
    Vx = V_total[:, 0].reshape(X.shape) - V_fs[0]
    Vz = V_total[:, 2].reshape(X.shape) - V_fs[2]
    speed = np.hypot(Vx, Vz)
    return xg, zg, Vx, Vz, speed


def _velocity_grid_yz(vlm, x_slice: float, n_y=70, n_z=50):
    """Compute induced velocity on a YZ cross-section.

    The freestream is subtracted so the tip-vortex swirl is clearly visible.
    Returns (y_grid, z_grid, Vy_ind, Vz_ind, speed_ind).
    """
    mn, mx = _wing_bounds(vlm)
    span_half = (mx[1] - mn[1]) * 0.65
    chord_ref = mx[0] - mn[0]
    z_half    = max(chord_ref * 1.2, span_half * 0.20)

    yg = np.linspace(-span_half, span_half, n_y)
    zg = np.linspace(-z_half, z_half, n_z)
    Y, Z = np.meshgrid(yg, zg)
    X = np.full_like(Y, x_slice)

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    V_total = vlm.get_velocity_at_points(points)

    V_fs = vlm.steady_freestream_velocity
    Vy = V_total[:, 1].reshape(Y.shape) - V_fs[1]
    Vz = V_total[:, 2].reshape(Y.shape) - V_fs[2]
    speed = np.hypot(Vy, Vz)
    return yg, zg, Vy, Vz, speed


def _add_wing_outline_xz(ax, vlm):
    """Overlay the wing cross-section outline on an XZ axes."""
    fl = vlm.front_left_vertices
    bl = vlm.back_left_vertices
    fr = vlm.front_right_vertices
    br = vlm.back_right_vertices
    # root chord (minimum absolute y)
    y_abs = np.abs(fl[:, 1])
    root_mask = y_abs < (y_abs.max() * 0.15)
    if root_mask.any():
        x_le = fl[root_mask, 0].min()
        x_te = bl[root_mask, 0].max()
        z_ctr = fl[root_mask, 2].mean()
        ax.plot([x_le, x_te], [z_ctr, z_ctr],
                color="white", lw=1.5, ls="--", alpha=0.6, zorder=5)


def _add_wing_outline_yz(ax, vlm):
    """Overlay the wing tips on a YZ axes (dots at tip y, z positions)."""
    fr = vlm.front_right_vertices
    fl = vlm.front_left_vertices
    y_right = fr[:, 1].max()
    y_left  = fl[:, 1].min()
    z_mean  = fr[fr[:, 1] > y_right * 0.95, 2].mean() if np.any(fr[:, 1] > y_right * 0.95) else 0.0
    ax.axvline(y_right, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(y_left,  color="white", lw=0.8, ls=":", alpha=0.5)
    ax.text(y_right, z_mean, " tip", color="white", fontsize=7, va="center")
    ax.text(y_left,  z_mean, "tip ", color="white", fontsize=7, va="center", ha="right")


# ------------------------------------------------------------------ #
# Public functions
# ------------------------------------------------------------------ #

def plot_surface_pressure(result, show: bool = True, save_path: str | None = None):
    """Plot pressure coefficient on VLM wing panels.

    Creates a 2-panel figure:
    - Left: 3-D isometric view with Cp colour map
    - Right: top-down view (y vs x) with Cp colour map

    Parameters
    ----------
    result : AeroResult
        Must have been produced by method="vlm".
    show : bool
        Call plt.show() after drawing.
    save_path : str or None
        Save figure to path if given.
    """
    vlm = _require_vlm(result)
    Cp = _panel_cp(vlm, result.dynamic_pressure)

    fl = vlm.front_left_vertices
    bl = vlm.back_left_vertices
    br = vlm.back_right_vertices
    fr = vlm.front_right_vertices

    verts = np.stack([fl, bl, br, fr], axis=1)   # (N, 4, 3)
    cmap = "RdBu_r"
    vmax = np.percentile(np.abs(Cp), 98)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        f"VLM surface pressure   α={result.alpha:.1f}°  V={result.velocity:.0f} m/s  "
        f"CL={result.CL:.3f}",
        fontsize=12, fontweight="bold",
    )

    # ── 3-D view ─────────────────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")
    poly3d = Poly3DCollection(verts, cmap=cmap, norm=norm, linewidths=0.2,
                               edgecolors="k", alpha=0.95)
    poly3d.set_array(Cp)
    ax3d.add_collection3d(poly3d)
    mn, mx = _wing_bounds(vlm)
    pad = (mx - mn).max() * 0.1
    ax3d.set_xlim(mn[0] - pad, mx[0] + pad)
    ax3d.set_ylim(mn[1] - pad, mx[1] + pad)
    ax3d.set_zlim(mn[2] - pad, mx[2] + pad)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("3-D view")
    ax3d.view_init(elev=25, azim=-60)
    # equal aspect
    extent = (mx - mn).max() / 2
    mid = (mn + mx) / 2
    ax3d.set_xlim(mid[0]-extent, mid[0]+extent)
    ax3d.set_ylim(mid[1]-extent, mid[1]+extent)
    ax3d.set_zlim(mid[2]-extent, mid[2]+extent)

    # ── Top-down view ─────────────────────────────────────────────────
    ax2d = fig.add_subplot(122)
    # y vs −x so leading edge is at top
    verts_2d = verts[:, :, [1, 0]]    # (N, 4, [y, x])
    verts_2d[:, :, 1] = -verts_2d[:, :, 1]  # flip x so LE is top
    poly2d = PolyCollection(verts_2d, cmap=cmap, norm=norm,
                            linewidths=0.2, edgecolors="k")
    poly2d.set_array(Cp)
    ax2d.add_collection(poly2d)
    ax2d.set_xlim(mn[1] - pad, mx[1] + pad)
    ax2d.set_ylim(-mx[0] - pad, -mn[0] + pad)
    ax2d.set_xlabel("y [m]  (span)")
    ax2d.set_ylabel("−x [m]  (←LE)")
    ax2d.set_title("Top-down view")
    ax2d.set_aspect("equal")

    # ── Shared colorbar ──────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="ΔCp  (suction +)")

    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.08, wspace=0.25)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_streamlines(
    result,
    plane: str = "xz",
    x_slice: float | None = None,
    show: bool = True,
    save_path: str | None = None,
):
    """Plot 2-D streamlines on a cross-section of the VLM flow field.

    Parameters
    ----------
    result : AeroResult
        Must have been produced by method="vlm".
    plane : str
        ``"xz"`` — symmetry plane (y = 0): shows upwash ahead and
        downwash/wake behind the wing.
        ``"yz"`` — rear cross-section at ``x_slice``: shows the two
        counter-rotating tip vortices.
    x_slice : float or None
        x-position [m] for the ``"yz"`` slice.  Defaults to 2 chord
        lengths behind the trailing edge.
    show : bool
        Call plt.show() after drawing.
    save_path : str or None
        Save figure to path if given.
    """
    vlm = _require_vlm(result)
    mn, mx = _wing_bounds(vlm)
    chord = mx[0] - mn[0]

    if plane == "xz":
        xg, zg, Vx, Vz, speed = _velocity_grid_xz(vlm)
        V_inf = result.velocity

        fig, ax = plt.subplots(figsize=(12, 5))
        strm = ax.streamplot(
            xg, zg, Vx, Vz,
            color=speed / V_inf,
            cmap="plasma",
            linewidth=1.2,
            density=1.8,
            arrowsize=0.8,
        )
        cbar = fig.colorbar(strm.lines, ax=ax, label="|V_induced| [m/s]")

        # Wing chord line at root
        _add_wing_outline_xz(ax, vlm)

        ax.set_xlabel("x [m]  (chord direction)")
        ax.set_ylabel("z [m]  (height)")
        ax.set_title(
            f"Induced flow — symmetry plane (y = 0)\n"
            f"α={result.alpha:.1f}°  V={result.velocity:.0f} m/s  CL={result.CL:.3f}"
        )
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.tick_params(labelcolor="white")
        cbar.set_label("|V| / V∞", color="white")

    elif plane == "yz":
        if x_slice is None:
            x_slice = mx[0] + 2.0 * chord
        yg, zg, Vy, Vz, speed = _velocity_grid_yz(vlm, x_slice)
        V_inf = result.velocity

        fig, ax = plt.subplots(figsize=(7, 6))
        strm = ax.streamplot(
            yg, zg, Vy, Vz,
            color=speed / V_inf,
            cmap="plasma",
            linewidth=1.2,
            density=1.8,
            arrowsize=0.8,
        )
        cbar = fig.colorbar(strm.lines, ax=ax, label="|V_induced_yz| [m/s]")
        _add_wing_outline_yz(ax, vlm)

        ax.set_xlabel("y [m]  (span)")
        ax.set_ylabel("z [m]  (height)")
        ax.set_title(
            f"Induced crossflow  x = {x_slice:.2f} m  (≈ {(x_slice-mx[0])/chord:.1f}c behind TE)\n"
            f"α={result.alpha:.1f}°  V={result.velocity:.0f} m/s"
        )
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.tick_params(labelcolor="white")
        cbar.set_label("|V_yz| / V∞", color="white")
    else:
        raise ValueError(f"Unknown plane '{plane}'. Choose 'xz' or 'yz'.")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_flow(result, show: bool = True, save_path: str | None = None):
    """Combined 4-panel flow visualisation figure.

    Panels:
    - Top-left:  Cp on wing surface (top-down)
    - Top-right: 3-D Cp surface view
    - Bottom-left:  XZ streamlines (symmetry plane)
    - Bottom-right: YZ crossflow behind TE

    Parameters
    ----------
    result : AeroResult
        Must have been produced by method="vlm".
    show : bool
        Call plt.show() after drawing.
    save_path : str or None
        Save figure to path if given.
    """
    vlm = _require_vlm(result)
    Cp = _panel_cp(vlm, result.dynamic_pressure)
    mn, mx = _wing_bounds(vlm)
    chord = mx[0] - mn[0]

    fl = vlm.front_left_vertices
    bl = vlm.back_left_vertices
    br = vlm.back_right_vertices
    fr = vlm.front_right_vertices
    verts = np.stack([fl, bl, br, fr], axis=1)

    vmax = np.percentile(np.abs(Cp), 98)
    norm_cp = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap_cp = "RdBu_r"

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle(
        f"VLM flow field   α={result.alpha:.1f}°   V={result.velocity:.0f} m/s   "
        f"h={result.altitude:.0f} m   CL={result.CL:.3f}   CD={result.CD:.4f}",
        fontsize=12, fontweight="bold", color="white",
    )
    fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.07,
                        hspace=0.38, wspace=0.32)

    # ── Panel A: top-down Cp ─────────────────────────────────────────
    ax_td = fig.add_subplot(2, 2, 1)
    ax_td.set_facecolor("#0f0f1a")
    pad = (mx - mn).max() * 0.1
    verts_2d = verts[:, :, [1, 0]].copy()
    verts_2d[:, :, 1] = -verts_2d[:, :, 1]
    poly2d = PolyCollection(verts_2d, cmap=cmap_cp, norm=norm_cp,
                            linewidths=0.15, edgecolors="#444")
    poly2d.set_array(Cp)
    ax_td.add_collection(poly2d)
    ax_td.set_xlim(mn[1] - pad, mx[1] + pad)
    ax_td.set_ylim(-mx[0] - pad, -mn[0] + pad)
    ax_td.set_aspect("equal")
    ax_td.set_xlabel("y [m]", color="white"); ax_td.set_ylabel("−x [m]", color="white")
    ax_td.set_title("Cp — top-down", color="white")
    ax_td.tick_params(colors="white")
    for sp in ax_td.spines.values(): sp.set_edgecolor("#555")
    sm_cp = plt.cm.ScalarMappable(cmap=cmap_cp, norm=norm_cp)
    sm_cp.set_array([])
    cb1 = fig.colorbar(sm_cp, ax=ax_td, shrink=0.85, pad=0.03)
    cb1.set_label("ΔCp", color="white"); cb1.ax.tick_params(labelcolor="white")

    # ── Panel B: 3-D Cp ──────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 2, 2, projection="3d")
    ax3d.set_facecolor("#0f0f1a")
    poly3d = Poly3DCollection(verts, cmap=cmap_cp, norm=norm_cp,
                               linewidths=0.15, edgecolors="#444", alpha=0.95)
    poly3d.set_array(Cp)
    ax3d.add_collection3d(poly3d)
    extent = (mx - mn).max() / 2
    mid = (mn + mx) / 2
    ax3d.set_xlim(mid[0]-extent, mid[0]+extent)
    ax3d.set_ylim(mid[1]-extent, mid[1]+extent)
    ax3d.set_zlim(mid[2]-extent, mid[2]+extent)
    ax3d.set_xlabel("x", color="white"); ax3d.set_ylabel("y", color="white")
    ax3d.set_zlabel("z", color="white")
    ax3d.set_title("Cp — 3-D view", color="white")
    ax3d.view_init(elev=22, azim=-55)
    ax3d.tick_params(colors="white")
    ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#333"); ax3d.yaxis.pane.set_edgecolor("#333")
    ax3d.zaxis.pane.set_edgecolor("#333")

    # ── Panel C: XZ streamlines ──────────────────────────────────────
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.set_facecolor("#1a1a2e")
    xg, zg, Vx, Vz, speed = _velocity_grid_xz(vlm, n_x=55, n_z=35)
    strm_xz = ax_xz.streamplot(
        xg, zg, Vx, Vz,
        color=speed / result.velocity, cmap="plasma",
        linewidth=1.1, density=1.6, arrowsize=0.7,
    )
    _add_wing_outline_xz(ax_xz, vlm)
    ax_xz.set_xlabel("x [m]", color="white"); ax_xz.set_ylabel("z [m]", color="white")
    ax_xz.set_title("Streamlines — symmetry plane  (y = 0)", color="white")
    ax_xz.tick_params(colors="white")
    for sp in ax_xz.spines.values(): sp.set_edgecolor("#555")
    cb3 = fig.colorbar(strm_xz.lines, ax=ax_xz, shrink=0.85, pad=0.03)
    cb3.set_label("|V_induced| [m/s]", color="white"); cb3.ax.tick_params(labelcolor="white")

    # ── Panel D: YZ crossflow ────────────────────────────────────────
    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.set_facecolor("#1a1a2e")
    x_slice = mx[0] + 2.0 * chord
    yg, zg, Vy, Vz, speed_yz = _velocity_grid_yz(vlm, x_slice, n_y=55, n_z=35)
    strm_yz = ax_yz.streamplot(
        yg, zg, Vy, Vz,
        color=speed_yz / result.velocity, cmap="plasma",
        linewidth=1.1, density=1.6, arrowsize=0.7,
    )
    _add_wing_outline_yz(ax_yz, vlm)
    ax_yz.set_xlabel("y [m]", color="white"); ax_yz.set_ylabel("z [m]", color="white")
    ax_yz.set_title(
        f"Tip vortices — x = {x_slice:.2f} m  ({2.0:.1f}c behind TE)", color="white"
    )
    ax_yz.tick_params(colors="white")
    for sp in ax_yz.spines.values(): sp.set_edgecolor("#555")
    cb4 = fig.colorbar(strm_yz.lines, ax=ax_yz, shrink=0.85, pad=0.03)
    cb4.set_label("|V_induced_yz| [m/s]", color="white"); cb4.ax.tick_params(labelcolor="white")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()


# ------------------------------------------------------------------ #
# Interactive Plotly visualisation
# ------------------------------------------------------------------ #

def _trace_streamline_3d(vlm, seed: np.ndarray, n_steps: int = 300,
                          ds: float = 0.01,
                          x_min: float = -np.inf, x_max: float = np.inf) -> np.ndarray:
    """Trace a 3-D streamline using 4th-order Runge-Kutta.

    Uses the *total* velocity (induced + freestream) so the line follows the
    actual flow, not just the perturbation.

    Parameters
    ----------
    vlm : VortexLatticeMethod solver
    seed : (3,) array — starting point [m]
    n_steps : int — maximum integration steps
    ds : float — arc-length step size [m]
    x_min, x_max : float — stop integration outside this x range

    Returns
    -------
    pts : (M, 3) array — streamline path
    """
    V_fs = np.asarray(vlm.steady_freestream_velocity)

    def velocity(p):
        V_ind = vlm.get_velocity_at_points(p.reshape(1, 3))[0]
        return V_ind + V_fs

    p = np.asarray(seed, dtype=float)
    pts = [p.copy()]
    for _ in range(n_steps):
        if p[0] < x_min or p[0] > x_max:
            break
        v1 = velocity(p)
        spd = np.linalg.norm(v1)
        if spd < 1e-6:
            break
        h = ds / spd          # normalise step to arc length
        k1 = v1
        k2 = velocity(p + 0.5 * h * k1)
        k3 = velocity(p + 0.5 * h * k2)
        k4 = velocity(p + h * k3)
        p = p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pts.append(p.copy())
    return np.array(pts)


def _make_seed_points(vlm, n_span: int = 6, n_z: int = 3) -> list[np.ndarray]:
    """Generate a grid of seed points upstream of the leading edge.

    Seeds are placed one chord length upstream, distributed across the span
    and at a few z offsets so streamlines capture both the upper-surface
    acceleration and the lower-surface stagnation.
    """
    mn, mx = _wing_bounds(vlm)
    chord = mx[0] - mn[0]
    span_half = mx[1]                # semi-span (positive y side)

    x_seed = mn[0] - 0.8 * chord    # just upstream of LE
    z_center = (mn[2] + mx[2]) / 2
    z_offsets = np.linspace(-0.25 * chord, 0.25 * chord, n_z)
    y_positions = np.linspace(-span_half * 0.95, span_half * 0.95, n_span)

    seeds = []
    for y in y_positions:
        for dz in z_offsets:
            seeds.append(np.array([x_seed, y, z_center + dz]))
    return seeds


def plot_interactive(
    result,
    n_streamlines: int = 40,
    show: bool = True,
    save_path: str | None = None,
):
    """Interactive 3-D Plotly figure: wing Cp surface + 3-D streamlines.

    Requires ``plotly`` (``pip install plotly``).

    Parameters
    ----------
    result : AeroResult
        Must have been produced by ``method='vlm'``.
    n_streamlines : int
        Approximate number of streamlines to trace.  They are seeded on a
        span × z grid one chord upstream of the leading edge.  Default 40.
    show : bool
        Call ``fig.show()`` (opens browser / inline in Jupyter).
    save_path : str or None
        If given, save an interactive HTML file to this path.
    """
    import plotly.graph_objects as go

    vlm = _require_vlm(result)
    Cp = _panel_cp(vlm, result.dynamic_pressure)

    # ── Wing surface mesh ────────────────────────────────────────────
    fl = np.asarray(vlm.front_left_vertices)
    bl = np.asarray(vlm.back_left_vertices)
    br = np.asarray(vlm.back_right_vertices)
    fr = np.asarray(vlm.front_right_vertices)
    N = len(fl)

    # Triangulate each quad panel into 2 triangles.
    # Vertex layout: block 0 = fl, 1 = bl, 2 = br, 3 = fr
    all_v = np.vstack([fl, bl, br, fr])    # (4N, 3)
    idx0 = np.arange(N)
    idx1 = idx0 + N
    idx2 = idx0 + 2 * N
    idx3 = idx0 + 3 * N

    tri_i = np.concatenate([idx0, idx0])
    tri_j = np.concatenate([idx1, idx2])
    tri_k = np.concatenate([idx2, idx3])
    tri_c = np.concatenate([Cp, Cp])       # one value per triangle

    vmax = float(np.percentile(np.abs(Cp), 98))

    wing_trace = go.Mesh3d(
        x=all_v[:, 0], y=all_v[:, 1], z=all_v[:, 2],
        i=tri_i, j=tri_j, k=tri_k,
        intensity=tri_c,
        colorscale="RdBu_r",
        cmin=-vmax, cmid=0.0, cmax=vmax,
        colorbar=dict(
            title=dict(text="ΔCp", font=dict(color="white")),
            tickfont=dict(color="white"),
            x=1.02, thickness=15, len=0.75,
        ),
        name="Wing ΔCp",
        showscale=True,
        opacity=0.95,
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1),
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Wing</extra>",
    )

    # ── Streamlines ──────────────────────────────────────────────────
    mn, mx_b = _wing_bounds(vlm)
    chord = mx_b[0] - mn[0]
    x_stop = mx_b[0] + 8.0 * chord   # stop tracing 8c behind TE

    # Determine seed grid size from n_streamlines
    n_span = max(3, int(round((n_streamlines / 3) ** 0.5 * 3)))
    n_z    = max(2, n_streamlines // n_span)
    seeds = _make_seed_points(vlm, n_span=n_span, n_z=n_z)

    # Arc-length step: ~1% of chord, 250 steps to cross ~8c downstream
    ds = chord * 0.015
    n_steps = int(9.0 * chord / ds)

    stream_traces = []
    for seed in seeds:
        pts = _trace_streamline_3d(
            vlm, seed, n_steps=n_steps, ds=ds,
            x_min=mn[0] - 2 * chord, x_max=x_stop,
        )
        if len(pts) < 3:
            continue
        # Colour by local speed magnitude to distinguish fast/slow regions
        V_fs = np.asarray(vlm.steady_freestream_velocity)
        # Thin out points for file size (every 3rd)
        pts = pts[::3]
        stream_traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            line=dict(color="rgba(120, 200, 255, 0.55)", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Layout ──────────────────────────────────────────────────────
    eye_dist = (mx_b - mn).max() * 3.5
    mid = (mn + mx_b) / 2
    scene_range = (mx_b - mn).max() * 0.75

    fig = go.Figure(data=[wing_trace] + stream_traces)
    fig.update_layout(
        title=dict(
            text=(
                f"VLM interactive flow — "
                f"α={result.alpha:.1f}°  V={result.velocity:.0f} m/s  "
                f"h={result.altitude:.0f} m  CL={result.CL:.3f}  CD={result.CD:.4f}"
            ),
            font=dict(color="white", size=14),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="x [m]", backgroundcolor="#0f0f1a",
                       gridcolor="#333", zerolinecolor="#444",
                       tickfont=dict(color="white"),
                       title_font=dict(color="white"),
                       range=[mid[0] - scene_range, mid[0] + scene_range]),
            yaxis=dict(title="y [m]", backgroundcolor="#0f0f1a",
                       gridcolor="#333", zerolinecolor="#444",
                       tickfont=dict(color="white"),
                       title_font=dict(color="white"),
                       range=[mid[1] - scene_range, mid[1] + scene_range]),
            zaxis=dict(title="z [m]", backgroundcolor="#0f0f1a",
                       gridcolor="#333", zerolinecolor="#444",
                       tickfont=dict(color="white"),
                       title_font=dict(color="white"),
                       range=[mid[2] - scene_range, mid[2] + scene_range]),
            bgcolor="#0f0f1a",
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=-1.8, z=0.8),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        paper_bgcolor="#0f0f1a",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=50, b=10),
        width=950, height=700,
    )

    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn")
    if show:
        fig.show()
