"""Component-based mass buildup from aircraft geometry and materials."""

from __future__ import annotations

import numpy as np

from aerisplane.core.aircraft import Aircraft
from aerisplane.core.fuselage import Fuselage
from aerisplane.core.wing import Wing
from aerisplane.weights.result import ComponentMass


# ---------------------------------------------------------------------------
# Wing structure
# ---------------------------------------------------------------------------

def _wing_panel_span(wing: Wing, i: int) -> float:
    """Euclidean distance between two adjacent wing sections [m]."""
    p0 = wing.xsecs[i].xyz_le
    p1 = wing.xsecs[i + 1].xyz_le
    return float(np.linalg.norm(p1 - p0))


def _wing_panel_wetted_area(wing: Wing, i: int) -> float:
    """Approximate wetted area of one panel (top + bottom) [m^2]."""
    c0 = wing.xsecs[i].chord
    c1 = wing.xsecs[i + 1].chord
    span = _wing_panel_span(wing, i)
    # Average chord * span * 2 surfaces (top + bottom)
    return (c0 + c1) / 2.0 * span * 2.0


def _wing_panel_midpoint(wing: Wing, i: int) -> np.ndarray:
    """Midpoint between two adjacent section LE positions [m]."""
    return (wing.xsecs[i].xyz_le + wing.xsecs[i + 1].xyz_le) / 2.0


def _wing_mass(wing: Wing) -> list[ComponentMass]:
    """Compute wing structural mass from spars and skin.

    Returns separate ComponentMass entries for spar and skin.
    For symmetric wings, mass is doubled and CG_y is set to 0.
    """
    spar_mass_total = 0.0
    spar_cg_weighted = np.zeros(3)
    skin_mass_total = 0.0
    skin_cg_weighted = np.zeros(3)

    n_panels = len(wing.xsecs) - 1
    if n_panels <= 0:
        return []

    for i in range(n_panels):
        sec0 = wing.xsecs[i]
        sec1 = wing.xsecs[i + 1]
        panel_span = _wing_panel_span(wing, i)
        midpoint = _wing_panel_midpoint(wing, i)

        # --- Spar mass ---
        if sec0.spar is not None or sec1.spar is not None:
            # Average mass_per_length if both sections have spars,
            # otherwise use whichever is defined
            mpl_values = []
            if sec0.spar is not None:
                mpl_values.append(sec0.spar.mass_per_length())
            if sec1.spar is not None:
                mpl_values.append(sec1.spar.mass_per_length())
            avg_mpl = sum(mpl_values) / len(mpl_values)

            panel_spar_mass = avg_mpl * panel_span
            spar_mass_total += panel_spar_mass

            # Spar CG at panel midpoint, offset to spar chordwise position
            spar_positions = []
            if sec0.spar is not None:
                spar_positions.append(sec0.spar.position)
            if sec1.spar is not None:
                spar_positions.append(sec1.spar.position)
            avg_spar_pos = sum(spar_positions) / len(spar_positions)
            avg_chord = (sec0.chord + sec1.chord) / 2.0

            spar_cg = midpoint.copy()
            spar_cg[0] += avg_spar_pos * avg_chord
            spar_cg_weighted += panel_spar_mass * spar_cg

        # --- Skin mass ---
        if sec0.skin is not None or sec1.skin is not None:
            mpa_values = []
            if sec0.skin is not None:
                mpa_values.append(sec0.skin.mass_per_area())
            if sec1.skin is not None:
                mpa_values.append(sec1.skin.mass_per_area())
            avg_mpa = sum(mpa_values) / len(mpa_values)

            wetted = _wing_panel_wetted_area(wing, i)
            panel_skin_mass = avg_mpa * wetted
            skin_mass_total += panel_skin_mass

            # Skin CG at panel midpoint, offset to ~40% chord
            skin_cg = midpoint.copy()
            avg_chord = (sec0.chord + sec1.chord) / 2.0
            skin_cg[0] += 0.4 * avg_chord
            skin_cg_weighted += panel_skin_mass * skin_cg

    # For symmetric wings: double mass, set CG_y = 0
    symmetry_factor = 2.0 if wing.symmetric else 1.0

    components = []

    if spar_mass_total > 0:
        spar_cg = spar_cg_weighted / spar_mass_total
        if wing.symmetric:
            spar_cg[1] = 0.0
        components.append(ComponentMass(
            name=f"{wing.name}_spar",
            mass=spar_mass_total * symmetry_factor,
            cg=spar_cg,
        ))

    if skin_mass_total > 0:
        skin_cg = skin_cg_weighted / skin_mass_total
        if wing.symmetric:
            skin_cg[1] = 0.0
        # Add rib estimate: 15% of skin mass
        rib_factor = 1.15
        components.append(ComponentMass(
            name=f"{wing.name}_skin",
            mass=skin_mass_total * symmetry_factor * rib_factor,
            cg=skin_cg,
        ))

    return components


# ---------------------------------------------------------------------------
# Fuselage structure
# ---------------------------------------------------------------------------

def _fuselage_mass(fuselage: Fuselage) -> list[ComponentMass]:
    """Compute fuselage shell mass from geometry and material.

    Returns empty list if fuselage has no material (must be overridden).
    """
    if fuselage.material is None:
        return []

    if len(fuselage.xsecs) < 2:
        return []

    shell_mass = (
        fuselage.wetted_area() * fuselage.wall_thickness * fuselage.material.density
    )

    if shell_mass <= 0:
        return []

    # CG at the area-weighted centroid along the fuselage axis
    x_stations = np.array([xsec.x for xsec in fuselage.xsecs])
    perimeters = np.array([xsec.perimeter() for xsec in fuselage.xsecs])

    # Weighted average of x-positions by local wetted area contribution
    # Approximate: use perimeter * dx as local area, weight x by midpoint
    cg_x_weighted = 0.0
    total_weight = 0.0
    for i in range(len(fuselage.xsecs) - 1):
        dx = x_stations[i + 1] - x_stations[i]
        avg_perim = (perimeters[i] + perimeters[i + 1]) / 2.0
        local_area = avg_perim * dx
        mid_x = (x_stations[i] + x_stations[i + 1]) / 2.0
        cg_x_weighted += local_area * mid_x
        total_weight += local_area

    cg_x = cg_x_weighted / total_weight if total_weight > 0 else 0.0

    # Offset by fuselage position in aircraft frame
    cg = np.array([fuselage.x_le + cg_x, fuselage.y_le, fuselage.z_le])

    return [ComponentMass(name=f"{fuselage.name}_shell", mass=shell_mass, cg=cg)]


# ---------------------------------------------------------------------------
# Hardware masses (propulsion + servos)
# ---------------------------------------------------------------------------

def _hardware_masses(aircraft: Aircraft) -> list[ComponentMass]:
    """Extract hardware component masses from the aircraft."""
    components = []

    if aircraft.propulsion is not None:
        ps = aircraft.propulsion
        pos = ps.position

        components.append(ComponentMass(
            name="motor", mass=ps.motor.mass, cg=pos.copy(),
        ))
        components.append(ComponentMass(
            name="propeller", mass=ps.propeller.mass, cg=pos.copy(),
        ))
        components.append(ComponentMass(
            name="battery", mass=ps.battery.mass, cg=pos.copy(),
        ))
        components.append(ComponentMass(
            name="esc", mass=ps.esc.mass, cg=pos.copy(),
        ))

    # Servos from control surfaces on all wings
    for wing in aircraft.wings:
        for cs in wing.control_surfaces:
            if cs.servo is None:
                continue

            # Estimate servo position: interpolate along wing span
            mid_frac = (cs.span_start + cs.span_end) / 2.0
            if len(wing.xsecs) >= 2:
                root_le = wing.xsecs[0].xyz_le
                tip_le = wing.xsecs[-1].xyz_le
                servo_cg = root_le + mid_frac * (tip_le - root_le)
            else:
                servo_cg = np.zeros(3)

            # For symmetric wings, add servos for both sides
            if wing.symmetric:
                servo_cg_r = servo_cg.copy()
                components.append(ComponentMass(
                    name=f"{cs.name}_servo_R",
                    mass=cs.servo.mass,
                    cg=servo_cg_r,
                ))
                servo_cg_l = servo_cg.copy()
                servo_cg_l[1] = -servo_cg_l[1]
                components.append(ComponentMass(
                    name=f"{cs.name}_servo_L",
                    mass=cs.servo.mass,
                    cg=servo_cg_l,
                ))
            else:
                components.append(ComponentMass(
                    name=f"{cs.name}_servo",
                    mass=cs.servo.mass,
                    cg=servo_cg.copy(),
                ))

    return components


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def _payload_mass(aircraft: Aircraft) -> list[ComponentMass]:
    """Extract payload mass."""
    if aircraft.payload is None:
        return []
    if aircraft.payload.mass <= 0:
        return []

    return [ComponentMass(
        name=aircraft.payload.name,
        mass=aircraft.payload.mass,
        cg=aircraft.payload.cg.copy(),
    )]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_buildup(aircraft: Aircraft) -> list[ComponentMass]:
    """Run the full mass buildup for an aircraft.

    Walks the aircraft tree and computes structural and hardware masses
    from geometry, materials, and component properties.

    Parameters
    ----------
    aircraft : Aircraft
        The aircraft to analyze.

    Returns
    -------
    list of ComponentMass
        All computed component masses.
    """
    components: list[ComponentMass] = []

    # Structural masses
    for wing in aircraft.wings:
        components.extend(_wing_mass(wing))

    for fuselage in aircraft.fuselages:
        components.extend(_fuselage_mass(fuselage))

    # Hardware
    components.extend(_hardware_masses(aircraft))

    # Payload
    components.extend(_payload_mass(aircraft))

    return components


def aggregate(components: list[ComponentMass], reference_area: float) -> tuple[
    float, np.ndarray, np.ndarray, float
]:
    """Compute total mass, CG, inertia tensor, and wing loading.

    Parameters
    ----------
    components : list of ComponentMass
        All component masses.
    reference_area : float
        Main wing planform area [m^2] for wing loading calculation.

    Returns
    -------
    total_mass : float
        Sum of all component masses [kg].
    cg : numpy array
        Mass-weighted CG [x, y, z] [m].
    inertia : numpy array
        3x3 inertia tensor about CG [kg*m^2].
    wing_loading : float
        Wing loading [g/dm^2].
    """
    if not components:
        return 0.0, np.zeros(3), np.zeros((3, 3)), 0.0

    total_mass = sum(c.mass for c in components)

    if total_mass <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3)), 0.0

    # Mass-weighted CG
    cg = sum(c.mass * c.cg for c in components) / total_mass

    # Inertia tensor via parallel axis theorem (point masses)
    inertia = np.zeros((3, 3))
    for c in components:
        r = c.cg - cg  # offset from overall CG
        r_sq = float(np.dot(r, r))
        inertia += c.mass * (r_sq * np.eye(3) - np.outer(r, r))

    # Wing loading: total mass in grams / wing area in dm^2
    if reference_area > 0:
        wing_loading = (total_mass * 1000.0) / (reference_area * 100.0)
    else:
        wing_loading = 0.0

    return total_mass, cg, inertia, wing_loading
