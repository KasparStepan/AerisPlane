"""Component placement geometry: bounding boxes, containment, and intersection checks.

Provides axis-aligned bounding boxes (AABB) for hardware components, and utilities
to check whether components fit inside fuselage/wing volumes and whether any two
components overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aerisplane.core.fuselage import Fuselage


@dataclass
class ComponentBox:
    """Axis-aligned bounding box for a hardware component.

    Parameters
    ----------
    name : str
        Component identifier (e.g., "battery", "flight_controller").
    dimensions : array-like
        [length_x, width_y, height_z] in meters.
    position : array-like
        [x, y, z] center position in aircraft frame [m].
    """

    name: str
    dimensions: np.ndarray
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.dimensions = np.asarray(self.dimensions, dtype=float)
        self.position = np.asarray(self.position, dtype=float)

    def min_corner(self) -> np.ndarray:
        """Lower corner of the bounding box [x, y, z] [m]."""
        return self.position - self.dimensions / 2.0

    def max_corner(self) -> np.ndarray:
        """Upper corner of the bounding box [x, y, z] [m]."""
        return self.position + self.dimensions / 2.0

    def volume(self) -> float:
        """Volume of the bounding box [m^3]."""
        return float(np.prod(self.dimensions))


@dataclass
class Collision:
    """Record of two overlapping components.

    Parameters
    ----------
    box_a : str
        Name of the first component.
    box_b : str
        Name of the second component.
    overlap : numpy array
        Overlap distance along each axis [x, y, z] [m].
        Positive values indicate overlap on that axis.
    """

    box_a: str
    box_b: str
    overlap: np.ndarray


def boxes_intersect(a: ComponentBox, b: ComponentBox) -> bool:
    """Check if two axis-aligned bounding boxes overlap.

    Parameters
    ----------
    a, b : ComponentBox
        The two boxes to test.

    Returns
    -------
    bool
        True if the boxes overlap in all three axes.
    """
    a_min, a_max = a.min_corner(), a.max_corner()
    b_min, b_max = b.min_corner(), b.max_corner()

    # Boxes overlap only if they overlap on ALL three axes
    return bool(
        np.all(a_min < b_max) and np.all(b_min < a_max)
    )


def overlap_distance(a: ComponentBox, b: ComponentBox) -> np.ndarray:
    """Compute the overlap distance between two boxes on each axis.

    Returns a 3-element array. Positive values mean overlap on that axis,
    negative values mean separation (gap).
    """
    a_min, a_max = a.min_corner(), a.max_corner()
    b_min, b_max = b.min_corner(), b.max_corner()

    # Overlap on each axis = min of the maxes - max of the mins
    return np.minimum(a_max, b_max) - np.maximum(a_min, b_min)


def check_intersections(boxes: list[ComponentBox]) -> list[Collision]:
    """Check all pairs of boxes for intersections.

    Parameters
    ----------
    boxes : list of ComponentBox
        All component boxes to check.

    Returns
    -------
    list of Collision
        All pairs that overlap. Empty if no collisions.
    """
    collisions = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_intersect(boxes[i], boxes[j]):
                ovlp = overlap_distance(boxes[i], boxes[j])
                collisions.append(Collision(
                    box_a=boxes[i].name,
                    box_b=boxes[j].name,
                    overlap=ovlp,
                ))
    return collisions


def fuselage_contains_point(fuselage: Fuselage, point: np.ndarray) -> bool:
    """Check if a point lies inside the fuselage cross-section envelope.

    Interpolates the fuselage radius at the point's x-position and checks
    if the point's y-z distance from the fuselage centerline is within
    that radius.

    Parameters
    ----------
    fuselage : Fuselage
        The fuselage to check against.
    point : array-like
        [x, y, z] position in aircraft frame [m].

    Returns
    -------
    bool
        True if the point is inside the fuselage envelope.
    """
    point = np.asarray(point, dtype=float)

    if len(fuselage.xsecs) < 2:
        return False

    # Convert point to fuselage-local coordinates
    local_x = point[0] - fuselage.x_le
    local_y = point[1] - fuselage.y_le
    local_z = point[2] - fuselage.z_le

    x_stations = np.array([xsec.x for xsec in fuselage.xsecs])

    # Point must be within the fuselage x-range
    if local_x < x_stations[0] or local_x > x_stations[-1]:
        return False

    # Interpolate radius at this x-position
    radii = np.array([xsec.equivalent_radius() for xsec in fuselage.xsecs])
    radius_at_x = float(np.interp(local_x, x_stations, radii))

    # Check if y-z distance from centerline is within radius
    dist_yz = np.sqrt(local_y**2 + local_z**2)
    return bool(dist_yz <= radius_at_x)


def fuselage_contains_box(fuselage: Fuselage, box: ComponentBox) -> bool:
    """Check if an entire bounding box fits inside the fuselage.

    Tests all 8 corners of the box against the fuselage envelope.

    Parameters
    ----------
    fuselage : Fuselage
        The fuselage to check against.
    box : ComponentBox
        The component box to test.

    Returns
    -------
    bool
        True if all 8 corners are inside the fuselage.
    """
    half = box.dimensions / 2.0
    # Generate all 8 corners of the AABB
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner = box.position + half * np.array([sx, sy, sz])
                if not fuselage_contains_point(fuselage, corner):
                    return False
    return True


@dataclass
class PlacementResult:
    """Result of a placement validation check.

    Parameters
    ----------
    collisions : list of Collision
        All pairs of overlapping component boxes.
    containment : dict
        Maps component name to bool: True if inside its parent volume.
    is_valid : bool
        True if no collisions and all components are contained.
    """

    collisions: list[Collision]
    containment: dict[str, bool]

    @property
    def is_valid(self) -> bool:
        """True if placement has no collisions and all components are contained."""
        return len(self.collisions) == 0 and all(self.containment.values())

    def report(self) -> str:
        """Human-readable placement check summary."""
        lines = []
        lines.append("Placement Validation")
        lines.append("=" * 60)

        if self.is_valid:
            lines.append("PASS — all components fit and no intersections")
        else:
            if self.collisions:
                lines.append(f"FAIL — {len(self.collisions)} collision(s):")
                for c in self.collisions:
                    lines.append(
                        f"  {c.box_a} <-> {c.box_b}: overlap "
                        f"[{c.overlap[0]*1000:.1f}, {c.overlap[1]*1000:.1f}, "
                        f"{c.overlap[2]*1000:.1f}] mm"
                    )
            not_contained = [
                name for name, inside in self.containment.items() if not inside
            ]
            if not_contained:
                lines.append(f"FAIL — {len(not_contained)} component(s) outside volume:")
                for name in not_contained:
                    lines.append(f"  {name}")

        return "\n".join(lines)


def validate_placement(
    boxes: list[ComponentBox],
    fuselage: Fuselage | None = None,
) -> PlacementResult:
    """Run full placement validation: intersection + containment checks.

    Parameters
    ----------
    boxes : list of ComponentBox
        All component boxes to validate.
    fuselage : Fuselage or None
        If provided, checks that all boxes fit inside the fuselage.

    Returns
    -------
    PlacementResult
        Validation result with collision details and containment status.
    """
    collisions = check_intersections(boxes)

    containment: dict[str, bool] = {}
    if fuselage is not None:
        for box in boxes:
            containment[box.name] = fuselage_contains_box(fuselage, box)

    return PlacementResult(collisions=collisions, containment=containment)
