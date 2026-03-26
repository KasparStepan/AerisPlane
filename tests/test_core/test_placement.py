"""Tests for component placement: bounding boxes, intersection, containment."""

import numpy as np
import numpy.testing as npt
import pytest

import aerisplane as ap
from aerisplane.core.placement import (
    ComponentBox,
    boxes_intersect,
    check_intersections,
    fuselage_contains_box,
    fuselage_contains_point,
    overlap_distance,
    validate_placement,
)


# ===================================================================
# ComponentBox
# ===================================================================

class TestComponentBox:
    def test_min_max_corners(self):
        box = ComponentBox("test", dimensions=[0.1, 0.04, 0.03], position=[0.5, 0, 0])
        npt.assert_array_almost_equal(box.min_corner(), [0.45, -0.02, -0.015])
        npt.assert_array_almost_equal(box.max_corner(), [0.55, 0.02, 0.015])

    def test_volume(self):
        box = ComponentBox("test", dimensions=[0.1, 0.04, 0.03])
        assert box.volume() == pytest.approx(0.1 * 0.04 * 0.03)


# ===================================================================
# Intersection checks
# ===================================================================

class TestBoxIntersection:
    def test_overlapping_boxes(self):
        a = ComponentBox("a", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        b = ComponentBox("b", dimensions=[0.1, 0.1, 0.1], position=[0.05, 0, 0])
        assert boxes_intersect(a, b) is True

    def test_separated_boxes(self):
        a = ComponentBox("a", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        b = ComponentBox("b", dimensions=[0.1, 0.1, 0.1], position=[0.2, 0, 0])
        assert boxes_intersect(a, b) is False

    def test_touching_boxes_not_overlapping(self):
        """Boxes touching on a face (zero gap) are not considered overlapping."""
        a = ComponentBox("a", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        b = ComponentBox("b", dimensions=[0.1, 0.1, 0.1], position=[0.1, 0, 0])
        assert boxes_intersect(a, b) is False

    def test_identical_boxes(self):
        a = ComponentBox("a", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        b = ComponentBox("b", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        assert boxes_intersect(a, b) is True

    def test_one_axis_separated(self):
        """Boxes overlap in x and y but separated in z."""
        a = ComponentBox("a", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        b = ComponentBox("b", dimensions=[0.1, 0.1, 0.1], position=[0.05, 0.05, 0.2])
        assert boxes_intersect(a, b) is False

    def test_overlap_distance_overlapping(self):
        a = ComponentBox("a", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        b = ComponentBox("b", dimensions=[0.1, 0.1, 0.1], position=[0.05, 0, 0])
        ovlp = overlap_distance(a, b)
        assert ovlp[0] == pytest.approx(0.05)  # 50mm overlap in x
        assert ovlp[1] == pytest.approx(0.1)   # full overlap in y
        assert ovlp[2] == pytest.approx(0.1)   # full overlap in z

    def test_overlap_distance_separated(self):
        a = ComponentBox("a", dimensions=[0.1, 0.1, 0.1], position=[0.0, 0, 0])
        b = ComponentBox("b", dimensions=[0.1, 0.1, 0.1], position=[0.2, 0, 0])
        ovlp = overlap_distance(a, b)
        assert ovlp[0] < 0  # negative = gap


class TestCheckIntersections:
    def test_no_collisions(self):
        boxes = [
            ComponentBox("a", [0.1, 0.04, 0.03], position=[0.2, 0, 0]),
            ComponentBox("b", [0.1, 0.04, 0.03], position=[0.5, 0, 0]),
            ComponentBox("c", [0.1, 0.04, 0.03], position=[0.8, 0, 0]),
        ]
        assert check_intersections(boxes) == []

    def test_one_collision(self):
        boxes = [
            ComponentBox("a", [0.1, 0.1, 0.1], position=[0.0, 0, 0]),
            ComponentBox("b", [0.1, 0.1, 0.1], position=[0.05, 0, 0]),
            ComponentBox("c", [0.1, 0.1, 0.1], position=[0.5, 0, 0]),
        ]
        collisions = check_intersections(boxes)
        assert len(collisions) == 1
        assert collisions[0].box_a == "a"
        assert collisions[0].box_b == "b"

    def test_multiple_collisions(self):
        """Three boxes all overlapping each other."""
        boxes = [
            ComponentBox("a", [0.1, 0.1, 0.1], position=[0.0, 0, 0]),
            ComponentBox("b", [0.1, 0.1, 0.1], position=[0.02, 0, 0]),
            ComponentBox("c", [0.1, 0.1, 0.1], position=[0.04, 0, 0]),
        ]
        collisions = check_intersections(boxes)
        assert len(collisions) == 3  # a-b, a-c, b-c

    def test_empty_list(self):
        assert check_intersections([]) == []


# ===================================================================
# Fuselage containment
# ===================================================================

class TestFuselageContainment:
    @pytest.fixture
    def cylinder_fuselage(self):
        """Simple cylindrical fuselage: radius 0.05m, length 1.0m."""
        return ap.Fuselage(
            name="cyl",
            xsecs=[
                ap.FuselageXSec(x=0.0, radius=0.05),
                ap.FuselageXSec(x=1.0, radius=0.05),
            ],
            x_le=0.0, y_le=0.0, z_le=0.0,
        )

    def test_point_inside(self, cylinder_fuselage):
        assert fuselage_contains_point(cylinder_fuselage, [0.5, 0.0, 0.0]) is True

    def test_point_on_centerline(self, cylinder_fuselage):
        assert fuselage_contains_point(cylinder_fuselage, [0.5, 0.0, 0.0]) is True

    def test_point_near_wall(self, cylinder_fuselage):
        assert fuselage_contains_point(cylinder_fuselage, [0.5, 0.04, 0.0]) is True

    def test_point_outside_radially(self, cylinder_fuselage):
        assert fuselage_contains_point(cylinder_fuselage, [0.5, 0.06, 0.0]) is False

    def test_point_outside_axially(self, cylinder_fuselage):
        assert fuselage_contains_point(cylinder_fuselage, [1.5, 0.0, 0.0]) is False

    def test_point_before_nose(self, cylinder_fuselage):
        assert fuselage_contains_point(cylinder_fuselage, [-0.1, 0.0, 0.0]) is False

    def test_small_box_inside(self, cylinder_fuselage):
        box = ComponentBox("bat", [0.1, 0.04, 0.03], position=[0.5, 0.0, 0.0])
        assert fuselage_contains_box(cylinder_fuselage, box) is True

    def test_large_box_outside(self, cylinder_fuselage):
        box = ComponentBox("big", [0.5, 0.12, 0.12], position=[0.5, 0.0, 0.0])
        assert fuselage_contains_box(cylinder_fuselage, box) is False

    def test_box_protruding_aft(self, cylinder_fuselage):
        """Box center inside but extends past the tail."""
        box = ComponentBox("aft", [0.3, 0.02, 0.02], position=[0.9, 0.0, 0.0])
        assert fuselage_contains_box(cylinder_fuselage, box) is False

    def test_tapered_fuselage(self):
        """Box that fits in the wide section but not in the tapered tail."""
        fus = ap.Fuselage(
            name="taper",
            xsecs=[
                ap.FuselageXSec(x=0.0, radius=0.01),
                ap.FuselageXSec(x=0.2, radius=0.06),
                ap.FuselageXSec(x=0.8, radius=0.06),
                ap.FuselageXSec(x=1.0, radius=0.01),
            ],
        )
        # Box in wide section: fits
        box_mid = ComponentBox("mid", [0.1, 0.08, 0.08], position=[0.5, 0, 0])
        assert fuselage_contains_box(fus, box_mid) is True

        # Same box in tapered tail: doesn't fit
        box_tail = ComponentBox("tail", [0.1, 0.08, 0.08], position=[0.9, 0, 0])
        assert fuselage_contains_box(fus, box_tail) is False


# ===================================================================
# validate_placement (full pipeline)
# ===================================================================

class TestValidatePlacement:
    def test_valid_placement(self):
        fus = ap.Fuselage(
            name="fus",
            xsecs=[
                ap.FuselageXSec(x=0.0, radius=0.05),
                ap.FuselageXSec(x=1.0, radius=0.05),
            ],
        )
        boxes = [
            ComponentBox("battery", [0.08, 0.04, 0.03], position=[0.3, 0, 0]),
            ComponentBox("fc", [0.04, 0.04, 0.02], position=[0.5, 0, 0]),
        ]
        result = validate_placement(boxes, fuselage=fus)
        assert result.is_valid
        assert len(result.collisions) == 0
        assert all(result.containment.values())

    def test_collision_detected(self):
        boxes = [
            ComponentBox("battery", [0.1, 0.04, 0.03], position=[0.3, 0, 0]),
            ComponentBox("fc", [0.1, 0.04, 0.02], position=[0.35, 0, 0]),
        ]
        result = validate_placement(boxes)
        assert not result.is_valid
        assert len(result.collisions) == 1

    def test_outside_fuselage_detected(self):
        fus = ap.Fuselage(
            name="fus",
            xsecs=[
                ap.FuselageXSec(x=0.0, radius=0.03),
                ap.FuselageXSec(x=0.5, radius=0.03),
            ],
        )
        boxes = [
            ComponentBox("huge", [0.1, 0.1, 0.1], position=[0.25, 0, 0]),
        ]
        result = validate_placement(boxes, fuselage=fus)
        assert not result.is_valid
        assert result.containment["huge"] is False

    def test_report_pass(self):
        result = validate_placement([])
        report = result.report()
        assert "PASS" in report

    def test_report_fail_collision(self):
        boxes = [
            ComponentBox("a", [0.1, 0.1, 0.1], position=[0.0, 0, 0]),
            ComponentBox("b", [0.1, 0.1, 0.1], position=[0.05, 0, 0]),
        ]
        result = validate_placement(boxes)
        report = result.report()
        assert "FAIL" in report
        assert "collision" in report.lower()
