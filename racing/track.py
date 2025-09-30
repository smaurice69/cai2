"""Track layout primitives and a pre-built Indianapolis-inspired oval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Point:
    """2D coordinate."""

    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class LineSegment:
    """Straight line segment between two points."""

    start: Point
    end: Point

    def direction(self) -> Tuple[float, float]:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = math.hypot(dx, dy)
        if length == 0:
            raise ValueError("Line segment has zero length")
        return (dx / length, dy / length)

    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)


@dataclass(frozen=True)
class Checkpoint:
    """Checkpoint that spans the track width at a specific centreline index."""

    label: str
    segment: LineSegment
    centerline_index: int


@dataclass(frozen=True)
class Track:
    """Representation of a closed race track."""

    name: str
    centerline: Sequence[Point]
    width: float
    goal_line: LineSegment
    checkpoints: Sequence[Checkpoint]
    start_finish_index: int

    def _wrapped_index(self, index: int) -> int:
        return index % len(self.centerline)

    def _centerline_vector(self, index: int) -> Tuple[float, float]:
        prev_point = self.centerline[self._wrapped_index(index - 1)]
        next_point = self.centerline[self._wrapped_index(index + 1)]
        dx = next_point.x - prev_point.x
        dy = next_point.y - prev_point.y
        length = math.hypot(dx, dy)
        if length == 0:
            raise ValueError("Degenerate centreline segment around index {index}")
        return (dx / length, dy / length)

    def tangent(self, index: int) -> Tuple[float, float]:
        """Return the unit tangent vector along the centreline."""

        return self._centerline_vector(index)

    def _segment_normal(self, index: int) -> Tuple[float, float]:
        tangent = self._centerline_vector(index)
        # Rotate tangent by 90 degrees clockwise to obtain an outward normal.
        return (tangent[1], -tangent[0])

    def validate_geometry(
        self,
        *,
        perpendicular_tolerance: float = 0.12,
        length_tolerance: float = 1e-3,
    ) -> None:
        """Validate perpendicular goal line/checkpoints and spanning width."""

        def _assert_perpendicular(tangent: Tuple[float, float], segment: LineSegment, label: str) -> None:
            seg_dir = segment.direction()
            dot = tangent[0] * seg_dir[0] + tangent[1] * seg_dir[1]
            if abs(dot) > perpendicular_tolerance:
                raise ValueError(f"{label} is not perpendicular to track tangent (dot={dot})")

        # Goal line perpendicularity
        start_tangent = self._centerline_vector(self.start_finish_index)
        _assert_perpendicular(start_tangent, self.goal_line, "Goal line")

        if self.goal_line.length() + length_tolerance < self.width:
            raise ValueError("Goal line does not span the full track width")

        for checkpoint in self.checkpoints:
            tangent = self._centerline_vector(checkpoint.centerline_index)
            _assert_perpendicular(tangent, checkpoint.segment, f"Checkpoint '{checkpoint.label}'")
            if checkpoint.segment.length() + length_tolerance < self.width:
                raise ValueError(f"Checkpoint '{checkpoint.label}' does not span the track width")

    def normals(self) -> List[Tuple[float, float]]:
        """Return outward normals along the centreline for visualisation."""

        return [self._segment_normal(i) for i in range(len(self.centerline))]


# Indianapolis-inspired oval track definition.
_CENTERLINE: Tuple[Point, ...] = (
    Point(70.0, -45.0),
    Point(35.0, -45.0),
    Point(0.0, -45.0),
    Point(-35.0, -45.0),
    Point(-70.0, -45.0),
    Point(-75.0, -25.0),
    Point(-75.0, 0.0),
    Point(-75.0, 25.0),
    Point(-70.0, 45.0),
    Point(-35.0, 45.0),
    Point(0.0, 45.0),
    Point(35.0, 45.0),
    Point(70.0, 45.0),
    Point(75.0, 25.0),
    Point(75.0, 0.0),
    Point(75.0, -25.0),
)

_TRACK_WIDTH = 18.0

_INDYCAR_GOAL_LINE = LineSegment(
    Point(0.0, -45.0 - _TRACK_WIDTH / 2),
    Point(0.0, -45.0 + _TRACK_WIDTH / 2),
)

_CHECKPOINTS: Tuple[Checkpoint, ...] = (
    Checkpoint(
        "Turn 1",
        LineSegment(
            Point(-75.0 - _TRACK_WIDTH / 2, -25.0),
            Point(-75.0 + _TRACK_WIDTH / 2, -25.0),
        ),
        5,
    ),
    Checkpoint(
        "North Short Chute",
        LineSegment(
            Point(-75.0 - _TRACK_WIDTH / 2, 0.0),
            Point(-75.0 + _TRACK_WIDTH / 2, 0.0),
        ),
        6,
    ),
    Checkpoint(
        "Backstretch",
        LineSegment(
            Point(0.0, 45.0 - _TRACK_WIDTH / 2),
            Point(0.0, 45.0 + _TRACK_WIDTH / 2),
        ),
        10,
    ),
    Checkpoint(
        "Turn 3",
        LineSegment(
            Point(75.0 - _TRACK_WIDTH / 2, 25.0),
            Point(75.0 + _TRACK_WIDTH / 2, 25.0),
        ),
        13,
    ),
    Checkpoint(
        "South Short Chute",
        LineSegment(
            Point(75.0 - _TRACK_WIDTH / 2, 0.0),
            Point(75.0 + _TRACK_WIDTH / 2, 0.0),
        ),
        14,
    ),
)

INDY_OVAL_TRACK = Track(
    name="Indy Oval",
    centerline=_CENTERLINE,
    width=_TRACK_WIDTH,
    goal_line=_INDYCAR_GOAL_LINE,
    checkpoints=_CHECKPOINTS,
    start_finish_index=2,
)

# Validate geometry eagerly so incorrect edits are caught during import.
INDY_OVAL_TRACK.validate_geometry()

