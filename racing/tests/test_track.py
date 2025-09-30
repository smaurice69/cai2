import math

from racing.track import INDY_OVAL_TRACK


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def test_goal_line_perpendicular_to_track():
    tangent = INDY_OVAL_TRACK.tangent(INDY_OVAL_TRACK.start_finish_index)
    line_dir = INDY_OVAL_TRACK.goal_line.direction()
    assert math.isclose(_dot(tangent, line_dir), 0.0, abs_tol=0.12)


def test_goal_line_spans_full_width():
    assert INDY_OVAL_TRACK.goal_line.length() >= INDY_OVAL_TRACK.width


def test_checkpoints_are_perpendicular_and_full_width():
    for checkpoint in INDY_OVAL_TRACK.checkpoints:
        tangent = INDY_OVAL_TRACK.tangent(checkpoint.centerline_index)
        direction = checkpoint.segment.direction()
        assert math.isclose(_dot(tangent, direction), 0.0, abs_tol=0.12)
        assert checkpoint.segment.length() >= INDY_OVAL_TRACK.width
