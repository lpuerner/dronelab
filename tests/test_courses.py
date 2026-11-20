"""Tests for the reference trajectories (no simulator needed)."""

import numpy as np
import pytest

from dronebench.config import ArenaConfig
from dronebench.tasks import COURSES, make_course


# The reference starts at the first waypoint and ends at rest on the last one.
@pytest.mark.parametrize("name", sorted(COURSES))
def test_reference_starts_and_ends_on_path(name):
    c = make_course(name)
    first = c.reference(0.0)
    last = c.reference(c.duration + 1.0)
    np.testing.assert_allclose(first.pos, c.waypoints[0])
    np.testing.assert_allclose(last.pos, c.waypoints[-1])
    np.testing.assert_allclose(last.vel, 0.0)


# While moving, the reference velocity always has exactly the course speed.
@pytest.mark.parametrize("name", ["square", "figure8", "gates"])
def test_constant_speed_along_path(name):
    c = make_course(name)
    for t in np.linspace(0.1, c.length / c.speed - 0.1, 25):
        assert np.linalg.norm(c.reference(t).vel) == pytest.approx(c.speed)


# The reference position never jumps (max. step = speed * dt).
def test_reference_is_continuous():
    c = make_course("square")
    ts = np.arange(0, c.duration, 0.01)
    pos = np.array([c.reference(t).pos for t in ts])
    assert np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)) <= c.speed * 0.01 + 1e-9


# Every waypoint lies inside the default arena with a 0.5 m safety margin.
@pytest.mark.parametrize("name", sorted(COURSES))
def test_course_inside_default_arena(name):
    arena = ArenaConfig()
    wp = make_course(name).waypoints
    assert np.all(np.abs(wp[:, :2]) < arena.half_size - 0.5)
    assert np.all(wp[:, 2] < arena.max_height)


# An unknown course name raises a clear error instead of failing later.
def test_unknown_course():
    with pytest.raises(ValueError):
        make_course("nope")
