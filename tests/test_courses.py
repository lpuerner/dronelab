"""Tests for the course generator and gates (no simulator needed)."""

import numpy as np
import pytest

from dronebench.tasks import FAMILIES, Gate, evaluate_gates, make_course

PATH_FAMILIES = [f for f in FAMILIES if f != "hover"]


# Same (family, speed, seed) always gives the same course; another seed a different one.
@pytest.mark.parametrize("family", FAMILIES)
def test_course_is_deterministic(family):
    a, b, c = make_course(family, 1.0, 3), make_course(family, 1.0, 3), make_course(family, 1.0, 4)
    np.testing.assert_allclose(a.reference(2.0).pos, b.reference(2.0).pos)
    assert a.params == b.params
    assert not np.allclose(a.reference(2.0).pos, c.reference(2.0).pos)


# The reference starts at the course start and ends at rest on the last point.
@pytest.mark.parametrize("family", PATH_FAMILIES)
def test_reference_starts_and_ends_at_rest(family):
    c = make_course(family, 1.0, 0)
    first, last = c.reference(0.0), c.reference(c.duration + 1.0)
    np.testing.assert_allclose(first.pos, c.start, atol=1e-9)
    np.testing.assert_allclose(first.vel, 0.0, atol=1e-9)
    np.testing.assert_allclose(last.vel, 0.0, atol=1e-9)


# During the cruise phase the reference moves at exactly the requested speed.
@pytest.mark.parametrize("family", PATH_FAMILIES)
def test_cruise_speed(family):
    c = make_course(family, 1.0, 0)
    t_mid = c.profile.t_acc + 0.5 * c.profile.t_cruise
    assert np.linalg.norm(c.reference(t_mid).vel) == pytest.approx(c.profile.peak, rel=1e-3)


# Position and velocity are continuous (smooth spline, no corners).
@pytest.mark.parametrize("family", PATH_FAMILIES)
def test_reference_is_smooth(family):
    c = make_course(family, 1.5, 0)
    ts = np.arange(0, c.duration, 0.01)
    refs = [c.reference(t) for t in ts]
    dp = np.linalg.norm(np.diff([r.pos for r in refs], axis=0), axis=1)
    dv = np.linalg.norm(np.diff([r.vel for r in refs], axis=0), axis=1)
    assert dp.max() <= 1.5 * 0.01 * 1.01
    assert dv.max() < 0.3  # bounded by acceleration * dt, no jumps


# Reference acceleration matches the derivative of the reference velocity.
def test_acceleration_is_consistent():
    c = make_course("figure8", 1.0, 2)
    t, h = 3.0, 1e-4
    num = (c.reference(t + h).vel - c.reference(t - h).vel) / (2 * h)
    np.testing.assert_allclose(c.reference(t).acc, num, atol=0.05)


# Feasible courses stay inside the arena and within thrust / tilt limits.
@pytest.mark.parametrize("family", PATH_FAMILIES)
def test_feasible_courses_respect_limits(family):
    for seed in range(3):
        c = make_course(family, 1.0, seed)
        assert c.feasible
        assert np.all(np.abs(c.path.pos[:, :2]) <= 4.5)
        assert c.difficulty["max_thrust_ratio"] <= 0.8
        assert c.difficulty["max_tilt_deg"] <= 45.0


# Faster flight on the same family means more required tilt.
def test_difficulty_grows_with_speed():
    slow = make_course("helix", 0.5, 0).difficulty["max_tilt_deg"]
    fast = make_course("helix", 2.0, 0).difficulty["max_tilt_deg"]
    assert fast > slow


# Hover is a set-point task.
def test_hover_is_setpoint():
    c = make_course("hover", seed=0)
    assert c.path is None and 0.5 <= c.setpoint[2] <= 1.5


# Gate pass: through the opening -> passed; beside it or backwards -> not.
def test_gate_crossing():
    g = Gate(center=np.zeros(3), normal=np.array([1.0, 0, 0]))
    through = np.array([[-0.5, 0.1, 0.0], [0.5, 0.1, 0.0]])
    beside = np.array([[-0.5, 0.6, 0.0], [0.5, 0.6, 0.0]])
    backwards = through[::-1]
    assert evaluate_gates(through, [g])["gates_passed"] == 1
    assert evaluate_gates(beside, [g])["gates_passed"] == 0
    assert evaluate_gates(backwards, [g])["gates_passed"] == 0


# Gate courses come with gate objects whose centres lie on the reference path.
def test_gate_course_has_gates():
    c = make_course("gates", 1.0, 0)
    assert 4 <= len(c.gates) <= 5
    for g in c.gates:
        assert np.min(np.linalg.norm(c.path.pos - g.center, axis=1)) < 1e-9
        assert abs(np.linalg.norm(g.normal) - 1) < 1e-9


# Unknown family names raise a clear error.
def test_unknown_course():
    with pytest.raises(ValueError):
        make_course("nope")


# The gate count of a seed does not depend on the speed (no selection bias by rejection).
def test_gate_count_independent_of_speed():
    for seed in range(5):
        counts = {len(make_course("gates", v, seed).gates) for v in (0.25, 1.0, 2.0)}
        assert len(counts) == 1
