"""Tests for gate frames, gate order / lap time and the A1 planner (no simulator needed)."""

import numpy as np

from dronelab.autonomy.planner import plan_course
from dronelab.core.tasks import DRONE_RADIUS, Gate, GateTracker, evaluate_gates, make_course
from dronelab.core.tasks.courses import _VariableSpeedProfile

GATE = Gate(center=np.zeros(3), normal=np.array([1.0, 0, 0]))


# Frame: centre and a safe offset are free, near the edge is a hit, far beside is no hit.
def test_frame_hit():
    assert not GATE.frame_hit(np.zeros(3))
    assert not GATE.frame_hit(np.array([0.0, 0.25 - DRONE_RADIUS - 0.01, 0.0]))
    assert GATE.frame_hit(np.array([0.0, 0.25 - DRONE_RADIUS + 0.01, 0.0]))
    assert GATE.frame_hit(np.array([0.0, 0.0, -0.26]))  # bottom bar
    assert not GATE.frame_hit(np.array([0.0, 1.0, 0.0]))
    assert not GATE.frame_hit(np.array([0.5, 0.0, 0.0]))  # in front of the gate


# Gates count only in their order; lap time = interpolated pass of the last gate.
def test_gate_order_and_lap_time():
    g2 = Gate(center=np.array([1.0, 0, 0]), normal=np.array([1.0, 0, 0]))
    t = np.arange(0.0, 2.01, 0.5)
    pos = np.stack([t - 0.25, np.zeros_like(t), np.zeros_like(t)], 1)  # x: -0.25 .. 1.75
    m = evaluate_gates(pos, [GATE, g2], t)
    assert m["gates_in_order"] == 2
    assert np.isclose(m["lap_time"], 1.25)
    assert np.isclose(m["gate_clearance_min"], 0.25)
    rev = evaluate_gates(pos, [g2, GATE], t)  # wrong order: the second gate never counts
    assert rev["gates_in_order"] == 1 and np.isnan(rev["lap_time"])
    beside = GateTracker([GATE])
    assert not beside.update(np.array([-0.1, 0.4, 0]), np.array([0.1, 0.4, 0]), 0.0, 0.1)


# Without curvature the profile is unchanged; with it the speed respects a = v^2 kappa.
def test_curvature_profile():
    base = _VariableSpeedProfile(5.0, np.array([5.0]), np.array([2.0]), 2.0)
    same = _VariableSpeedProfile(5.0, np.array([5.0]), np.array([2.0]), 2.0, curv=None)
    assert np.array_equal(base._v, same._v)
    k = 2.0  # circle of radius 0.5 m
    p = _VariableSpeedProfile(5.0, np.array([5.0]), np.array([9.0]), 4.0,
                              curv=lambda s: np.full_like(s, k))
    assert p._v.max() <= np.sqrt(4.0 / k) + 1e-9


# Planner: within the budget, faster with more budget, A1+ not slower than A1.
def test_planner():
    c = make_course("gates", 1.0, 100)
    slow, fast = plan_course(c, 0.3), plan_course(c, 0.6)
    assert slow.difficulty["max_acc"] <= 0.3 * 9.8 * 1.02
    assert fast.profile.t_end < slow.profile.t_end < c.profile.t_end
    plus = plan_course(c, 0.3, optimize=True)
    assert plus.profile.t_end <= slow.profile.t_end
    assert plus.difficulty["max_offset"] <= 0.25 - DRONE_RADIUS - 0.05 + 1e-9
    assert [g.center.tolist() for g in plus.gates] == [g.center.tolist() for g in c.gates]
