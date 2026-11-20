"""Tests for metrics and disturbance models on synthetic data (no simulator needed)."""

import numpy as np
import pytest

from dronebench.backends.base import DroneState
from dronebench.backends.disturbances import GustProcess, MotorLag, _rpy_to_quat, add_sensor_noise
from dronebench.config import DisturbanceConfig
from dronebench.eval.metrics import compute_metrics, settling_time
from dronebench.eval.runner import Trajectory


# Helper: hover trajectory at z = 1 m with a given x error per time step.
def _traj(err_x: np.ndarray) -> Trajectory:
    n = len(err_x)
    t = np.arange(n) * 0.1
    ref = np.tile([0.0, 0.0, 1.0], (n, 1))
    pos = ref + np.stack([err_x, np.zeros(n), np.zeros(n)], 1)
    return Trajectory(
        t=t, pos=pos, rpy=np.zeros((n, 3)), vel=np.zeros((n, 3)), ref_pos=ref,
        ref_vel=np.zeros((n, 3)), rpm=np.full((n, 4), 1000.0), crashed=False,
        crash_reason=None, max_rpm=2000.0, hover_rpm=1000.0, course_duration=t[-1],
    )


# Zero tracking error at hover rpm gives zero RMSE, error and effort.
def test_perfect_tracking():
    m = compute_metrics(_traj(np.zeros(50)))
    assert m["rmse_pos"] == 0 and m["max_err"] == 0 and m["effort"] == 0
    assert m["settling_time"] == 0


# Settling time = from when on the error *stays* below tol; never settled -> NaN.
def test_settling_time():
    t = np.arange(10.0)
    err = np.array([1, 1, 0.5, 0.01, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01])
    assert settling_time(t, err, tol=0.05) == 5.0
    assert np.isnan(settling_time(t, np.ones(10), tol=0.05))


# Over many steps the gust process has the configured standard deviation.
def test_gust_stationary_std():
    g = GustProcess(std=0.1, tau=0.5, rng=np.random.default_rng(1))
    samples = np.array([g.step(0.02) for _ in range(40000)])
    assert samples[5000:].std() == pytest.approx(0.1, rel=0.1)


# Motor lag responds only partially at first, then reaches the commanded rpm.
def test_motor_lag_converges():
    m = MotorLag(tau=0.05)
    m.reset(np.zeros(4))
    first = m.step(np.full(4, 100.0), 0.02)[0]
    assert 0 < first < 100
    for _ in range(100):
        out = m.step(np.full(4, 100.0), 0.02)
    np.testing.assert_allclose(out, 100.0, atol=1e-3)


# Zero angles give the identity quaternion; without noise the state passes unchanged.
def test_quat_identity_and_noise_free_passthrough():
    np.testing.assert_allclose(_rpy_to_quat(np.zeros(3)), [0, 0, 0, 1])
    s = DroneState(np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3), np.zeros(3))
    assert add_sensor_noise(s, DisturbanceConfig(), np.random.default_rng()) is s
