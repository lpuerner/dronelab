"""Rigid-body model, linearisation and mixer: math checks plus agreement with PyBullet."""

import numpy as np
import pytest

from dronebench.backends.base import DroneParams
from dronebench.controllers.lqr import LQRWeights, lqr_gain
from dronebench.controllers.mixer import Mixer
from dronebench.models.rigid_body import (
    linearize_hover,
    mixer_matrix,
    quat_to_rot,
    rk4_step,
    rot_from_rpy,
    state_derivative,
)

# Crazyflie CF2X values from gym-pybullet-drones (no simulator needed for these tests)
P = DroneParams(
    mass=0.027, arm=0.0397, kf=3.16e-10, km=7.94e-12,
    inertia=np.array([1.4e-5, 1.4e-5, 2.17e-5]), max_rpm=21702.6, hover_rpm=14468.4,
)


# Euler -> rotation matrix agrees with the quaternion route (PyBullet convention).
def test_rotation_conventions():
    rpy = np.array([0.2, -0.3, 1.1])
    cr, cp, cy = np.cos(rpy / 2)
    sr, sp, sy = np.sin(rpy / 2)
    q = [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
    np.testing.assert_allclose(rot_from_rpy(rpy), quat_to_rot(np.array(q)), atol=1e-12)


# Hover thrust exactly cancels gravity: all derivatives are zero.
def test_hover_equilibrium():
    x = np.zeros(12)
    u = np.array([P.mass * P.gravity, 0, 0, 0])
    np.testing.assert_allclose(state_derivative(x, u, P), 0.0, atol=1e-12)


# The analytic (A, B) match finite differences of the nonlinear model at hover.
def test_linearization_matches_finite_differences():
    A, B = linearize_hover(P)
    x0, u0 = np.zeros(12), np.array([P.mass * P.gravity, 0, 0, 0])
    eps = 1e-6
    A_fd = np.column_stack([(state_derivative(x0 + eps * e, u0, P)
                             - state_derivative(x0 - eps * e, u0, P)) / (2 * eps)
                            for e in np.eye(12)])
    B_fd = np.column_stack([(state_derivative(x0, u0 + eps * e, P)
                             - state_derivative(x0, u0 - eps * e, P)) / (2 * eps)
                            for e in np.eye(4)])
    np.testing.assert_allclose(A, A_fd, atol=1e-5)
    np.testing.assert_allclose(B, B_fd, rtol=1e-5, atol=1e-5)


# ENU sign check: pitching nose-down (positive pitch) accelerates +x, positive roll -y.
def test_enu_tilt_signs():
    u = np.array([P.mass * P.gravity, 0, 0, 0])
    acc_pitch = state_derivative(np.r_[np.zeros(6), 0, 0.1, 0, np.zeros(3)], u, P)[3:6]
    acc_roll = state_derivative(np.r_[np.zeros(6), 0.1, 0, 0, np.zeros(3)], u, P)[3:6]
    assert acc_pitch[0] > 0 and acc_roll[1] < 0


# Both LQR and LQI closed loops are stable (all eigenvalues inside the unit circle).
@pytest.mark.parametrize("integral", [False, True])
def test_lqr_closed_loop_stable(integral):
    _, A_cl = lqr_gain(P, 1 / 48, LQRWeights(), integral)
    assert np.max(np.abs(np.linalg.eigvals(A_cl))) < 1.0


# Mixer: wrench -> rpm -> wrench round trip when nothing saturates.
def test_mixer_round_trip():
    mix = Mixer(P)
    T, tau = 0.3, np.array([2e-4, -1e-4, 5e-5])
    np.testing.assert_allclose(mix.rpm_to_wrench(mix.wrench_to_rpm(T, tau)), [T, *tau], rtol=1e-6)


# Mixer sign per motor: more speed on a +y rotor rolls positive, on a -x rotor pitches positive.
def test_mixer_motor_signs():
    M = mixer_matrix(P)
    for i, (x, y) in enumerate(P.rotor_xy):
        assert np.sign(M[1, i]) == np.sign(y)  # tau_x
        assert np.sign(M[2, i]) == -np.sign(x)  # tau_y
    assert np.allclose(M[3], P.km * P.yaw_sign)


# Under saturation roll/pitch torque is kept and yaw is sacrificed first.
def test_mixer_saturation_priority():
    mix = Mixer(P)
    tau = np.array([3e-3, 0.0, 3e-3])  # large roll + large yaw at high thrust
    rpm = mix.wrench_to_rpm(0.9 * P.max_thrust, tau)
    assert np.all(rpm <= P.max_rpm + 1e-6) and np.all(rpm >= 0)
    got = mix.rpm_to_wrench(rpm)
    assert got[1] == pytest.approx(tau[0], rel=1e-6)  # roll fully delivered
    assert abs(got[3]) < tau[2]  # yaw reduced


@pytest.mark.sim
# One control step in PyBullet matches the nominal model (checks frames, signs, mixer).
def test_model_matches_pybullet_one_step():
    pytest.importorskip("gym_pybullet_drones")
    from dronebench.backends.pybullet_backend import PyBulletBackend

    rng = np.random.default_rng(0)
    with PyBulletBackend() as be:
        p = be.params
        for _ in range(5):
            rpy = rng.uniform(-0.3, 0.3, 3)
            vel = rng.uniform(-0.5, 0.5, 3)
            w_world = rng.uniform(-1, 1, 3)
            s0 = be.reset(np.array([0.0, 0.0, 2.0]), init_rpy=rpy, init_vel=vel,
                          init_ang_vel=w_world)
            rpm = p.hover_rpm * (1 + rng.uniform(-0.05, 0.05, 4))
            s1 = be.step(rpm)
            R0 = quat_to_rot(s0.quat)
            x = np.concatenate([s0.pos, s0.vel, s0.rpy, R0.T @ s0.ang_vel])
            u = mixer_matrix(p) @ rpm**2
            for _ in range(5):  # same sub-steps as PyBullet (240 Hz)
                x = rk4_step(x, u, p, 1 / 240)
            R1 = quat_to_rot(s1.quat)
            np.testing.assert_allclose(x[0:3], s1.pos, atol=2e-3)
            np.testing.assert_allclose(x[3:6], s1.vel, atol=2e-2)
            np.testing.assert_allclose(x[6:9], s1.rpy, atol=1e-2)
            np.testing.assert_allclose(x[9:12], R1.T @ s1.ang_vel, atol=0.15)
