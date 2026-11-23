"""Nominal quadrotor rigid-body model, consistent with gym-pybullet-drones ``Physics.PYB``.

Frames: world ENU (z up), body FLU. Euler angles are roll-pitch-yaw with
``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (PyBullet convention).
State ``x = [p(3), v(3), rpy(3), omega_body(3)]``, wrench ``u = [T, tau_x, tau_y, tau_z]``.

Note: the PINN papers in the literature (Gu 2024, Bianchi 2024) use NED/FRD; signs of
z and pitch differ from here.
"""

from __future__ import annotations

import numpy as np

from dronebench.backends.base import DroneParams


def rot_from_rpy(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> rotation matrix body -> world."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mixer_matrix(params: DroneParams) -> np.ndarray:
    """Maps squared motor speeds (rpm^2, shape 4) to [T, tau_x, tau_y, tau_z]."""
    kf, km = params.kf, params.km
    x, y = params.rotor_xy[:, 0], params.rotor_xy[:, 1]
    return np.vstack(
        [
            kf * np.ones(4),
            kf * y,  # tau_x = sum(y_i * F_i)
            -kf * x,  # tau_y = sum(-x_i * F_i)
            km * params.yaw_sign,
        ]
    )


def euler_rates(rpy: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    r, p, _ = rpy
    wx, wy, wz = omega_body
    return np.array(
        [
            wx + np.sin(r) * np.tan(p) * wy + np.cos(r) * np.tan(p) * wz,
            np.cos(r) * wy - np.sin(r) * wz,
            (np.sin(r) * wy + np.cos(r) * wz) / np.cos(p),
        ]
    )


def state_derivative(x: np.ndarray, u: np.ndarray, params: DroneParams) -> np.ndarray:
    v, rpy, w = x[3:6], x[6:9], x[9:12]
    J = params.inertia
    R = rot_from_rpy(rpy)
    acc = R[:, 2] * u[0] / params.mass - np.array([0.0, 0.0, params.gravity])
    w_dot = (u[1:] - np.cross(w, J * w)) / J
    return np.concatenate([v, acc, euler_rates(rpy, w), w_dot])


def rk4_step(x: np.ndarray, u: np.ndarray, params: DroneParams, dt: float) -> np.ndarray:
    k1 = state_derivative(x, u, params)
    k2 = state_derivative(x + 0.5 * dt * k1, u, params)
    k3 = state_derivative(x + 0.5 * dt * k2, u, params)
    k4 = state_derivative(x + dt * k3, u, params)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def linearize_hover(params: DroneParams) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time (A, B) around hover with yaw 0; input u = [dT, tau_x, tau_y, tau_z]."""
    g, m, J = params.gravity, params.mass, params.inertia
    A = np.zeros((12, 12))
    A[0:3, 3:6] = np.eye(3)
    A[3, 7] = g  # v_x' =  g * pitch
    A[4, 6] = -g  # v_y' = -g * roll
    A[6:9, 9:12] = np.eye(3)  # euler rates = body rates at hover
    B = np.zeros((12, 4))
    B[5, 0] = 1.0 / m
    B[9:12, 1:4] = np.diag(1.0 / J)
    return A, B
