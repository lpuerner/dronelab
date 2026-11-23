"""Discrete-time LQR / LQI around hover (12-state rigid-body model).

* Linearisation: :func:`dronelab.core.models.rigid_body.linearize_hover`, discretised with a
  zero-order hold at the control rate.
* Tracking: position/velocity errors are expressed in the yaw frame; the reference
  acceleration is fed forward as attitude (pitch = a_x/g, roll = -a_y/g) and thrust.
* LQI adds the integrated position error as 3 extra states, which removes steady-state
  errors caused by model mismatch (e.g. a heavier drone).
* Weights are parametrised as multipliers on Bryson's rule (1 / max_value^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from dronelab.control.mixer import Mixer
from dronelab.core.backends.base import DroneParams, DroneState
from dronelab.core.controller import Controller, Reference
from dronelab.core.models.rigid_body import linearize_hover, quat_to_rot


@dataclass
class LQRWeights:
    # Bryson scales: the largest acceptable value of each state / input
    pos_xy: float = 0.10  # [m]
    pos_z: float = 0.10
    vel_xy: float = 0.50  # [m/s]
    vel_z: float = 0.50
    att_rp: float = 0.20  # [rad]
    att_yaw: float = 0.30
    rate_rp: float = 2.0  # [rad/s]
    rate_yaw: float = 2.0
    thrust: float = 0.08  # [N]  (~30 % of the weight)
    torque_rp: float = 1e-3  # [N m]
    torque_yaw: float = 3e-4
    int_xy: float = 0.20  # [m s]  (LQI only)
    int_z: float = 0.20

    @classmethod
    def from_dict(cls, d: dict | None) -> LQRWeights:
        return cls(**(d or {}))


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def lqr_gain(
    params: DroneParams, dt: float, w: LQRWeights, integral: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, closed-loop matrix) for the discrete system."""
    A, B = linearize_hover(params)
    Ad, Bd = discretize(A, B, dt)
    q = [w.pos_xy, w.pos_xy, w.pos_z, w.vel_xy, w.vel_xy, w.vel_z,
         w.att_rp, w.att_rp, w.att_yaw, w.rate_rp, w.rate_rp, w.rate_yaw]
    r = [w.thrust, w.torque_rp, w.torque_rp, w.torque_yaw]
    if integral:
        n = 12
        Ad = np.block([[Ad, np.zeros((n, 3))], [dt * np.eye(3, n), np.eye(3)]])
        Bd = np.vstack([Bd, np.zeros((3, 4))])
        q += [w.int_xy, w.int_xy, w.int_z]
    Q = np.diag(1.0 / np.square(q))
    R = np.diag(1.0 / np.square(r))
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, Ad - Bd @ K


class LQRController(Controller):
    def __init__(
        self,
        params: DroneParams,
        ctrl_dt: float,
        weights: LQRWeights | dict | None = None,
        integral: bool = False,
        max_tilt_deg: float = 45.0,
    ):
        self.p = params
        self.dt = ctrl_dt
        self.integral = integral
        self.name = "lqi" if integral else "lqr"
        self.w = weights if isinstance(weights, LQRWeights) else LQRWeights.from_dict(weights)
        self.K, self.A_cl = lqr_gain(params, ctrl_dt, self.w, integral)
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.mixer = Mixer(params)
        self.reset()

    def reset(self) -> None:
        self.z = np.zeros(3)

    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        g = self.p.gravity
        yaw = state.rpy[2]
        c, s = np.cos(yaw), np.sin(yaw)
        Rz_T = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # world -> yaw frame
        e_p = Rz_T @ (state.pos - ref.pos)
        e_v = Rz_T @ (state.vel - ref.vel)
        a_ref = Rz_T @ ref.acc
        lim = np.tan(self.max_tilt)
        pitch_ff = np.arctan(np.clip(a_ref[0] / g, -lim, lim))
        roll_ff = -np.arctan(np.clip(a_ref[1] / g, -lim, lim))
        e_yaw = np.arctan2(np.sin(yaw - ref.yaw), np.cos(yaw - ref.yaw))
        R = quat_to_rot(state.quat)
        w_body = R.T @ state.ang_vel
        x = np.concatenate(
            [e_p, e_v, [state.rpy[0] - roll_ff, state.rpy[1] - pitch_ff, e_yaw], w_body]
        )
        if self.integral:
            self.z = np.clip(self.z + e_p * self.dt, -1.0, 1.0)
            x = np.concatenate([x, self.z])
        u = -self.K @ x
        thrust = self.p.mass * (g + a_ref[2]) + u[0]
        # compensate the thrust loss when tilted
        thrust /= max(np.cos(state.rpy[0]) * np.cos(state.rpy[1]), 0.5)
        thrust = float(np.clip(thrust, 0.0, 0.95 * self.p.max_thrust))
        return self.mixer.wrench_to_rpm(thrust, u[1:])
