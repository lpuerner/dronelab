"""Own cascaded PID: position -> thrust vector -> attitude -> body rates -> torques -> mixer.

The innermost loop (:meth:`CascadedPID.rate_control`) is usable on its own, e.g. below an
RL policy that outputs collective thrust + body rates (Stage 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dronebench.backends.base import DroneParams, DroneState
from dronebench.controllers.base import Controller, Reference
from dronebench.controllers.mixer import Mixer
from dronebench.models.rigid_body import quat_to_rot


@dataclass
class PIDGains:
    # Position loop (acceleration command per m / (m/s) / (m s)); defaults from pole
    # placement on a double integrator: xy w_n=2 rad/s, z w_n=3 rad/s, zeta=0.9
    pos_kp_xy: float = 4.0
    pos_kd_xy: float = 3.6
    pos_ki_xy: float = 0.5
    pos_kp_z: float = 9.0
    pos_kd_z: float = 5.4
    pos_ki_z: float = 1.5
    # Attitude loop: desired body rate per rad of attitude error
    att_kp_rp: float = 8.0
    att_kp_yaw: float = 3.0
    # Rate loop: angular acceleration per rad/s of rate error (multiplied by inertia)
    rate_kp_rp: float = 25.0
    rate_ki_rp: float = 5.0
    rate_kp_yaw: float = 10.0

    @classmethod
    def from_dict(cls, d: dict | None) -> PIDGains:
        return cls(**(d or {}))


class CascadedPID(Controller):
    name = "pid"

    def __init__(
        self,
        params: DroneParams,
        ctrl_dt: float,
        gains: PIDGains | dict | None = None,
        max_tilt_deg: float = 45.0,
        max_rate: float = 10.0,
    ):
        self.p = params
        self.dt = ctrl_dt
        self.g = gains if isinstance(gains, PIDGains) else PIDGains.from_dict(gains)
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.max_rate = max_rate
        self.mixer = Mixer(params)
        self.reset()

    def reset(self) -> None:
        self.i_pos = np.zeros(3)
        self.i_rate = np.zeros(3)

    # ----------------------------------------------------------------- loops
    def position_control(self, s: DroneState, ref: Reference) -> np.ndarray:
        """Desired thrust vector in the world frame [N]."""
        g = self.g
        e_p, e_v = ref.pos - s.pos, ref.vel - s.vel
        self.i_pos = np.clip(self.i_pos + e_p * self.dt, -1.0, 1.0)
        kp = np.array([g.pos_kp_xy, g.pos_kp_xy, g.pos_kp_z])
        kd = np.array([g.pos_kd_xy, g.pos_kd_xy, g.pos_kd_z])
        ki = np.array([g.pos_ki_xy, g.pos_ki_xy, g.pos_ki_z])
        acc = ref.acc + kp * e_p + kd * e_v + ki * self.i_pos
        f = self.p.mass * (acc + np.array([0.0, 0.0, self.p.gravity]))
        f[2] = max(f[2], 0.1 * self.p.mass * self.p.gravity)
        # limit tilt: shrink the horizontal part
        horiz = np.linalg.norm(f[:2])
        max_h = f[2] * np.tan(self.max_tilt)
        if horiz > max_h:
            f[:2] *= max_h / horiz
        # limit total thrust
        fn = np.linalg.norm(f)
        f_max = 0.95 * self.p.max_thrust
        return f * (f_max / fn) if fn > f_max else f

    @staticmethod
    def desired_rotation(f: np.ndarray, yaw: float) -> np.ndarray:
        z_b = f / np.linalg.norm(f)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        return np.column_stack([np.cross(y_b, z_b), y_b, z_b])

    def attitude_control(self, R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        """Desired body rates from the geometric attitude error (Lee et al.)."""
        E = R_des.T @ R - R.T @ R_des
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        k = np.array([self.g.att_kp_rp, self.g.att_kp_rp, self.g.att_kp_yaw])
        return np.clip(-k * e_R, -self.max_rate, self.max_rate)

    def rate_control(self, thrust: float, w_des: np.ndarray, s: DroneState) -> np.ndarray:
        """Collective thrust [N] + desired body rates [rad/s] -> motor RPM."""
        R = quat_to_rot(s.quat)
        w = R.T @ s.ang_vel  # world -> body
        e = w_des - w
        self.i_rate[:2] = np.clip(self.i_rate[:2] + e[:2] * self.dt, -1.0, 1.0)
        J = self.p.inertia
        kp = np.array([self.g.rate_kp_rp, self.g.rate_kp_rp, self.g.rate_kp_yaw])
        ki = np.array([self.g.rate_ki_rp, self.g.rate_ki_rp, 0.0])
        tau = J * (kp * e + ki * self.i_rate) + np.cross(w, J * w)
        return self.mixer.wrench_to_rpm(thrust, tau)

    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        f = self.position_control(state, ref)
        R = quat_to_rot(state.quat)
        R_des = self.desired_rotation(f, ref.yaw)
        thrust = max(float(f @ R[:, 2]), 0.0)  # project onto the current body z axis
        w_des = self.attitude_control(R, R_des)
        return self.rate_control(thrust, w_des, state)
