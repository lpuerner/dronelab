"""INDI: incremental nonlinear dynamic inversion (Smeur et al.; Tal & Karaman).

Instead of inverting a model of the forces and torques, INDI changes the last applied wrench
by what is still missing for the desired acceleration, using the *measured* acceleration:

    tau_cmd = tau_f + J (w_dot_des - w_dot_f)            (rotation, body frame)
    f_cmd   = f_f + m (a_des - a_f)                       (translation, world frame)

Unmodelled torques and forces (drag, wind, payload, ground effect) are in the measurement and
cancel out; only the inertia, the mass and the motor time constant are used. ``w_dot`` and
``a`` are finite differences of the observed body rates and velocity (no accelerometer in
the state, the same observation every controller gets). The applied wrench ``tau_f, f_f`` comes
from the sent motor commands through a model of the motor lag and is filtered with the same
low-pass as the measurement, so that both are synchronous (the standard INDI requirement).

Outer loops as in :class:`~dronelab.control.pid.CascadedPID` (same tilt / rate limits): PD
position law -> desired acceleration -> thrust vector -> attitude (Lee et al.) -> body rates
-> desired angular acceleration. No integrators: steady disturbances are measured instead.

Variant ``indi_sg`` (``acc_window > 0``): the finite difference of the noisy velocity amplifies
its noise by sqrt(2)/dt. ``f_f - m a_f`` is the estimate of the external force (gravity, wind,
payload), so filtering it harder only slows the disturbance estimate, not the tracking
(``m a_des`` passes unfiltered). The translation then uses the slope of a straight line
through the last ``acc_window`` velocities (Savitzky-Golay, order 1) instead of the low-pass;
the same weights are applied to the thrust vector, so both stay synchronous. The rotation
keeps the low-pass (the gyro is not noisy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dronelab.control.mixer import Mixer
from dronelab.control.pid import CascadedPID
from dronelab.core.backends.base import DroneParams, DroneState
from dronelab.core.controller import Reference
from dronelab.core.models.rigid_body import quat_to_rot

MOTOR_TAU = 0.02  # [s] nominal motor time constant of the `real` world (as the MPPI baseline)
ACC_WIN = 9  # indi_sg: velocities in the Savitzky-Golay (straight-line fit) derivative


@dataclass
class INDIGains:
    # position loop: desired acceleration per m / (m/s); defaults = PID defaults
    pos_kp_xy: float = 4.0
    pos_kd_xy: float = 3.6
    pos_kp_z: float = 9.0
    pos_kd_z: float = 5.4
    # attitude loop: desired body rate per rad of attitude error
    att_kp_rp: float = 8.0
    att_kp_yaw: float = 3.0
    # rate loop: desired angular acceleration per rad/s of rate error
    rate_kp_rp: float = 25.0
    rate_kp_yaw: float = 10.0
    filt_hz: float = 10.0  # first-order low-pass on measurement and wrench (indi_sg: rotation)


class INDIController(CascadedPID):
    name = "indi"

    def __init__(self, params: DroneParams, ctrl_dt: float, gains: dict | None = None,
                 max_tilt_deg: float = 45.0, max_rate: float = 10.0,
                 motor_tau: float = MOTOR_TAU, acc_window: int = 0):
        self.p, self.dt = params, ctrl_dt
        self.w_sg = None
        if acc_window:
            self.name = "indi_sg"
            c = np.arange(acc_window) - (acc_window - 1) / 2
            c /= c @ c  # slope of the line fit: sum c_i v_i (per step)
            # = weighted mean of the acc_window - 1 differences, weights sum to 1
            self.w_sg = np.cumsum(c[::-1])[::-1][1:]
        self.g = INDIGains(**(gains or {}))
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.max_rate = max_rate
        self.mixer = Mixer(params)
        self.a_motor = math.exp(-ctrl_dt / motor_tau) if motor_tau > 0 else 0.0
        self.beta = 1.0 - math.exp(-2 * math.pi * self.g.filt_hz * ctrl_dt)
        self.reset()

    def reset(self) -> None:
        p = self.p
        self.prev = None  # (velocity, body rates) of the previous step
        self.R_prev = np.eye(3)
        self.rpm_est = np.full(4, p.hover_rpm)  # motor speeds acting since the last step
        self.acc_f = np.zeros(3)  # filtered measured acceleration (world)
        self.wdot_f = np.zeros(3)  # filtered measured angular acceleration (body)
        self.f_f = np.array([0.0, 0.0, p.mass * p.gravity])  # filtered thrust vector (world)
        self.tau_f = np.zeros(3)  # filtered torque (body)
        if self.w_sg is not None:  # external force f - m a per past interval (world)
            self.d_hist = np.tile(self.f_f, (len(self.w_sg), 1))

    def act(self, s: DroneState, ref: Reference) -> np.ndarray:
        g, p = self.g, self.p
        R = quat_to_rot(s.quat)
        w = R.T @ s.ang_vel
        wrench = self.mixer.rpm_to_wrench(self.rpm_est)  # acted during the last interval
        if self.prev is None:
            acc, wdot = np.zeros(3), np.zeros(3)
        else:
            acc, wdot = (s.vel - self.prev[0]) / self.dt, (w - self.prev[1]) / self.dt
        b = self.beta
        self.wdot_f += b * (wdot - self.wdot_f)
        self.tau_f += b * (wrench[1:] - self.tau_f)
        f = self.R_prev[:, 2] * wrench[0]
        if self.w_sg is None:
            self.acc_f += b * (acc - self.acc_f)
            self.f_f += b * (f - self.f_f)
            f_ext = self.f_f - p.mass * self.acc_f
        else:
            if self.prev is not None:
                self.d_hist = np.vstack([self.d_hist[1:], f - p.mass * acc])
            f_ext = self.w_sg @ self.d_hist

        kp = np.array([g.pos_kp_xy, g.pos_kp_xy, g.pos_kp_z])
        kd = np.array([g.pos_kd_xy, g.pos_kd_xy, g.pos_kd_z])
        a_des = ref.acc + kp * (ref.pos - s.pos) + kd * (ref.vel - s.vel)
        f_cmd = self.limit_thrust_vector(f_ext + p.mass * a_des)
        thrust = max(float(f_cmd @ R[:, 2]), 0.0)
        w_des = self.attitude_control(R, self.desired_rotation(f_cmd, ref.yaw))
        k_rate = np.array([g.rate_kp_rp, g.rate_kp_rp, g.rate_kp_yaw])
        tau = self.tau_f + p.inertia * (k_rate * (w_des - w) - self.wdot_f)
        rpm = self.mixer.wrench_to_rpm(thrust, tau)

        self.rpm_est = self.a_motor * self.rpm_est + (1 - self.a_motor) * rpm
        self.prev, self.R_prev = (np.array(s.vel, float), w), R
        return rpm
