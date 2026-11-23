"""Wrench <-> motor speed allocation with prioritised saturation.

Clipping each motor separately distorts the torques exactly when they matter most
(aggressive manoeuvres). Instead, as e.g. ``DynAviary`` in gym-pybullet-drones solves a
bounded allocation, we keep priorities: roll/pitch torque > collective thrust > yaw torque.
"""

from __future__ import annotations

import numpy as np

from dronelab.core.backends.base import DroneParams
from dronelab.core.models.rigid_body import mixer_matrix


class Mixer:
    def __init__(self, params: DroneParams):
        self.params = params
        self.M = mixer_matrix(params)
        self.M_inv = np.linalg.inv(self.M)
        self.w_max = params.max_rpm**2

    def rpm_to_wrench(self, rpm: np.ndarray) -> np.ndarray:
        return self.M @ (np.asarray(rpm, float) ** 2)

    def wrench_to_rpm(self, thrust: float, torque: np.ndarray) -> np.ndarray:
        Mi, w_max = self.M_inv, self.w_max
        w_rp = Mi @ np.array([0.0, torque[0], torque[1], 0.0])
        w_yaw = Mi @ np.array([0.0, 0.0, 0.0, torque[2]])

        # 1) roll/pitch alone must fit into [0, w_max]
        span = w_rp.max() - w_rp.min()
        if span > w_max:
            w_rp = w_rp * (w_max / span)

        # 2) collective offset as close to the requested thrust as the bounds allow
        c_des = max(thrust, 0.0) / (4 * self.params.kf)
        c = float(np.clip(c_des, -w_rp.min(), w_max - w_rp.max()))
        w = w_rp + c

        # 3) as much yaw torque as still fits
        alpha = 1.0
        for wi, yi in zip(w, w_yaw, strict=True):
            if yi > 0:
                alpha = min(alpha, (w_max - wi) / yi)
            elif yi < 0:
                alpha = min(alpha, wi / -yi)
        w = w + max(alpha, 0.0) * w_yaw
        return np.sqrt(np.clip(w, 0.0, w_max))
