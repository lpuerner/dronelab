"""Simulator-independent disturbance models."""

from __future__ import annotations

import numpy as np

from dronelab.core.backends.base import DroneState
from dronelab.core.config import DisturbanceConfig


class GustProcess:
    """Ornstein-Uhlenbeck process for a 3D gust force (stationary std = ``std``)."""

    def __init__(self, std: float, tau: float, rng: np.random.Generator):
        self.std, self.tau, self.rng = std, tau, rng
        self.value = np.zeros(3)

    def reset(self) -> None:
        self.value = np.zeros(3)

    def step(self, dt: float) -> np.ndarray:
        if self.std > 0:
            a = np.exp(-dt / self.tau)
            self.value = a * self.value + self.std * np.sqrt(1 - a**2) * self.rng.standard_normal(3)
        return self.value


class MotorLag:
    """First-order lag on commanded RPM (exact discretisation)."""

    def __init__(self, tau: float):
        self.tau = tau
        self.rpm: np.ndarray | None = None

    def reset(self, rpm0: np.ndarray) -> None:
        self.rpm = rpm0.astype(float).copy()

    def step(self, cmd: np.ndarray, dt: float) -> np.ndarray:
        if self.tau <= 0 or self.rpm is None:
            self.rpm = np.asarray(cmd, dtype=float).copy()
            return self.rpm
        a = np.exp(-dt / self.tau)
        self.rpm = a * self.rpm + (1 - a) * np.asarray(cmd, dtype=float)
        return self.rpm


def add_sensor_noise(
    state: DroneState, cfg: DisturbanceConfig, rng: np.random.Generator
) -> DroneState:
    if cfg.pos_noise_std == cfg.vel_noise_std == cfg.att_noise_std == 0:
        return state
    rpy = state.rpy + cfg.att_noise_std * rng.standard_normal(3)
    return DroneState(
        pos=state.pos + cfg.pos_noise_std * rng.standard_normal(3),
        quat=_rpy_to_quat(rpy),
        rpy=rpy,
        vel=state.vel + cfg.vel_noise_std * rng.standard_normal(3),
        ang_vel=state.ang_vel,
    )


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """ZYX Euler -> quaternion (x, y, z, w), same convention as PyBullet."""
    cr, cp, cy = np.cos(rpy / 2)
    sr, sp, sy = np.sin(rpy / 2)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )
