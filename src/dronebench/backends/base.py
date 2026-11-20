"""Simulator-agnostic backend interface.

Every simulator (PyBullet now, maybe a vectorised NumPy sim later) implements
:class:`Backend`. Controllers, environments and benchmarks only talk to this API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DroneState:
    pos: np.ndarray  # (3,) world frame [m]
    quat: np.ndarray  # (4,) x, y, z, w
    rpy: np.ndarray  # (3,) roll, pitch, yaw [rad]
    vel: np.ndarray  # (3,) world frame [m/s]
    ang_vel: np.ndarray  # (3,) world frame [rad/s]

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.pos, self.quat, self.rpy, self.vel, self.ang_vel])


@dataclass(frozen=True)
class DroneParams:
    """Nominal parameters (what a model-based controller may 'know')."""

    mass: float  # [kg]
    arm: float  # [m]
    kf: float  # thrust coefficient, F = kf * rpm^2
    km: float  # torque coefficient, M = km * rpm^2
    inertia: np.ndarray  # (3,) diagonal [kg m^2]
    max_rpm: float
    hover_rpm: float
    gravity: float = 9.8


class Backend(ABC):
    """One simulated quadrotor. Actions are motor speeds in RPM, shape (4,)."""

    @property
    @abstractmethod
    def ctrl_freq(self) -> int: ...

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_freq

    @property
    @abstractmethod
    def params(self) -> DroneParams: ...

    @property
    @abstractmethod
    def time(self) -> float:
        """Simulated time since the last reset [s]."""

    @abstractmethod
    def reset(self, init_pos: np.ndarray, seed: int | None = None) -> DroneState: ...

    @abstractmethod
    def step(self, rpm: np.ndarray) -> DroneState:
        """Advance one control step and return the *true* state."""

    @abstractmethod
    def observe(self) -> DroneState:
        """Current state as seen by the controller (sensor noise applied)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
