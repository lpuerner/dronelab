"""Simulator-agnostic backend interface.

Every simulator (PyBullet now, maybe a vectorised NumPy sim later) implements
:class:`Backend`. Controllers, environments and benchmarks only talk to this API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DroneState:
    pos: np.ndarray  # (3,) world frame [m]
    quat: np.ndarray  # (4,) x, y, z, w
    rpy: np.ndarray  # (3,) roll, pitch, yaw [rad]
    vel: np.ndarray  # (3,) world frame [m/s]
    ang_vel: np.ndarray  # (3,) world frame [rad/s]


def _cf2x_rotors() -> np.ndarray:
    # Propeller positions (x, y) in the body frame, order 0..3 as in gym-pybullet-drones cf2x.urdf
    return np.array([[0.028, -0.028], [-0.028, -0.028], [-0.028, 0.028], [0.028, 0.028]])


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
    rotor_xy: np.ndarray = field(default_factory=_cf2x_rotors)  # (4, 2) [m]
    # sign of each rotor's reaction torque about body z (PyBullet: -T0 + T1 - T2 + T3)
    yaw_sign: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0, -1.0, 1.0]))

    @property
    def max_thrust(self) -> float:
        return 4 * self.kf * self.max_rpm**2


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
    def reset(
        self,
        init_pos: np.ndarray,
        seed: int | None = None,
        init_rpy: np.ndarray | None = None,
        init_vel: np.ndarray | None = None,
        init_ang_vel: np.ndarray | None = None,
    ) -> DroneState: ...

    @property
    @abstractmethod
    def last_applied_rpm(self) -> np.ndarray:
        """Motor speeds actually applied in the last step (after motor lag)."""

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
