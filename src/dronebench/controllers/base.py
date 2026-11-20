"""Common controller interface: every method (PID, RL, MPC, ...) implements ``act``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from dronebench.backends.base import DroneState


@dataclass(frozen=True)
class Reference:
    pos: np.ndarray  # (3,) desired position [m]
    vel: np.ndarray  # (3,) desired velocity (feed-forward) [m/s]
    yaw: float = 0.0


class Controller(ABC):
    name: str = "controller"

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Clear internal state (integrators, hidden states, warm starts)."""

    @abstractmethod
    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        """Return motor commands in RPM, shape (4,)."""


class ConstantRPMController(Controller):
    """Open-loop baseline: all motors at a fixed RPM (useful for tests/speed checks)."""

    name = "constant_rpm"

    def __init__(self, rpm: float = 0.0):
        self.rpm = float(rpm)

    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        return np.full(4, self.rpm)
