"""Common controller interface: every method (PID, RL, MPC, ...) implements ``act``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from dronebench.backends.base import DroneState


@dataclass(frozen=True)
class Reference:
    pos: np.ndarray  # (3,) desired position [m]
    vel: np.ndarray  # (3,) desired velocity (feed-forward) [m/s]
    yaw: float = 0.0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))  # feed-forward [m/s^2]


class Controller(ABC):
    name: str = "controller"

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Clear internal state (integrators, hidden states, warm starts)."""

    @abstractmethod
    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        """Return motor commands in RPM, shape (4,)."""
