"""Flight logs as torch tensors, and training segments cut from them.

A flight log (see :mod:`dronelab.core.eval.flightlog`) holds the true state at the control
rate (48 Hz) and the motor speeds the controller *commanded* at each step. Step ``k`` maps
``(x_k, u_k) -> x_{k+1}``. The applied (lagged) motor speeds are kept for analysis only;
models never see them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dronelab.core.backends.base import DroneParams

DT = 1.0 / 48.0  # control / logging step [s]

# Nominal Crazyflie 2.x parameters from gym-pybullet-drones' cf2x.urdf ("datasheet" values a
# modeller would have; the effective thrust and drag in the real world differ).
CF2X = DroneParams(
    mass=0.027,
    arm=0.0397,
    kf=3.16e-10,
    km=7.94e-12,
    inertia=np.array([1.4e-5, 1.4e-5, 2.17e-5]),
    max_rpm=float(np.sqrt(2.25 * 9.8 * 0.027 / (4 * 3.16e-10))),
    hover_rpm=float(np.sqrt(9.8 * 0.027 / (4 * 3.16e-10))),
    gravity=9.8,
)


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (..., 4) as (x, y, z, w) -> rotation matrix body -> world (..., 3, 3)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


@dataclass
class Flight:
    name: str
    meta: dict
    pos: torch.Tensor  # (T, 3) world
    vel: torch.Tensor  # (T, 3) world
    rot: torch.Tensor  # (T, 3, 3) body -> world
    omega: torch.Tensor  # (T, 3) body rates
    u: torch.Tensor  # (T, 4) commanded rpm
    applied: torch.Tensor = field(repr=False)  # (T, 4) applied rpm (analysis only)

    def __len__(self) -> int:
        return len(self.pos)

    @property
    def max_speed(self) -> float:
        """Measured maximum speed of the whole flight (used for the T2 selection)."""
        return float(self.vel.norm(dim=1).max())


def load_flight(path: str | Path) -> Flight:
    path = Path(path)
    with np.load(path) as d:
        def t(k):
            return torch.as_tensor(np.asarray(d[k]), dtype=torch.float32)

        return Flight(
            name=path.stem,
            meta=json.loads(str(d["meta"])),
            pos=t("pos"),
            vel=t("vel"),
            rot=quat_to_rot(t("quat")),
            omega=t("ang_vel_body"),
            u=t("rpm_cmd"),
            applied=t("rpm_applied"),
        )


def load_split(folder: str | Path, pattern: str = "*.npz", min_len: int = 2,
               exclude: tuple[str, ...] = ()) -> list[Flight]:
    """All flights of a folder (sorted by name); ``exclude`` drops names containing a token."""
    files = sorted(Path(folder).glob(pattern))
    flights = [load_flight(f) for f in files if not any(e in f.stem for e in exclude)]
    return [f for f in flights if len(f) >= min_len]


@dataclass
class Segments:
    """Stacked windows of ``length`` consecutive steps, shapes (N, length, ...)."""

    pos: torch.Tensor
    vel: torch.Tensor
    rot: torch.Tensor
    omega: torch.Tensor
    u: torch.Tensor
    applied: torch.Tensor

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, idx) -> Segments:
        return Segments(*(getattr(self, k)[idx] for k in _SEG_KEYS))


_SEG_KEYS = ("pos", "vel", "rot", "omega", "u", "applied")


def make_segments(flights: list[Flight], length: int, stride: int) -> Segments:
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in _SEG_KEYS}
    for f in flights:
        n = len(f) - length + 1
        if n <= 0:
            continue
        starts = torch.arange(0, n, stride)
        idx = starts[:, None] + torch.arange(length)[None]
        for k in _SEG_KEYS:
            parts[k].append(getattr(f, k)[idx])
    return Segments(*(torch.cat(parts[k]) for k in _SEG_KEYS))
