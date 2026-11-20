"""Flight courses as time-parametrised reference trajectories.

A course is a polyline that is traversed at constant speed, followed by a hold
phase at the final point. Hover is the special case of a single point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dronebench.controllers.base import Reference


@dataclass(frozen=True)
class Course:
    name: str
    start: np.ndarray  # (3,) initial drone position
    waypoints: np.ndarray  # (N, 3) path to follow
    speed: float = 0.5  # [m/s] along the path
    hold_time: float = 3.0  # [s] at the last waypoint

    def __post_init__(self):
        object.__setattr__(self, "start", np.asarray(self.start, dtype=float))
        wp = np.atleast_2d(np.asarray(self.waypoints, dtype=float))
        object.__setattr__(self, "waypoints", wp)
        seg = np.linalg.norm(np.diff(wp, axis=0), axis=1) if len(wp) > 1 else np.zeros(0)
        object.__setattr__(self, "_cum", np.concatenate([[0.0], np.cumsum(seg)]))

    @property
    def length(self) -> float:
        return float(self._cum[-1])

    @property
    def duration(self) -> float:
        return self.length / self.speed + self.hold_time

    def reference(self, t: float) -> Reference:
        wp, cum = self.waypoints, self._cum
        if len(wp) == 1:
            return Reference(pos=wp[0].copy(), vel=np.zeros(3))
        s = self.speed * t
        if s >= cum[-1]:
            return Reference(pos=wp[-1].copy(), vel=np.zeros(3))
        i = int(np.searchsorted(cum, s, side="right") - 1)
        d = wp[i + 1] - wp[i]
        seg_len = cum[i + 1] - cum[i]
        u = d / seg_len
        return Reference(pos=wp[i] + u * (s - cum[i]), vel=u * self.speed)


def hover(height: float = 1.0, hold_time: float = 8.0) -> Course:
    return Course("hover", start=[0, 0, 0.1], waypoints=[[0, 0, height]], hold_time=hold_time)


def square(size: float = 2.0, height: float = 1.0, speed: float = 0.5) -> Course:
    h = size / 2
    wp = [[-h, -h, height], [h, -h, height], [h, h, height], [-h, h, height], [-h, -h, height]]
    return Course("square", start=wp[0], waypoints=wp, speed=speed)


def figure8(
    size: float = 1.5, height: float = 1.0, speed: float = 0.5, n: int = 200
) -> Course:
    th = np.linspace(0, 2 * np.pi, n)
    # Lemniscate of Gerono, starts and ends at the origin
    wp = np.stack([size * np.sin(th), size * np.sin(th) * np.cos(th), np.full(n, height)], 1)
    return Course("figure8", start=wp[0], waypoints=wp, speed=speed)


def gates(speed: float = 0.6) -> Course:
    """Slalom through gates at different heights (gates themselves come with obstacles)."""
    wp = [
        [-3.0, 0.0, 1.0],
        [-1.5, 0.8, 1.2],
        [0.0, -0.8, 0.7],
        [1.5, 0.8, 1.5],
        [3.0, 0.0, 1.0],
    ]
    return Course("gates", start=wp[0], waypoints=wp, speed=speed)


COURSES: dict[str, Callable[..., Course]] = {
    "hover": hover,
    "square": square,
    "figure8": figure8,
    "gates": gates,
}


def make_course(name: str, **kwargs) -> Course:
    try:
        return COURSES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown course {name!r}, choose from {sorted(COURSES)}") from None
