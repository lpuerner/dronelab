"""Racing gates: a rectangular opening with a pass-through direction.

Optionally the opening has a frame of four bars (``bar`` thick, just outside the opening) that the
drone, a sphere of radius ``DRONE_RADIUS``, must not touch (autonomy subproject only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DRONE_RADIUS = 0.06  # [m] collision cylinder of the CF2X URDF (gym-pybullet-drones)


@dataclass(frozen=True)
class Gate:
    center: np.ndarray  # (3,) [m]
    normal: np.ndarray  # (3,) unit vector, horizontal; direction of a valid pass
    width: float = 0.5  # horizontal opening [m]
    height: float = 0.5  # vertical opening [m]
    bar: float = 0.03  # frame bar thickness [m]

    @property
    def lateral(self) -> np.ndarray:
        """Horizontal unit vector in the gate plane."""
        lat = np.cross([0.0, 0.0, 1.0], self.normal)
        return lat / np.linalg.norm(lat)

    def frame_hit(self, pos: np.ndarray, radius: float = DRONE_RADIUS) -> bool:
        """True if a sphere at ``pos`` touches one of the four frame bars."""
        lat = self.lateral
        up = np.array([0.0, 0.0, 1.0])
        hw, hh = (self.width + self.bar) / 2, (self.height + self.bar) / 2  # bar centre lines
        corners = [self.center + a * hw * lat + b * hh * up for a, b in
                   ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            ab = b - a
            q = a + np.clip(np.dot(pos - a, ab) / np.dot(ab, ab), 0.0, 1.0) * ab
            if np.linalg.norm(pos - q) < radius + self.bar / 2:
                return True
        return False

    def clearance(self, q: np.ndarray) -> float:
        """Distance of a point in the gate plane to the nearest opening edge (< 0: outside)."""
        rel = q - self.center
        return min(self.width / 2 - abs(float(np.dot(rel, self.lateral))),
                   self.height / 2 - abs(float(rel[2])))

    def crossing(self, p0: np.ndarray, p1: np.ndarray) -> tuple[bool, float] | None:
        """Check the segment p0 -> p1 against the gate plane.

        Returns ``None`` if the segment does not cross the plane in pass direction,
        otherwise ``(inside_opening, miss_distance)`` where the miss distance is the
        distance of the crossing point from the gate centre.
        """
        d0 = float(np.dot(p0 - self.center, self.normal))
        d1 = float(np.dot(p1 - self.center, self.normal))
        if not (d0 < 0.0 <= d1):
            return None
        q = p0 + (d0 / (d0 - d1)) * (p1 - p0)
        rel = q - self.center
        lat = abs(float(np.dot(rel, self.lateral)))
        vert = abs(float(rel[2]))
        inside = lat <= self.width / 2 and vert <= self.height / 2
        return inside, float(np.linalg.norm(rel))


class GateTracker:
    """Passes the gates in their order, step by step (online: runner, RL environment)."""

    def __init__(self, gates: list[Gate]):
        self.gates = gates
        self.next = 0  # index of the next gate to pass
        self.times: list[float] = []  # pass time per gate, in order
        self.clearances: list[float] = []

    @property
    def done(self) -> bool:
        return self.next >= len(self.gates)

    def update(self, p0: np.ndarray, p1: np.ndarray, t0: float, dt: float) -> bool:
        """Advance with the segment p0 -> p1 (time t0 .. t0 + dt); True if a gate was passed."""
        if self.done:
            return False
        g = self.gates[self.next]
        d0 = float(np.dot(p0 - g.center, g.normal))
        d1 = float(np.dot(p1 - g.center, g.normal))
        if not (d0 < 0.0 <= d1):
            return False
        f = d0 / (d0 - d1)
        c = g.clearance(p0 + f * (p1 - p0))
        if c < 0:
            return False  # through the plane beside the opening: not passed
        self.times.append(t0 + f * dt)
        self.clearances.append(c)
        self.next += 1
        return True


def evaluate_gates(pos: np.ndarray, gates: list[Gate],
                   t: np.ndarray | None = None) -> dict[str, float]:
    """Count gates passed (each at most once) along a position trace of shape (T, 3).

    With the time stamps ``t`` also the gates passed in their order, the lap time (pass of the
    last gate, nan if not all were passed in order) and the smallest clearance to the opening.
    """
    if not gates:
        return {}
    passed = np.zeros(len(gates), dtype=bool)
    miss = np.full(len(gates), np.nan)
    for k in range(len(pos) - 1):
        for i, g in enumerate(gates):
            res = g.crossing(pos[k], pos[k + 1])
            if res is None:
                continue
            inside, dist = res
            passed[i] |= inside
            miss[i] = dist if np.isnan(miss[i]) else min(miss[i], dist)
    out = {
        "gates_passed": float(passed.sum()),
        "gates_total": float(len(gates)),
        "all_gates": float(passed.all()),
        "gate_miss_mean": float(np.nanmean(miss)) if np.any(~np.isnan(miss)) else float("nan"),
    }
    if t is not None and len(t) > 1:
        tr = GateTracker(gates)
        dt = float(t[1] - t[0])
        for k in range(len(pos) - 1):
            tr.update(pos[k], pos[k + 1], float(t[k]), dt)
        out["gates_in_order"] = float(tr.next)
        out["lap_time"] = tr.times[-1] if tr.done else float("nan")
        out["gate_clearance_min"] = min(tr.clearances) if tr.clearances else float("nan")
    return out
