"""Racing gates: a rectangular opening with a pass-through direction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gate:
    center: np.ndarray  # (3,) [m]
    normal: np.ndarray  # (3,) unit vector, horizontal; direction of a valid pass
    width: float = 0.5  # horizontal opening [m]
    height: float = 0.5  # vertical opening [m]

    @property
    def lateral(self) -> np.ndarray:
        """Horizontal unit vector in the gate plane."""
        lat = np.cross([0.0, 0.0, 1.0], self.normal)
        return lat / np.linalg.norm(lat)

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


def evaluate_gates(pos: np.ndarray, gates: list[Gate]) -> dict[str, float]:
    """Count gates passed (each at most once) along a position trace of shape (T, 3)."""
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
    return {
        "gates_passed": float(passed.sum()),
        "gates_total": float(len(gates)),
        "all_gates": float(passed.all()),
        "gate_miss_mean": float(np.nanmean(miss)) if np.any(~np.isnan(miss)) else float("nan"),
    }
