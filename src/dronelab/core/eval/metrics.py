"""Scalar metrics used to compare controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dronelab.core.tasks.gates import evaluate_gates

if TYPE_CHECKING:
    from dronelab.core.eval.runner import Trajectory


def settling_time(t: np.ndarray, err: np.ndarray, tol: float = 0.05) -> float:
    """First time after which the error stays below ``tol`` (nan if never)."""
    outside = np.nonzero(err > tol)[0]
    if len(outside) == 0:
        return float(t[0]) if len(t) else float("nan")
    last = outside[-1]
    return float(t[last + 1]) if last + 1 < len(t) else float("nan")


def compute_metrics(traj: Trajectory, settle_tol: float = 0.05) -> dict[str, float]:
    err = np.linalg.norm(traj.pos - traj.ref_pos, axis=1)
    dt = traj.dt or (float(traj.t[1] - traj.t[0]) if len(traj.t) > 1 else 0.0)
    rpm_n = traj.rpm / traj.max_rpm
    d_rpm = np.diff(rpm_n, axis=0) if len(rpm_n) > 1 else np.zeros((1, 4))
    hold = traj.t >= traj.course_duration - 2.0  # last 2 s: steady-state window
    m = {
        "rmse_pos": float(np.sqrt(np.mean(err**2))),
        "max_err": float(err.max()),
        "final_err": float(err[hold].mean()) if hold.any() else float("nan"),
        "settling_time": settling_time(traj.t, err, settle_tol),
        # literature-style integral criteria (Bianchi 2024, Madebo 2026)
        "iae": float(np.sum(err) * dt),
        "ise": float(np.sum(err**2) * dt),
        "itae": float(np.sum(traj.t * err) * dt),
        "effort": float(np.mean(np.abs(traj.rpm - traj.hover_rpm)) / traj.max_rpm),
        "smoothness": float(np.mean(np.abs(d_rpm))),
        "crashed": float(traj.crashed),
        "flight_time": float(traj.t[-1]) if len(traj.t) else 0.0,
        "ctrl_ms_per_step": 1e3 * traj.wall_time_per_step,
    }
    m.update(evaluate_gates(traj.pos, traj.gates, traj.t))
    return m
