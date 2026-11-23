"""Save flights as training data for the learned dynamics models (Stage 3).

Each episode becomes one ``.npz`` with everything a forward model ``f(x, u) -> x'`` or a
physics + residual model (NeuroBEM-style) needs: states, body-frame velocities, finite-
difference accelerations, commanded and applied motor speeds, and the force / torque the
nominal quadratic rotor model predicts for the applied motor speeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dronebench.backends.base import DroneParams
from dronebench.eval.runner import Trajectory
from dronebench.models.rigid_body import mixer_matrix, quat_to_rot


def flight_arrays(traj: Trajectory, params: DroneParams) -> dict[str, np.ndarray]:
    R = np.array([quat_to_rot(q) for q in traj.quat])  # (T, 3, 3) body -> world
    vel_body = np.einsum("tji,tj->ti", R, traj.vel)
    angvel_body = np.einsum("tji,tj->ti", R, traj.ang_vel)
    dt = traj.dt
    acc = np.gradient(traj.vel, dt, axis=0) if len(traj.t) > 2 else np.zeros_like(traj.vel)
    ang_acc_body = (
        np.gradient(angvel_body, dt, axis=0) if len(traj.t) > 2 else np.zeros_like(angvel_body)
    )
    wrench = (mixer_matrix(params) @ (traj.rpm_applied**2).T).T  # (T, 4): thrust, tau_xyz
    force_model_world = R[:, :, 2] * wrench[:, :1]  # thrust along body z
    return {
        "t": traj.t,
        "pos": traj.pos,
        "quat": traj.quat,
        "rpy": traj.rpy,
        "vel": traj.vel,
        "vel_body": vel_body,
        "ang_vel": traj.ang_vel,
        "ang_vel_body": angvel_body,
        "acc": acc,
        "ang_acc_body": ang_acc_body,
        "rpm_cmd": traj.rpm,
        "rpm_applied": traj.rpm_applied,
        "thrust_model": wrench[:, 0],
        "torque_model_body": wrench[:, 1:],
        "force_model_world": force_model_world,
        "ref_pos": traj.ref_pos,
        "ref_vel": traj.ref_vel,
    }


def save_flight(path: str | Path, traj: Trajectory, params: DroneParams, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = flight_arrays(traj, params)
    info = {**traj.meta, **meta, "dt": traj.dt, "crashed": traj.crashed}
    np.savez_compressed(path, meta=json.dumps(info, default=str), **arrays)
    return path
