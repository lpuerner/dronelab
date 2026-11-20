"""Fly one course with one controller on one backend and record everything."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dronebench.backends.base import Backend
from dronebench.config import ArenaConfig
from dronebench.controllers.base import Controller
from dronebench.tasks.courses import Course


@dataclass
class Trajectory:
    t: np.ndarray  # (T,)
    pos: np.ndarray  # (T, 3) true states
    rpy: np.ndarray
    vel: np.ndarray
    ref_pos: np.ndarray  # (T, 3)
    ref_vel: np.ndarray
    rpm: np.ndarray  # (T, 4) commanded
    crashed: bool
    crash_reason: str | None
    max_rpm: float
    hover_rpm: float
    course_duration: float


def check_crash(t: float, pos: np.ndarray, rpy: np.ndarray, arena: ArenaConfig) -> str | None:
    if np.any(np.abs(rpy[:2]) > np.deg2rad(arena.max_tilt_deg)):
        return "tilt"
    if np.any(np.abs(pos[:2]) > arena.half_size) or pos[2] > arena.max_height:
        return "out_of_bounds"
    if t > arena.ground_grace_time and pos[2] < 0.03:
        return "ground"
    return None


def run_episode(
    backend: Backend,
    controller: Controller,
    course: Course,
    arena: ArenaConfig | None = None,
    seed: int = 0,
    stop_on_crash: bool = True,
) -> Trajectory:
    arena = arena or ArenaConfig()
    state = backend.reset(course.start, seed=seed)
    controller.reset()
    n_steps = int(np.ceil(course.duration * backend.ctrl_freq))

    log: dict[str, list] = {k: [] for k in ("t", "pos", "rpy", "vel", "rp", "rv", "rpm")}
    crash = None
    for k in range(n_steps):
        t = k * backend.dt
        ref = course.reference(t)
        rpm = controller.act(backend.observe(), ref)
        for key, val in zip(
            log, (t, state.pos, state.rpy, state.vel, ref.pos, ref.vel, rpm), strict=True
        ):
            log[key].append(val)
        state = backend.step(rpm)
        crash = check_crash(t + backend.dt, state.pos, state.rpy, arena)
        if crash and stop_on_crash:
            break

    a = {k: np.asarray(v) for k, v in log.items()}
    return Trajectory(
        t=a["t"],
        pos=a["pos"],
        rpy=a["rpy"],
        vel=a["vel"],
        ref_pos=a["rp"],
        ref_vel=a["rv"],
        rpm=a["rpm"],
        crashed=crash is not None,
        crash_reason=crash,
        max_rpm=backend.params.max_rpm,
        hover_rpm=backend.params.hover_rpm,
        course_duration=course.duration,
    )
