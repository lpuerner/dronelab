"""Fly one course with one controller on one backend and record everything."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dronelab.core.backends.base import Backend
from dronelab.core.config import ArenaConfig, InitNoiseConfig
from dronelab.core.controller import Controller
from dronelab.core.tasks.courses import Course
from dronelab.core.tasks.gates import Gate, GateTracker


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
    dt: float = 0.0
    quat: np.ndarray | None = None  # (T, 4)
    ang_vel: np.ndarray | None = None  # (T, 3) world frame
    rpm_applied: np.ndarray | None = None  # (T, 4) after motor lag (applied during step k)
    ref_acc: np.ndarray | None = None
    gates: list[Gate] = field(default_factory=list)
    wall_time_per_step: float = 0.0  # controller compute time [s]
    meta: dict = field(default_factory=dict)


def check_crash(t: float, pos: np.ndarray, rpy: np.ndarray, arena: ArenaConfig) -> str | None:
    if np.any(np.abs(rpy[:2]) > np.deg2rad(arena.max_tilt_deg)):
        return "tilt"
    if np.any(np.abs(pos[:2]) > arena.half_size) or pos[2] > arena.max_height:
        return "out_of_bounds"
    if t > arena.ground_grace_time and pos[2] < 0.03:
        return "ground"
    return None


def sample_initial_state(
    start: np.ndarray, seed: int, cfg: InitNoiseConfig | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random start offset, attitude and velocity; depends only on the seed."""
    if cfg is None or not cfg.enabled:
        return np.asarray(start, float), np.zeros(3), np.zeros(3)
    rng = np.random.default_rng([seed, 7919])  # independent of controller and course
    pos = np.asarray(start, float) + rng.uniform(-cfg.pos, cfg.pos, 3)
    pos[2] = max(pos[2], 0.05)
    rp = np.deg2rad(cfg.rp_deg)
    rpy = np.array(
        [rng.uniform(-rp, rp), rng.uniform(-rp, rp), np.deg2rad(rng.uniform(-1, 1) * cfg.yaw_deg)]
    )
    vel = rng.uniform(-cfg.vel, cfg.vel, 3)
    return pos, rpy, vel


def run_episode(
    backend: Backend,
    controller: Controller,
    course: Course,
    arena: ArenaConfig | None = None,
    seed: int = 0,
    stop_on_crash: bool = True,
    init_noise: InitNoiseConfig | None = None,
    gate_frames: bool = False,
    finish_time: float = 0.5,
) -> Trajectory:
    """Fly ``course`` to its end (or the first crash).

    ``gate_frames`` (autonomy): touching a gate frame is a crash, and the episode ends
    ``finish_time`` after the last gate was passed in order.
    """
    arena = arena or ArenaConfig()
    p0, rpy0, v0 = sample_initial_state(course.start, seed, init_noise)
    state = backend.reset(p0, seed=seed, init_rpy=rpy0, init_vel=v0)
    controller.reset()
    if hasattr(controller, "set_course"):  # controllers with a reference preview (MPPI)
        controller.set_course(course)
    n_steps = int(np.ceil(course.duration * backend.ctrl_freq))

    keys = ("t", "pos", "rpy", "vel", "quat", "angv", "rp", "rv", "ra", "rpm", "rpm_app")
    log: dict[str, list] = {k: [] for k in keys}
    crash = None
    ctrl_time = 0.0
    tracker = GateTracker(list(course.gates)) if gate_frames else None
    t_finish = None
    for k in range(n_steps):
        t = k * backend.dt
        ref = course.reference(t)
        tic = time.perf_counter()
        rpm = controller.act(backend.observe(), ref)
        ctrl_time += time.perf_counter() - tic
        values = (t, state.pos, state.rpy, state.vel, state.quat, state.ang_vel,
                  ref.pos, ref.vel, ref.acc, rpm)
        for key, val in zip(keys[:-1], values, strict=True):
            log[key].append(val)
        p_prev = log["pos"][-1]
        state = backend.step(rpm)
        log["rpm_app"].append(backend.last_applied_rpm)
        crash = check_crash(t + backend.dt, state.pos, state.rpy, arena)
        if tracker is not None:
            if not crash and any(g.frame_hit(state.pos) for g in tracker.gates):
                crash = "gate_frame"
            if tracker.update(p_prev, state.pos, t, backend.dt) and tracker.done:
                t_finish = t + finish_time
        if crash and stop_on_crash:
            break
        if t_finish is not None and t >= t_finish:
            break

    a = {k: np.asarray(v) for k, v in log.items()}
    n = max(len(a["t"]), 1)
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
        dt=backend.dt,
        quat=a["quat"],
        ang_vel=a["angv"],
        rpm_applied=a["rpm_app"],
        ref_acc=a["ra"],
        gates=list(course.gates),
        wall_time_per_step=ctrl_time / n,
        meta={
            "family": course.family,
            "speed": course.speed,
            "course_seed": course.seed,
            "feasible": course.feasible,
            **{f"course_{k}": v for k, v in course.difficulty.items()},
        },
    )
