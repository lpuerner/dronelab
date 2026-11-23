"""Run many episodes in worker processes, each keeping its own simulator alive.

An episode is described by a plain, picklable :class:`EpisodeSpec`. Every worker builds a
PyBullet backend once per physics mode and only swaps the disturbances between episodes,
so the expensive simulator start-up is paid once per process.

Workers use one BLAS thread each (set before numpy is imported in the child); with
2 cores this measured ~1.7x the throughput of a single process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from multiprocessing import get_context

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from dronelab.core.world import DEFAULT_WORLD  # noqa: E402


@dataclass(frozen=True)
class EpisodeSpec:
    controller: str
    family: str
    speed: float
    seed: int
    gains: dict | None = None
    disturbances: dict = field(default_factory=dict)  # scenario on top of the world
    world: str = DEFAULT_WORLD  # see dronelab.core.world
    init_noise: bool = False
    log_path: str | None = None  # write a flight log (.npz) for the model stage
    tags: dict = field(default_factory=dict)  # copied into the result row (e.g. scenario)
    speed_profile: str = "constant"  # "variable": speed is the maximum (surrogate data)
    dither: dict | None = None  # excitation on the commands, see dronelab.control.dither


_BACKENDS: dict = {}


def _backend(physics: str):
    from dronelab.core.backends.pybullet_backend import PyBulletBackend
    from dronelab.core.config import SimConfig

    if physics not in _BACKENDS:
        _BACKENDS[physics] = PyBulletBackend(sim=SimConfig(physics=physics))
    return _BACKENDS[physics]


def run_spec(spec: EpisodeSpec) -> dict:
    """Fly one episode and return a flat result row (metrics + spec + course info)."""
    from dronelab.control import make_controller
    from dronelab.core.config import DisturbanceConfig, InitNoiseConfig
    from dronelab.core.eval.flightlog import save_flight
    from dronelab.core.eval.metrics import compute_metrics
    from dronelab.core.eval.runner import run_episode
    from dronelab.core.tasks import make_course
    from dronelab.core.world import resolve

    physics, dist = resolve(spec.world, spec.disturbances)
    row = {"controller": spec.controller, "family": spec.family, "speed": spec.speed,
           "seed": spec.seed, "world": spec.world, "physics": physics,
           "speed_profile": spec.speed_profile, "dither": bool(spec.dither), **spec.tags}
    course = make_course(spec.family, spec.speed, spec.seed, speed_profile=spec.speed_profile)
    row.update({"feasible": course.feasible, "rejections": course.rejections,
                **{f"course_{k}": v for k, v in course.difficulty.items()}})
    if not course.feasible:
        return row  # counted as infeasible, not flown
    be = _backend(physics)
    be.set_disturbances(DisturbanceConfig(**dist))
    try:
        ctrl = make_controller(spec.controller, be.params, be.dt, spec.gains, spec.dither)
    except (ValueError, ArithmeticError) as exc:  # e.g. Riccati failed (LinAlgError)
        row.update({"crashed": 1.0, "error": str(exc)})
        return row
    noise = InitNoiseConfig(enabled=spec.init_noise)
    traj = run_episode(be, ctrl, course, seed=spec.seed, init_noise=noise)
    row.update(compute_metrics(traj))
    row["crash_reason"] = traj.crash_reason
    if spec.log_path:
        save_flight(spec.log_path, traj, be.params,
                    {**row, "disturbances": dist, "scenario_disturbances": spec.disturbances})
    return row


class EpisodePool:
    """``with EpisodePool(2) as pool: rows = pool.map(specs)`` (n=1 runs in-process)."""

    def __init__(self, processes: int = 2):
        self.n = processes
        self._pool = get_context("spawn").Pool(processes) if processes > 1 else None

    def map(self, specs: list[EpisodeSpec]) -> list[dict]:
        if self._pool is None:
            return [run_spec(s) for s in specs]
        return self._pool.map(run_spec, specs, chunksize=1)

    def imap(self, specs: list[EpisodeSpec]):
        if self._pool is None:
            return (run_spec(s) for s in specs)
        return self._pool.imap_unordered(run_spec, specs, chunksize=1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
