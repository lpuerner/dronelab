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


@dataclass(frozen=True)
class EpisodeSpec:
    controller: str
    family: str
    speed: float
    seed: int
    gains: dict | None = None
    disturbances: dict = field(default_factory=dict)  # DisturbanceConfig fields
    physics: str = "pyb"
    init_noise: bool = False
    log_path: str | None = None  # write a flight log (.npz) for the model stage
    tags: dict = field(default_factory=dict)  # copied into the result row (e.g. scenario)


_BACKENDS: dict = {}


def _backend(physics: str):
    from dronebench.backends.pybullet_backend import PyBulletBackend
    from dronebench.config import SimConfig

    if physics not in _BACKENDS:
        _BACKENDS[physics] = PyBulletBackend(sim=SimConfig(physics=physics))
    return _BACKENDS[physics]


def run_spec(spec: EpisodeSpec) -> dict:
    """Fly one episode and return a flat result row (metrics + spec + course info)."""
    from dronebench.config import DisturbanceConfig, InitNoiseConfig
    from dronebench.controllers import make_controller
    from dronebench.eval.flightlog import save_flight
    from dronebench.eval.metrics import compute_metrics
    from dronebench.eval.runner import run_episode
    from dronebench.tasks import make_course

    row = {"controller": spec.controller, "family": spec.family, "speed": spec.speed,
           "seed": spec.seed, **spec.tags}
    course = make_course(spec.family, spec.speed, spec.seed)
    row.update({"feasible": course.feasible, "rejections": course.rejections,
                **{f"course_{k}": v for k, v in course.difficulty.items()}})
    if not course.feasible:
        return row  # counted as infeasible, not flown
    be = _backend(spec.physics)
    be.set_disturbances(DisturbanceConfig(**spec.disturbances))
    try:
        ctrl = make_controller(spec.controller, be.params, be.dt, spec.gains)
    except (ValueError, ArithmeticError) as exc:  # e.g. Riccati failed (LinAlgError)
        row.update({"crashed": 1.0, "error": str(exc)})
        return row
    noise = InitNoiseConfig(enabled=spec.init_noise)
    traj = run_episode(be, ctrl, course, seed=spec.seed, init_noise=noise)
    row.update(compute_metrics(traj))
    row["crash_reason"] = traj.crash_reason
    if spec.log_path:
        save_flight(spec.log_path, traj, be.params, {**row, "disturbances": spec.disturbances})
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
