"""Flight datasets of the surrogate subproject (rules: projects/surrogate/DATASETS.md).

A split is a folder of ``.npz`` flight logs (see :mod:`dronelab.core.eval.flightlog`).
Generated splits (train, val, T1) consist of two kinds of flights, each with its own seed
range:

* ``excitation``: tuned LQI on a random excitation path (variable speed, low passes for the
  ground effect) plus a dither signal on the commanded wrench,
* ``courses``: the tuning families (hover, figure8, helix, waypoints) with a *variable*
  speed profile, flown in turn by the tuned PID, LQR and LQI, without dither.

T3 / T4 come from the control benchmark (``projects/control/configs/benchmark/r1.yaml``),
T2 is a selection of train / T1 by maximum speed, not a folder of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dronelab.core.eval.parallel import EpisodeSpec
from dronelab.core.tasks import TUNE_FAMILIES

COURSE_CONTROLLERS = ("pid", "lqr", "lqi")
EXCITATION_CONTROLLER = "lqi"
SPLIT_DIRS = {
    "train": "train",
    "val": "val",
    "T1": "T1_in_distribution",
    "T3": "T3_courses",
    "T4": "T4_disturbances",
    "T5": "T5_controllers",
}
T2_MAX_SPEED = 1.5  # "slow" training flights for the extrapolation test T2


@dataclass(frozen=True)
class SplitConfig:
    excitation: tuple[int, int]  # seed range [start, stop)
    courses: tuple[int, int]


def max_speed(seed: int, v_range: tuple[float, float]) -> float:
    """Requested maximum speed of a generated flight (own stream, depends only on the seed)."""
    return round(float(np.random.default_rng([seed, 555]).uniform(*v_range)), 3)


def course_assignment(seed: int) -> tuple[str, str]:
    """(family, controller) of a course flight: every family with every controller in turn."""
    family = TUNE_FAMILIES[seed % len(TUNE_FAMILIES)]
    controller = COURSE_CONTROLLERS[(seed // len(TUNE_FAMILIES)) % len(COURSE_CONTROLLERS)]
    return family, controller


def split_specs(
    split: SplitConfig,
    gains: dict[str, dict | None],
    out_dir: str | Path,
    world: str = "real",
    v_range: tuple[float, float] = (0.5, 3.0),
    dither_amp: float = 0.1,
) -> list[EpisodeSpec]:
    out = Path(out_dir)
    specs = []
    for seed in range(*split.excitation):
        v = max_speed(seed, v_range)
        c = EXCITATION_CONTROLLER
        specs.append(EpisodeSpec(
            c, "excitation", v, seed, gains.get(c), world=world, speed_profile="variable",
            dither={"seed": seed, "amp": dither_amp},
            log_path=str(out / f"excitation_{c}_{seed}.npz"),
            tags={"kind": "excitation"},
        ))
    for seed in range(*split.courses):
        family, c = course_assignment(seed)
        v = 0.0 if family == "hover" else max_speed(seed, v_range)
        specs.append(EpisodeSpec(
            c, family, v, seed, gains.get(c), world=world, speed_profile="variable",
            log_path=str(out / f"course_{c}_{family}_{seed}.npz"),
            tags={"kind": "course"},
        ))
    return specs


def overlapping_seeds(splits: dict[str, SplitConfig]) -> set[int]:
    """Seeds that appear in more than one split or kind (must be empty)."""
    seen: dict[int, str] = {}
    dup = set()
    for name, sp in splits.items():
        for kind, rng in (("excitation", sp.excitation), ("courses", sp.courses)):
            for s in range(*rng):
                if s in seen:
                    dup.add(s)
                seen[s] = f"{name}/{kind}"
    return dup
