"""A1 planner baseline: a spline through the gates, flown as fast as an acceleration budget allows.

* **A1** keeps the course's spline through the gate centres and replaces its speed profile by a
  curvature-limited one: v <= sqrt(a / kappa), forward / backward pass with the tangential
  remainder of the budget (friction circle). ``acc_frac`` scales the budget a = acc_frac * g
  (a = g is 45 deg tilt in level flight), it is the only parameter and is swept on validation seeds.
* **A1+** (``optimize=True``) also moves the crossing point of every gate inside its opening
  (lateral and vertical offset, keeping ``DRONE_RADIUS + margin`` to the frame) so that the lap
  time of that profile is minimal; the gates themselves stay where they are.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from dronelab.core.tasks.courses import (
    Course,
    CourseLimits,
    G,
    _difficulty,
    _Path,
    _VariableSpeedProfile,
)
from dronelab.core.tasks.gates import DRONE_RADIUS


def _profile(path: _Path, a: float, v_max: float) -> _VariableSpeedProfile:
    kappa = np.linalg.norm(path.curv, axis=1)
    return _VariableSpeedProfile(path.length, np.array([path.length]), np.array([v_max]), a,
                                 curv=lambda s: np.interp(s, path.s, kappa))


def _points(course: Course, offsets: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    pts = [course.path.pos[0]]
    for g, (dl, dz) in zip(course.gates, offsets.reshape(-1, 2), strict=True):
        pts.append(g.center + dl * g.lateral + dz * up)
    pts.append(course.path.pos[-1])
    return np.array(pts)


def _optimize(course: Course, a: float, v_max: float, margin: float) -> np.ndarray:
    lim = np.array([[g.width / 2, g.height / 2] for g in course.gates]) - DRONE_RADIUS - margin
    bounds = [(-b, b) for b in np.maximum(lim, 0.0).ravel()]

    def lap(x):
        return _profile(_Path(_points(course, x)), a, v_max).t_end

    x0 = np.zeros(2 * len(course.gates))
    res = minimize(lap, x0, method="Powell", bounds=bounds,
                   options={"maxfev": 400, "xtol": 1e-3, "ftol": 1e-4})
    return res.x if res.fun < lap(x0) else x0


@lru_cache(maxsize=256)
def _offsets(family: str, seed: int, speed: float, a: float, v_max: float,
             margin: float) -> tuple:
    from dronelab.core.tasks.courses import make_course

    return tuple(_optimize(make_course(family, speed, seed), a, v_max, margin))


def plan_course(course: Course, acc_frac: float, v_max: float = 5.0, optimize: bool = False,
                margin: float = 0.05) -> Course:
    """Re-plan a gate course (see module doc). The returned course is always flown:
    the acceleration budget, not the course generator's check, is the limit here."""
    if not course.gates:
        raise ValueError("plan_course needs a course with gates")
    a = acc_frac * G
    path = course.path
    offsets = None
    if optimize:  # cached per worker: every controller flies the same plan of a seed
        offsets = np.array(_offsets(course.family, course.seed, course.speed, a, v_max, margin))
        path = _Path(_points(course, offsets))
    out = replace(course, path=path, profile=_profile(path, a, v_max), speed=v_max,
                  speed_profile="curvature", feasible=True,
                  params={**course.params, "acc_frac": acc_frac, "optimized": optimize})
    out.difficulty = _difficulty(out, CourseLimits())
    if offsets is not None:
        out.difficulty["max_offset"] = float(np.abs(offsets).max())
    return out
