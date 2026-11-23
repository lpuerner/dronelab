"""Randomised, smooth and feasibility-checked flight courses.

A course is generated deterministically from ``(family, speed, seed, speed_profile)``:

* the shape parameters are drawn from a family-specific range,
* the path is a cubic spline through the shape points, re-parametrised by arc length,
* it is flown with a trapezoidal speed profile (accelerate, cruise, decelerate) and
  then held at the last point; with ``speed_profile="variable"`` the cruise speed changes
  along the path instead (segments of 1-3 m with speeds in [0.25, speed]), used for the
  data of the surrogate model,
* courses that leave the arena or would need more than ``thrust_margin`` of the
  maximum thrust / more than ``max_tilt_deg`` of tilt are rejected and redrawn.

``hover`` is a set-point task (step to a target height) instead of a path.
``excitation`` (not a benchmark family) is a random path for system identification: 6-10
points in a 6 x 6 m box from 0.15 m (ground effect) to 2.5 m, always with a variable
speed profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from dronelab.core.controller import Reference
from dronelab.core.tasks.gates import Gate

G = 9.8
FAMILIES = ("hover", "figure8", "helix", "waypoints", "gates")
TUNE_FAMILIES = ("hover", "figure8", "helix", "waypoints")  # gates = unseen test family
EXTRA_FAMILIES = ("excitation",)  # surrogate data only
_FAMILY_ID = {name: i for i, name in enumerate(FAMILIES + EXTRA_FAMILIES)}
SPEED_PROFILES = ("constant", "variable")
MIN_SEGMENT_SPEED = 0.25


@dataclass(frozen=True)
class CourseLimits:
    thrust_to_weight: float = 2.25  # Crazyflie CF2X (gym-pybullet-drones URDF)
    thrust_margin: float = 0.8  # max. fraction of full thrust the reference may need
    max_tilt_deg: float = 45.0
    arena_half: float = 4.5  # |x|, |y| of the path (arena is 5 m, keep a margin)
    z_min: float = 0.3
    z_max: float = 3.0
    ramp_acc: float = 2.0  # [m/s^2] acceleration of the speed profile
    max_tries: int = 50


# --------------------------------------------------------------------------- path
class _Path:
    """Arc-length parametrised spline with tangent and curvature vector tables."""

    def __init__(self, points: np.ndarray, ds: float = 0.005):
        pts = np.asarray(points, dtype=float)
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        spline = CubicSpline(chord, pts, axis=0)
        d1, d2 = spline.derivative(1), spline.derivative(2)
        u = np.linspace(0.0, chord[-1], max(2000, int(chord[-1] / ds) * 2))
        seg = np.linalg.norm(np.diff(spline(u), axis=0), axis=1)
        s_of_u = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(s_of_u[-1])
        self.ds = ds
        n = int(np.ceil(self.length / ds)) + 1
        self.s = np.linspace(0.0, self.length, n)
        uu = np.interp(self.s, s_of_u, u)
        p1, p2 = d1(uu), d2(uu)
        speed = np.linalg.norm(p1, axis=1, keepdims=True)
        self.pos = spline(uu)
        self.tan = p1 / speed
        # d^2 p / ds^2 = (p'' - (p''.T) T) / |p'|^2
        self.curv = (p2 - np.sum(p2 * self.tan, axis=1, keepdims=True) * self.tan) / speed**2

    def at(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = min(max(s, 0.0), self.length) / self.ds
        i = min(int(x), len(self.s) - 2)
        f = x - i
        lerp = lambda a: (1 - f) * a[i] + f * a[i + 1]  # noqa: E731
        return lerp(self.pos), lerp(self.tan), lerp(self.curv)


@dataclass(frozen=True)
class _SpeedProfile:
    """Trapezoidal (or triangular) speed profile over a path of given length."""

    length: float
    v: float
    a: float

    @property
    def peak(self) -> float:
        return min(self.v, np.sqrt(self.a * self.length))

    @property
    def t_acc(self) -> float:
        return self.peak / self.a

    @property
    def t_cruise(self) -> float:
        return max(0.0, (self.length - self.peak**2 / self.a) / self.peak)

    @property
    def t_end(self) -> float:
        return 2 * self.t_acc + self.t_cruise

    def many(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised version of ``__call__``."""
        out = np.array([self(float(x)) for x in np.atleast_1d(t)])  # cheap scalar maths
        return out[:, 0], out[:, 1], out[:, 2]

    def __call__(self, t: float) -> tuple[float, float, float]:
        """Arc length, speed and tangential acceleration at time t."""
        a, vp, ta, tc = self.a, self.peak, self.t_acc, self.t_cruise
        if t <= 0:
            return 0.0, 0.0, 0.0
        if t < ta:
            return 0.5 * a * t**2, a * t, a
        d_acc = 0.5 * a * ta**2
        if t < ta + tc:
            return d_acc + vp * (t - ta), vp, 0.0
        if t < self.t_end:
            r = t - ta - tc
            return d_acc + vp * tc + vp * r - 0.5 * a * r**2, vp - a * r, -a
        return self.length, 0.0, 0.0


class _VariableSpeedProfile:
    """Piecewise-constant target speed along the arc length, made feasible for a maximum
    tangential acceleration ``a`` by a forward and a backward pass (starts and ends at rest).
    Same interface as :class:`_SpeedProfile`."""

    def __init__(self, length: float, seg_ends: np.ndarray, seg_speeds: np.ndarray, a: float,
                 ds: float = 0.01):
        self.length, self.a = float(length), float(a)
        n = max(int(np.ceil(length / ds)), 2) + 1
        self._s = np.linspace(0.0, length, n)
        h = self._s[1] - self._s[0]
        idx = np.minimum(np.searchsorted(seg_ends, self._s, side="right"), len(seg_speeds) - 1)
        v = np.asarray(seg_speeds, float)[idx]
        v[0] = v[-1] = 0.0
        for i in range(1, n):  # acceleration limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a * h))
        for i in range(n - 2, -1, -1):  # braking limit
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a * h))
        self._v = v
        dt = 2 * h / np.maximum(v[1:] + v[:-1], 1e-9)
        self._t = np.concatenate([[0.0], np.cumsum(dt)])
        self.v = float(v.max())  # peak speed (like _SpeedProfile.v)

    @property
    def t_end(self) -> float:
        return float(self._t[-1])

    def many(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = np.array([self(float(x)) for x in np.atleast_1d(t)])
        return out[:, 0], out[:, 1], out[:, 2]

    def __call__(self, t: float) -> tuple[float, float, float]:
        if t <= 0:
            return 0.0, 0.0, 0.0
        if t >= self._t[-1]:
            return self.length, 0.0, 0.0
        i = int(np.searchsorted(self._t, t, side="right")) - 1
        v0, v1 = self._v[i], self._v[i + 1]
        h = self._s[i + 1] - self._s[i]
        acc = (v1**2 - v0**2) / (2 * h)  # constant within the cell
        tau = t - self._t[i]
        return float(self._s[i] + v0 * tau + 0.5 * acc * tau**2), float(v0 + acc * tau), float(acc)


def _variable_profile(rng: np.random.Generator, length: float, v_max: float,
                      a: float) -> _VariableSpeedProfile:
    ends, s = [], 0.0
    while s < length:
        s += rng.uniform(1.0, 3.0)
        ends.append(min(s, length))
    speeds = rng.uniform(min(MIN_SEGMENT_SPEED, v_max), v_max, len(ends))
    speeds[rng.integers(len(ends))] = v_max  # the requested speed is reached somewhere
    return _VariableSpeedProfile(length, np.array(ends), speeds, a)


# ------------------------------------------------------------------------- course
@dataclass
class Course:
    family: str
    seed: int
    speed: float
    start: np.ndarray
    params: dict
    hold_time: float
    setpoint: np.ndarray | None = None  # hover target
    path: _Path | None = None
    profile: _SpeedProfile | _VariableSpeedProfile | None = None
    speed_profile: str = "constant"
    gates: list[Gate] = field(default_factory=list)
    feasible: bool = True
    rejections: int = 0
    difficulty: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        flight = self.profile.t_end if self.profile else 0.0
        return flight + self.hold_time

    @property
    def length(self) -> float:
        return self.path.length if self.path else 0.0

    def reference(self, t: float) -> Reference:
        if self.path is None:
            return Reference(pos=self.setpoint.copy(), vel=np.zeros(3))
        s, sd, sdd = self.profile(t)
        pos, tan, curv = self.path.at(s)
        return Reference(pos=pos, vel=sd * tan, acc=sdd * tan + sd**2 * curv)

    @property
    def waypoints(self) -> np.ndarray:
        """Start and end of the reference (compatibility / plotting)."""
        if self.path is None:
            return np.atleast_2d(self.setpoint)
        return self.path.pos[[0, -1]]


def _difficulty(course: Course, lim: CourseLimits) -> dict:
    ts = np.arange(0.0, course.profile.t_end, 0.02)
    s, sd, sdd = course.profile.many(ts)
    idx = np.clip(np.rint(s / course.path.ds).astype(int), 0, len(course.path.s) - 1)
    acc = sdd[:, None] * course.path.tan[idx] + (sd**2)[:, None] * course.path.curv[idx]
    f = acc + np.array([0.0, 0.0, G])
    fn = np.linalg.norm(f, axis=1)
    tilt = np.degrees(np.arccos(np.clip(f[:, 2] / fn, -1, 1)))
    return {
        "max_acc": float(np.linalg.norm(acc, axis=1).max()),
        "max_tilt_deg": float(tilt.max()),
        "max_thrust_ratio": float((fn / (lim.thrust_to_weight * G)).max()),
    }


def _inside_arena(pts: np.ndarray, lim: CourseLimits) -> bool:
    return bool(
        np.all(np.abs(pts[:, :2]) <= lim.arena_half)
        and np.all(pts[:, 2] >= lim.z_min)
        and np.all(pts[:, 2] <= lim.z_max)
    )


# ---------------------------------------------------------------- shape samplers
def _figure8(rng):
    a = rng.uniform(0.5, 2.5)  # half width -> width 1..5 m (large ones stay feasible when fast)
    # Lobe-top radius of this lemniscate is a / (8 * aspect): flat eights stay feasible at speed
    aspect = rng.uniform(0.3, 0.7)
    rot = rng.uniform(0, 2 * np.pi)
    alt = rng.uniform(0.8, 1.5)
    th = np.linspace(0, 2 * np.pi, 120)
    x, y = a * np.sin(th), a * aspect * np.sin(2 * th)
    c, s = np.cos(rot), np.sin(rot)
    pts = np.stack([c * x - s * y, s * x + c * y, np.full_like(th, alt)], 1)
    return pts, {"width": 2 * a, "aspect": aspect, "rotation": rot, "altitude": alt}, []


def _helix(rng):
    r = rng.uniform(0.5, 2.0)
    turns = int(rng.integers(1, 3))
    climb = rng.uniform(0.0, 0.5)
    alt = rng.uniform(0.8, 1.2)
    phi0 = rng.uniform(0, 2 * np.pi)
    direction = rng.choice([-1.0, 1.0])
    th = np.linspace(0, 2 * np.pi * turns, 80 * turns)
    pts = np.stack(
        [
            r * np.cos(phi0 + direction * th),
            r * np.sin(phi0 + direction * th),
            alt + climb * th / (2 * np.pi),
        ],
        1,
    )
    return pts, {"radius": r, "turns": turns, "climb": climb, "altitude": alt}, []


def _waypoints(rng):
    n = int(rng.integers(4, 7))
    pts = [np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.7, 2.0)])]
    while len(pts) < n:
        cand = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.7, 2.0)])
        if np.linalg.norm(cand - pts[-1]) >= 1.0:
            pts.append(cand)
    return np.array(pts), {"n_waypoints": n}, []


def _gates(rng, n: int):
    """Slalom through ``n`` gates with random spacing, side and height.

    ``n`` is drawn once per seed outside the rejection loop, so infeasible geometries are
    re-drawn without biasing the gate count towards easier courses.
    """
    gaps = rng.uniform(1.2, 1.8, n - 1)
    margins = rng.uniform(0.6, 0.9, 2)
    xs = np.concatenate([[0.0], np.cumsum(gaps)])
    total = margins[0] + xs[-1] + margins[1]
    xs = xs + (-total / 2 + margins[0])  # centre the whole course on x = 0
    side = rng.choice([-1.0, 1.0])
    ys = []
    for i in range(n):
        if i > 0 and rng.random() > 0.25:  # usually alternate, sometimes same side twice
            side = -side
        ys.append(side * rng.uniform(0.2, 1.2))
    zs = rng.uniform(0.7, 1.6, n)
    centers = np.stack([xs, np.array(ys), zs], 1)
    start = np.array([xs[0] - margins[0], 0.0, rng.uniform(0.8, 1.2)])
    end = np.array([xs[-1] + margins[1], 0.0, rng.uniform(0.8, 1.2)])
    pts = np.vstack([start, centers, end])
    return pts, {"n_gates": n, "gaps": [round(float(g), 3) for g in gaps]}, list(centers)


def _excitation(rng):
    """Random points for system identification, incl. low passes and large height changes."""
    n = int(rng.integers(6, 11))
    pts = [np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 2.0)])]
    while len(pts) < n:
        low = rng.random() < 0.25  # ground effect
        z = rng.uniform(0.15, 0.3) if low else rng.uniform(0.4, 2.5)
        cand = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), z])
        if np.linalg.norm(cand - pts[-1]) >= 1.0:
            pts.append(cand)
    return np.array(pts), {"n_points": n}, []


_SAMPLERS = {"figure8": _figure8, "helix": _helix, "waypoints": _waypoints, "gates": _gates,
             "excitation": _excitation}
_EXCITATION_Z_MIN = 0.1


def _gate_objects(path: _Path, centers: list[np.ndarray]) -> list[Gate]:
    gates = []
    for c in centers:
        i = int(np.argmin(np.linalg.norm(path.pos - c, axis=1)))
        n = path.tan[i].copy()
        n[2] = 0.0
        gates.append(Gate(center=path.pos[i].copy(), normal=n / np.linalg.norm(n)))
    return gates


def make_course(
    family: str,
    speed: float = 1.0,
    seed: int = 0,
    limits: CourseLimits | None = None,
    speed_profile: str = "constant",
) -> Course:
    """Deterministic course for ``(family, speed, seed, speed_profile)``.

    With ``speed_profile="variable"``, ``speed`` is the maximum cruise speed.
    """
    if family not in _FAMILY_ID:
        raise ValueError(f"unknown course family {family!r}, choose from "
                         f"{FAMILIES + EXTRA_FAMILIES}")
    if speed_profile not in SPEED_PROFILES:
        raise ValueError(f"unknown speed profile {speed_profile!r}, choose from {SPEED_PROFILES}")
    lim = limits or CourseLimits()
    if family == "excitation":
        speed_profile = "variable"
        if limits is None:
            lim = CourseLimits(z_min=_EXCITATION_Z_MIN)
    rng = np.random.default_rng([seed, _FAMILY_ID[family]])
    # speed segments from their own stream: same shapes for constant and variable profiles
    rng_speed = np.random.default_rng([seed, _FAMILY_ID[family], 77])

    if family == "hover":
        target = np.array([0.0, 0.0, rng.uniform(0.5, 1.5)])
        return Course(
            family="hover",
            seed=seed,
            speed=0.0,
            start=np.array([0.0, 0.0, 0.1]),
            params={"height": float(target[2])},
            hold_time=8.0,
            setpoint=target,
        )

    course = None
    # gate count drawn from its own stream, before (and independent of) the rejection loop
    n_gates = int(np.random.default_rng([seed, _FAMILY_ID[family], 99]).integers(4, 6))
    for attempt in range(lim.max_tries):
        pts, params, centers = (
            _gates(rng, n_gates) if family == "gates" else _SAMPLERS[family](rng)
        )
        path = _Path(pts)
        if not _inside_arena(path.pos, lim):
            continue
        course = Course(
            family=family,
            seed=seed,
            speed=float(speed),
            start=path.pos[0].copy(),
            params=params,
            hold_time=2.0,
            path=path,
            profile=(
                _variable_profile(rng_speed, path.length, float(speed), lim.ramp_acc)
                if speed_profile == "variable"
                else _SpeedProfile(path.length, float(speed), lim.ramp_acc)
            ),
            speed_profile=speed_profile,
            gates=_gate_objects(path, centers),
            rejections=attempt,
        )
        course.difficulty = _difficulty(course, lim)
        if (
            course.difficulty["max_thrust_ratio"] <= lim.thrust_margin
            and course.difficulty["max_tilt_deg"] <= lim.max_tilt_deg
        ):
            return course
    if course is None:
        raise RuntimeError(f"could not generate a {family} course inside the arena")
    course.feasible = False  # last candidate is returned but flagged
    course.rejections = lim.max_tries
    return course
