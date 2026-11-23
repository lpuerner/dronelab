"""Data generation for the surrogate model: variable speed, excitation paths, dither, splits."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from dronelab.core.tasks import TUNE_FAMILIES, make_course
from dronelab.core.tasks.courses import CourseLimits
from dronelab.surrogate.datasets import (
    SplitConfig,
    course_assignment,
    max_speed,
    overlapping_seeds,
    split_specs,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample(course, dt=0.002):
    t = np.arange(0.0, course.profile.t_end, dt)
    return (t, *course.profile.many(t))


# Variable profile: starts and ends at rest, respects the ramp acceleration and the
# requested maximum, and position / speed / acceleration are consistent.
@pytest.mark.parametrize("family", ["figure8", "helix", "waypoints", "gates", "excitation"])
def test_variable_profile_consistent(family):
    lim = CourseLimits()
    c = make_course(family, 2.5, 20001, speed_profile="variable")
    t, s, v, a = _sample(c)
    assert v[0] == 0 and c.profile(c.profile.t_end + 1) == (c.length, 0.0, 0.0)
    assert np.all(v <= 2.5 + 1e-9) and np.all(v >= 0)
    assert np.abs(a).max() <= lim.ramp_acc + 1e-6
    assert np.allclose(np.gradient(s, t)[2:-2], v[2:-2], atol=5e-3)
    assert np.all(np.diff(s) >= -1e-12)


# The speed really varies within one flight (that is the point of the profile).
def test_variable_profile_varies():
    c = make_course("waypoints", 3.0, 20003, speed_profile="variable")
    _, _, v, _ = _sample(c, 0.02)
    cruise = v[(v > 0.2)]
    assert cruise.max() - cruise.min() > 0.5


# Shape and gates do not depend on the speed profile (same seed -> same path).
def test_variable_profile_same_shape():
    c0 = make_course("gates", 1.0, 1000)
    c1 = make_course("gates", 1.0, 1000, speed_profile="variable")
    assert np.allclose(c0.path.pos, c1.path.pos)
    assert len(c0.gates) == len(c1.gates)


# Excitation courses: deterministic, always variable, and they visit the ground-effect zone.
def test_excitation_course():
    a = make_course("excitation", 2.0, 20005)
    b = make_course("excitation", 2.0, 20005)
    assert np.allclose(a.path.pos, b.path.pos) and a.speed_profile == "variable"
    low = [make_course("excitation", 2.0, s).path.pos[:, 2].min() for s in range(20000, 20020)]
    assert min(low) < 0.3 and min(low) >= 0.1 - 1e-9


# Unknown speed profile is an error.
def test_unknown_speed_profile():
    with pytest.raises(ValueError):
        make_course("figure8", 1.0, 0, speed_profile="random")


# Dither: off before t_on (pure base controller), then a bounded, seed-determined signal.
def test_dither():
    pytest.importorskip("gym_pybullet_drones")
    from dronelab.control import make_controller
    from dronelab.control.mixer import Mixer
    from dronelab.core.backends.base import DroneParams, DroneState
    from dronelab.core.controller import Reference

    p = DroneParams(mass=0.027, arm=0.0397, kf=3.16e-10, km=7.94e-12,
                    inertia=np.array([1.4e-5, 1.4e-5, 2.17e-5]), max_rpm=21702.0,
                    hover_rpm=14468.0)
    s = DroneState(pos=np.array([0, 0, 1.0]), quat=np.array([0, 0, 0, 1.0]), rpy=np.zeros(3),
                   vel=np.zeros(3), ang_vel=np.zeros(3))
    ref = Reference(pos=np.array([0, 0, 1.0]), vel=np.zeros(3))
    base = make_controller("lqr", p, 1 / 48)
    runs = []
    for _ in range(2):
        d = make_controller("lqr", p, 1 / 48, dither={"seed": 7, "amp": 0.1})
        d.reset()
        runs.append(np.array([d.act(s, ref) for _ in range(150)]))
    assert np.allclose(runs[0], runs[1])  # deterministic
    base.reset()
    r_base = np.array([base.act(s, ref) for _ in range(150)])
    assert np.allclose(runs[0][:48], r_base[:48])  # first second: no dither
    m = Mixer(p)
    dw = np.array([m.rpm_to_wrench(a) - m.rpm_to_wrench(b) for a, b in zip(runs[0], r_base,
                                                                           strict=True)])
    assert np.abs(dw[48:, 0]).max() <= 0.1 * p.mass * p.gravity * 1.0001
    assert np.abs(dw[48:, 0]).max() > 0.02 * p.mass * p.gravity  # it does something
    other = make_controller("lqr", p, 1 / 48, dither={"seed": 8, "amp": 0.1})
    other.reset()
    assert not np.allclose(runs[0], [other.act(s, ref) for _ in range(150)])


# Every family is flown by every controller within 12 consecutive seeds.
def test_course_assignment_balanced():
    combos = {course_assignment(s) for s in range(20300, 20312)}
    assert len(combos) == len(TUNE_FAMILIES) * 3


# The seed ranges of the configured splits never overlap (no leakage between splits).
def test_config_seeds_disjoint():
    cfg = yaml.safe_load((ROOT / "projects/surrogate/configs/data.yaml").read_text())
    splits = {k: SplitConfig(tuple(v["excitation"]), tuple(v["courses"]))
              for k, v in cfg["splits"].items()}
    assert overlapping_seeds(splits) == set()
    assert overlapping_seeds({"a": SplitConfig((0, 5), (5, 10)),
                              "b": SplitConfig((9, 12), (12, 13))}) == {9}


# Specs: one file per flight, unique names, dither only on excitation flights.
def test_split_specs(tmp_path):
    specs = split_specs(SplitConfig((100, 104), (200, 208)), {}, tmp_path)
    assert len(specs) == 12 and len({s.log_path for s in specs}) == 12
    assert all(s.dither for s in specs if s.family == "excitation")
    assert not any(s.dither for s in specs if s.family != "excitation")
    assert all(s.world == "real" and s.speed_profile == "variable" for s in specs)
    assert max_speed(5, (0.5, 3.0)) == max_speed(5, (0.5, 3.0))


# End to end: one dithered excitation flight in the real world writes a complete log.
@pytest.mark.sim
def test_generate_one_flight(tmp_path):
    pytest.importorskip("gym_pybullet_drones")
    from dronelab.core.eval.parallel import run_spec

    spec = split_specs(SplitConfig((20000, 20001), (0, 0)), {}, tmp_path)[0]
    row = run_spec(spec)
    assert row["world"] == "real" and row["physics"] == "pyb_gnd_drag_dw" and row["dither"]
    with np.load(spec.log_path) as d:
        assert {"t", "pos", "quat", "vel", "ang_vel", "rpm_cmd", "rpm_applied"} <= set(d.files)
        lag = np.abs(d["rpm_applied"] - d["rpm_cmd"]).max()
    assert lag > 0  # 20 ms motor lag of the real world is active
