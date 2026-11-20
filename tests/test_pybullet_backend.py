"""Physics sanity checks against the real PyBullet backend."""

import numpy as np
import pytest

pytest.importorskip("gym_pybullet_drones")

from dronebench.backends.pybullet_backend import PyBulletBackend  # noqa: E402
from dronebench.config import DisturbanceConfig  # noqa: E402
from dronebench.controllers.dsl_pid import DSLPIDController  # noqa: E402
from dronebench.eval import compute_metrics, run_episode  # noqa: E402
from dronebench.tasks import make_course  # noqa: E402

pytestmark = pytest.mark.sim


def _fly_open_loop(backend, rpm, seconds, start=(0.0, 0.0, 3.0)):
    backend.reset(np.array(start))
    for _ in range(int(seconds * backend.ctrl_freq)):
        s = backend.step(np.full(4, rpm))
    return s


def test_free_fall():
    with PyBulletBackend() as be:
        s = _fly_open_loop(be, 0.0, 0.5)
        drop = 3.0 - s.pos[2]
        assert drop == pytest.approx(0.5 * be.params.gravity * 0.5**2, rel=0.05)


def test_wind_force_acts_on_every_substep():
    f = 0.027  # N
    with PyBulletBackend(disturbances=DisturbanceConfig(wind_force=(f, 0.0, 0.0))) as be:
        s = _fly_open_loop(be, 0.0, 0.5)
        expected_x = 0.5 * f / be.params.mass * 0.5**2
        assert s.pos[0] == pytest.approx(expected_x, rel=0.1)


def test_heavier_drone_sinks_at_nominal_hover_rpm():
    with PyBulletBackend() as be:
        z_nominal = _fly_open_loop(be, be.params.hover_rpm, 0.5).pos[2]
    with PyBulletBackend(disturbances=DisturbanceConfig(mass_scale=1.3)) as be:
        z_heavy = _fly_open_loop(be, be.params.hover_rpm, 0.5).pos[2]
    assert abs(z_nominal - 3.0) < 0.05
    assert z_heavy < 3.0 - 0.1


def test_dsl_pid_hovers():
    with PyBulletBackend() as be:
        traj = run_episode(be, DSLPIDController(be.dt), make_course("hover", hold_time=6.0))
    m = compute_metrics(traj)
    assert not traj.crashed
    assert m["final_err"] < 0.05
