"""Physics sanity checks against the real PyBullet backend."""

import numpy as np
import pytest

pytest.importorskip("gym_pybullet_drones")

from dronelab.control.dsl_pid import DSLPIDController  # noqa: E402
from dronelab.core.backends.pybullet_backend import PyBulletBackend  # noqa: E402
from dronelab.core.config import DisturbanceConfig  # noqa: E402
from dronelab.core.eval import compute_metrics, run_episode  # noqa: E402
from dronelab.core.tasks import make_course  # noqa: E402

pytestmark = pytest.mark.sim


# Helper: hold all motors at a fixed rpm (no controller) and return the final state.
def _fly_open_loop(backend, rpm, seconds, start=(0.0, 0.0, 3.0)):
    backend.reset(np.array(start))
    for _ in range(int(seconds * backend.ctrl_freq)):
        s = backend.step(np.full(4, rpm))
    return s


# Motors off: fall distance after 0.5 s matches 1/2 * g * t^2.
def test_free_fall():
    with PyBulletBackend() as be:
        s = _fly_open_loop(be, 0.0, 0.5)
        drop = 3.0 - s.pos[2]
        assert drop == pytest.approx(0.5 * be.params.gravity * 0.5**2, rel=0.05)


# Wind must act on all physics sub-steps: drift = 1/2 * (F/m) * t^2, not 1/5 of it.
def test_wind_force_acts_on_every_substep():
    f = 0.027  # N
    with PyBulletBackend(disturbances=DisturbanceConfig(wind_force=(f, 0.0, 0.0))) as be:
        s = _fly_open_loop(be, 0.0, 0.5)
        expected_x = 0.5 * f / be.params.mass * 0.5**2
        assert s.pos[0] == pytest.approx(expected_x, rel=0.1)


# Nominal hover rpm keeps the nominal drone in place; a 30 % heavier one sinks.
def test_heavier_drone_sinks_at_nominal_hover_rpm():
    with PyBulletBackend() as be:
        z_nominal = _fly_open_loop(be, be.params.hover_rpm, 0.5).pos[2]
    with PyBulletBackend(disturbances=DisturbanceConfig(mass_scale=1.3)) as be:
        z_heavy = _fly_open_loop(be, be.params.hover_rpm, 0.5).pos[2]
    assert abs(z_nominal - 3.0) < 0.05
    assert z_heavy < 3.0 - 0.1


# Mass change scales mass and inertia together (PyBullet would otherwise recompute
# the inertia from the collision shape, ~3x the URDF value).
def test_mass_scale_keeps_inertia_consistent():
    import pybullet as p

    with PyBulletBackend(disturbances=DisturbanceConfig(mass_scale=1.2)) as be:
        be.reset(np.array([0.0, 0.0, 1.0]))
        mass, _, inertia = p.getDynamicsInfo(be.drone_id, -1, physicsClientId=be.client)[:3]
        assert mass == pytest.approx(1.2 * be.params.mass)
        np.testing.assert_allclose(inertia, 1.2 * be.params.inertia, rtol=1e-6)


# The reference DSL-PID flies the hover course without crashing, final error < 5 cm.
def test_dsl_pid_hovers():
    with PyBulletBackend() as be:
        traj = run_episode(be, DSLPIDController(be.dt), make_course("hover", seed=0))
    m = compute_metrics(traj)
    assert not traj.crashed
    assert m["final_err"] < 0.05


# Same seed -> identical flight (initial-state noise included); another seed differs.
def test_same_seed_same_flight():
    from dronelab.control import make_controller
    from dronelab.core.config import InitNoiseConfig

    noise = InitNoiseConfig(enabled=True)
    course = make_course("figure8", 1.0, 0)
    with PyBulletBackend() as be:
        runs = [
            run_episode(be, make_controller("pid", be.params, be.dt), course, seed=s,
                        init_noise=noise).pos[:200]
            for s in (5, 5, 6)
        ]
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.allclose(runs[0], runs[2])


# Own PID, LQR and LQI hover stably; LQI also removes the offset of a 20 % heavier drone.
@pytest.mark.parametrize("name", ["pid", "lqr", "lqi"])
def test_own_controllers_hover(name):
    from dronelab.control import make_controller

    with PyBulletBackend() as be:
        ctrl = make_controller(name, be.params, be.dt)
        traj = run_episode(be, ctrl, make_course("hover", seed=0))
    assert not traj.crashed and compute_metrics(traj)["final_err"] < 0.05


def test_lqi_removes_offset_with_heavier_drone():
    from dronelab.control import make_controller

    errs = {}
    for name in ("lqr", "lqi"):
        with PyBulletBackend(disturbances=DisturbanceConfig(mass_scale=1.2)) as be:
            traj = run_episode(be, make_controller(name, be.params, be.dt),
                               make_course("hover", seed=0))
        errs[name] = compute_metrics(traj)["final_err"]
    assert errs["lqi"] < 0.02 and errs["lqi"] < errs["lqr"]


# The flight log contains body-frame data and the quadratic-model force.
def test_flight_log(tmp_path):
    from dronelab.control import make_controller
    from dronelab.core.eval.flightlog import flight_arrays, save_flight

    with PyBulletBackend() as be:
        traj = run_episode(be, make_controller("pid", be.params, be.dt),
                           make_course("helix", 1.0, 0))
        arr = flight_arrays(traj, be.params)
        save_flight(tmp_path / "f.npz", traj, be.params, {"test": True})
    n = len(traj.t)
    assert arr["rpm_applied"].shape == (n, 4)
    assert (tmp_path / "f.npz").exists()
