"""INDI controller: steady hover without integrators, payload rejection in closed loop."""

import numpy as np
import pytest

from dronelab.control.indi import INDIController
from dronelab.core.backends.base import DroneState
from dronelab.core.controller import Reference
from dronelab.surrogate.data import CF2X, DT


# At rest on the set-point with synchronous measurement and wrench, the increment is zero:
# the command stays at hover for many steps (no drift from the filters or the motor model).
@pytest.mark.parametrize("window", [0, 9])
def test_steady_hover_is_a_fixed_point(window):
    ctrl = INDIController(CF2X, DT, acc_window=window)
    s = DroneState(pos=np.array([0.0, 0.0, 1.0]), quat=np.array([0.0, 0.0, 0.0, 1.0]),
                   rpy=np.zeros(3), vel=np.zeros(3), ang_vel=np.zeros(3))
    ref = Reference(pos=s.pos.copy(), vel=np.zeros(3))
    for _ in range(50):
        rpm = ctrl.act(s, ref)
    assert np.allclose(rpm, CF2X.hover_rpm, rtol=1e-6)


# 30 % payload: the measured acceleration carries the missing thrust, so INDI holds the
# altitude without an integrator (a pure PD law would sag by m g / kp).
@pytest.mark.sim
@pytest.mark.parametrize("name", ["indi", "indi_sg"])
def test_payload_rejection_closed_loop(name):
    pytest.importorskip("gym_pybullet_drones")
    from dronelab.core.eval.parallel import EpisodeSpec, run_spec

    row = run_spec(EpisodeSpec(name, "hover", 0.0, 0, disturbances={"mass_scale": 1.3}))
    assert not row["crashed"]
    assert row["final_err"] < 0.02


# The Savitzky-Golay weights reproduce the slope of a straight line exactly and cut white
# velocity noise by sqrt(12 / (N (N^2 - 1))) instead of the sqrt(2) of a finite difference.
def test_sg_weights_exact_slope_and_noise_gain():
    w = INDIController(CF2X, DT, acc_window=9).w_sg
    assert np.isclose(w.sum(), 1.0)
    v = 0.3 * np.arange(9.0)  # constant acceleration 0.3 per step
    assert np.isclose(w @ np.diff(v), 0.3)
    c = np.diff(np.concatenate([[0.0], w, [0.0]]))  # weights on the velocities themselves
    assert np.isclose(np.linalg.norm(c), np.sqrt(12 / (9 * 80)))