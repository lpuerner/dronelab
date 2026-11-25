"""INDI controller: steady hover without integrators, payload rejection in closed loop."""

import numpy as np
import pytest

from dronelab.control.indi import INDIController
from dronelab.core.backends.base import DroneState
from dronelab.core.controller import Reference
from dronelab.surrogate.data import CF2X, DT


# At rest on the set-point with synchronous measurement and wrench, the increment is zero:
# the command stays at hover for many steps (no drift from the filters or the motor model).
def test_steady_hover_is_a_fixed_point():
    ctrl = INDIController(CF2X, DT)
    s = DroneState(pos=np.array([0.0, 0.0, 1.0]), quat=np.array([0.0, 0.0, 0.0, 1.0]),
                   rpy=np.zeros(3), vel=np.zeros(3), ang_vel=np.zeros(3))
    ref = Reference(pos=s.pos.copy(), vel=np.zeros(3))
    for _ in range(50):
        rpm = ctrl.act(s, ref)
    assert np.allclose(rpm, CF2X.hover_rpm, rtol=1e-6)


# 30 % payload: the measured acceleration carries the missing thrust, so INDI holds the
# altitude without an integrator (a pure PD law would sag by m g / kp).
@pytest.mark.sim
def test_payload_rejection_closed_loop():
    pytest.importorskip("gym_pybullet_drones")
    from dronelab.core.eval.parallel import EpisodeSpec, run_spec

    row = run_spec(EpisodeSpec("indi", "hover", 0.0, 0, disturbances={"mass_scale": 1.3}))
    assert not row["crashed"]
    assert row["final_err"] < 0.02
