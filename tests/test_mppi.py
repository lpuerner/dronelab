"""MPPI controller: batched wrench -> rpm, closed-loop hover."""

import numpy as np
import pytest
import torch

from dronelab.control.mixer import Mixer
from dronelab.control.mppi import MPPIController
from dronelab.control.tuning.protocol import DEFAULTS
from dronelab.surrogate.data import CF2X, DT


# Inside the bounds the batched linear allocation of the rollouts equals the Mixer.
def test_batched_allocation_matches_mixer():
    ctrl = MPPIController(CF2X, DT, model="urdf")
    wrench = np.array([0.28, 1e-3, -5e-4, 2e-4])
    ref = Mixer(CF2X).wrench_to_rpm(wrench[0], wrench[1:])
    rpm = torch.sqrt((torch.tensor(wrench, dtype=torch.float32) @ ctrl.M_inv_T).clamp(0))
    assert np.allclose(rpm.numpy(), ref, rtol=1e-4)


# Closed loop: take off from 0.1 m and hold the set-point (URDF model, ideal world), using
# the reference preview the runner hands over.
@pytest.mark.sim
def test_hover_closed_loop():
    pytest.importorskip("gym_pybullet_drones")
    from dronelab.core.eval.parallel import EpisodeSpec, run_spec

    gains = {**DEFAULTS["mppi_urdf"], "samples": 128}
    row = run_spec(EpisodeSpec("mppi_urdf", "hover", 0.0, 0, gains, world="ideal"))
    assert not row["crashed"]
    assert row["final_err"] < 0.05
