"""Reference baseline: the tuned cascaded PID shipped with gym-pybullet-drones."""

from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from dronelab.core.backends.base import DroneState
from dronelab.core.controller import Controller, Reference


class DSLPIDController(Controller):
    name = "dsl_pid"

    def __init__(self, ctrl_dt: float):
        self.ctrl_dt = ctrl_dt
        self._pid = DSLPIDControl(drone_model=DroneModel.CF2X)

    def reset(self) -> None:
        self._pid.reset()

    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        rpm, _, _ = self._pid.computeControl(
            control_timestep=self.ctrl_dt,
            cur_pos=state.pos,
            cur_quat=state.quat,
            cur_vel=state.vel,
            cur_ang_vel=state.ang_vel,
            target_pos=ref.pos,
            target_rpy=np.array([0.0, 0.0, ref.yaw]),
            target_vel=ref.vel,
        )
        return np.asarray(rpm, dtype=float)
