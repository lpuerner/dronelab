"""PyBullet backend built on gym-pybullet-drones (Crazyflie 2.x, CF2X)."""

from __future__ import annotations

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from dronebench.backends.base import Backend, DroneParams, DroneState
from dronebench.backends.disturbances import GustProcess, MotorLag, add_sensor_noise
from dronebench.config import DisturbanceConfig, SimConfig

_PHYSICS = {
    "pyb": Physics.PYB,
    "pyb_drag": Physics.PYB_DRAG,
    "pyb_gnd": Physics.PYB_GND,
    "pyb_gnd_drag_dw": Physics.PYB_GND_DRAG_DW,
}


class _DisturbedAviary(CtrlAviary):
    """CtrlAviary that applies an external (wind) force on *every* physics sub-step.

    PyBullet clears external forces after each ``stepSimulation``; CtrlAviary runs
    several sub-steps per control step, so the force has to be re-applied inside
    the sub-step loop, i.e. in ``_physics``.
    """

    external_force: np.ndarray = np.zeros(3)

    def _physics(self, rpm, nth_drone):
        super()._physics(rpm, nth_drone)
        if np.any(self.external_force):
            body = self.DRONE_IDS[nth_drone]
            pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self.CLIENT)
            p.applyExternalForce(
                body,
                -1,
                forceObj=self.external_force.tolist(),
                posObj=pos,  # apply at the centre of mass -> no extra torque
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )


class PyBulletBackend(Backend):
    def __init__(self, sim: SimConfig | None = None, disturbances: DisturbanceConfig | None = None):
        self.sim = sim or SimConfig()
        self.dist = disturbances or DisturbanceConfig()
        self._rng = np.random.default_rng(0)
        self._env = _DisturbedAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
            initial_rpys=np.zeros((1, 3)),
            physics=_PHYSICS[self.sim.physics],
            pyb_freq=self.sim.pyb_freq,
            ctrl_freq=self.sim.ctrl_freq,
            gui=self.sim.gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        e = self._env
        self._params = DroneParams(
            mass=float(e.M),
            arm=float(e.L),
            kf=float(e.KF),
            km=float(e.KM),
            inertia=np.array([e.J[0, 0], e.J[1, 1], e.J[2, 2]], dtype=float),
            max_rpm=float(e.MAX_RPM),
            hover_rpm=float(e.HOVER_RPM),
            gravity=float(e.G),
        )
        self._gust = GustProcess(self.dist.gust_std, self.dist.gust_tau, self._rng)
        self._motor = MotorLag(self.dist.motor_tau)
        self._state: DroneState | None = None
        self._steps = 0

    # ------------------------------------------------------------------ API
    @property
    def ctrl_freq(self) -> int:
        return self.sim.ctrl_freq

    @property
    def params(self) -> DroneParams:
        return self._params

    @property
    def time(self) -> float:
        return self._steps * self.dt

    @property
    def client(self) -> int:
        return self._env.CLIENT

    @property
    def drone_id(self) -> int:
        return self._env.DRONE_IDS[0]

    def reset(self, init_pos: np.ndarray, seed: int | None = None) -> DroneState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._gust.rng = self._rng
        self._env.INIT_XYZS = np.asarray(init_pos, dtype=float).reshape(1, 3)
        obs, _ = self._env.reset(seed=seed)
        if self.dist.mass_scale != 1.0:
            p.changeDynamics(
                self.drone_id,
                -1,
                mass=self._params.mass * self.dist.mass_scale,
                physicsClientId=self.client,
            )
        self._gust.reset()
        self._motor.reset(np.zeros(4))
        self._steps = 0
        self._state = self._parse(obs)
        return self._state

    def step(self, rpm: np.ndarray) -> DroneState:
        rpm = np.clip(np.asarray(rpm, dtype=float), 0.0, self._params.max_rpm)
        applied = self._motor.step(rpm, self.dt)
        gust = self._gust.step(self.dt)
        self._env.external_force = np.asarray(self.dist.wind_force, dtype=float) + gust
        obs, *_ = self._env.step(applied.reshape(1, 4))
        self._steps += 1
        self._state = self._parse(obs)
        return self._state

    def observe(self) -> DroneState:
        assert self._state is not None, "call reset() first"
        return add_sensor_noise(self._state, self.dist, self._rng)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _parse(obs: np.ndarray) -> DroneState:
        s = np.asarray(obs, dtype=float)[0]
        return DroneState(pos=s[0:3], quat=s[3:7], rpy=s[7:10], vel=s[10:13], ang_vel=s[13:16])
