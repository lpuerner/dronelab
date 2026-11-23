"""PyBullet backend built on gym-pybullet-drones (Crazyflie 2.x, CF2X)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import gym_pybullet_drones
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from dronelab.core.backends.base import Backend, DroneParams, DroneState
from dronelab.core.backends.disturbances import GustProcess, MotorLag, add_sensor_noise
from dronelab.core.config import DisturbanceConfig, SimConfig

_PHYSICS = {
    "pyb": Physics.PYB,
    "pyb_drag": Physics.PYB_DRAG,
    "pyb_gnd": Physics.PYB_GND,
    "pyb_gnd_drag_dw": Physics.PYB_GND_DRAG_DW,
}


def _rotor_positions() -> np.ndarray:
    """Propeller (x, y) positions from the installed cf2x.urdf (single source of truth)."""
    urdf = Path(gym_pybullet_drones.__file__).parent / "assets" / "cf2x.urdf"
    root = ET.parse(urdf).getroot()
    xy = []
    for i in range(4):
        # the prop position is stored as the inertial origin of prop{i}_link (fixed joints)
        origin = root.find(f"link[@name='prop{i}_link']/inertial/origin")
        x, y, _ = (float(v) for v in origin.get("xyz").split())
        xy.append([x, y])
    return np.array(xy)


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
            rotor_xy=_rotor_positions(),
        )
        self.set_disturbances(self.dist)
        self._state: DroneState | None = None
        self._steps = 0
        self._applied = np.zeros(4)

    def set_disturbances(self, disturbances: DisturbanceConfig) -> None:
        """Change disturbances without recreating the simulator (takes effect at reset)."""
        self.dist = disturbances
        self._gust = GustProcess(self.dist.gust_std, self.dist.gust_tau, self._rng)
        self._motor = MotorLag(self.dist.motor_tau)

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

    @property
    def last_applied_rpm(self) -> np.ndarray:
        return self._applied.copy()

    def reset(
        self,
        init_pos: np.ndarray,
        seed: int | None = None,
        init_rpy: np.ndarray | None = None,
        init_vel: np.ndarray | None = None,
        init_ang_vel: np.ndarray | None = None,
    ) -> DroneState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._gust.rng = self._rng
        e = self._env
        e.INIT_XYZS = np.asarray(init_pos, dtype=float).reshape(1, 3)
        e.INIT_RPYS = np.zeros((1, 3)) if init_rpy is None else np.reshape(init_rpy, (1, 3))
        obs, _ = e.reset(seed=seed)
        if init_vel is not None or init_ang_vel is not None:
            p.resetBaseVelocity(
                self.drone_id,
                linearVelocity=list(np.zeros(3) if init_vel is None else init_vel),
                angularVelocity=list(np.zeros(3) if init_ang_vel is None else init_ang_vel),
                physicsClientId=self.client,
            )
            e._updateAndStoreKinematicInformation()
            obs = e._computeObs()
        if self.dist.mass_scale != 1.0:
            # Setting only `mass` makes PyBullet recompute the inertia from the collision
            # shape (~3x the URDF value). Pass the inertia explicitly: a payload with the
            # same mass distribution scales the inertia by the same factor.
            k = self.dist.mass_scale
            p.changeDynamics(
                self.drone_id,
                -1,
                mass=self._params.mass * k,
                localInertiaDiagonal=(self._params.inertia * k).tolist(),
                physicsClientId=self.client,
            )
        self._gust.reset()
        # motors are already spinning at hover speed when the episode starts
        self._motor.reset(np.full(4, self._params.hover_rpm))
        self._applied = np.full(4, self._params.hover_rpm)
        self._steps = 0
        self._state = self._parse(obs)
        return self._state

    def step(self, rpm: np.ndarray) -> DroneState:
        rpm = np.clip(np.asarray(rpm, dtype=float), 0.0, self._params.max_rpm)
        applied = self._motor.step(rpm, self.dt)
        self._applied = applied.copy()
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
