"""R3: a trained PPO policy behind the common :class:`Controller` interface.

``gains`` holds the paths written by ``projects/control/scripts/train_rl.py``:
``{"model": ".../model.zip", "vecnormalize": ".../vecnormalize.pkl"}``. The observation is
normalised with the frozen running statistics of training, the action is the deterministic
policy mean, mapped to motor speeds exactly as in :class:`~dronelab.control.rl_env.TrackingEnv`.
"""

from __future__ import annotations

import pickle

import numpy as np

from dronelab.control.rl_env import PREVIEW, ActionMap, observation
from dronelab.core.backends.base import DroneParams, DroneState
from dronelab.core.controller import Controller, Reference


class PPOController(Controller):
    name = "ppo"

    def __init__(self, params: DroneParams, ctrl_dt: float, gains: dict):
        from stable_baselines3 import PPO

        if not gains or "model" not in gains:
            raise ValueError("ppo needs gains={'model': ..., 'vecnormalize': ...}")
        self.model = PPO.load(gains["model"], device="cpu")
        with open(gains["vecnormalize"], "rb") as f:
            vn = pickle.load(f)
        self.mean, self.std = vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon)
        self.clip = vn.clip_obs
        self.dt = ctrl_dt
        self.act_map = ActionMap(params, ctrl_dt)
        self.course = None
        self.reset()

    def reset(self) -> None:
        self.k = 0
        self.last_action = np.zeros(4, np.float32)
        self.act_map.reset()

    def set_course(self, course) -> None:
        self.course = course

    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        t = self.k * self.dt
        if self.course is None:  # no preview available: hold the current reference
            preview = [ref.pos] * len(PREVIEW)
        else:
            preview = [self.course.reference(t + tau).pos for tau in PREVIEW]
        obs = observation(state, ref, preview, self.last_action)
        obs = np.clip((obs - self.mean) / self.std, -self.clip, self.clip)
        a, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
        self.last_action = np.clip(a, -1.0, 1.0).astype(np.float32)
        self.k += 1
        return self.act_map(self.last_action, state)
