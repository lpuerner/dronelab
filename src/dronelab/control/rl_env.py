"""R3: gymnasium environment for a PPO path-tracking policy (thrust + body rates).

The policy outputs a in [-1, 1]^4: collective thrust T = m g (1 + a_0) and desired body rates
a_1..3 * MAX_RATE; the rate loop of the tuned cascaded PID (:meth:`CascadedPID.rate_control`)
turns them into motor speeds, as in Kaufmann et al. 2023 and the MPPI of R4.

Observation (all world vectors rotated into the yaw frame): position / velocity error,
reference acceleration, reference positions 0.2 / 0.4 / 0.6 s ahead relative to the drone,
attitude as 6D rotation (first two columns of R in the yaw frame), body rates, height, last
action. :func:`observation` is shared with the benchmark wrapper :mod:`dronelab.control.rl_policy`.

Episodes: a random tuning-family course (seeds 10 000-10 999), speed U(0.25, 1.5) m/s, a random
start time on the course and a random start offset, ``EPISODE_S`` long, `real` world, nominal.
``randomize=True`` (domain randomisation variant) also draws mass, constant wind and motor lag.

Subclasses (the gate environment of the autonomy subproject) override ``_sample_task``,
``_obs`` and ``_reward``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import yaml

from dronelab.control.pid import CascadedPID
from dronelab.core.backends.base import DroneState
from dronelab.core.config import ArenaConfig, DisturbanceConfig, InitNoiseConfig, SimConfig
from dronelab.core.controller import Reference
from dronelab.core.eval.runner import check_crash, sample_initial_state
from dronelab.core.models.rigid_body import quat_to_rot
from dronelab.core.tasks.courses import TUNE_FAMILIES, make_course
from dronelab.core.world import resolve

PID_GAINS = "projects/control/results/tuning_real/pid/best_params.yaml"
MAX_RATE = np.array([6.0, 6.0, 3.0])  # [rad/s] body-rate command range (roll, pitch, yaw)
PREVIEW = (0.2, 0.4, 0.6)  # [s]
OBS_DIM = 32
EPISODE_S = 10.0
SEED0, N_SEEDS = 10_000, 1_000
SPEED_RANGE = (0.25, 1.5)
CRASH_PENALTY = 50.0
MAX_ERR = 1.0  # [m] tracking lost -> end of episode (counted as a crash)
W_ERR, W_DACT, W_RATE = 1 / 0.3, 0.1, 0.005  # reward weights (see TrackingEnv._reward)


def load_pid_gains() -> dict:
    with open(PID_GAINS) as f:
        return yaml.safe_load(f)


def _yaw_frame(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def observation(s: DroneState, ref: Reference, preview: list[np.ndarray],
                last_action: np.ndarray) -> np.ndarray:
    R = quat_to_rot(s.quat)
    Ry = _yaw_frame(float(s.rpy[2])).T  # world -> yaw frame
    rel = [Ry @ (p - s.pos) for p in preview]
    return np.concatenate([
        Ry @ (ref.pos - s.pos), Ry @ (ref.vel - s.vel), Ry @ ref.acc, *rel,
        (Ry @ R)[:, :2].T.ravel(), R.T @ s.ang_vel, [s.pos[2]], last_action,
    ]).astype(np.float32)


class ActionMap:
    """Policy action -> motor speeds through the tuned PID rate loop."""

    def __init__(self, params, dt: float):
        self.p = params
        self.pid = CascadedPID(params, dt, load_pid_gains())

    def reset(self) -> None:
        self.pid.reset()

    def __call__(self, a: np.ndarray, s: DroneState) -> np.ndarray:
        a = np.clip(a, -1.0, 1.0)
        thrust = self.p.mass * self.p.gravity * (1.0 + a[0])
        return self.pid.rate_control(thrust, a[1:] * MAX_RATE, s)


class TrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rank: int = 0, world: str = "real", randomize: bool = False,
                 seed0: int = SEED0, stream: int = 0, w_dact: float = W_DACT):
        """``rank``: worker index; ``stream``: training seed (own episode sequence per seed);
        ``w_dact``: weight of the action-change penalty (R3 ablation)."""
        from dronelab.core.backends.pybullet_backend import PyBulletBackend

        self.physics, self.world_dist = resolve(world)
        self.be = PyBulletBackend(sim=SimConfig(physics=self.physics))
        self.p, self.dt = self.be.params, self.be.dt
        self.act_map = ActionMap(self.p, self.dt)
        self.rank, self.randomize, self.seed0, self.stream = rank, randomize, seed0, stream
        self.w_dact = w_dact
        self.arena = ArenaConfig()
        self.n_episodes = 0
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    # ------------------------------------------------------------- hooks
    def _sample_task(self, rng: np.random.Generator) -> None:
        """Set ``self.course`` and the start time ``self.t0`` on it."""
        family = TUNE_FAMILIES[int(rng.integers(len(TUNE_FAMILIES)))]
        speed = float(rng.uniform(*SPEED_RANGE)) if family != "hover" else 0.0
        seed = self.seed0 + int(rng.integers(N_SEEDS))
        self.course = make_course(family, speed, seed)
        self.t0 = float(rng.uniform(0.0, max(self.course.duration - EPISODE_S, 0.0)))

    def _obs(self, s: DroneState) -> np.ndarray:
        t = self.t0 + self.k * self.dt
        prev = [self.course.reference(t + tau).pos for tau in PREVIEW]
        return observation(s, self.course.reference(t), prev, self.last_action)

    def _reward(self, s: DroneState, a: np.ndarray) -> tuple[float, bool]:
        """Analog of the tuning objective J: tracking error, command change, body rates."""
        ref = self.course.reference(self.t0 + self.k * self.dt)
        err = float(np.linalg.norm(ref.pos - s.pos))
        w = quat_to_rot(s.quat).T @ s.ang_vel
        r = (1.0 - min(W_ERR * err, 1.0) - self.w_dact * float(np.linalg.norm(a - self.last_action))
             - W_RATE * float(w @ w))
        return r, err > MAX_ERR

    def _disturbances(self, rng: np.random.Generator) -> dict:
        dist = dict(self.world_dist)
        if self.randomize:
            phi = rng.uniform(0, 2 * np.pi)
            wind = rng.uniform(0.0, 0.09)  # [N], up to the strongest benchmark wind
            dist.update(mass_scale=float(rng.uniform(0.8, 1.3)),
                        wind_force=(wind * np.cos(phi), wind * np.sin(phi), 0.0),
                        motor_tau=float(rng.uniform(0.02, 0.06)))
        return dist

    # ------------------------------------------------------------- gym API
    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng([self.seed0, self.stream, self.rank, self.n_episodes])
        self.n_episodes += 1
        self._sample_task(rng)
        self.be.set_disturbances(DisturbanceConfig(**self._disturbances(rng)))
        ref = self.course.reference(self.t0)
        ep_seed = int(rng.integers(2**31))
        p0, rpy0, v0 = sample_initial_state(ref.pos, ep_seed, InitNoiseConfig(enabled=True))
        self.state = self.be.reset(p0, seed=ep_seed, init_rpy=rpy0, init_vel=ref.vel + v0)
        self.act_map.reset()
        self.k = 0
        self.n_steps = int(min(EPISODE_S, self.course.duration - self.t0) / self.dt)
        self.last_action = np.zeros(4, np.float32)
        return self._obs(self.be.observe()), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        self.state = self.be.step(self.act_map(a, self.be.observe()))
        self.k += 1
        r, lost = self._reward(self.state, a)
        self.last_action = a
        crash = check_crash(self.k * self.dt, self.state.pos, self.state.rpy, self.arena)
        terminated = bool(crash) or lost
        if terminated:
            r -= CRASH_PENALTY
        truncated = not terminated and self.k >= self.n_steps
        return self._obs(self.be.observe()), float(r), terminated, truncated, {"crash": crash}

    def close(self):
        self.be.close()
