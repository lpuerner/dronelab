"""Tests for the R3 tracking environment and the PPO wrapper (need PyBullet)."""

import numpy as np
import pytest

from dronelab.control.rl_env import OBS_DIM, TrackingEnv

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def env():
    e = TrackingEnv(rank=0)
    yield e
    e.close()


# Shapes, and the same episode sequence for the same (stream, rank).
def test_reset_step_and_determinism(env):
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
    for _ in range(5):
        obs, r, term, trunc, _ = env.step(np.zeros(4))
        assert obs.shape == (OBS_DIM,) and np.isfinite(r)
    other = TrackingEnv(rank=0)
    o1, _ = other.reset()
    env.n_episodes = 0
    o2, _ = env.reset()
    other.close()
    assert np.allclose(o1, o2)


# Zero action = hover thrust and zero rates: the drone neither falls nor climbs much in 0.5 s.
def test_zero_action_hovers(env):
    env.reset()
    z0 = env.state.pos[2]
    for _ in range(24):
        env.step(np.zeros(4))
    assert abs(env.state.pos[2] - z0) < 0.3


# Falling (no thrust) ends the episode with the crash penalty.
def test_crash_terminates(env):
    env.reset()
    for _ in range(400):
        _, r, term, trunc, _ = env.step(np.array([-1.0, 0, 0, 0]))
        if term or trunc:
            break
    assert term and r < -40
