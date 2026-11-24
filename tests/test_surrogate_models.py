"""Surrogate models: rotation maths, rollout interface, physics model against PyBullet."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dronelab.surrogate.data import CF2X, load_flight, make_segments
from dronelab.surrogate.models import Physics, Surrogate, rot_angle, so3_exp
from dronelab.surrogate.train import TrainConfig, rollout_loss


def _batch(n=6, steps=12):
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(n, steps, 3, generator=g)
    vel = torch.randn(n, steps, 3, generator=g)
    rot = so3_exp(0.3 * torch.randn(n, steps, 3, generator=g))
    omega = torch.randn(n, steps, 3, generator=g)
    u = CF2X.hover_rpm * (1 + 0.05 * torch.randn(n, steps, 4, generator=g))
    return pos, vel, rot, omega, u


# Rodrigues gives proper rotations, also for tiny angles.
def test_so3_exp_orthonormal():
    phi = torch.tensor([[0.0, 0.0, 0.0], [1e-7, 0, 0], [0.3, -1.2, 2.0]])
    R = so3_exp(phi)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(3, 3, 3), atol=1e-6)
    assert torch.allclose(torch.det(R), torch.ones(3), atol=1e-6)
    assert abs(float(rot_angle(R[2], torch.eye(3))) - float(phi[2].norm())) < 1e-5


# The numba inference path (no_grad) gives the same physics step as the torch path.
def test_fast_step_matches_torch():
    pytest.importorskip("numba")
    pos, vel, rot, omega, u = _batch(n=32, steps=1)
    ph = Physics(n_sub=5)
    with torch.no_grad():
        ph.kd.copy_(torch.tensor([1.6, 2.9]))
    res = (0.01 * torch.randn(32, 1, 3), 1e-4 * torch.randn(32, 1, 3))
    for f_res, t_res in ((None, None), res):
        ref = ph.step(pos, vel, rot, omega, u, f_res, t_res)  # grad enabled -> torch
        with torch.no_grad():
            fast = ph.step(pos, vel, rot, omega, u, f_res, t_res)
        for a, b in zip(ref, fast, strict=True):
            assert a.shape == b.shape
            assert torch.allclose(a.detach(), b, atol=1e-5, rtol=1e-5)


# Every model returns (B, H, ...) and is deterministic.
@pytest.mark.parametrize("kind", ["phys", "phys+mlp", "mlp"])
def test_rollout_shapes(kind):
    m = Surrogate(kind, window=4)
    pos, vel, rot, omega, u = _batch()
    out = m.rollout(pos, vel, rot, omega, u[:, :11], warmup=4, horizon=7)
    assert [o.shape for o in out] == [(6, 7, 3), (6, 7, 3), (6, 7, 3, 3), (6, 7, 3)]
    out2 = m.rollout(pos, vel, rot, omega, u[:, :11], warmup=4, horizon=7)
    assert all(torch.equal(a, b) for a, b in zip(out, out2, strict=True))


# Before training, the hybrid model is exactly its physics model (zero output layer).
def test_hybrid_starts_as_physics():
    pos, vel, rot, omega, u = _batch()
    a = Surrogate("phys").rollout(pos, vel, rot, omega, u[:, :10], 4, 6)
    b = Surrogate("phys+mlp", window=2).rollout(pos, vel, rot, omega, u[:, :10], 4, 6)
    assert all(torch.allclose(x, y) for x, y in zip(a, b, strict=True))


# A training step reduces the loss and the gradients reach the physics parameters.
def test_loss_backward_reaches_physics():
    pos, vel, rot, omega, u = _batch(steps=14)
    from dronelab.surrogate.data import Segments

    seg = Segments(pos, vel, rot, omega, u, u)
    m = Surrogate("phys+mlp", window=2)
    loss = rollout_loss(m, seg, warmup=4, horizon=8)
    loss.backward()
    assert m.phys.log_kf.grad is not None and torch.isfinite(m.phys.log_kf.grad)
    assert m.net[-1].weight.grad.abs().sum() > 0
    assert TrainConfig().horizon == 20


# Segments never cross flight boundaries and keep the step order.
def test_segments():
    class F:
        def __init__(self, n, off):
            self.pos = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3) + off
            self.vel = self.omega = self.pos
            self.rot = torch.eye(3).expand(n, 3, 3)
            self.u = self.applied = self.pos[:, :1].repeat(1, 4)

        def __len__(self):
            return len(self.pos)

    seg = make_segments([F(10, 0), F(5, 100), F(3, 200)], length=4, stride=2)
    assert len(seg) == 4 + 1  # starts 0, 2, 4, 6 | 0 | none (flight shorter than a segment)
    assert set(seg.pos[:, 0, 0].tolist()) == {0, 2, 4, 6, 100}
    assert torch.all(seg.pos[:, 1:, 0] - seg.pos[:, :-1, 0] == 1)


# With the simulator's own parameters (ideal world: no drag, no lag, no extra aerodynamics),
# the physics model reproduces a PyBullet flight step by step.
def test_physics_matches_pybullet(tmp_path):
    pytest.importorskip("gym_pybullet_drones")
    from dronelab.core.eval.parallel import EpisodeSpec, run_spec

    path = tmp_path / "f.npz"
    run_spec(EpisodeSpec("lqr", "figure8", 1.0, 3, None, world="ideal", log_path=str(path)))
    f = load_flight(path)
    ph = Physics()
    with torch.no_grad():
        ph.log_tau.fill_(math.log(1e-6))  # no motor lag
        k = slice(10, len(f) - 1)  # away from the start transient
        p, v, r, w = ph.step(f.pos[k], f.vel[k], f.rot[k], f.omega[k], f.u[k])
    dv = (v - f.vel[10:][1:]).norm(dim=1)
    dw = (w - f.omega[10:][1:]).norm(dim=1)
    true_dv = (f.vel[11:] - f.vel[10:-1]).norm(dim=1)
    assert float(dv.mean()) < 0.05 * float(true_dv.mean()) + 1e-4
    assert float(dw.mean()) < 0.01
    assert np.isfinite(float(rot_angle(r, f.rot[11:]).max()))


# A constant extra force/torque (like wind) is recovered from the warm-up steps, and the
# prediction with the estimate follows the disturbed flight.
def test_disturbance_estimate():
    from dronelab.surrogate.evaluate import estimate_disturbance
    from dronelab.surrogate.models import urdf_physics

    m = urdf_physics()
    g = torch.Generator().manual_seed(1)
    n, steps = 3, 30
    u = CF2X.hover_rpm * (1 + 0.02 * torch.randn(n, steps, 4, generator=g))
    pos0 = torch.tensor([0.0, 0.0, 1.0]).expand(n, 1, 3)
    z = torch.zeros(n, 1, 3)
    eye = torch.eye(3).expand(n, 1, 3, 3)
    dist = (torch.tensor([[0.5, -0.3, 0.2]]).expand(n, 3),
            torch.tensor([[2.0, -1.0, 0.5]]).expand(n, 3))
    with torch.no_grad():
        x = m.rollout(pos0, z, eye, z, u, 0, steps - 1, dist)  # "true" disturbed flight
    pos, vel, rot, omega = (torch.cat([a, b], 1) for a, b in zip((pos0, z, eye, z), x, strict=True))
    acc, alpha = estimate_disturbance(m, pos, vel, rot, omega, u, warmup=8)
    assert torch.allclose(acc, dist[0], atol=1e-3) and torch.allclose(alpha, dist[1], atol=1e-2)
    with torch.no_grad():
        p_est = m.rollout(pos, vel, rot, omega, u, 8, 20, (acc, alpha))[0]
        p_none = m.rollout(pos, vel, rot, omega, u, 8, 20)[0]
    err = lambda p: float((p - pos[:, 9:29]).norm(dim=-1).max())  # noqa: E731
    assert err(p_est) < 1e-3 < err(p_none)
