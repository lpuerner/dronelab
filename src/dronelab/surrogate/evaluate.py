"""Open-loop rollout evaluation on test flights.

From many start points of every test flight, a model predicts up to ``HORIZONS[-1]`` steps
with the recorded commands. Errors are stored per rollout, together with the conditions at
the start (speed, acceleration, height), so they can be grouped afterwards.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from dronelab.surrogate.data import DT, Flight, make_segments
from dronelab.surrogate.models import Surrogate, rot_angle

HORIZONS = (1, 10, 20, 48, 96)  # steps: 21 ms, 0.21 s, 0.42 s (MPC horizon), 1 s, 2 s
WARMUP = 8
EVERY = 12  # steps between start points (0.25 s)


@torch.no_grad()
def evaluate(model: Surrogate, flights: list[Flight], batch: int = 2048) -> dict:
    """Per-rollout errors; flights shorter than warmup + 2 s are skipped (counted)."""
    model.eval()
    hmax = HORIZONS[-1]
    length = WARMUP + hmax + 1
    rows: dict[str, list] = {k: [] for k in ("flight", "e_pos", "e_vel", "e_att", "e_omega",
                                             "speed", "acc", "z")}
    skipped = 0
    hi = np.array(HORIZONS)
    for fi, f in enumerate(flights):
        if len(f) < length:
            skipped += 1
            continue
        seg = make_segments([f], length, EVERY)
        for i in range(0, len(seg), batch):
            s = seg[slice(i, i + batch)]
            p, v, r, w = model.rollout(s.pos, s.vel, s.rot, s.omega, s.u[:, :WARMUP + hmax],
                                       WARMUP, hmax)
            sl = slice(WARMUP + 1, WARMUP + hmax + 1)
            e_pos = (p - s.pos[:, sl]).norm(dim=-1)[:, hi - 1]
            e_vel = (v - s.vel[:, sl]).norm(dim=-1)[:, hi - 1]
            e_att = torch.rad2deg(rot_angle(r, s.rot[:, sl]))[:, hi - 1]
            e_om = (w - s.omega[:, sl]).norm(dim=-1)[:, hi - 1]
            v0 = s.vel[:, WARMUP]
            acc0 = (s.vel[:, WARMUP + 1] - s.vel[:, WARMUP - 1]).norm(dim=-1) / (2 * DT)
            n = len(s)
            rows["flight"].append(np.full(n, fi))
            rows["e_pos"].append(e_pos.numpy())
            rows["e_vel"].append(e_vel.numpy())
            rows["e_att"].append(e_att.numpy())
            rows["e_omega"].append(e_om.numpy())
            rows["speed"].append(v0.norm(dim=-1).numpy())
            rows["acc"].append(acc0.numpy())
            rows["z"].append(s.pos[:, WARMUP, 2].numpy())
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}
    out["names"] = np.array([f.name for f in flights])
    out["skipped"] = skipped
    return out


def per_flight(res: dict, key: str) -> np.ndarray:
    """Mean error of each flight (flights x horizons) - the unit for medians and tests."""
    fl = res["flight"]
    ids = np.unique(fl)
    return np.stack([res[key][fl == i].mean(0) for i in ids])


@torch.no_grad()
def time_batch(model: Surrogate, n: int = 512, horizon: int = 20, repeats: int = 5) -> float:
    """Wall time [ms] to roll out ``n`` command sequences of ``horizon`` steps (warm state)."""
    model.eval()
    eye = torch.eye(3).expand(n, WARMUP + 1, 3, 3).contiguous()
    pos = torch.zeros(n, WARMUP + 1, 3)
    pos[..., 2] = 1.0
    z3 = torch.zeros(n, WARMUP + 1, 3)
    u = torch.full((n, WARMUP + horizon, 4), model.hover)
    model.rollout(pos, z3, eye, z3, u, WARMUP, horizon)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.rollout(pos, z3, eye, z3, u, WARMUP, horizon)
        best = min(best, time.perf_counter() - t0)
    return 1000 * best
