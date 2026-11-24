"""Training with a multi-step rollout loss and early stopping on validation rollouts."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import torch

from dronelab.surrogate.data import Flight, Segments, make_segments
from dronelab.surrogate.models import Surrogate, rot_error

# error scales of the loss (roughly the size of a "noticeable" error per quantity)
LOSS_SCALES = {"pos": 0.02, "vel": 0.05, "att": 0.02, "omega": 0.2}  # m, m/s, rad, rad/s


@dataclass
class TrainConfig:
    warmup: int = 8  # steps that fill the window and settle the motor model
    horizon: int = 20  # rollout length of the loss (20 steps = 0.42 s, the MPC horizon)
    horizon_start: int = 5  # curriculum: the loss horizon grows linearly from here ...
    ramp_epochs: int = 3  # ... to ``horizon`` over these epochs (unstable early rollouts)
    stride: int = 2  # between training segments
    batch: int = 512
    lr: float = 1e-3  # network
    lr_phys: float = 3e-3  # physics parameters
    epochs: int = 40
    steps_per_epoch: int = 150
    patience: int = 5
    val_segments: int = 3000
    seed: int = 0


def rollout_loss(model: Surrogate, seg: Segments, warmup: int, horizon: int) -> torch.Tensor:
    p, v, r, w = model.rollout(seg.pos, seg.vel, seg.rot, seg.omega,
                               seg.u[:, :warmup + horizon], warmup, horizon)
    sl = slice(warmup + 1, warmup + horizon + 1)
    s = LOSS_SCALES
    err = (((p - seg.pos[:, sl]) / s["pos"]) ** 2).sum(-1) \
        + (((v - seg.vel[:, sl]) / s["vel"]) ** 2).sum(-1) \
        + ((rot_error(r, seg.rot[:, sl]) / s["att"]) ** 2).sum(-1) \
        + (((w - seg.omega[:, sl]) / s["omega"]) ** 2).sum(-1)
    return err.mean()


@torch.no_grad()
def evaluate_loss(model: Surrogate, seg: Segments, cfg: TrainConfig, batch: int = 1500) -> float:
    model.eval()
    tot = 0.0
    for i in range(0, len(seg), batch):
        part = seg[slice(i, i + batch)]
        tot += float(rollout_loss(model, part, cfg.warmup, cfg.horizon)) * len(part)
    model.train()
    return tot / len(seg)


def train(model: Surrogate, train_flights: list[Flight], val_flights: list[Flight],
          cfg: TrainConfig, log=print) -> dict:
    """Train in place; returns the history. The best validation state is restored."""
    torch.manual_seed(cfg.seed)
    length = cfg.warmup + cfg.horizon + 1
    tr = make_segments(train_flights, length, cfg.stride)
    va = make_segments(val_flights, length, length)
    g = torch.Generator().manual_seed(1234)  # same validation subset for every model
    va = va[torch.randperm(len(va), generator=g)[:cfg.val_segments]]
    model.fit_normalizer(tr)

    groups = []
    if model.phys is not None:
        groups.append({"params": list(model.phys.parameters()), "lr": cfg.lr_phys})
    if model.net is not None:
        groups.append({"params": list(model.net.parameters()), "lr": cfg.lr})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs * cfg.steps_per_epoch)

    best, best_state, bad = float("inf"), None, 0
    hist = {"epoch": [], "train": [], "val": [], "time_s": []}
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(cfg.seed)
    for ep in range(cfg.epochs):
        h = cfg.horizon if ep >= cfg.ramp_epochs else round(
            cfg.horizon_start + (cfg.horizon - cfg.horizon_start) * ep / cfg.ramp_epochs)
        run = 0.0
        for _ in range(cfg.steps_per_epoch):
            idx = torch.randint(len(tr), (cfg.batch,), generator=gen)
            loss = rollout_loss(model, tr[idx], cfg.warmup, h)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run += loss.item()
        val = evaluate_loss(model, va, cfg)
        hist["epoch"].append(ep)
        hist["train"].append(run / cfg.steps_per_epoch)
        hist["val"].append(val)
        hist["time_s"].append(round(time.perf_counter() - t0, 1))
        log(f"  epoch {ep:2d}  train {hist['train'][-1]:9.3f}  val {val:9.3f}  "
            f"({hist['time_s'][-1]:.0f} s)")
        if ep < cfg.ramp_epochs:  # early stopping only on the full horizon
            best_state = copy.deepcopy(model.state_dict())
        elif val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return {"history": hist, "best_val": best, "config": asdict(cfg),
            "n_train_segments": len(tr), "n_val_segments": len(va),
            "train_time_s": round(time.perf_counter() - t0, 1)}
