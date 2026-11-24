"""Open-loop rollout errors of every trained model on the test sets (resumable).

    python projects/surrogate/scripts/evaluate.py

Writes ``<results>/eval/<model>__<set>.npz`` (one row per rollout, see
dronelab.surrogate.evaluate), ``flights_<set>.json`` (name, measured max speed, controller,
scenario, family per flight) and ``timing.json`` (batch of 512 x 20 steps per model kind).
T2 is not evaluated separately: it is the subset of T1 flights faster than T2_MAX_SPEED.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from dronelab.surrogate.data import load_flight, load_split, make_segments
from dronelab.surrogate.datasets import SPLIT_DIRS
from dronelab.surrogate.evaluate import WARMUP, evaluate, time_batch
from dronelab.surrogate.models import load_model

TEST_SETS = {  # set -> (split folder key, excluded name tokens)
    "T1": ("T1", ()),
    "T3": ("T3", ()),
    "T4": ("T4", ("_ideal_",)),  # the ideal-world cross-check is another world, not T4
}


def _flights(data_dir: Path, s: str):
    key, excl = TEST_SETS[s]
    return load_split(data_dir / SPLIT_DIRS[key], exclude=excl)


def _job(args) -> str:
    model_path, s, data_dir, out = args
    torch.set_num_threads(1)
    dst = out / f"{model_path.stem}__{s}.npz"
    if dst.exists():
        return f"{dst.name} (exists)"
    model, _ = load_model(model_path)
    res = evaluate(model, _flights(data_dir, s))
    np.savez_compressed(dst, **res)
    return f"{dst.name}: {len(res['e_pos'])} rollouts, {res['skipped']} short flights skipped"


@torch.no_grad()
def _examples(models_dir: Path, data_dir: Path, out: Path, horizon: int = 48) -> None:
    """Predicted 1 s snippets every 0.5 s along one fast T3 figure-eight (for a figure)."""
    folder = data_dir / SPLIT_DIRS["T3"]
    files = sorted(folder.glob("A_lqi_nominal_figure8_2_1000.npz")) or sorted(folder.glob("*.npz"))
    if not files:
        return
    f = load_flight(files[0])
    seg = make_segments([f], WARMUP + horizon + 1, 24)
    ex = {"name": np.array(f.name), "pos": f.pos.numpy(), "horizon_s": np.array(horizon / 48)}
    for name in ("phys_full", "phys_mlp_full_s0", "mlp_full_s0"):
        p = models_dir / f"{name}.pt"
        if p.exists():
            model, _ = load_model(p)
            pred = model.rollout(seg.pos, seg.vel, seg.rot, seg.omega,
                                 seg.u[:, :WARMUP + horizon], WARMUP, horizon)[0]
            ex[name] = torch.cat([seg.pos[:, WARMUP:WARMUP + 1], pred], 1).numpy()
    np.savez_compressed(out / "examples.npz", **ex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="projects/surrogate/results/models")
    ap.add_argument("--data", default="data/surrogate")
    ap.add_argument("--out", default="projects/surrogate/results/eval")
    ap.add_argument("--sets", nargs="+", default=list(TEST_SETS))
    ap.add_argument("--processes", type=int, default=2)
    args = ap.parse_args()
    out, data_dir = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)

    for s in args.sets:
        info = []
        for f in _flights(data_dir, s):
            m = f.meta
            info.append({"name": f.name, "max_speed": round(f.max_speed, 3), "steps": len(f),
                         "controller": m.get("controller"), "family": m.get("family"),
                         "scenario": m.get("scenario", "nominal"), "crashed": m.get("crashed")})
        (out / f"flights_{s}.json").write_text(json.dumps(info))

    models = sorted(Path(args.models).glob("*.pt"))
    jobs = [(m, s, data_dir, out) for m in models for s in args.sets]
    with Pool(args.processes) as pool:
        for msg in pool.imap_unordered(_job, jobs):
            print(msg, flush=True)

    torch.set_num_threads(1)
    _examples(Path(args.models), data_dir, out)
    timing = {}
    for m in models:
        model, _ = load_model(m)
        kind = model.kind
        if kind not in timing:
            timing[kind] = {"ms_512x20": round(time_batch(model), 2), "threads": 1,
                            "params": sum(p.numel() for p in model.parameters())}
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
