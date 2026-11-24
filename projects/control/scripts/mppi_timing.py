"""R4.0: rollout time of the surrogate models for MPPI and the accuracy cost of fewer substeps.

    python projects/control/scripts/mppi_timing.py

Time: batch of K sequences x 20 steps (1 thread) for K in {128, 256, 512}, each model and number
of physics substeps. Accuracy: median per-flight position error at 0.42 s on T1 for each number
of substeps. Results -> projects/control/results/mppi/timing.{json,md}
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from dronelab.surrogate.data import load_split
from dronelab.surrogate.datasets import SPLIT_DIRS
from dronelab.surrogate.evaluate import HORIZONS, evaluate, per_flight, time_batch
from dronelab.surrogate.models import load_model, urdf_physics

MODELS = "projects/surrogate/results/models"
OUT = Path("projects/control/results/mppi")
KS = (128, 256, 512)
SUBSTEPS = (5, 3, 2)
H_MPC = HORIZONS.index(20)  # 0.42 s


def _model(name: str, n_sub: int):
    model = urdf_physics() if name == "urdf" else load_model(f"{MODELS}/{name}.pt")[0]
    if model.phys is not None:
        model.phys.n_sub = n_sub
    return model


def _accuracy(args: tuple[str, int]) -> dict:
    name, n_sub = args
    torch.set_num_threads(1)
    res = evaluate(_model(name, n_sub), load_split(Path("data/surrogate") / SPLIT_DIRS["T1"]))
    e = per_flight(res, "e_pos")[:, H_MPC] * 100
    return {"model": name, "n_sub": n_sub, "e_pos_cm_median": round(float(np.median(e)), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    timing = []
    for name in ("urdf", "phys_mlp_full_s0", "mlp_full_s0"):
        for n_sub in SUBSTEPS if name != "mlp_full_s0" else (0,):
            model = _model(name, n_sub)
            for k in KS:
                ms = time_batch(model, n=k, repeats=7)
                timing.append({"model": name, "n_sub": n_sub, "K": k, "ms": round(ms, 1)})
                print(timing[-1], flush=True)

    jobs = [(m, n) for m in ("urdf", "phys_mlp_full_s0") for n in SUBSTEPS]
    with Pool(2) as pool:
        acc = []
        for row in pool.imap_unordered(_accuracy, jobs):
            acc.append(row)
            print(row, flush=True)
    acc.sort(key=lambda r: (r["model"], -r["n_sub"]))

    (OUT / "timing.json").write_text(json.dumps({"timing": timing, "accuracy": acc}, indent=1))
    lines = ["Rollout time [ms], K sequences x 20 steps, 1 thread (best of 7)\n",
             "| model | substeps | " + " | ".join(f"K={k}" for k in KS) + " |",
             "|---|---|" + "---:|" * len(KS)]
    for name, n_sub in dict.fromkeys((r["model"], r["n_sub"]) for r in timing):
        ms = [str(r["ms"]) for r in timing if (r["model"], r["n_sub"]) == (name, n_sub)]
        lines.append(f"| {name} | {n_sub or '-'} | " + " | ".join(ms) + " |")
    lines += ["\nT1 median position error at 0.42 s [cm]\n", "| model | substeps | error |",
              "|---|---|---:|"]
    lines += [f"| {r['model']} | {r['n_sub']} | {r['e_pos_cm_median']} |" for r in acc]
    (OUT / "timing.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
