"""Train the surrogate models (resumable: finished models are skipped).

    python projects/surrogate/scripts/train.py                 # physics fit, search, finals
    python projects/surrogate/scripts/train.py --stage phys     # only one stage

Stages
1. ``phys``: fit the 5 physics parameters (one model per data variant).
2. ``search``: history window of both networks (seed 0, full data, shorter training).
3. ``final``: ``phys+mlp`` and ``mlp`` with the chosen window, every seed and data variant.
   The hybrid model starts from the fitted physics parameters of its variant.

Output per model in ``out_dir``: ``<name>.pt`` (weights + config) and ``<name>.json``
(history, identified parameters, data used).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import torch
import yaml

from dronelab.surrogate.data import load_split
from dronelab.surrogate.datasets import SPLIT_DIRS
from dronelab.surrogate.models import Surrogate, load_model, save_model, urdf_physics
from dronelab.surrogate.train import TrainConfig, train


def _flights(cfg: dict, split: str, v_max: float | None):
    fl = load_split(Path(cfg["data_dir"]) / SPLIT_DIRS[split])
    return fl if v_max is None else [f for f in fl if f.max_speed <= v_max]


def run_job(job: dict) -> dict:
    torch.set_num_threads(1)
    cfg, name = job["cfg"], job["name"]
    out = Path(cfg["out_dir"])
    if (out / f"{name}.pt").exists():
        return json.loads((out / f"{name}.json").read_text())
    v_max = cfg["variants"][job["variant"]]
    tr, va = _flights(cfg, "train", v_max), _flights(cfg, "val", v_max)
    if not tr or not va:
        raise SystemExit(f"{name}: no train or val flights (v_max={v_max})")
    tc = TrainConfig(**cfg["train"])
    if job["kind"] == "phys":
        tc = replace(tc, **cfg["phys"])
    if "epochs" in job:
        tc = replace(tc, epochs=job["epochs"])
    tc = replace(tc, seed=job["seed"])
    model = Surrogate(job["kind"], window=job["window"], **cfg["net"])
    if job["kind"] == "phys+mlp":
        phys, _ = load_model(out / f"phys_{job['variant']}.pt")
        model.phys.load_state_dict(phys.phys.state_dict())
    log_path = out / f"{name}.log"
    with log_path.open("w") as lf:
        def log(msg):
            lf.write(msg + "\n")
            lf.flush()

        log(f"{name}: {len(tr)} train / {len(va)} val flights")
        info = train(model, tr, va, tc, log)
    info.update(name=name, kind=job["kind"], window=job["window"], variant=job["variant"],
                seed=job["seed"], n_train_flights=len(tr), n_val_flights=len(va),
                train_steps=int(sum(len(f) for f in tr)))
    if model.phys is not None:
        info["identified"] = model.phys.identified()
    save_model(model, out / f"{name}.pt", {"info": info})
    (out / f"{name}.json").write_text(json.dumps(info, indent=1))
    return info


def _run(jobs: list[dict], processes: int) -> list[dict]:
    if processes <= 1:
        return [run_job(j) for j in jobs]
    with Pool(processes) as pool:
        return pool.map(run_job, jobs, chunksize=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/surrogate/configs/train.yaml")
    ap.add_argument("--stage", nargs="+", default=["phys", "search", "final"])
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["out_dir"])
    (out / "search").mkdir(parents=True, exist_ok=True)
    nproc = cfg["processes"]

    if "phys" in args.stage:
        save_model(urdf_physics(), out / "phys_urdf.pt", {"info": {"name": "phys_urdf"}})
        jobs = [dict(cfg=cfg, name=f"phys_{v}", kind="phys", window=1, variant=v, seed=0)
                for v in cfg["variants"]]
        for info in _run(jobs, nproc):
            print(info["name"], {k: f"{x:.3e}" for k, x in info["identified"].items()},
                  f"val {info['best_val']:.2f}  {info['train_time_s']:.0f} s", flush=True)

    search_file = out / "search" / "search.json"
    if "search" in args.stage:
        scfg = {**cfg, "out_dir": str(out / "search")}
        jobs = [dict(cfg=scfg, name=f"{k.replace('+', '_')}_w{w}", kind=k, window=w,
                     variant="full", seed=0, epochs=cfg["search"]["epochs"])
                for k in ("phys+mlp", "mlp") for w in cfg["search"]["windows"]]
        # the hybrid models need the fitted physics in the search folder too
        for v in cfg["variants"]:
            for ext in ("pt", "json"):
                src = out / f"phys_{v}.{ext}"
                if src.exists() and not (out / "search" / src.name).exists():
                    (out / "search" / src.name).write_bytes(src.read_bytes())
        res = _run(jobs, nproc)
        best = {}
        for k in ("phys+mlp", "mlp"):
            cand = [r for r in res if r["kind"] == k]
            best[k] = min(cand, key=lambda r: r["best_val"])["window"]
        table = {r["name"]: round(r["best_val"], 3) for r in res}
        search_file.write_text(json.dumps({"val_loss": table, "best_window": best}, indent=1))
        print(json.dumps({"val_loss": table, "best_window": best}, indent=1), flush=True)

    if "final" in args.stage:
        best = json.loads(search_file.read_text())["best_window"]
        jobs = [dict(cfg=cfg, name=f"{k.replace('+', '_')}_{v}_s{s}", kind=k, window=best[k],
                     variant=v, seed=s)
                for k in ("phys+mlp", "mlp") for v in cfg["variants"] for s in cfg["seeds"]]
        for info in _run(jobs, nproc):
            print(info["name"], f"val {info['best_val']:.2f}  {info['train_time_s']:.0f} s",
                  flush=True)


if __name__ == "__main__":
    main()
