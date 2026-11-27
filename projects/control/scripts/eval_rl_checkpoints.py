"""R3: validate the saved PPO checkpoints and pick the best one per run.

    python projects/control/scripts/eval_rl_checkpoints.py --every 200000 [--runs final_s0 ...]

Each checkpoint (``data/rl_checkpoints/<cfg>/<run>/model_<n>_steps.zip``) is flown on the 40
validation episodes of the tuning protocol, as in ``train_rl.py``. Rows go to
``results/rl/<cfg>/checkpoints.csv`` (resumable). Per run, the best checkpoint is copied to
``<run>/best_model.zip`` + ``best_vecnormalize.pkl`` (``best_checkpoint.json``). The best run
of each group (``final_*`` -> ``tuning_real/ppo``, ``dr_*`` -> ``tuning_real/ppo_dr``,
``abl_<name>_*`` -> ``tuning_real/ppo_abl_<name>``) becomes the benchmark parameter set.
Figure ``j_over_steps.png``: validation J over training steps, LQI / INDI for reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import yaml

OUT = Path("projects/control/results/rl")
CKPT = Path("data/rl_checkpoints")
PARAMS = Path("projects/control/results/tuning_real")
FIELDS = ["run", "steps", "J", "rmse", "smooth_rel", "crash_rate", "eval_s"]


def _ckpt(root: Path, n: int) -> tuple[Path, Path]:
    return root / f"model_{n}_steps.zip", root / f"model_vecnormalize_{n}_steps.pkl"


def evaluate(cfg: dict, runs: list[str], every: int, table: Path) -> None:
    from dronelab.control.tuning.protocol import objective_value, validation_episodes
    from dronelab.core.eval.parallel import EpisodePool

    done = set()
    if table.exists():
        with open(table) as f:
            done = {(r["run"], int(r["steps"])) for r in csv.DictReader(f)}
    new = not table.exists()
    with open(table, "a", newline="") as f, EpisodePool(2) as pool:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for run in runs:
            root = CKPT / cfg["name"] / run
            for n in sorted(int(m.stem.split("_")[1]) for m in root.glob("model_*_steps.zip")):
                if n % every or (run, n) in done:
                    continue
                m, vn = _ckpt(root, n)
                tic = time.time()
                o = objective_value(pool.map(validation_episodes(
                    "ppo", {"model": str(m), "vecnormalize": str(vn)})))
                w.writerow({"run": run, "steps": n, **{k: o[k] for k in FIELDS[2:6]},
                            "eval_s": time.time() - tic})
                f.flush()
                print(f"[{run}] {n}: J {o['J']:.3f} crash {o['crash_rate']:.2f}", flush=True)


def select(cfg: dict, rows: list[dict]) -> None:
    best = {}
    for run in {r["run"] for r in rows}:
        b = min((r for r in rows if r["run"] == run), key=lambda r: float(r["J"]))
        d = OUT / cfg["name"] / run
        m, vn = _ckpt(CKPT / cfg["name"] / run, int(b["steps"]))
        if d.exists() and m.exists():
            shutil.copy(m, d / "best_model.zip")
            shutil.copy(vn, d / "best_vecnormalize.pkl")
            (d / "best_checkpoint.json").write_text(json.dumps(b, indent=2))
            best[run] = b
            print(f"[{run}] best: {b['steps']} steps, J {float(b['J']):.3f}", flush=True)
    groups = defaultdict(list)
    for run in best:
        name = ("ppo" if run.startswith("final_") else "ppo_dr" if run.startswith("dr_")
                else "ppo_" + run.rsplit("_s", 1)[0] if run.startswith("abl_") else None)
        if name:
            groups[name].append(run)
    for name, rr in groups.items():
        run = min(rr, key=lambda r: float(best[r]["J"]))
        d = PARAMS / name
        d.mkdir(parents=True, exist_ok=True)
        src = OUT / cfg["name"] / run
        (d / "best_params.yaml").write_text(yaml.safe_dump({
            "model": (src / "best_model.zip").as_posix(),
            "vecnormalize": (src / "best_vecnormalize.pkl").as_posix(),
            "run": run, "steps": int(best[run]["steps"]), "J": float(best[run]["J"])}))
        print(f"{name} <- {run} @ {best[run]['steps']}", flush=True)


def plot(cfg: dict, rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dronelab.core.eval.plots import save_captioned

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cmap = plt.get_cmap("tab10")
    for i, run in enumerate(sorted({r["run"] for r in rows})):
        rr = sorted((int(r["steps"]), float(r["J"]), float(r["crash_rate"]))
                    for r in rows if r["run"] == run)
        x = [s / 1e6 for s, _, _ in rr]
        ax.plot(x, [j for _, j, _ in rr], marker="o", ms=3.5, lw=1.2, color=cmap(i), label=run)
        cx = [s / 1e6 for s, _, c in rr if c > 0]
        ax.scatter(cx, [j for _, j, c in rr if c > 0], s=40, facecolors="none",
                   edgecolors=cmap(i), lw=1.0)
    for c, ls in (("lqi", "--"), ("indi", ":")):
        f = PARAMS / c / "summary.json"
        if f.exists():
            j = json.loads(f.read_text())["validation_selected"]["J"]
            ax.axhline(j, color="#6b6a66", ls=ls, lw=1.0, label=f"{c.upper()} (tuned) {j:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("training steps [M]")
    ax.set_ylabel("validation J (lower is better)")
    ax.grid(True, color="#e4e3df", lw=0.8)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    save_captioned(fig, path, "R3: validation J over training",
                   "Test: every saved PPO checkpoint flown on the 40 validation episodes of the "
                   "tuning protocol (seeds 100-139), same objective J as for PID / LQI / INDI. "
                   "Shown: J per checkpoint (log scale); open circles = checkpoints with crashes; "
                   "dashed / dotted = tuned LQI / INDI.")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/control/configs/rl/r3.yaml")
    ap.add_argument("--every", type=int, default=200_000, help="checkpoint stride [steps]")
    ap.add_argument("--runs", nargs="*", default=None, help="default: final_s<seeds_final>")
    ap.add_argument("--no-eval", action="store_true", help="only select + plot")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    table = OUT / cfg["name"] / "checkpoints.csv"
    if not args.no_eval:
        evaluate(cfg, args.runs or [f"final_s{s}" for s in cfg["seeds_final"]], args.every,
                 table)
    with open(table) as f:
        rows = list(csv.DictReader(f))
    select(cfg, rows)
    plot(cfg, rows, OUT / cfg["name"] / "j_over_steps.png")


if __name__ == "__main__":
    main()
