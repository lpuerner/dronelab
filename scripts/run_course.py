"""Fly one course with one controller, print metrics, save a plot.

    python scripts/run_course.py --config configs/hover_pid.yaml
    python scripts/run_course.py --config configs/hover_pid.yaml --course figure8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dronebench.backends import make_backend
from dronebench.config import load_config
from dronebench.controllers import make_controller
from dronebench.eval import compute_metrics, run_episode
from dronebench.eval.plots import plot_trajectory
from dronebench.tasks import make_course


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--course", default=None, help="override course from config")
    ap.add_argument("--controller", default=None, help="override controller from config")
    ap.add_argument("--gui", action="store_true", help="open the PyBullet window")
    ap.add_argument("--out", default="results/runs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.course:
        cfg.course = args.course
    if args.controller:
        cfg.controller = args.controller
    if args.gui:
        cfg.sim.gui = True

    course = make_course(cfg.course)
    with make_backend("pybullet", sim=cfg.sim, disturbances=cfg.disturbances) as backend:
        ctrl = make_controller(cfg.controller, ctrl_dt=backend.dt)
        traj = run_episode(backend, ctrl, course, arena=cfg.arena, seed=cfg.seed)

    metrics = compute_metrics(traj)
    print(json.dumps(metrics, indent=2))
    tag = Path(args.config).stem if args.config else "default"
    name = f"{cfg.course}_{cfg.controller}_{tag}"
    png = plot_trajectory(traj, Path(args.out) / f"{name}.png", title=name)
    print(f"plot -> {png}")


if __name__ == "__main__":
    main()
