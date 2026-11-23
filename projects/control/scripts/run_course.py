"""Fly one course with one controller, print metrics, save a plot.

    python projects/control/scripts/run_course.py \
        --config projects/control/configs/hover_pid.yaml [--course figure8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dronelab.control import make_controller
from dronelab.core.backends import make_backend
from dronelab.core.config import load_config
from dronelab.core.eval import compute_metrics, run_episode
from dronelab.core.eval.plots import plot_trajectory
from dronelab.core.tasks import make_course


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--course", default=None, help="override course from config")
    ap.add_argument("--controller", default=None, help="override controller from config")
    ap.add_argument("--speed", type=float, default=None, help="override path speed [m/s]")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    ap.add_argument("--gui", action="store_true", help="open the PyBullet window")
    ap.add_argument("--out", default="projects/control/results/runs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.course:
        cfg.course = args.course
    if args.controller:
        cfg.controller = args.controller
    if args.speed is not None:
        cfg.speed = args.speed
    if args.seed is not None:
        cfg.seed = args.seed
    if args.gui:
        cfg.sim.gui = True

    gains = None
    if cfg.controller_params:
        gains = yaml.safe_load(Path(cfg.controller_params).read_text())
    course = make_course(cfg.course, cfg.speed, cfg.seed)
    if not course.feasible:
        print(f"warning: course not feasible at {cfg.speed} m/s (flown anyway)")
    with make_backend("pybullet", sim=cfg.sim, disturbances=cfg.disturbances) as backend:
        ctrl = make_controller(cfg.controller, backend.params, backend.dt, gains)
        traj = run_episode(backend, ctrl, course, arena=cfg.arena, seed=cfg.seed,
                           init_noise=cfg.init_noise)

    metrics = compute_metrics(traj)
    print(json.dumps(metrics, indent=2))
    tag = Path(args.config).stem if args.config else "default"
    name = f"{cfg.course}_{cfg.speed:g}ms_{cfg.controller}_{tag}"
    png = plot_trajectory(traj, Path(args.out) / f"{name}.png", title=name)
    print(f"plot -> {png}")


if __name__ == "__main__":
    main()
