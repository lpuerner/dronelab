"""Fly several controllers on the same course and plot it in 3D plus the three side views.

    python projects/control/scripts/plot_course.py --course gates --speed 1.5 --seed 1000

Gates are drawn as their 0.5 x 0.5 m frames, so the views show whether a controller
flew through the opening (top view: lateral miss, front/side view: height miss).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from dronelab.control import make_controller  # noqa: E402
from dronelab.core.backends.pybullet_backend import PyBulletBackend  # noqa: E402
from dronelab.core.config import DisturbanceConfig, SimConfig  # noqa: E402
from dronelab.core.eval import compute_metrics, run_episode  # noqa: E402
from dronelab.core.tasks import make_course  # noqa: E402
from dronelab.core.world import DEFAULT_WORLD, WORLDS, resolve  # noqa: E402

COLORS = {"dsl_pid": "#2a78d6", "pid": "#eb6834", "lqr": "#1baf7a", "lqi": "#eda100"}
LABELS = {"dsl_pid": "DSL-PID", "pid": "PID", "lqr": "LQR", "lqi": "LQI"}
INK, MUTED, GRID, GATE = "#1f1f1e", "#6b6a66", "#e4e3df", "#3a3a38"
VIEWS = [((0, 1), "top view (x–y)"), ((0, 2), "front view (x–z)"), ((1, 2), "side view (y–z)")]
AXN = "xyz"


def gate_frame(g) -> np.ndarray:
    """Closed polygon (5 x 3) of the gate opening."""
    lat, up = g.lateral * g.width / 2, np.array([0.0, 0.0, g.height / 2])
    c = g.center
    return np.array([c - lat - up, c + lat - up, c + lat + up, c - lat + up, c - lat - up])


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default="gates")
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--controllers", nargs="+", default=list(COLORS))
    ap.add_argument("--params-dir", default="projects/control/results/tuning_real")
    ap.add_argument("--world", default=DEFAULT_WORLD, choices=list(WORLDS))
    ap.add_argument("--out", default="projects/control/results/figures")
    args = ap.parse_args()

    course = make_course(args.course, args.speed, args.seed)
    trajs, mets = {}, {}
    physics, dist = resolve(args.world)
    with PyBulletBackend(SimConfig(physics=physics), DisturbanceConfig(**dist)) as be:
        for c in args.controllers:
            f = Path(args.params_dir) / c / "best_params.yaml"
            gains = yaml.safe_load(f.read_text()) if f.exists() else None
            trajs[c] = run_episode(be, make_controller(c, be.params, be.dt, gains), course,
                                   seed=args.seed)
            mets[c] = compute_metrics(trajs[c])
    ref = np.array([course.reference(t).pos for t in np.arange(0, course.duration, 0.02)])
    frames = [gate_frame(g) for g in course.gates]

    fig = plt.figure(figsize=(11, 8.2))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax3.plot(*ref.T, color=MUTED, lw=1, ls="--")
    for fr in frames:
        ax3.plot(*fr.T, color=GATE, lw=1.5)
    for c, tr in trajs.items():
        ax3.plot(*tr.pos.T, color=COLORS[c], lw=1.0)
    # not true scale: lateral and vertical motion would be invisible next to the 8 m length
    ax3.set_box_aspect((2.2, 1.0, 0.7))
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax3.set_xlabel("x [m]", color=MUTED, fontsize=8)
    ax3.set_ylabel("y [m]", color=MUTED, fontsize=8)
    ax3.set_zlabel("z [m]", color=MUTED, fontsize=8)
    ax3.tick_params(colors=MUTED, labelsize=7)
    ax3.view_init(elev=28, azim=-60)
    ax3.set_title("3D", color=INK, fontsize=10, loc="left")

    for k, ((i, j), title) in enumerate(VIEWS, start=2):
        ax = fig.add_subplot(2, 2, k)
        ax.plot(ref[:, i], ref[:, j], color=MUTED, lw=1, ls="--", label="reference")
        for n, fr in enumerate(frames):
            ax.plot(fr[:, i], fr[:, j], color=GATE, lw=1.5, label="gate" if n == 0 else None)
        for c, tr in trajs.items():
            m = mets[c]
            lab = (f"{LABELS[c]}  RMSE {100 * m['rmse_pos']:.1f} cm, "
                   f"gates {int(m['gates_passed'])}/{int(m['gates_total'])}")
            ax.plot(tr.pos[:, i], tr.pos[:, j], color=COLORS[c], lw=1.0, label=lab)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"{AXN[i]} [m]", color=MUTED, fontsize=9)
        ax.set_ylabel(f"{AXN[j]} [m]", color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        style(ax)
        if k == 2:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(f"{args.course} @ {args.speed:g} m/s, seed {args.seed}", color=INK,
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.course}_{args.speed:g}ms_seed{args.seed}.png"
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
