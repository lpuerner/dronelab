"""Static plots of a flown trajectory."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dronelab.core.eval.runner import Trajectory  # noqa: E402


def plot_trajectory(traj: Trajectory, path: str | Path, title: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*traj.ref_pos.T, "--", color="0.5", label="reference")
    ax.plot(*traj.pos.T, color="tab:blue", label="flown")
    ax.scatter(*traj.pos[0], color="tab:green", s=20, label="start")
    if traj.crashed:
        ax.scatter(*traj.pos[-1], color="tab:red", marker="x", s=60, label="crash")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")

    err = np.linalg.norm(traj.pos - traj.ref_pos, axis=1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(traj.t, err)
    ax2.set_ylabel("position error [m]")
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)
    ax3.plot(traj.t, traj.rpm, lw=0.8)
    ax3.axhline(traj.hover_rpm, color="0.5", ls=":", label="hover rpm")
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("motor [rpm]")
    ax3.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
