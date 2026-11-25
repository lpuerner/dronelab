"""Diagnosis plots: one hover and one figure8 tuning episode, LQI (tuned) vs MPPI hybrid.

    python projects/control/scripts/mppi_diag.py [--gains defaults|tuned] [--out DIR]

Panels: z or position error, thrust, motor-1 command, |delta rpm| per step (the smoothness
metric). Used in R4.3 to find the slow take-off and the command jitter of plain MPPI.
Default output: projects/control/results/mppi/diag_<family>.png
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

TUNED = Path("projects/control/results/tuning_real")
FAMILIES = ("hover", "figure8")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dronelab.control.tuning.protocol import DEFAULTS, tuning_episodes
    from dronelab.core.eval.parallel import EpisodePool
    from dronelab.core.tasks import make_course
    from dronelab.surrogate.data import CF2X

    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", choices=["defaults", "tuned"], default="tuned")
    ap.add_argument("--out", default="projects/control/results/mppi")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())

    mppi = (DEFAULTS["mppi_hybrid"] if args.gains == "defaults"
            else yaml.safe_load((TUNED / "mppi_hybrid/best_params.yaml").read_text()))
    runs = {"lqi": yaml.safe_load((TUNED / "lqi/best_params.yaml").read_text()),
            "mppi_hybrid": mppi}
    specs = []
    for name, gains in runs.items():
        for fam in FAMILIES:  # first tuning episode of the family
            s = next(s for s in tuning_episodes(name, gains) if s.family == fam)
            specs.append(replace(s, log_path=str(tmp / f"{name}_{fam}.npz")))
    with EpisodePool(2) as pool:
        for r in pool.map(specs):
            print(f"{r['controller']:12s} {r['family']:8s} rmse {100 * r['rmse_pos']:.1f} cm  "
                  f"smooth {r['smoothness']:.5f}  crashed {r['crashed']}")

    for fam in FAMILIES:
        fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        for color, s in zip(("C0", "C3"), [s for s in specs if s.family == fam], strict=True):
            d = np.load(s.log_path)
            t, name = d["t"], s.controller
            course = make_course(fam, s.speed, s.seed)
            ref = np.array([course.reference(ti).pos for ti in t])
            if fam == "hover":
                axs[0].plot(t, d["pos"][:, 2], color, label=f"{name} z")
                if name == "lqi":
                    axs[0].plot(t, ref[:, 2], "k--", label="reference z")
                axs[0].set_ylabel("z [m]")
            else:
                axs[0].plot(t, 100 * np.linalg.norm(d["pos"] - ref, axis=1), color, label=name)
                axs[0].set_ylabel("|position error| [cm]")
            axs[1].plot(t, d["thrust_model"], color, label=name)
            axs[1].set_ylabel("thrust [N]")
            axs[2].plot(t, d["rpm_cmd"][:, 0], color, lw=0.8, label=name)
            axs[2].set_ylabel("rpm command motor 1")
            drpm = np.abs(np.diff(d["rpm_cmd"], axis=0)).mean(1) / CF2X.max_rpm
            axs[3].plot(t[1:], drpm, color, lw=0.8, label=f"{name} (mean {drpm.mean():.4f})")
            axs[3].set_ylabel("|Δrpm| / max rpm per step")
        for a in axs:
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
        axs[-1].set_xlabel("t [s]")
        axs[0].set_title(f"{fam}, tuning episode seed {s.seed}: LQI (tuned) vs "
                         f"MPPI hybrid ({args.gains})")
        fig.tight_layout()
        fig.savefig(out / f"diag_{fam}.png", dpi=100)


if __name__ == "__main__":
    main()
