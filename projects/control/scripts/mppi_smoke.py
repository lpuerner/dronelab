"""R4.2 smoke test: MPPI with its default gains on the 8 tuning episodes, against LQI.

    python projects/control/scripts/mppi_smoke.py

Same episodes and objective J as one tuning trial (dronelab/control/tuning/protocol.py), so
the numbers are the tuning starting point. Results -> projects/control/results/mppi/smoke.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dronelab.control.tuning.protocol import DEFAULTS, objective_value, tuning_episodes
from dronelab.core.eval.parallel import EpisodePool

OUT = Path("projects/control/results/mppi")
RUNS = {  # label -> (controller, gains)
    "lqi (tuned)": ("lqi", yaml.safe_load(
        Path("projects/control/results/tuning_real/lqi/best_params.yaml").read_text())),
    "mppi_urdf K=256": ("mppi_urdf", {**DEFAULTS["mppi_urdf"], "samples": 256}),
    "mppi_hybrid K=256": ("mppi_hybrid", {**DEFAULTS["mppi_hybrid"], "samples": 256}),
    "mppi_hybrid K=512": ("mppi_hybrid", {**DEFAULTS["mppi_hybrid"], "samples": 512}),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["MPPI defaults on the 8 tuning episodes (nominal, real world); J as in tuning\n",
             "| run | J | RMSE rel | smooth rel | crash rate | RMSE per family [cm] | ms/step |",
             "|---|---:|---:|---:|---:|---|---:|"]
    with EpisodePool(2) as pool:
        for label, (ctrl, gains) in RUNS.items():
            rows = pool.map(tuning_episodes(ctrl, gains))
            o = objective_value(rows)
            fams = {r["family"]: [] for r in rows}
            for r in rows:
                fams[r["family"]].append(100 * r.get("rmse_pos", np.nan))
            per = ", ".join(f"{f} {np.mean(v):.1f}" for f, v in fams.items())
            ms = np.mean([r.get("ctrl_ms_per_step", np.nan) for r in rows])
            lines.append(f"| {label} | {o['J']:.3f} | {o['rmse_rel']:.3f} | {o['smooth_rel']:.3f}"
                         f" | {o['crash_rate']:.2f} | {per} | {ms:.1f} |")
            print(lines[-1], flush=True)
    (OUT / "smoke.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
