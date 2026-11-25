"""R4.3c ablation: tuned MPPI hybrid gains on the 8 tuning + 40 validation episodes.

    python projects/control/scripts/mppi_ablation.py

Variant: the reference candidate alone (``mppi_hybrid_cand``, a geometric PD law without
rollouts). The tuned MPPI and LQI are taken from their tuning summaries (same episodes, same
objective). Results -> projects/control/results/mppi/ablation.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from dronelab.control.tuning.protocol import (
    objective_value,
    tuning_episodes,
    validation_episodes,
)
from dronelab.core.eval.parallel import EpisodePool

OUT = Path("projects/control/results/mppi")
TUNED = Path("projects/control/results/tuning_real")
VARIANTS = {"candidate only (geometric PD)": "mppi_hybrid_cand"}


def _row(label: str, tune: dict | None, val: dict, ms: float | str) -> str:
    t = f"{tune['J']:.3f}" if tune else "–"
    return (f"| {label} | {t} | {val['J']:.3f} | {val['rmse_rel']:.3f} | {val['smooth_rel']:.3f}"
            f" | {val['crash_rate']:.2f} | {ms} |")


def main() -> None:
    gains = yaml.safe_load((TUNED / "mppi_hybrid/best_params.yaml").read_text())
    lines = ["Tuned MPPI hybrid gains (trial 47); J on the 8 tuning and 40 validation episodes\n",
             "| variant | J tune | J val | rmse_rel val | smooth_rel val | crash val | ms/step |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("lqi", "LQI (tuned)"), ("mppi_hybrid", "mppi_hybrid (tuned)")):
        s = json.loads((TUNED / name / "summary.json").read_text())
        lines.append(_row(label, None, s["validation_selected"], "–"))
    with EpisodePool(2) as pool:
        for label, ctrl in VARIANTS.items():
            tune_rows = pool.map(tuning_episodes(ctrl, gains))
            val_rows = pool.map(validation_episodes(ctrl, gains))
            ms = np.mean([r.get("ctrl_ms_per_step", np.nan) for r in val_rows])
            lines.append(_row(label, objective_value(tune_rows), objective_value(val_rows),
                              f"{ms:.1f}"))
            print(lines[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
