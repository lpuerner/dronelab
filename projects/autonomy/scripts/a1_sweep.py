"""A1 / A1+: sweep the acceleration budget of the gate planner on validation seeds (resumable).

    python projects/autonomy/scripts/a1_sweep.py --processes 2

Every (controller, variant, acc_frac, seed) is one episode on the `gates` course of that seed
(geometry of the 1 m/s course), `real` world, nominal, gate frames on. Writes
``results/a1_sweep.csv`` and ``results/a1_selected.yaml``: per controller and variant the
fastest budget with >= 95 % success (all gates in order, no crash).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import yaml

from dronelab.core.eval.parallel import EpisodePool, EpisodeSpec

GAINS = "projects/control/results/tuning_real/{}/best_params.yaml"
OUT = Path("projects/autonomy/results")
KEY = ("controller", "variant", "acc_frac", "seed")
FIELDS = [*KEY, "crashed", "crash_reason", "gates_in_order", "gates_total", "lap_time",
          "gate_clearance_min", "rmse_pos", "smoothness", "ctrl_ms_per_step", "course_max_acc",
          "course_max_tilt_deg", "course_max_thrust_ratio", "course_max_offset"]
MIN_SUCCESS = 0.95


def success(r: dict) -> bool:
    return float(r["crashed"]) == 0 and float(r["gates_in_order"]) == float(r["gates_total"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["indi", "lqi"])
    ap.add_argument("--acc", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--seeds", type=int, nargs=2, default=[100, 140])
    ap.add_argument("--processes", type=int, default=2)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "a1_sweep.csv"
    done = set()
    if path.exists():
        done = {tuple(r[k] for k in KEY) for r in csv.DictReader(path.open())}
    specs = []
    for c in args.controllers:
        gains = yaml.safe_load(open(GAINS.format(c)))
        for variant in ("a1", "a1plus"):
            for f in args.acc:
                for seed in range(*args.seeds):
                    if (c, variant, str(f), str(seed)) in done:
                        continue
                    specs.append(EpisodeSpec(
                        c, "gates", 1.0, seed, gains=gains, gate_frames=True,
                        planner={"acc_frac": f, "optimize": variant == "a1plus"},
                        tags={"variant": variant, "acc_frac": f}))
    print(f"{len(specs)} episodes to fly ({len(done)} done)", flush=True)
    new = not path.exists()
    with path.open("a", newline="") as fh, EpisodePool(args.processes) as pool:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap(specs), 1):
            w.writerow(row)
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(specs)}", flush=True)
    summarize(path)


def summarize(path: Path) -> None:
    rows = list(csv.DictReader(path.open()))
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["controller"], r["variant"], float(r["acc_frac"])), []).append(r)
    lines = ["| controller | variant | acc_frac | success | median lap time [s] |",
             "|---|---|---:|---:|---:|"]
    selected: dict = {}
    for (c, v, f), rs in sorted(groups.items()):
        ok = [r for r in rs if success(r)]
        rate = len(ok) / len(rs)
        lap = statistics.median(float(r["lap_time"]) for r in ok) if ok else float("nan")
        lines.append(f"| {c} | {v} | {f:.1f} | {rate:.0%} | {lap:.2f} |")
        best = selected.get(c, {}).get(v)
        if rate >= MIN_SUCCESS and (best is None or lap < best["median_lap_time"]):
            selected.setdefault(c, {})[v] = {"acc_frac": f, "success": rate,
                                             "median_lap_time": round(lap, 3)}
    (OUT / "a1_sweep.md").write_text(
        "A1 sweep (validation seeds, nominal, gate frames on); selection: fastest budget with "
        f">= {MIN_SUCCESS:.0%} success\n\n" + "\n".join(lines) + "\n")
    (OUT / "a1_selected.yaml").write_text(yaml.safe_dump(selected, sort_keys=True))
    print("\n".join(lines))
    print(yaml.safe_dump(selected))


if __name__ == "__main__":
    main()
