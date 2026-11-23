"""Run a benchmark config and append one CSV row per episode (resumable).

    python projects/control/scripts/benchmark.py \
        --config projects/control/configs/benchmark/stage1.yaml [--log-flights]

Rows already present in the CSV (same controller, part, scenario, family, speed, seed) are
skipped, so an interrupted run simply continues. Every row stores the git commit and a hash
of config + controller parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

from dronelab.core.eval.parallel import EpisodePool, EpisodeSpec
from dronelab.core.eval.scenarios import SCENARIOS, disturbance_fields

PARAMS_DIR = "projects/control/results/tuning"
RESULTS = "projects/control/results/benchmark"
KEY = ("controller", "part", "scenario", "family", "speed", "seed")
FIELDS = [
    *KEY, "kind", "level", "physics", "feasible", "rejections",
    "course_max_acc", "course_max_tilt_deg", "course_max_thrust_ratio",
    "crashed", "crash_reason", "error", "rmse_pos", "max_err", "final_err", "settling_time",
    "iae", "ise", "itae", "effort", "smoothness", "flight_time", "ctrl_ms_per_step",
    "gates_passed", "gates_total", "all_gates", "gate_miss_mean",
    "git", "config_hash",
]


def _git() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src"]) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_gains(cfg: dict) -> dict[str, dict | None]:
    gains = {}
    for c in cfg["controllers"]:
        path = Path(cfg.get("params_dir", PARAMS_DIR)) / c / "best_params.yaml"
        gains[c] = yaml.safe_load(path.read_text()) if path.exists() else None
    return gains


def build_specs(cfg: dict, gains: dict, chash: str) -> list[EpisodeSpec]:
    specs = []
    nominal = SCENARIOS["nominal"]

    def add(ctrl, part, scen, family, speed, seed):
        specs.append(EpisodeSpec(
            ctrl, family, float(speed), int(seed), gains[ctrl],
            disturbances=disturbance_fields(scen, seed), physics=scen.physics,
            tags={"part": part, "scenario": scen.name, "kind": scen.kind, "level": scen.level,
                  "physics": scen.physics, "config_hash": chash},
        ))

    for ctrl in cfg["controllers"]:
        for seed in cfg["seeds"]:
            a = cfg.get("part_a")
            if a:
                if a.get("hover"):
                    add(ctrl, "A", nominal, "hover", 0.0, seed)
                for fam in a["families"]:
                    for v in a["speeds"]:
                        add(ctrl, "A", nominal, fam, v, seed)
            b = cfg.get("part_b")
            if b:
                names = list(SCENARIOS) if b["scenarios"] == "all" else b["scenarios"]
                for name in names:
                    for fam in b["families"]:
                        add(ctrl, "B", SCENARIOS[name], fam,
                            0.0 if fam == "hover" else b["speed"], seed)
    return specs


def _key(d: dict) -> tuple:
    return tuple(str(float(d[k])) if k == "speed" else str(d[k]) for k in KEY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help=f"CSV path (default {RESULTS}/<name>.csv)")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--log-flights", action="store_true", help="save .npz flight logs to data/")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    gains = _load_gains(cfg)
    chash = hashlib.sha1(json.dumps([cfg, gains], sort_keys=True).encode()).hexdigest()[:10]
    out = Path(args.out or f"{RESULTS}/{cfg['name']}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with out.open() as f:
            done = {_key(r) for r in csv.DictReader(f)}
    specs = [s for s in build_specs(cfg, gains, chash)
             if _key({"controller": s.controller, "family": s.family, "speed": s.speed,
                      "seed": s.seed, **s.tags}) not in done]
    if args.log_flights:
        flights = Path("data/flights") / cfg["name"]
        flights.mkdir(parents=True, exist_ok=True)
        specs = [EpisodeSpec(**{**s.__dict__, "log_path": str(
            flights / f"{s.tags['part']}_{s.controller}_{s.tags['scenario']}_{s.family}_"
            f"{s.speed:g}_{s.seed}.npz")})
            for s in specs]
    print(f"{len(done)} rows done, {len(specs)} to run; untuned: "
          f"{[c for c, g in gains.items() if g is None]}", flush=True)

    git = _git()
    new = not out.exists()
    t0 = time.perf_counter()
    with out.open("a", newline="") as f, EpisodePool(args.processes) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap(specs), 1):
            w.writerow({**row, "git": git})
            f.flush()
            if i % 50 == 0 or i == len(specs):
                el = time.perf_counter() - t0
                print(f"{i}/{len(specs)}  {el:.0f} s  (eta {el / i * (len(specs) - i):.0f} s)",
                      flush=True)


if __name__ == "__main__":
    main()
