"""Generate the flight datasets of the surrogate subproject (resumable).

    python projects/surrogate/scripts/generate_data.py                       # all splits
    python projects/surrogate/scripts/generate_data.py --splits val --limit 4 # quick check

Every split folder gets a ``manifest.json`` (config, git commit, seeds, flight time, crashes).
Flights already on disk are skipped. Rules: projects/surrogate/DATASETS.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from dronelab.core.eval.parallel import EpisodePool
from dronelab.surrogate.datasets import (
    COURSE_CONTROLLERS,
    EXCITATION_CONTROLLER,
    SPLIT_DIRS,
    SplitConfig,
    overlapping_seeds,
    split_specs,
)

DOC = Path("projects/surrogate/DATASETS.md")


def _git() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src"]) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_gains(params_dir: Path, allow_default: bool) -> dict:
    gains = {}
    for c in {*COURSE_CONTROLLERS, EXCITATION_CONTROLLER}:
        p = params_dir / c / "best_params.yaml"
        if p.exists():
            gains[c] = yaml.safe_load(p.read_text())
        elif allow_default:
            gains[c] = None
        else:
            raise SystemExit(f"missing {p} - tune first (or --allow-default-gains for a test)")
    return gains


def _flight_seconds(path: Path) -> float:
    with np.load(path) as d:
        return float(len(d["t"])) * float(d["t"][1] - d["t"][0]) if len(d["t"]) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/surrogate/configs/data.yaml")
    ap.add_argument("--splits", nargs="+", default=None, help="default: all in the config")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N flights of each kind per split (test runs)")
    ap.add_argument("--out", default=None, help="override out_dir (e.g. for a test run)")
    ap.add_argument("--allow-default-gains", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = {k: SplitConfig(tuple(v["excitation"]), tuple(v["courses"]))
              for k, v in cfg["splits"].items()}
    if dup := overlapping_seeds(splits):
        raise SystemExit(f"seed ranges overlap: {sorted(dup)[:10]} ...")
    root = Path(args.out or cfg["out_dir"])
    gains = _load_gains(Path(cfg["params_dir"]), args.allow_default_gains)
    root.mkdir(parents=True, exist_ok=True)
    if DOC.exists():
        shutil.copy(DOC, root / "README.md")  # the rules travel with the data

    for name in args.splits or list(splits):
        out = root / SPLIT_DIRS[name]
        out.mkdir(parents=True, exist_ok=True)
        specs = split_specs(splits[name], gains, out, cfg["world"], tuple(cfg["v_max_range"]),
                            cfg["dither_amp"])
        if args.limit is not None:
            kinds = [s.tags["kind"] for s in specs]
            specs = [s for i, s in enumerate(specs) if kinds[:i].count(kinds[i]) < args.limit]
        todo = [s for s in specs if not Path(s.log_path).exists()]
        print(f"[{name}] {len(specs) - len(todo)} on disk, {len(todo)} to fly", flush=True)
        t0, rows = time.perf_counter(), []
        with EpisodePool(args.processes) as pool:
            for i, row in enumerate(pool.imap(todo), 1):
                rows.append(row)
                if i % 25 == 0 or i == len(todo):
                    el = time.perf_counter() - t0
                    print(f"  {i}/{len(todo)}  {el:.0f} s  (eta {el / i * (len(todo) - i):.0f} s)",
                          flush=True)
        files = sorted(out.glob("*.npz"))
        # rows arrive in completion order (imap_unordered): count, don't zip with ``todo``
        infeasible = sum(not r.get("feasible", True) for r in rows)
        manifest = {
            "split": name,
            "folder": out.as_posix(),
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": _git(),
            "config": cfg,
            "seeds": {"excitation": list(splits[name].excitation),
                      "courses": list(splits[name].courses)},
            "limit": args.limit,
            "default_gains_used": [c for c, g in gains.items() if g is None],
            "flights": len(files),
            "flight_time_h": round(sum(_flight_seconds(f) for f in files) / 3600, 3),
            "crashed_this_run": int(sum(bool(r.get("crashed", 0)) for r in rows)),
            "infeasible_this_run": int(infeasible),
            "wall_time_s": round(time.perf_counter() - t0, 1),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({k: manifest[k] for k in ("flights", "flight_time_h",
                                                   "crashed_this_run", "wall_time_s")}))


if __name__ == "__main__":
    main()
