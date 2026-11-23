"""Measure the frozen reference scales of the tuning objective for one world.

Default LQR on 48 nominal episodes (seeds 0-199, all tuning families). Paste the printed
dictionaries into REF_RMSE / REF_SMOOTH in src/dronelab/control/tuning/protocol.py *before*
tuning starts; they must not change afterwards.

    python projects/control/scripts/reference_scales.py --world real
"""

from __future__ import annotations

import argparse
import json

from dronelab.control.tuning.protocol import reference_episodes, reference_scales
from dronelab.core.eval.parallel import EpisodePool
from dronelab.core.world import DEFAULT_WORLD, WORLDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default=DEFAULT_WORLD, choices=list(WORLDS))
    ap.add_argument("--processes", type=int, default=2)
    args = ap.parse_args()
    with EpisodePool(args.processes) as pool:
        rows = pool.map(reference_episodes(args.world))
    crashed = sum(bool(r.get("crashed", 0)) for r in rows)
    rmse, smooth = reference_scales(rows)
    print(json.dumps({"world": args.world, "episodes": len(rows), "crashed": crashed,
                      "REF_RMSE": {k: round(v, 4) for k, v in rmse.items()},
                      "REF_SMOOTH": {k: round(v, 5) for k, v in smooth.items()}}, indent=2))


if __name__ == "__main__":
    main()
