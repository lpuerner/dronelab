"""Tune controllers with the fixed protocol (see dronelab/control/tuning/protocol.py).

    python projects/control/scripts/tune.py --controller pid lqr lqi --trials 200 --processes 2
    python projects/control/scripts/tune.py --controller lqi --world ideal \
        --out projects/control/results/tuning          # stage-1 setting (ideal world)
"""

from __future__ import annotations

import argparse
import json

from dronelab.control.tuning import tune
from dronelab.core.world import DEFAULT_WORLD, WORLDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", nargs="+", required=True, choices=["pid", "lqr", "lqi"])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--world", default=DEFAULT_WORLD, choices=list(WORLDS))
    ap.add_argument("--out", default=None,
                    help="default projects/control/results/tuning_<world> "
                         "(stage-1 results of the ideal world: projects/control/results/tuning)")
    args = ap.parse_args()
    out = args.out or f"projects/control/results/tuning_{args.world}"
    for name in args.controller:
        print(json.dumps(tune(name, args.trials, args.processes, out, world=args.world),
                         indent=2), flush=True)


if __name__ == "__main__":
    main()
