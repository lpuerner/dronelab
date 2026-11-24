"""Tune controllers with the fixed protocol (see dronelab/control/tuning/protocol.py).

    python projects/control/scripts/tune.py --controller pid lqr lqi --trials 200 --processes 2
    python projects/control/scripts/tune.py --controller lqi --world ideal \
        --out projects/control/results/tuning          # stage-1 setting (ideal world)
"""

from __future__ import annotations

import argparse
import json

from dronelab.control.tuning import SEARCH_SPACES, tune
from dronelab.core.world import DEFAULT_WORLD, WORLDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", nargs="+", required=True, choices=list(SEARCH_SPACES))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--world", default=DEFAULT_WORLD, choices=list(WORLDS))
    ap.add_argument("--no-validate", action="store_true",
                    help="pilot: only run the trials, no top-k validation (study continues later)")
    ap.add_argument("--out", default=None,
                    help="default projects/control/results/tuning_<world> "
                         "(stage-1 results of the ideal world: projects/control/results/tuning)")
    args = ap.parse_args()
    out = args.out or f"projects/control/results/tuning_{args.world}"
    for name in args.controller:
        res = tune(name, args.trials, args.processes, out, world=args.world,
                   validate=not args.no_validate)
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
