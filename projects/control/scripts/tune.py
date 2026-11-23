"""Tune controllers with the fixed protocol (see dronelab/control/tuning/protocol.py).

    python projects/control/scripts/tune.py --controller pid --trials 200
    python projects/control/scripts/tune.py --controller pid lqr lqi --trials 200 --processes 2
"""

from __future__ import annotations

import argparse
import json

from dronelab.control.tuning import tune


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", nargs="+", required=True, choices=["pid", "lqr", "lqi"])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--out", default="projects/control/results/tuning")
    args = ap.parse_args()
    for name in args.controller:
        print(json.dumps(tune(name, args.trials, args.processes, args.out), indent=2), flush=True)


if __name__ == "__main__":
    main()
