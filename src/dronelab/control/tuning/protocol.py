"""The tuning protocol, identical for every controller (fixed before any result exists).

* Objective (normalised so that every family counts equally)::

      J = mean_f(RMSE_f / RMSE_ref_f) + 0.3 * mean_f(smooth_f / smooth_ref_f) + 1.0 * crash_rate

  ``RMSE_f`` / ``smooth_f`` are means over the episodes of family f; the reference scales are
  measured once per world with the default LQR weights (the 48 tuning + validation
  episodes, nominal, start on the reference, :func:`reference_episodes`) and frozen below.
  ``w_s = 0.3``: 10 % smoother motor commands are worth 3 % more tracking error. A crash costs
  as much as doubling the tracking error everywhere.
* Episodes start exactly on the reference (no initial-state noise); hover takes off from 0.1 m.
* Tuning episodes: 8 fixed episodes (2 per family hover/figure8/helix/waypoints) from the
  tuning seeds 0-99, nominal conditions, path speed drawn from U(0.25, 1.5) m/s.
* Validation: the best 5 trials are re-flown on 40 episodes from seeds 100-199; the one
  with the lowest validation objective is selected.
* Search space: each parameter is searched log-uniformly in [default / f, default * f]
  around its hand-set default (f = 5, f = 10 for integral gains and the MPPI weights w_vel,
  w_att; the MPPI weight w_pos is fixed since the cost is normalised).
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np

from dronelab.control.lqr import LQRWeights
from dronelab.control.mppi import MPPIGains
from dronelab.control.pid import PIDGains
from dronelab.core.eval.parallel import EpisodeSpec
from dronelab.core.tasks import TUNE_FAMILIES
from dronelab.core.world import DEFAULT_WORLD

W_SMOOTH = 0.3
W_CRASH = 1.0
# frozen reference scales per world and family (default LQR, see module docstring)
REF_RMSE = {
    "ideal": {"hover": 0.182, "figure8": 0.0147, "helix": 0.0131, "waypoints": 0.0126},
    "real": {"hover": 0.1908, "figure8": 0.0321, "helix": 0.0379, "waypoints": 0.0456},
}
REF_SMOOTH = {
    "ideal": {"hover": 0.00164, "figure8": 0.00123, "helix": 0.00091, "waypoints": 0.00082},
    "real": {"hover": 0.00132, "figure8": 0.0014, "helix": 0.00112, "waypoints": 0.00096},
}
SPEED_RANGE = (0.25, 1.5)
TUNE_SEEDS = range(0, 100)
VAL_SEEDS = range(100, 200)
N_TUNE = 8
N_VAL = 40
TOP_K = 5


def _log_space(defaults: dict, wide: tuple[str, ...] = ()) -> dict[str, tuple[float, float]]:
    return {k: (v / (10.0 if k in wide else 5.0), v * (10.0 if k in wide else 5.0))
            for k, v in defaults.items()}


def _defaults(cls) -> dict:
    return {f.name: f.default for f in fields(cls)}


_PID = _defaults(PIDGains)
_LQR = _defaults(LQRWeights)
_LQR_ONLY = {k: v for k, v in _LQR.items() if not k.startswith("int_")}
_MPPI = {k: v for k, v in _defaults(MPPIGains).items() if k != "w_pos"}  # w_pos fixed
_MPPI_WIDE = ("w_vel", "w_att")  # at the lower bound in the R4.3 pilot

SEARCH_SPACES: dict[str, dict[str, tuple[float, float]]] = {
    "pid": _log_space(_PID, wide=tuple(k for k in _PID if "_ki_" in k)),
    "lqr": _log_space(_LQR_ONLY),
    "lqi": _log_space(_LQR),
    "mppi_hybrid": _log_space(_MPPI, wide=_MPPI_WIDE),
    "mppi_urdf": _log_space(_MPPI, wide=_MPPI_WIDE),
}
DEFAULTS = {"pid": _PID, "lqr": _LQR_ONLY, "lqi": _LQR, "mppi_hybrid": _MPPI, "mppi_urdf": _MPPI}


def _episodes(controller: str, seeds: range, n: int, gains: dict | None,
              world: str) -> list[EpisodeSpec]:
    specs = []
    for i, seed in enumerate(list(seeds)[:n]):
        family = TUNE_FAMILIES[i % len(TUNE_FAMILIES)]
        speed = 0.0
        if family != "hover":
            speed = float(np.random.default_rng([seed, 4242]).uniform(*SPEED_RANGE))
        specs.append(EpisodeSpec(controller, family, round(speed, 3), seed, gains, world=world))
    return specs


def tuning_episodes(controller: str, gains: dict | None = None,
                    world: str = DEFAULT_WORLD) -> list[EpisodeSpec]:
    return _episodes(controller, TUNE_SEEDS, N_TUNE, gains, world)


def validation_episodes(controller: str, gains: dict | None = None,
                        world: str = DEFAULT_WORLD) -> list[EpisodeSpec]:
    return _episodes(controller, VAL_SEEDS, N_VAL, gains, world)


def reference_episodes(world: str = DEFAULT_WORLD) -> list[EpisodeSpec]:
    """Default LQR on the 8 tuning + 40 validation episodes (48 in total, seeds 0-199);
    this is how the frozen stage-1 scales were measured."""
    gains = DEFAULTS["lqr"]
    return tuning_episodes("lqr", gains, world) + validation_episodes("lqr", gains, world)


def reference_scales(rows: list[dict]) -> tuple[dict, dict]:
    """Mean RMSE and smoothness per family from :func:`reference_episodes` rows."""
    flown = [r for r in rows if r.get("feasible", True) and not r.get("crashed", 0)]
    fams = sorted({r["family"] for r in flown})
    rmse = {f: float(np.mean([r["rmse_pos"] for r in flown if r["family"] == f])) for f in fams}
    smooth = {f: float(np.mean([r["smoothness"] for r in flown if r["family"] == f]))
              for f in fams}
    return rmse, smooth


def objective_value(rows: list[dict], world: str = DEFAULT_WORLD) -> dict[str, float]:
    """Aggregate episode rows into the protocol objective (failed setups count as crashes)."""
    ref_rmse, ref_smooth = REF_RMSE[world], REF_SMOOTH[world]
    if ref_rmse is None or ref_smooth is None:
        raise RuntimeError(f"reference scales for world {world!r} not measured yet "
                           "(projects/control/scripts/reference_scales.py)")
    flown = [r for r in rows if r.get("feasible", True)]
    fams = sorted({r["family"] for r in flown})
    rmse_rel, smooth_rel = [], []
    for f in fams:
        rs = [r for r in flown if r["family"] == f]
        rmse_rel.append(np.mean([r.get("rmse_pos", 1.0) for r in rs]) / ref_rmse[f])
        smooth_rel.append(np.mean([r.get("smoothness", 1.0) for r in rs]) / ref_smooth[f])
    out = {
        "rmse": float(np.mean([r.get("rmse_pos", 1.0) for r in flown])),
        "smoothness": float(np.mean([r.get("smoothness", 1.0) for r in flown])),
        "rmse_rel": float(np.mean(rmse_rel)),
        "smooth_rel": float(np.mean(smooth_rel)),
        "crash_rate": float(np.mean([r.get("crashed", 1.0) for r in flown])),
    }
    out["J"] = out["rmse_rel"] + W_SMOOTH * out["smooth_rel"] + W_CRASH * out["crash_rate"]
    if not math.isfinite(out["J"]):
        out["J"] = 100.0
    return out
