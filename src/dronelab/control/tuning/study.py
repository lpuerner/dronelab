"""Optuna study (TPE) following :mod:`dronelab.control.tuning.protocol`.

Each trial flies its 8 tuning episodes in parallel worker processes. The study is
stored in SQLite, so an interrupted run continues where it stopped.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import optuna
import yaml

from dronelab.control.tuning.protocol import (
    DEFAULTS,
    N_TUNE,
    N_VAL,
    SEARCH_SPACES,
    TOP_K,
    W_CRASH,
    W_SMOOTH,
    objective_value,
    tuning_episodes,
    validation_episodes,
)
from dronelab.core.eval.parallel import EpisodePool


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def tune(
    controller: str,
    n_trials: int = 200,
    processes: int = 2,
    out_dir: str | Path = "projects/control/results/tuning",
    sampler_seed: int = 0,
) -> dict:
    space = SEARCH_SPACES[controller]
    out = Path(out_dir) / controller
    out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=controller,
        storage=f"sqlite:///{(out / 'study.db').as_posix()}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULTS[controller])  # trial 0 = hand-set defaults

    def n_done() -> int:
        return sum(t.state.is_finished() for t in study.trials)

    t0 = time.perf_counter()
    with EpisodePool(processes) as pool:
        while n_done() < n_trials:
            trial = study.ask()
            gains = {k: trial.suggest_float(k, lo, hi, log=True) for k, (lo, hi) in space.items()}
            res = objective_value(pool.map(tuning_episodes(controller, gains)))
            for k, v in res.items():
                trial.set_user_attr(k, v)
            study.tell(trial, res["J"])
            if trial.number % 10 == 0:
                print(f"[{controller}] trial {trial.number:3d}  J={res['J']:.4f}  "
                      f"best={study.best_value:.4f}  ({time.perf_counter() - t0:.0f} s)",
                      flush=True)
        t_tune = time.perf_counter() - t0

        # validation of the top-k trials (and the defaults) on unseen seeds
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        top = sorted(done, key=lambda t: t.value)[:TOP_K]
        val_rows, ep_rows = [], []
        for t in top:
            rows = pool.map(validation_episodes(controller, t.params))
            res = objective_value(rows)
            val_rows.append({"trial": t.number, "J_tune": t.value,
                             **{f"{k}_val": v for k, v in res.items()}})
            ep_rows += [{"trial": t.number, **r} for r in rows]
        default_val = objective_value(pool.map(validation_episodes(controller,
                                                                   DEFAULTS[controller])))
    t_total = time.perf_counter() - t0

    best = min(val_rows, key=lambda r: r["J_val"])
    best_params = next(t.params for t in top if t.number == best["trial"])
    (out / "best_params.yaml").write_text(yaml.safe_dump(best_params, sort_keys=True))
    _write_csv(out / "validation.csv", val_rows)
    _write_csv(out / "validation_episodes.csv", ep_rows)
    _write_csv(out / "trials.csv",
               [{"trial": t.number, "J": t.value, **t.user_attrs, **t.params} for t in done])
    summary = {
        "controller": controller,
        "n_trials": len(done),
        "episodes_per_trial": N_TUNE,
        "validation_episodes": N_VAL,
        "top_k": TOP_K,
        "objective": f"mean_f(rmse/ref) + {W_SMOOTH}*mean_f(smooth/ref) + {W_CRASH}*crash_rate",
        "selected_trial": best["trial"],
        "validation_selected": {k[:-4]: v for k, v in best.items() if k.endswith("_val")},
        "validation_defaults": default_val,
        "wall_time_tuning_s": round(t_tune, 1),
        "wall_time_total_s": round(t_total, 1),
        "cpu_hours": round(t_total * processes / 3600, 3),
        "processes": processes,
        "machine": platform.processor(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
