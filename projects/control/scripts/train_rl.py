"""R3: train the PPO tracking policy (resumable).

    python projects/control/scripts/train_rl.py --stage smoke --steps 100000   # throughput
    python projects/control/scripts/train_rl.py --stage all                    # search, final, dr

Stages: ``search`` (every configuration of the config, seed 0, ``steps_search``), ``final``
(best search configuration by validation J, ``seeds_final`` x ``steps_final``), ``dr`` (best
configuration with domain randomisation, ``steps_dr``), ``ablation`` (best configuration with
the changes of ``ablation``, seed 0, ``steps_ablation``), ``all`` = search, final, dr. A run whose
``info.json`` exists is skipped; an interrupted run continues from its last checkpoint
(``data/rl_checkpoints/``). Every run is validated on the 40 validation episodes of the tuning
protocol; ``info.json`` holds J, wall time and CPU-h. The benchmark parameter sets
(``tuning_real/ppo*/``) come from the best checkpoint, see ``eval_rl_checkpoints.py``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

import yaml

OUT = Path("projects/control/results/rl")
CKPT = Path("data/rl_checkpoints")
CKPT_EVERY = 100_000  # [steps]


def _make_env(rank: int, randomize: bool, stream: int, w_dact: float | None):
    from dronelab.control.rl_env import W_DACT, TrackingEnv

    return TrackingEnv(rank=rank, randomize=randomize, stream=stream,
                       w_dact=W_DACT if w_dact is None else w_dact)


def _hp(cfg: dict, variant: dict) -> dict:
    return {**cfg["base"], **{k: v for k, v in variant.items() if k != "name"}}


def train(cfg: dict, run: str, hp: dict, steps: int, seed: int, randomize: bool) -> dict:
    """Train one run (or finish it from its last checkpoint) and validate it."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

    out, ckpt = OUT / cfg["name"] / run, CKPT / cfg["name"] / run
    if (out / "info.json").exists():
        return json.loads((out / "info.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    n = cfg["n_envs"]
    venv = VecMonitor(SubprocVecEnv([functools.partial(_make_env, i, randomize, seed,
                                                       hp.get("w_dact"))
                                     for i in range(n)], start_method="spawn"))
    models = sorted(ckpt.glob("model_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    t_prev = json.loads((ckpt / "time.json").read_text())["wall_s"] if models else 0.0
    if models:  # resume
        last = models[-1]
        venv = VecNormalize.load(str(last.with_name(
            last.name.replace("model_", "model_vecnormalize_").replace(".zip", ".pkl"))), venv)
        model = PPO.load(str(last), env=venv, device="cpu")
        print(f"[{run}] resume at {model.num_timesteps} steps", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=hp["gamma"])
        model = PPO(
            "MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
            learning_rate=hp["learning_rate"], n_steps=hp["n_steps"],
            batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"],
            policy_kwargs={"net_arch": list(hp["net"]), "log_std_init": hp["log_std_init"]})
    from stable_baselines3.common.logger import configure

    model.set_logger(configure(str(out), ["csv"]))
    cb = CheckpointCallback(max(CKPT_EVERY // n, 1), str(ckpt), "model",
                            save_vecnormalize=True)
    tic = time.time()

    class _Clock(BaseCallback):  # keeps the wall time across resumes
        def _on_step(self) -> bool:
            if self.n_calls % cb.save_freq == 0:
                (ckpt / "time.json").write_text(json.dumps({"wall_s": t_prev + time.time() - tic}))
            return True

    model.learn(steps - model.num_timesteps, callback=[cb, _Clock()], reset_num_timesteps=False)
    wall = t_prev + time.time() - tic
    model.save(out / "model.zip")
    venv.save(str(out / "vecnormalize.pkl"))
    venv.close()
    info = {"run": run, "hp": hp, "seed": seed, "randomize": randomize, "steps": steps,
            "wall_h": wall / 3600, "cpu_h": wall / 3600 * n,  # n env workers busy throughout
            "steps_per_s": steps / wall, **validate(out)}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[{run}] done: J {info['J']:.3f}, {info['steps_per_s']:.0f} steps/s", flush=True)
    return info


def validate(out: Path) -> dict:
    from dronelab.control.tuning.protocol import objective_value, validation_episodes
    from dronelab.core.eval.parallel import EpisodePool

    gains = {"model": str(out / "model.zip"), "vecnormalize": str(out / "vecnormalize.pkl")}
    with EpisodePool(2) as pool:
        rows = pool.map(validation_episodes("ppo", gains))
    return objective_value(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/control/configs/rl/r3.yaml")
    ap.add_argument("--stage", default="all",
                    choices=["smoke", "search", "final", "dr", "ablation", "all"])
    ap.add_argument("--steps", type=int, default=100_000, help="smoke only")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if args.stage == "smoke":
        info = train(cfg, f"smoke_{args.steps}", _hp(cfg, {}), args.steps, 0, False)
        print(json.dumps(info, indent=2))
        return
    search = []
    if args.stage != "smoke":
        for v in cfg["search"]:
            if args.stage in ("search", "all") or (OUT / cfg["name"] / f"search_{v['name']}"
                                                  / "info.json").exists():
                search.append(train(cfg, f"search_{v['name']}", _hp(cfg, v),
                                    cfg["steps_search"], 0, False))
    if args.stage == "search":
        return
    best = min(search, key=lambda i: i["J"])
    print(f"best configuration: {best['run']} (J {best['J']:.3f})", flush=True)
    if args.stage in ("final", "all"):
        for s in cfg["seeds_final"]:
            train(cfg, f"final_s{s}", best["hp"], cfg["steps_final"], s, False)
    if args.stage in ("dr", "all"):
        for s in cfg["seeds_dr"]:
            train(cfg, f"dr_s{s}", best["hp"], cfg["steps_dr"], s, True)
    if args.stage == "ablation":
        for v in cfg["ablation"]:
            train(cfg, f"abl_{v['name']}_s0", _hp({"base": best["hp"]}, v),
                  cfg["steps_ablation"], 0, False)


if __name__ == "__main__":
    main()
