"""Measure simulator throughput on this machine.

Answers: how many control steps per second does PyBullet deliver, single
process and with several worker processes (like SubprocVecEnv), and what does
that mean for RL training time? Results -> projects/control/results/speed_test.{json,md}

    python projects/control/scripts/speed_test.py --steps 3000
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per process. Without this, every worker spawns one thread
# per core and they fight over the CPU: on a 2-core laptop 4 workers were *slower*
# than 1 (1.1k vs 1.5k steps/s); with it they reach ~3.2k steps/s.
# Must be set before numpy is imported; spawned workers inherit it.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from dronelab.core.backends.pybullet_backend import PyBulletBackend  # noqa: E402
from dronelab.core.config import SimConfig  # noqa: E402
from dronelab.core.controller import Reference  # noqa: E402


def _worker(args: tuple[str, int, bool, int]) -> float:
    """Returns control steps per second for one process."""
    physics, n_steps, with_pid, ctrl_freq = args
    sim = SimConfig(physics=physics, ctrl_freq=ctrl_freq)
    be = PyBulletBackend(sim)
    pid = None
    if with_pid:
        from dronelab.control.dsl_pid import DSLPIDController

        pid = DSLPIDController(be.dt)
    ref = Reference(pos=np.array([0.0, 0.0, 1.0]), vel=np.zeros(3))
    hover = np.full(4, be.params.hover_rpm)
    state = be.reset(np.array([0.0, 0.0, 1.0]))
    for _ in range(50):  # warm-up
        state = be.step(hover)
    t0 = time.perf_counter()
    for k in range(n_steps):
        rpm = pid.act(state, ref) if pid else hover
        state = be.step(rpm)
        if k % 2000 == 1999:  # keep the open-loop drone from drifting away
            state = be.reset(np.array([0.0, 0.0, 1.0]))
    dt = time.perf_counter() - t0
    be.close()
    return n_steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ctrl-freq", type=int, default=48)
    ap.add_argument("--out", default="projects/control/results")
    args = ap.parse_args()

    n_cpu = os.cpu_count() or 1
    rows = []

    def record(name: str, physics: str, procs: int, with_pid: bool) -> None:
        jobs = [(physics, args.steps, with_pid, args.ctrl_freq)] * procs
        if procs == 1:
            per_proc = [_worker(jobs[0])]
        else:
            with mp.get_context("spawn").Pool(procs) as pool:
                per_proc = pool.map(_worker, jobs)
        total = float(np.sum(per_proc))
        rows.append(
            {
                "setup": name,
                "physics": physics,
                "processes": procs,
                "steps_per_s": round(total, 1),
                "realtime_factor": round(total / args.ctrl_freq, 1),
                "hours_per_1M_steps": round(1e6 / total / 3600, 2),
            }
        )
        print(f"{name:<34} {total:>9.0f} steps/s  ({total / args.ctrl_freq:5.1f}x realtime)")

    print(f"CPU: {platform.processor()} | logical cores: {n_cpu} | ctrl_freq={args.ctrl_freq} Hz")
    record("sim only, 1 proc", "pyb", 1, False)
    record("sim + DSL-PID, 1 proc", "pyb", 1, True)
    record("sim only, 1 proc, drag+ground+dw", "pyb_gnd_drag_dw", 1, False)
    for procs in sorted({2, n_cpu} - {1}):
        record(f"sim only, {procs} procs", "pyb", procs, False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"cpu": platform.processor(), "logical_cores": n_cpu, "ctrl_freq": args.ctrl_freq}
    (out / "speed_test.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    lines = [
        f"CPU: {meta['cpu']} ({n_cpu} logical cores), control rate {args.ctrl_freq} Hz\n",
        "| setup | steps/s | x realtime | h per 1M steps (sim only) |",
        "|---|---:|---:|---:|",
    ]
    keys = ("setup", "steps_per_s", "realtime_factor", "hours_per_1M_steps")
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    (out / "speed_test.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved -> {out / 'speed_test.md'}")
    print("Note: RL training adds policy inference + gradient updates on top of these numbers.")


if __name__ == "__main__":
    main()
