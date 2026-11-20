# drone-control-benchmark

**How do classical, learned and physics-informed controllers cope when the world doesn't match the model?**

This project compares quadrotor controllers on the same flight courses and under the same disturbances:

| Family | Methods |
|---|---|
| Classical | cascaded PID (reference: gym-pybullet-drones `DSLPIDControl`), own PID, LQR |
| Learned | PPO / SAC (stable-baselines3) |
| Physics-informed | learned dynamics model (black-box vs. physics-informed NN) + sampling MPC |
| Hybrid | residual RL on top of PID |

Disturbances: aerodynamic drag, ground effect, constant wind + gusts, mass change (payload), motor lag, sensor noise, harder courses.

Everything runs on a **CPU only** (developed on a 2-core laptop).

> Status: **Stage 0 – foundation** (simulator backend, controller interface, courses, metrics, speed test).

## Architecture

```
controllers (PID / RL / MPC) ──act(state, ref)──►  Backend interface  ◄── PyBullet (gym-pybullet-drones)
        ▲                                               │                 (NumPy sim possible later)
        └──────── courses: reference(t) ◄── runner ─────┴──► metrics / plots / benchmark
```

All controllers talk to a simulator-agnostic `Backend` (`reset`, `step(rpm)`, `observe`), so a second simulator can be plugged in without touching any controller.

## Quickstart

Requires Python ≥ 3.12. On Windows, PyBullet has no pip wheel – use conda-forge:

```bash
conda create -n dronebench -c conda-forge python=3.12 pybullet   # Linux/macOS: a venv + pip works too
conda activate dronebench
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -e ".[dev]"

pytest                                                   # unit + physics sanity tests
python scripts/run_course.py --config configs/hover_pid.yaml
python scripts/run_course.py --config configs/hover_pid.yaml --course figure8
python scripts/speed_test.py                             # simulator throughput on your machine
```

## Courses

`hover` (step from 0.1 m to 1 m), `square`, `figure8`, `gates` – time-parametrised references traversed at constant speed, followed by a hold phase.

## Metrics

Position RMSE, max error, steady-state error, settling time, control effort, command smoothness, crash rate.

## Roadmap

- [x] 0 – Foundation: backend interface, PyBullet backend with disturbances, courses, metrics, speed test, CI
- [ ] 1 – Classical: own PID cascade, LQR, baseline numbers + GIFs
- [ ] 2 – RL: Gymnasium env, PPO/SAC, curriculum hover → waypoints
- [ ] 3 – PINN + MPC: black-box vs. physics-informed dynamics, MPPI
- [ ] 4 – Hybrid: residual RL
- [ ] 5 – Disturbance benchmark (method × disturbance)
- [ ] 6 – Results write-up
- [ ] (optional) vectorised NumPy simulator + parity check against PyBullet

## License

MIT

## References

1. J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, A. P. Schoellig, "Learning to Fly—a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control," *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2021. [arXiv:2103.02142](https://arxiv.org/abs/2103.02142) · Code: [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
