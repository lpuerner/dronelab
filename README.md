# dronelab

Autonomy stack for a simulated Crazyflie 2.x quadrotor, built one layer at a time. Each layer is
its own subproject with a fixed question, a benchmark that every method runs under the same
conditions, and a README with the results and their limits.

The drone flies in PyBullet (gym-pybullet-drones [1]). The default `real` world adds effects the
controllers' model does not contain: drag, ground effect, downwash, 20 ms motor lag, quadratic
body drag, thrust loss in axial inflow and a blade-flapping moment. Everything runs on a CPU
(developed on a 2-core laptop).

| # | Subproject | Question | Status |
|---|---|---|---|
| 1 | [Control](projects/control/README.md) | Which controller follows a given path most accurately and robustly? | PID, LQI, INDI, MPPI and PPO benchmarked |
| 2 | [Surrogate model](projects/surrogate/README.md) | How well does a learned model predict the dynamics outside its training data? | first round done |
| 3 | Autonomy | How does the drone get through a gate course fastest without a given path? | planner baseline done, RL in progress |
| 4 | Path planning | Is the shortest path around known obstacles also the fastest flight? | planned |
| 5 | State estimation | How much does flying on an estimated state cost? | planned |
| 6 | Vision | Can the drone find the gates in camera images and fly through them? | planned |

Stage codes, current plans and project history: [docs/STATUS.md](docs/STATUS.md).

> **Work in progress.** This is a personal research project, not a finished library. Some parts
> are incomplete or not yet well optimised: the PPO controller still has training problems
> (it gets worse with longer training, and its reward does not match the evaluation objective
> well), MPPI is far from real-time on a CPU, and several results come from a first round with
> a small compute budget. The subproject READMEs list the known limitations; numbers may change
> as stages are reworked.

## Results so far

**Control.** Eight controller variants fly the same seeded courses (figure-eight, helix,
waypoints, gates) at 0.25 to 3 m/s, then 20 disturbance scenarios (mass, wind, gusts, motor lag,
sensor noise). Tuned controllers got the same Optuna budget and objective. The main ones:

| Controller | RMSE at 1 m/s | RMSE at 3 m/s | success at 3 m/s | compute per step |
|---|---|---|---|---|
| PID (untuned, gym-pybullet-drones) | 6.6 cm | 39.8 cm | 20 % | 1.2 ms |
| PID | 2.2 cm | 5.1 cm | 100 % | 0.7 ms |
| LQI | 1.3 cm | 3.5 cm | 100 % | 0.3 ms |
| INDI | 1.1 cm | 3.1 cm | 100 % | 0.7 ms |
| MPPI with learned model | 1.2 cm | 2.8 cm | 100 % | 78 ms |
| PPO (thrust + body rates) | 2.1 cm | 4.8 cm | 50 % | 1.2 ms |

Median over the four path families, 5 test seeds each, nominal conditions.

![tracking error vs. speed](projects/control/results/benchmark/r1/speed_sweep.png)

- INDI is the most accurate and the most robust under wind and payload; LQI is close behind,
  thanks to its integrator.
- MPPI with the learned model is the only controller that survives 100 ms motor lag on every
  course (PPO survives it on the gate course). It needs about 250 times the compute of LQI and
  loses accuracy under heavy payload.
- PPO tracks well at low speed but is less smooth and less reliable than the classical
  controllers. Longer training made it worse, not better (details in the control README).

**Surrogate model.** A physics model with a learned MLP residual predicts the drone 0.42 s
ahead (one MPC horizon) to about 0.1 cm in position. That is 2 to 4 times better than a pure MLP
and about 100 times better than the fitted physics alone, which misses the blade-flapping moment [2].
Neither model can predict unknown wind or payload (about 8 cm error); adding an estimated
disturbance force lowers that error by a factor of 10 to 100 (offline test).

## Repository layout

```
src/dronelab/
  core/         PyBullet backend, disturbances, courses and gates, evaluation
  control/      controllers, PPO environment, tuning protocol
  surrogate/    learned dynamics models (physics, physics + MLP residual, MLP)
  autonomy/     gate flight without a reference path
  planning/     path planning around known obstacles        (planned)
  estimation/   IMU + UWB model, EKF, surrogate-based EKF   (planned)
  vision/       camera rendering, gate detection            (planned)
projects/<subproject>/
  README.md  configs/  scripts/  results/
tests/
```

Seeds, courses, disturbance scenarios and the tuning budget are defined once in `core` and shared
by all subprojects. Course, wind and noise depend only on (seed, scenario), so every comparison
between two methods is paired.

## Quickstart

Python 3.12 or newer. On Windows PyBullet has no pip wheel, so install it from conda-forge:

```bash
conda create -n dronelab -c conda-forge python=3.12 pybullet   # Linux/macOS: venv + pip also works
conda activate dronelab
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -e ".[dev]"

pytest
python projects/control/scripts/run_course.py --course gates --speed 1.5 --controller lqi --seed 1000
```

Run scripts from the repository root. Each subproject README lists its commands.

## License

MIT

## References

1. J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, A. P. Schoellig, "Learning to Fly—a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control," *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2021. [arXiv:2103.02142](https://arxiv.org/abs/2103.02142) · Code: [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

2. L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, D. Scaramuzza, "NeuroBEM: Hybrid Aerodynamic Quadrotor Model," *Robotics: Science and Systems (RSS)*, 2021. [arXiv:2106.08015](https://arxiv.org/abs/2106.08015)
