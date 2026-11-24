# dronelab

**A CPU-only quadrotor autonomy stack – one layer per subproject, each with its own fair benchmark.**

A Crazyflie 2.x (CF2X) flies in PyBullet (gym-pybullet-drones [1]). Every layer of the stack answers its own question:

```
 Vision            →  Where is the next gate?                  (subproject 5)
 State estimation  →  Where am I, how fast, how tilted?        (subproject 4)
 Autonomy          →  Where do I fly?                          (subproject 3)
 Control           →  How do I follow the path?                (subproject 1)
 Surrogate model   →  How does the drone respond?              (subproject 2)
```

| # | Subproject | Question | Status |
|---|---|---|---|
| 1 | [Control](projects/control/README.md) | Which controller follows a given path most accurately and robustly? | R1 done (PID / LQI, fair tuning, benchmark in the `real` world) |
| 2 | [Surrogate modelling](projects/surrogate/README.md) | How well does a learned model predict the dynamics, especially outside its training data? | first round done (physics, physics + MLP residual, MLP) |
| 3 | Autonomy | How does the drone get through the gates fastest and safest without a given path? | planned |
| 4 | State estimation | How well do the controllers fly when the state has to be estimated from sensors? | planned |
| 5 | Vision | Can the drone find and fly through the gates from camera images? | planned |

Project history and a glossary of the stage and test-set codes (R1–R4, T1–T5, …): [docs/STATUS.md](docs/STATUS.md).

**Key result (control, R1):** in a simulated world with aerodynamics and motor lag that the controllers' model does not contain, the tuned LQI tracks at 1.3 cm RMSE at 1 m/s (3.5 cm at 3 m/s), the tuned PID at 2.2 cm (5.1 cm) and the untuned reference PID at 6.6 cm (only 20 % of the flights survive 3 m/s); thanks to its integrator LQI is by far the most robust under wind and payload, and no controller survives 100 ms motor lag on path courses. Details in the [control README](projects/control/README.md).

**Key result (surrogate):** a physics prior with a learned MLP residual predicts the drone 0.42 s ahead (one MPC horizon) to ~0.1 cm in position, 2–4× better than a pure MLP and ~100× better than the fitted physics alone, which misses the blade-flapping moment; it also generalises better to an unseen controller. Neither model can predict unknown wind or payload (~8 cm). Details in the [surrogate README](projects/surrogate/README.md).

Everything runs on a **CPU only** (developed on a 2-core laptop).

## Repository layout

```
src/dronelab/
  core/         simulator backend (PyBullet), disturbances, courses and gates, evaluation, rigid-body model
  control/      controllers (PID, LQR/LQI, ...) and the tuning protocol
  surrogate/    learned dynamics models (physics prior, residual MLP)
  autonomy/     gate flight without a reference path        (planned)
  estimation/   IMU model, EKF, learned estimator           (planned)
  vision/       camera rendering, gate detection            (planned)
projects/<subproject>/
  README.md  configs/  scripts/  results/
tests/
```

Shared protocols (seeds, course generator, disturbance scenarios, tuning budget) are defined once in `core` and used by every subproject, so results stay comparable across layers.

## Quickstart

Requires Python ≥ 3.12. On Windows, PyBullet has no pip wheel – use conda-forge:

```bash
conda create -n dronelab -c conda-forge python=3.12 pybullet   # Linux/macOS: a venv + pip works too
conda activate dronelab
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -e ".[dev]"

pytest                                                          # unit + physics sanity tests
python projects/control/scripts/run_course.py --course gates --speed 1.5 --controller lqi --seed 1000
```

Scripts are run from the repository root; each subproject README lists its commands.

## License

MIT

## References

1. J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, A. P. Schoellig, "Learning to Fly—a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control," *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2021. [arXiv:2103.02142](https://arxiv.org/abs/2103.02142) · Code: [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)

2. L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, D. Scaramuzza, "NeuroBEM: Hybrid Aerodynamic Quadrotor Model," *Robotics: Science and Systems (RSS)*, 2021. [arXiv:2106.08015](https://arxiv.org/abs/2106.08015)
