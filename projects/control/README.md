# Subproject 1 – Control

**Which controller follows a given path most accurately and most robustly when the world doesn't match the model?**

Part of [dronelab](../../README.md). All controllers fly the same randomised courses under the same disturbances and are tuned with the same fixed protocol:

| Stage | Methods | Status |
|---|---|---|
| R1 | cascaded PID (reference: gym-pybullet-drones `DSLPIDControl` [1]), own PID, LQI – tuned and benchmarked in the `real` world | done |
| R2 | INDI (incremental nonlinear dynamic inversion) | planned |
| R3 | learned controller: PPO policy (collective thrust + body rates) | planned |
| R4 | sampling MPC (MPPI) with the learned dynamics model from subproject 2 | planned |

Disturbances: mass change (payload), constant wind + gusts, motor lag, sensor noise, drag + ground effect, combinations.

## Architecture

```
controllers (PID / LQR / ...) ──act(state, ref)──►  Backend interface  ◄── PyBullet (gym-pybullet-drones)
        ▲                                                │
        └──────── courses: reference(t) ◄── runner ──────┴──► metrics / plots / benchmark
```

Controllers live in `src/dronelab/control/`, the shared simulator backend, courses, disturbances and evaluation in `src/dronelab/core/`. All controllers talk to a simulator-agnostic `Backend` (`reset`, `step(rpm)`, `observe`).

## Commands

Run from the repository root (installation: see the [main README](../../README.md#quickstart)).

```bash
python projects/control/scripts/run_course.py --config projects/control/configs/hover_pid.yaml
python projects/control/scripts/run_course.py --course gates --speed 1.5 --controller lqi --seed 1000
python projects/control/scripts/speed_test.py                     # simulator throughput on your machine

python projects/control/scripts/tune.py --controller pid lqi --trials 200            # fixed tuning protocol
python projects/control/scripts/benchmark.py --config projects/control/configs/benchmark/r1.yaml
python projects/control/scripts/analyze.py      # tables, plots and the results block of this README
python projects/control/scripts/plot_course.py --course gates --speed 1.5 --seed 1000 --controllers dsl_pid pid lqi   # 3D + side views
```

## Courses

Every course is generated deterministically from `(family, speed, seed)`:

| Family | Random parameters | Use |
|---|---|---|
| `hover` | target altitude 0.5–1.5 m (take-off from 0.1 m) | tuning + test |
| `figure8` | width, aspect ratio, rotation, altitude | tuning + test |
| `helix` | radius 0.5–2 m, 1–2 turns, climb | tuning + test |
| `waypoints` | 4–6 random points, ≥ 1 m apart | tuning + test |
| `gates` | 4–5 gates (0.5 × 0.5 m) as objects: spacing 1.2–1.8 m, lateral offset 0.2–1.2 m (usually alternating sides), height 0.7–1.6 m | **test only** (unseen family) |

The path is a cubic spline re-parametrised by arc length and flown at constant speed with a 2 m/s² ramp at start and end; the reference provides position, velocity and acceleration. A course is accepted only if it needs ≤ 80 % of the maximum thrust and ≤ 45° tilt (otherwise re-drawn, at most 50 times, else marked *infeasible* and not flown). A gate counts as passed only if the drone crosses the gate plane inside the opening in the flight direction. Path courses start exactly on the reference at rest, so the RMSE measures tracking, not the recovery from a random start offset.

## Metrics

Success rate (no crash, inside the arena, tilt < 80°) and – over successful flights – position RMSE, max error, steady-state error, settling time, IAE / ISE / ITAE, control effort, command smoothness, gates passed and controller compute time per step.

## Simulated world

| World | Contents | Used for |
|---|---|---|
| `ideal` | gym-pybullet-drones without aerodynamics, instant motors – identical to the textbook model (thrust ∝ rpm²) | cross-check scenario in the benchmark; the first R1 run (incl. LQR) is in the git history |
| `real` | PyBullet rotor drag + ground effect + downwash, first-order motor lag τ = 20 ms, plus three extra effects | default: R1 results below, R2–R4, all other subprojects |

The extra effects of the `real` world (`src/dronelab/core/world.py`, `AeroConfig`) are physically motivated but their coefficients are **assumptions** sized for the Crazyflie, so that each effect is noticeable at 3 m/s while nominal courses stay flyable:

| Effect | Model | Size |
|---|---|---|
| Quadratic body drag | F = −½ ρ C_d A \|v\| v, C_d A = 1·10⁻³ m² | 5.5 mN at 3 m/s (PyBullet's linear rotor drag: 16.7 mN) |
| Axial inflow | T_i = k_F Ω_i² (1 − 0.3 · v_ax,i / v_h), v_h = 4.0 m/s (momentum theory), v_ax,i includes the rotation | −7.7 % thrust when climbing at 1 m/s; damps roll and pitch rates |
| Blade flapping | M = −2.5·10⁻⁸ · ΣΩ · (e_z × v_⊥) | nose-up, 5.5 % of the maximum roll torque at 3 m/s |

With these effects the default LQR tracks 2–4× worse than in the ideal world (reference scales below), without crashing.

## Tuning and evaluation protocol

Published comparisons often pit a carefully designed method against a PID baseline whose tuning is not reported. Here every tunable controller gets **the same, fixed budget and objective**:

* **Objective** `J = mean_f(RMSE_f / RMSE_ref,f) + 0.3 · mean_f(smooth_f / smooth_ref,f) + 1.0 · crash rate`, nominal conditions only. Each course family f is normalised by fixed reference scales (default LQR), so hover take-off and path tracking count equally; *smooth* is the mean motor-command change per step, so overly aggressive gains are penalised (10 % smoother ≙ 3 % more error).
* **Optuna TPE**, 200 trials × 8 episodes (2 per family hover / figure8 / helix / waypoints, speed U(0.25, 1.5) m/s, tuning seeds 0–99). Each parameter is searched log-uniformly within ×/÷ 5 of its hand-set default (×/÷ 10 for integral gains); trial 0 is the default.
* **Selection**: the best 5 trials are re-flown on 40 validation episodes (seeds 100–199); the best on validation wins – this guards against lucky trials.
* **Anchor**: the original `DSLPIDControl` of gym-pybullet-drones [1] is *not* tuned. If the tuned own PID were worse than this anchor, the tuning would be broken.
* **Test**: frozen controllers on seeds 1000+ that were never used before. All controllers fly exactly the same courses, start states, wind directions and noise realisations, so results are compared pairwise (Wilcoxon signed-rank).
* For comparison: Madebo et al. [2] tune their controller with PSO, 20 particles × 30 iterations = 600 evaluations.

R1 benchmark (5 test seeds, `real` world): **Part A** – speed sweep 0.25 / 0.5 / 1 / 1.5 m/s and *extrapolation* to 2 / 3 m/s on figure8, helix, waypoints, gates (+ hover); **Part B** – 20 disturbance scenarios (mass 0.8–1.3×, constant wind 0.03–0.09 N, gusts, motor lag 50 / 100 ms instead of the world's 20 ms, sensor noise, the `ideal` world as cross-check, two combinations) on hover, figure8 and gates at 1 m/s.

LQR is not part of R1 in the `real` world. To save tuning time (~1.5 CPU-h per controller) only PID and LQI were re-tuned: in the first, ideal-world run LQR was as accurate as LQI but clearly less robust, because it has no integral action.

## Results (R1)

**Key findings (`real` world):**

* **LQI is the most accurate controller at every speed:** 1.3 cm median RMSE at 1 m/s and 3.5 cm at 3 m/s, against 2.2 / 5.1 cm for the tuned PID. Paired on the same flights, LQI beats PID in 87 % of cases. The untuned DSL-PID anchor reaches 6.6 cm at 1 m/s and survives only 20 % of the flights at 3 m/s.
* **The model gap costs accuracy:** in the ideal world the same controllers tracked at ~0.6 cm at 1 m/s (git history); with the extra aerodynamics and the motor lag it is 2–4× more.
* **The integrator decides the robustness:** under constant wind (0.09 N) LQI stays at 4.2 cm on figure-eights, PID degrades to 15.8 cm (×8); with 30 % extra mass 6.3 against 12.1 cm.
* **Motor lag is the hard limit:** 50 ms costs almost nothing, at 100 ms no controller finishes a figure-eight or gate course, and the strong combination is flown by nobody.

![gates at 1.5 m/s: 3D and side views](results/figures/gates_1.5ms_seed1000.png)

<!-- STAGE1-RESULTS:START -->
### Tuning (validation set, 40 episodes, seeds 100–139)

| Controller | Trials | J default | J tuned | mean RMSE [cm] | CPU-h |
|---|---|---|---|---|---|
| PID | 200 | 2.7098 | 0.9206 | 14.3 → 6.1 | 1.40 |
| LQI | 200 | 1.1292 | 0.8021 | 6.8 → 5.6 | 1.52 |

Tuning convergence – PID: best J 1.088, 3.7 % gained in the second half; LQI: best J 0.902, 2.7 % gained in the second half.

![tuning convergence](results/benchmark/r1/tuning_convergence.png)

### Part A – speed sweep (nominal)

Median position RMSE [cm] of successful flights / success rate, over 4 families × 5 seeds. Speeds above 1.5 m/s are outside the tuning range.

| Controller | 0.25 m/s | 0.5 m/s | 1 m/s | 1.5 m/s | 2 m/s | 3 m/s |
|---|---|---|---|---|---|---|
| DSL-PID (untuned anchor) | 0.8 / 100 % | 2.3 / 100 % | 6.6 / 100 % | 11.4 / 100 % | 15.0 / 85 % | 39.8 / 20 % |
| PID (tuned) | 0.6 / 100 % | 1.2 / 100 % | 2.2 / 100 % | 3.1 / 100 % | 3.9 / 100 % | 5.1 / 100 % |
| LQI (tuned) | 0.2 / 100 % | 0.5 / 100 % | 1.3 / 100 % | 1.9 / 100 % | 2.3 / 100 % | 3.5 / 100 % |

Infeasible courses (rejected by the thrust/tilt check, not flown): 3 m/s: 10.

Gate course – all gates passed (successful flights): DSL-PID 96 %; PID 100 %; LQI 100 %.

![speed sweep](results/benchmark/r1/speed_sweep.png)

### Accuracy vs. smoothness (Part A, one dot per flight)

Lower left is better. A controller can buy accuracy with more aggressive motor commands; the tuning objective weighs this trade-off (w_s = 0.3).

![accuracy vs smoothness](results/benchmark/r1/accuracy_vs_smoothness.png)

### Error vs. required acceleration (per control step)

Every control step of the nominal speed sweep is sorted by the reference acceleration at that moment (straights ≈ 0, tight turns and braking > 4 m/s²). Bars: median error, whiskers: interquartile range.

![error vs acceleration](results/benchmark/r1/error_vs_acceleration.png)

### Part B – disturbances (1 m/s), per course family

Success rate and median RMSE [cm] of successful flights; in brackets the RMSE relative to the nominal scenario of the same family (1.0 = no degradation). Hover includes the take-off from 0.1 m, so its RMSE is an order of magnitude larger.

<details><summary><b>hover</b></summary>

| Scenario | DSL-PID | PID | LQI |
|---|---|---|---|
| nominal | 100 %, 14.6 (1.00) | 100 %, 14.7 (1.00) | 100 %, 15.3 (1.00) |
| mass_0.8 | 100 %, 14.2 (0.97) | 100 %, 16.3 (1.11) | 100 %, 15.0 (0.98) |
| mass_0.9 | 100 %, 14.3 (0.98) | 100 %, 15.1 (1.03) | 100 %, 15.0 (0.98) |
| mass_1.1 | 100 %, 15.3 (1.04) | 100 %, 15.2 (1.03) | 100 %, 15.8 (1.03) |
| mass_1.2 | 100 %, 16.1 (1.10) | 100 %, 16.4 (1.12) | 100 %, 16.5 (1.08) |
| mass_1.3 | 100 %, 17.2 (1.18) | 100 %, 18.3 (1.24) | 100 %, 17.4 (1.13) |
| wind_0.03 | 100 %, 15.5 (1.06) | 100 %, 15.5 (1.06) | 100 %, 15.4 (1.00) |
| wind_0.06 | 100 %, 17.8 (1.21) | 100 %, 17.8 (1.21) | 100 %, 15.5 (1.01) |
| wind_0.09 | 100 %, 21.0 (1.44) | 100 %, 21.1 (1.43) | 100 %, 15.6 (1.02) |
| gust_0.01 | 100 %, 14.9 (1.02) | 100 %, 14.8 (1.01) | 100 %, 15.4 (1.00) |
| gust_0.02 | 100 %, 15.7 (1.07) | 100 %, 15.5 (1.05) | 100 %, 15.6 (1.02) |
| gust_0.04 | 100 %, 18.4 (1.25) | 100 %, 18.1 (1.23) | 100 %, 16.5 (1.08) |
| motor_0.05 | 100 %, 15.1 (1.03) | 100 %, 15.1 (1.03) | 100 %, 15.7 (1.03) |
| motor_0.1 | 100 %, 15.6 (1.06) | 100 %, 15.7 (1.07) | 100 %, 16.3 (1.06) |
| noise_low | 100 %, 14.7 (1.00) | 100 %, 14.7 (1.00) | 100 %, 15.3 (1.00) |
| noise_mid | 100 %, 14.9 (1.02) | 100 %, 14.7 (1.00) | 100 %, 15.3 (1.00) |
| noise_high | 0 %, – | 100 %, 14.7 (1.00) | 100 %, 15.3 (1.00) |
| ideal | 100 %, 14.2 (0.97) | 100 %, 14.1 (0.96) | 100 %, 14.9 (0.97) |
| combo_mid | 100 %, 19.6 (1.34) | 100 %, 19.1 (1.30) | 100 %, 16.1 (1.05) |
| combo_strong | 0 %, – | 0 %, – | 0 %, – |

</details>

<details><summary><b>figure8</b></summary>

| Scenario | DSL-PID | PID | LQI |
|---|---|---|---|
| nominal | 100 %, 10.7 (1.00) | 100 %, 1.9 (1.00) | 100 %, 2.6 (1.00) |
| mass_0.8 | 100 %, 9.3 (0.87) | 100 %, 7.6 (3.93) | 100 %, 4.3 (1.61) |
| mass_0.9 | 100 %, 9.9 (0.92) | 100 %, 4.0 (2.08) | 100 %, 3.1 (1.16) |
| mass_1.1 | 100 %, 12.0 (1.12) | 100 %, 4.6 (2.37) | 100 %, 3.3 (1.26) |
| mass_1.2 | 100 %, 13.7 (1.28) | 100 %, 8.3 (4.28) | 100 %, 4.7 (1.76) |
| mass_1.3 | 100 %, 15.6 (1.45) | 100 %, 12.1 (6.24) | 100 %, 6.3 (2.37) |
| wind_0.03 | 100 %, 12.0 (1.12) | 100 %, 5.5 (2.83) | 100 %, 2.9 (1.11) |
| wind_0.06 | 100 %, 14.6 (1.36) | 100 %, 10.5 (5.45) | 100 %, 3.5 (1.33) |
| wind_0.09 | 100 %, 18.5 (1.72) | 100 %, 15.8 (8.19) | 100 %, 4.2 (1.60) |
| gust_0.01 | 100 %, 10.9 (1.01) | 100 %, 3.0 (1.55) | 100 %, 3.0 (1.15) |
| gust_0.02 | 100 %, 11.0 (1.03) | 100 %, 4.8 (2.50) | 100 %, 4.1 (1.55) |
| gust_0.04 | 100 %, 14.7 (1.37) | 100 %, 9.2 (4.78) | 100 %, 6.2 (2.33) |
| motor_0.05 | 100 %, 11.5 (1.07) | 100 %, 1.9 (0.98) | 100 %, 3.2 (1.21) |
| motor_0.1 | 0 %, – | 0 %, – | 0 %, – |
| noise_low | 100 %, 10.8 (1.01) | 100 %, 1.9 (0.99) | 100 %, 2.6 (0.98) |
| noise_mid | 80 %, 11.4 (1.06) | 100 %, 2.0 (1.01) | 100 %, 2.5 (0.96) |
| noise_high | 0 %, – | 100 %, 2.2 (1.12) | 100 %, 2.5 (0.95) |
| ideal | 100 %, 11.8 (1.10) | 100 %, 2.5 (1.31) | 100 %, 3.2 (1.21) |
| combo_mid | 80 %, 16.2 (1.51) | 100 %, 12.0 (6.18) | 100 %, 4.6 (1.73) |
| combo_strong | 0 %, – | 0 %, – | 0 %, – |

</details>

<details><summary><b>gates</b></summary>

| Scenario | DSL-PID | PID | LQI |
|---|---|---|---|
| nominal | 100 %, 6.6 (1.00) | 100 %, 2.3 (1.00) | 100 %, 1.4 (1.00) |
| mass_0.8 | 100 %, 6.4 (0.98) | 100 %, 6.3 (2.69) | 100 %, 3.4 (2.48) |
| mass_0.9 | 100 %, 6.1 (0.94) | 100 %, 3.6 (1.53) | 100 %, 2.0 (1.50) |
| mass_1.1 | 100 %, 7.4 (1.13) | 100 %, 4.0 (1.73) | 100 %, 2.1 (1.57) |
| mass_1.2 | 100 %, 8.6 (1.31) | 100 %, 6.8 (2.92) | 100 %, 3.5 (2.57) |
| mass_1.3 | 100 %, 10.3 (1.56) | 100 %, 9.8 (4.22) | 100 %, 5.0 (3.68) |
| wind_0.03 | 100 %, 7.3 (1.12) | 100 %, 4.8 (2.06) | 100 %, 1.5 (1.07) |
| wind_0.06 | 100 %, 9.6 (1.46) | 100 %, 9.3 (4.00) | 100 %, 1.9 (1.39) |
| wind_0.09 | 100 %, 12.9 (1.97) | 100 %, 14.4 (6.16) | 100 %, 2.4 (1.81) |
| gust_0.01 | 100 %, 7.0 (1.07) | 100 %, 3.4 (1.44) | 100 %, 2.1 (1.58) |
| gust_0.02 | 100 %, 8.5 (1.29) | 100 %, 5.2 (2.24) | 100 %, 3.6 (2.65) |
| gust_0.04 | 100 %, 12.8 (1.95) | 100 %, 10.2 (4.40) | 100 %, 6.8 (5.02) |
| motor_0.05 | 100 %, 6.7 (1.02) | 100 %, 2.3 (0.98) | 100 %, 1.3 (1.00) |
| motor_0.1 | 0 %, – | 0 %, – | 0 %, – |
| noise_low | 100 %, 7.2 (1.10) | 100 %, 2.4 (1.02) | 100 %, 1.4 (1.04) |
| noise_mid | 100 %, 8.8 (1.34) | 100 %, 2.4 (1.03) | 100 %, 1.5 (1.11) |
| noise_high | 0 %, – | 100 %, 2.5 (1.07) | 100 %, 1.7 (1.23) |
| ideal | 100 %, 6.1 (0.93) | 100 %, 1.0 (0.41) | 100 %, 1.4 (1.02) |
| combo_mid | 100 %, 13.3 (2.03) | 100 %, 11.3 (4.85) | 100 %, 3.6 (2.65) |
| combo_strong | 0 %, – | 0 %, – | 0 %, – |

</details>

![heatmap](results/benchmark/r1/disturbance_heatmap.png)

Degradation curves (median, bars = interquartile range of the 5 seeds; missing points = no successful flight):

![degradation](results/benchmark/r1/degradation.png)

### Motor saturation

Share of control steps in which at least one motor command sits at its limit (max rpm, or the minimum: 0 rpm for own controllers, PWM floor ≈ 9 440 rpm for DSL-PID). Mean over flights; crashed flights included.

| Controller | 0.25 m/s | 0.5 m/s | 1 m/s | 1.5 m/s | 2 m/s | 3 m/s |
|---|---|---|---|---|---|---|
| DSL-PID | 0 % | 0 % | 0 % | 0 % | 1 % | 7 % |
| PID | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| LQI | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |

Nominal flights stay below the limits at every speed – the feasibility check admits only courses that need ≤ 80 % of the maximum thrust. Saturation appears under disturbances, above all with motor lag:

![saturation](results/benchmark/r1/saturation.png)

### Paired comparison (same course, seed and disturbance; both controllers finished)

Hover and path courses are tested separately because their RMSE scales differ.

| Courses | A vs B | n | median(RMSE_A − RMSE_B) [cm] | A better in | Wilcoxon p |
|---|---|---|---|---|---|
| hover | dsl_pid vs pid | 95 | -0.08 | 65 % | 1.2e-04 |
| hover | dsl_pid vs lqi | 95 | -0.37 | 69 % | 1.2e-01 |
| hover | pid vs lqi | 100 | -0.09 | 57 % | 2.8e-01 |
| path courses | dsl_pid vs pid | 267 | +4.10 | 4 % | 5.7e-44 |
| path courses | dsl_pid vs lqi | 267 | +6.47 | 0 % | 1.5e-45 |
| path courses | pid vs lqi | 290 | +1.00 | 13 % | 7.9e-38 |

Controller compute time per step (median): dsl_pid 1.244 ms, pid 0.731 ms, lqi 0.303 ms.

_1275 episodes, git 15c6cca-dirty, config 5aad03090e_
<!-- STAGE1-RESULTS:END -->

## References

1. J. Panerati, H. Zheng, S. Zhou, J. Xu, A. Prorok, A. P. Schoellig, "Learning to Fly—a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control," *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2021. [arXiv:2103.02142](https://arxiv.org/abs/2103.02142) · Code: [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
2. M. M. Madebo, M. Y. Erten, H. Aydilek, "PSO-optimized physics-informed neural network-based fuzzy-sliding mode controller design for quadrotor altitude stabilization and trajectory tracking," *Ain Shams Engineering Journal*, vol. 17, art. 104312, 2026. [doi:10.1016/j.asej.2026.104312](https://doi.org/10.1016/j.asej.2026.104312)
