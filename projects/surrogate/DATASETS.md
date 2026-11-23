# Surrogate datasets – generation rules

All data live in `data/surrogate/` (not in git). Every generated folder contains a `manifest.json` with the exact configuration, git commit, seed ranges, number of flights, total flight time and crashes; a copy of this file is placed in `data/surrogate/README.md`.

## Common rules (all splits)

| Rule | Value |
|---|---|
| World | `real`: PyBullet drag + ground effect + downwash, motor lag τ = 20 ms, extra aerodynamics (quadratic body drag, axial-inflow thrust change, blade-flapping moment) – see `src/dronelab/core/world.py` |
| Drone | Crazyflie 2.x (CF2X), nominal mass |
| Rates | physics 240 Hz, control and logging 48 Hz (the model predicts one control step, Δt ≈ 20.8 ms) |
| Logged per step | true state (position, quaternion, velocity, angular velocity), commanded and applied motor speeds, reference; metadata (seed, controller, course, world, disturbances) in `meta` |
| Model inputs | commanded motor speeds only – the applied (lagged) speeds are logged for analysis, a model must not use them |
| Start | on the reference at rest, motors at hover speed |
| Crashes | the flight stops at the crash, data up to the crash are kept (`meta.crashed`) |
| Infeasible courses | not flown, no file (counted in the manifest) |
| Split unit | whole flights; a seed belongs to exactly one split (checked by the generator and a test) |

## Generated splits (`projects/surrogate/configs/data.yaml`)

| Folder | Split | Excitation flights (seeds) | Course flights (seeds) | ≈ flight time |
|---|---|---|---|---|
| `train/` | training | 300 (20 000–20 299) | 300 (20 300–20 599) | 3 h |
| `val/` | validation, early stopping, hyper-parameters | 50 (21 000–21 049) | 50 (21 050–21 099) | 30 min |
| `T1_in_distribution/` | **T1** – same kinds of flights, unseen seeds | 50 (22 000–22 049) | 50 (22 050–22 099) | 30 min |

**Excitation flights** (`excitation_lqi_<seed>.npz`)
* Course family `excitation`: 6–10 random points in a 6 × 6 m box, height 0.4–2.5 m, each point with 25 % probability in the ground-effect zone (0.15–0.3 m), consecutive points ≥ 1 m apart.
* Variable speed profile: segments of 1–3 m, segment speed U(0.25, v_max), v_max reached in one segment, ramp acceleration 2 m/s².
* v_max per flight U(0.5, 3.0) m/s from its own random stream (`[seed, 555]`).
* Controller: tuned LQI + **dither** on the commanded wrench from t = 1 s: per channel (thrust, roll, pitch, yaw torque) 50 % multisine (4 frequencies in 0.3–6 Hz) + 50 % PRBS (hold 0.05–0.4 s), amplitude 10 % of hover thrust / maximum torque, seeded with `[seed, 31337]`.

**Course flights** (`course_<controller>_<family>_<seed>.npz`)
* Families hover, figure8, helix, waypoints (seed mod 4); controller PID, LQR, LQI in turn (⌊seed / 4⌋ mod 3) → every family × controller pair within 12 consecutive seeds.
* Same shape generator as the control benchmark, but with the **variable** speed profile (v_max as above; hover is a take-off to 0.5–1.5 m).
* Tuned gains from `projects/control/results/tuning_real/`, no dither.

Both kinds pass the feasibility check of the course generator (≤ 80 % thrust, ≤ 45° tilt along the reference).

## Test sets from other sources

| Folder | Set | Source | Rules |
|---|---|---|---|
| – | **T2** speed extrapolation | no folder: models are additionally trained only on `train/` flights whose measured maximum speed is ≤ 1.5 m/s (`T2_MAX_SPEED`), and tested on the T1 flights above 1.5 m/s | selection by the *measured* speed of the whole flight, never by time step |
| `T3_courses/` | **T3** benchmark courses | part A of the R1 benchmark (`projects/control/configs/benchmark/r1.yaml`, `--log-flights`): figure8, helix, waypoints, **gates** at constant 0.25–3 m/s + hover, test seeds 1000–1004, DSL-PID / PID / LQR / LQI | constant cruise speed, unseen family (gates), file `A_<ctrl>_<scenario>_<family>_<speed>_<seed>.npz` |
| `T4_disturbances/` | **T4** unknown conditions | part B of the same benchmark: 20 scenarios (mass, wind, gusts, motor lag 50 / 100 ms, sensor noise, combinations) on hover, figure8, gates at 1 m/s | the model has never seen disturbances; `B_*_ideal_*` flights are in another world and are not part of T4 |
| `T5_controllers/` | **T5** other controllers | later: flights of INDI and the learned controller | same courses as T3 |

## What is not allowed

* No seed of T1, T3, T4 or T5 may be used for training, model selection or hyper-parameters.
* No disturbance flights (T4) in training.
* No mixing of worlds: everything except the `ideal` cross-check is flown in the `real` world.

## Archive

`data/archive/stage1_ideal_world/` – the 1 660 flight logs of the stage-1 benchmark in the old ideal world (no aerodynamics, no motor lag). Not used by the surrogate; kept for the stage-1 analysis.
