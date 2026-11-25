# Project status, history and glossary

Short overview of where each subproject stands and what the stage and test-set codes mean.
Details are in the subproject READMEs.

## Status

| # | Subproject | Stages | Status |
|---|---|---|---|
| 1 | [Control](../projects/control/README.md) | R1 done · R2 + R4 tuned, benchmark next · R3 planned | R2: INDI tuned, J 0.70 on validation, most accurate controller so far (rmse_rel 0.43). R4: MPPI with the `phys+mlp` surrogate, tuned J 0.65 on validation (LQI 0.80): about as accurate as LQI (rmse_rel 0.50 vs 0.47) with less than half the command change. R1 in the `real` world: LQI 1.3 cm RMSE at 1 m/s (3.5 cm at 3 m/s), PID 2.2 cm (5.1 cm), DSL-PID 6.6 cm (only 20 % of the flights survive 3 m/s). LQI is the most robust under wind and payload; no controller survives 100 ms motor lag on path courses. LQR is not re-tuned in the `real` world. |
| 2 | [Surrogate](../projects/surrogate/README.md) | S1–S3 done (first round) · GRU, T5 open | `phys+mlp` ≈ 0.1 cm after 0.42 s on T1–T3, 2–4× better than `mlp`; physics alone tilts away. Under disturbances (T4) ~8 cm; disturbance estimation lowers wind/payload errors by a factor of 10–100. Rollout 512 × 20 steps: 32–103 ms, not real-time yet. |
| 3 | Autonomy | A1–A4 | planned |
| 4 | State estimation | Z1–Z4 | planned |
| 5 | Vision | V1–V4 | planned |

R4 is done before R2/R3 because it needs the finished surrogate; R2 and R3 do not.

## History

| Date | Commit | Content |
|---|---|---|
| 2026-11-20 | be53f6b | Stage 0: PyBullet backend, PID baseline, courses, metrics |
| 2026-11-23 | 92a3a30 | Stage 1 in the old `ideal` world: PID, LQR, LQI, tuning protocol, first benchmark |
| 2026-11-23 | d2cf164 | 0a: restructure into `dronelab`, one subproject per stack layer |
| 2026-11-23 | 15c6cca | 0b: `real` world with model gap, surrogate data generation |
| 2026-11-24 | 21852c2 | R1 again: PID and LQI re-tuned in the `real` world, benchmark `r1` flown (→ T3/T4) |
| 2026-11-24 | 0fd15a1 | Surrogate S2/S3: `phys`, `phys+mlp`, `mlp` |
| 2026-11-24 | 6e4a4b8 | R1 results: control README, analysis, gate figure |
| 2026-11-24 | 5667788 | Surrogate: URDF physics baseline, offline disturbance estimation |
| 2026-11-24 | f464238 | R4.0: numba physics step for MPPI rollouts, timing study, this status page |
| 2026-11-24 | 432a42f | R4.1–R4.2: MPPI controller (thrust + body rates), smoke test |
| 2026-11-25 | 12c4e35 | R4.3: reference candidate + smoothing after the pilot diagnosis, `mppi_hybrid` tuned (80 trials) |
| 2026-11-25 | – | R4.3c: candidate-only ablation (tilt cost relative to the reference tested, < 1 %, not kept), comparison with two MPPI papers, benchmark configs |

## Glossary

### Stages

| Code | Meaning |
|---|---|
| 0a / 0b | restructure into `dronelab` / new `real` world, R1 re-tuned and re-flown |
| R1–R4 | control: R1 PID / LQR / LQI · R2 INDI · R3 learned controller (PPO) · R4 MPPI with the surrogate |
| S1–S3 | surrogate: S1 data · S2 models and training · S3 test sets and evaluation |
| A1–A4 | autonomy: A1 planner + best controller · A2 RL through gates · A3 MPPI with gate progress · A4 comparison |
| Z1–Z4 | state estimation: Z1 IMU model · Z2 EKF · Z3 learned estimator · Z4 controllers on the estimated state |
| V1–V4 | vision: V1 camera images · V2 gate detection · V3 gate pose · V4 gate flight with detected gates |

### Benchmark parts

Config `projects/control/configs/benchmark/r1.yaml`, test seeds 1000–1004.

| Part | Content |
|---|---|
| A | speed sweep: figure8, helix, waypoints, gates at 0.25 / 0.5 / 1 / 1.5 m/s and extrapolation to 2 / 3 m/s, plus hover; nominal |
| B | disturbances: 20 scenarios on hover, figure8, gates at 1 m/s |

Benchmark part A is not the same as the autonomy stages A1–A4.

Part B scenarios: nominal · mass ×0.8 / 0.9 / 1.1 / 1.2 / 1.3 · wind 0.03 / 0.06 / 0.09 N ·
gusts 0.01 / 0.02 / 0.04 N · motor lag 50 / 100 ms · sensor noise low / mid / high · `ideal` world
(cross-check) · combo_mid · combo_strong.

### Surrogate test sets

| Set | Content | Flights | Question |
|---|---|---|---|
| T1 | new excitation and course flights (seeds 22 000+) | 99 | does the model learn? |
| T2 | T1 flights faster than 1.5 m/s, models trained on flights ≤ 1.5 m/s only | 65 | speed extrapolation |
| T3 | flights of benchmark part A (DSL-PID, PID, LQI) | 345 | MPC-like courses, unseen controller |
| T4 | flights of benchmark part B (without `ideal`) | 803 | unknown disturbances |
| T5 | flights of INDI and the learned controller | – | later: does the model depend on the controller? |

Training: 583 flights (seeds 20 000–20 599), validation: 98 flights (21 000–21 099). See
[DATASETS.md](../projects/surrogate/DATASETS.md).

### Worlds, seeds, controllers, models

| Term | Meaning |
|---|---|
| `ideal` world | plain PyBullet, cross-check only |
| `real` world | default: PyBullet drag, ground effect, downwash, 20 ms motor lag, quadratic body drag, axial-inflow thrust loss, blade-flapping moment |
| seeds | tuning 0–99 · validation 100–199 · test 1000+ · RL from 10 000 · surrogate 20 000+ |
| `dsl_pid` | untuned reference PID of gym-pybullet-drones |
| `pid` | cascaded PID, tuned |
| `lqr` / `lqi` | LQR (not re-tuned in the `real` world) / LQR with integrator, tuned |
| `indi` | incremental nonlinear dynamic inversion (force + torque from measured accelerations), tuned |
| `mppi_hybrid` / `mppi_urdf` | MPPI with the `phys+mlp` surrogate (tuned) / with URDF physics (hybrid gains) |
| `_nodob` / `_cand` | MPPI ablations: disturbance estimate off / reference candidate alone (geometric PD) |
| `phys` | physics model with fitted parameters |
| `phys (URDF)` | physics with data-sheet parameters, not fitted; physics baseline for R4 |
| `phys+mlp` | physics + MLP residual, most accurate model |
| `mlp` | pure MLP, fastest model |
