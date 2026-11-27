# Project status, history and glossary

Short overview of where each subproject stands and what the stage and test-set codes mean.
Details are in the subproject READMEs.

## Status

| # | Subproject | Stages | Status |
|---|---|---|---|
| 1 | [Control](../projects/control/README.md) | R1–R4 benchmarked | R1 (`real` world): LQI 1.3 cm RMSE at 1 m/s (3.5 cm at 3 m/s), PID 2.2 cm (5.1 cm), untuned DSL-PID 6.6 cm (20 % survive 3 m/s). R2: INDI tuned to J 0.70 on validation, most accurate and most robust under wind and payload; INDI-SG variant (Savitzky-Golay acceleration) about equal. R3: PPO (thrust + body rates), best checkpoint J 1.62; tracks about as well as LQI on validation (6.2 vs 5.6 cm) but with three times the command change; training longer than ~1 M steps made it worse (two ablations running). R4: MPPI with the `phys+mlp` surrogate, J 0.65, only controller that survives 100 ms motor lag on every course, ~80 ms per step. Only MPPI survives `combo_strong` at all regularly (60–80 %). |
| 2 | [Surrogate](../projects/surrogate/README.md) | S1–S3 done (first round) · GRU, T5 open | `phys+mlp` ≈ 0.1 cm after 0.42 s on T1–T3, 2–4× better than `mlp`; physics alone tilts away. Under disturbances (T4) ~8 cm; an estimated disturbance force lowers wind / payload errors by a factor of 10–100. Rollout 512 × 20 steps: 32–103 ms, not real-time yet. |
| 3 | Autonomy | A0, A1 done · A2, A4 next · A3 dropped | Gate racing on the `gates` courses without a given path; touching a gate frame counts as a crash. A1 (spline through the gates + curvature-limited speed profile): best budget 0.6 g, median lap time 3.7 s at 100 % success for INDI and LQI (40 validation seeds). A1+ (crossing points optimised inside the opening): 3.67 s for INDI (98 %); LQI hits the frames more often (best 4.1 s at 0.4 g). A2: PPO with gate-progress reward. A3 (MPPI with gate progress) dropped: compute time and no clear gain over the classical controllers in R4. |
| 4 | Path planning | P1–P4 planned | Pillar forest with known map: front-ends A* and RRT* × back-ends spline + curvature profile and minimum snap, flown by INDI/LQI. Question: is the shortest path also the fastest flight? |
| 5 | State estimation | Z1–Z4 planned | IMU (noise, bias random walk) + UWB-like position (~30 Hz, 5–10 cm, dropouts); error-state EKF against an EKF with the `phys+mlp` surrogate as process model (with / without external-force state); controllers in closed loop on the estimate. |
| 6 | Vision | V1–V4 planned | Grayscale 96 × 96 camera, rough gate map (±0.5 m / ±15°) corrected by vision; corner-heatmap CNN against a classical threshold + contour baseline, PnP gate pose, gate flight with detected gates (true state and full stack with the EKF). |

R4 was done before R2 and R3 because it needs the finished surrogate.

## Next steps

| Order | Stage | Content | Budget |
|---|---|---|---|
| 1 | R3 | finish: domain-randomisation run (1 × 1 M), two ablations of the late instability (learning rate 1e-4; stronger action-change penalty), decide on a rerun with a smoothness penalty in the units of J; afterwards close surrogate test set T5 (INDI + PPO flights) | running, ~5 h |
| 2 | A | A2 PPO through the gates (gate-progress reward, smoothness penalty as in J, gate-observation noise, 3 seeds × 1 M, checkpoint chosen by a gate-race score J_A) · A4 benchmark of A1, A1+, A2 | ~3 h training |
| 3 | P | arena generator, A* / RRT*, minimum snap, flight benchmark; afterwards A1 re-flown with minimum snap | not yet estimated |
| 4 | Z | IMU + UWB sensor model, EKF, surrogate EKF, closed loop | not yet estimated |
| 5 | V | camera images + dataset, detector, gate pose, gate flight (needs OpenCV) | not yet estimated |

R3 throughput: 340 steps/s with 2 environments, i.e. 0.8 h per 1 M steps. The R3 runs showed that
the training reward hardly reacts to command smoothness and that validation J stops improving
after 0.2–1 M steps; A2 therefore uses 1 M steps per seed and selects checkpoints on validation,
not on the reward.

Z3 (surrogate EKF) gets its own detailed plan first: the filter state must carry the surrogate's
motor state and history window, and the 240 Hz IMU is averaged per control step. Paired Wilcoxon
tests in A4, P4 and Z4 use a Holm correction for the many pairs.

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
| 2026-11-25 | 970565e | R2: INDI tuned · R4.3c: candidate-only ablation (tilt cost relative to the reference tested, < 1 %, not kept), comparison with two MPPI papers, benchmark configs |
| 2026-11-26 | a4f61c9 | R2/R4.4: INDI, MPPI hybrid and MPPI urdf benchmarked, plots with all controllers |
| 2026-11-26 | 1596a52 | R2: INDI-SG variant (Savitzky-Golay acceleration), tuned and benchmarked |

## Glossary

### Stages

| Code | Meaning |
|---|---|
| 0a / 0b | restructure into `dronelab` / new `real` world, R1 re-tuned and re-flown |
| R1–R4 | control: R1 PID / LQR / LQI · R2 INDI · R3 learned controller (PPO) · R4 MPPI with the surrogate |
| S1–S3 | surrogate: S1 data · S2 models and training · S3 test sets and evaluation |
| A0–A4 | autonomy: A0 gate frames, gate order, lap time · A1 planner + best controller · A2 RL through gates · A3 MPPI with gate progress (dropped) · A4 comparison |
| P1–P4 | path planning: P1 pillar-forest arena · P2 planners (A*, RRT* × spline, minimum snap) · P3 flight · P4 comparison |
| Z1–Z4 | state estimation: Z1 IMU + UWB-like position model · Z2 error-state EKF · Z3 EKF with the surrogate as process model · Z4 controllers on the estimated state |
| V1–V4 | vision: V1 grayscale camera images · V2 gate corner detection (CNN vs classical) · V3 gate pose (PnP) · V4 gate flight with detected gates |

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
| seeds | tuning 0–99 · validation 100–199 · test 1000+ · RL from 10 000 (R3) / 11 000 (A2) · surrogate 20 000+ · vision 30 000+ · planning 40 000+ · estimation 50 000+ |
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
