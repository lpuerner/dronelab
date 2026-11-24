Rollout time [ms], K sequences x 20 steps, 1 thread (best of 7)

| model | substeps | K=128 | K=256 | K=512 |
|---|---|---:|---:|---:|
| urdf | 5 | 6.6 | 7.2 | 8.7 |
| urdf | 3 | 5.8 | 6.5 | 7.6 |
| urdf | 2 | 5.8 | 5.8 | 7.5 |
| phys_mlp_full_s0 | 5 | 17.8 | 23.8 | 34.8 |
| phys_mlp_full_s0 | 3 | 17.6 | 22.6 | 76.8 |
| phys_mlp_full_s0 | 2 | 42.3 | 42.5 | 53.0 |
| mlp_full_s0 | - | 16.6 | 22.2 | 31.9 |

T1 median position error at 0.42 s [cm]

| model | substeps | error |
|---|---|---:|
| phys_mlp_full_s0 | 5 | 0.109 |
| phys_mlp_full_s0 | 3 | 0.178 |
| phys_mlp_full_s0 | 2 | 0.343 |
| urdf | 5 | 8.26 |
| urdf | 3 | 8.366 |
| urdf | 2 | 8.498 |
