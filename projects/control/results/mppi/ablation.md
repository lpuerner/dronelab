Tuned MPPI hybrid gains (trial 47); J on the 8 tuning and 40 validation episodes

| variant | J tune | J val | rmse_rel val | smooth_rel val | crash val | ms/step |
|---|---:|---:|---:|---:|---:|---:|
| LQI (tuned) | – | 0.802 | 0.468 | 1.114 | 0.00 | – |
| mppi_hybrid (tuned) | – | 0.652 | 0.507 | 0.482 | 0.00 | – |
| mppi_hybrid, tilt cost relative to the reference thrust direction (tested, not kept) | 0.750 | 0.648 | 0.503 | 0.481 | 0.00 | 72.9 |
| candidate only (geometric PD) | 2.903 | 2.344 | 2.185 | 0.528 | 0.00 | 4.4 |
