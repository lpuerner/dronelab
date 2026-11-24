MPPI defaults on the 8 tuning episodes (nominal, real world); J as in tuning

| run | J | RMSE rel | smooth rel | crash rate | RMSE per family [cm] | ms/step |
|---|---:|---:|---:|---:|---|---:|
| lqi (tuned) | 0.902 | 0.556 | 1.153 | 0.00 | hover 24.8, figure8 1.2, helix 0.7, waypoints 1.6 | 0.2 |
| mppi_urdf K=256 | 1.913 | 1.016 | 2.990 | 0.00 | hover 38.7, figure8 2.5, helix 2.2, waypoints 3.1 | 23.7 |
| mppi_hybrid K=256 | 1.607 | 0.807 | 2.666 | 0.00 | hover 38.7, figure8 1.5, helix 1.2, waypoints 1.9 | 65.7 |
| mppi_hybrid K=512 | 1.406 | 0.778 | 2.092 | 0.00 | hover 38.4, figure8 1.4, helix 1.1, waypoints 1.6 | 70.3 |
