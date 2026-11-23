# Subproject 2 – Surrogate modelling

**How well does a learned model predict the dynamics of the drone – above all outside its training data (faster, disturbed)?**

Part of [dronelab](../../README.md). Status: data generation ready, models planned.

* **World:** the `real` world (PyBullet aerodynamics, 20 ms motor lag and extra aerodynamics), so a standard physics model has a real model gap.
* **Models (first round):** physics only (rigid body, thrust ~ rpm², linear rotor drag, motor lag, learnable parameters), physics + MLP residual, MLP only. GRU variants later.
* **Data and test sets:** see [DATASETS.md](DATASETS.md).

```bash
python projects/surrogate/scripts/generate_data.py                 # train, val, T1 (needs tuned gains)
python projects/control/scripts/benchmark.py --config projects/control/configs/benchmark/r1.yaml --log-flights   # T3, T4
```
