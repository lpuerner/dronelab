"""The 20 disturbance scenarios of the benchmark (part B), on top of the "real" world.

Each scenario has a ``kind`` and a scalar ``level`` (used for degradation curves). The
wind direction is not part of the scenario: it is drawn per seed (horizontal, uniform),
so all controllers face the same wind for the same seed.

The real world already contains aerodynamics and 20 ms motor lag (see
:mod:`dronelab.core.world`), so there is no separate "aero" scenario any more and the
motor-lag scenarios replace the 20 ms by 50 / 100 ms. ``ideal`` flies in the old world
without aerodynamics and lag as a cross-check. ``LEGACY_ORDER`` keeps the scenario names
of the stage-1 benchmark (ideal world, 21 scenarios) so that its CSV can still be analysed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOISE = {  # position [m], velocity [m/s], attitude [rad]
    "low": (0.002, 0.01, np.deg2rad(0.5)),
    "mid": (0.005, 0.025, np.deg2rad(1.0)),
    "high": (0.010, 0.05, np.deg2rad(2.0)),
}


def _noise(level: str) -> dict:
    p, v, a = NOISE[level]
    return {"pos_noise_std": p, "vel_noise_std": v, "att_noise_std": float(a)}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    level: float
    dist: dict = field(default_factory=dict)  # DisturbanceConfig fields (wind as magnitude)
    world: str | None = None  # None = the world of the benchmark


def _build() -> dict[str, Scenario]:
    s = [Scenario("nominal", "nominal", 0.0)]
    s += [Scenario(f"mass_{m:g}", "mass", m, {"mass_scale": m}) for m in (0.8, 0.9, 1.1, 1.2, 1.3)]
    s += [Scenario(f"wind_{w:g}", "wind", w, {"wind": w}) for w in (0.03, 0.06, 0.09)]
    s += [Scenario(f"gust_{g:g}", "gust", g, {"gust_std": g, "gust_tau": 0.5})
          for g in (0.01, 0.02, 0.04)]
    s += [Scenario(f"motor_{t:g}", "motor_lag", t, {"motor_tau": t}) for t in (0.05, 0.10)]
    s += [Scenario(f"noise_{n}", "sensor_noise", i + 1, _noise(n))
          for i, n in enumerate(("low", "mid", "high"))]
    s += [Scenario("ideal", "world", 0.0, {}, world="ideal")]
    s += [
        Scenario("combo_mid", "combo", 1.0,
                 {"mass_scale": 1.1, "wind": 0.06, "gust_std": 0.02, "gust_tau": 0.5,
                  **_noise("mid")}),
        Scenario("combo_strong", "combo", 2.0,
                 {"mass_scale": 1.2, "wind": 0.09, "gust_std": 0.04, "gust_tau": 0.5,
                  "motor_tau": 0.10, **_noise("high")}),
    ]
    return {x.name: x for x in s}


SCENARIOS: dict[str, Scenario] = _build()
LEGACY_ORDER = (
    "nominal", "mass_0.8", "mass_0.9", "mass_1.1", "mass_1.2", "mass_1.3",
    "wind_0.03", "wind_0.06", "wind_0.09", "gust_0.01", "gust_0.02", "gust_0.04",
    "motor_0.02", "motor_0.05", "motor_0.1", "noise_low", "noise_mid", "noise_high",
    "aero", "combo_mid", "combo_strong",
)


def scenario_order(names) -> list[str]:
    """Protocol order of the scenario ``names`` found in a CSV (stage-1 or current set)."""
    names = set(names)
    order = LEGACY_ORDER if names - set(SCENARIOS) else tuple(SCENARIOS)
    return [n for n in order if n in names]


def disturbance_fields(scenario: Scenario, seed: int) -> dict:
    """DisturbanceConfig kwargs for one seed (wind magnitude -> seeded horizontal vector)."""
    d = dict(scenario.dist)
    w = d.pop("wind", 0.0)
    if w:
        ang = np.random.default_rng([seed, 1234]).uniform(0, 2 * np.pi)
        d["wind_force"] = (float(w * np.cos(ang)), float(w * np.sin(ang)), 0.0)
    return d
