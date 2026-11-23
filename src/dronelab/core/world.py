"""The simulated "true world" that every subproject is evaluated in.

``ideal``  gym-pybullet-drones as is: thrust ~ rpm^2, no aerodynamics, instant motors. This is
           exactly the textbook model, so a model-based method has no model error at all
           (stage R1 of the control subproject was tuned and benchmarked here).
``real``   PyBullet drag + ground effect + downwash, 20 ms motor lag and the extra
           aerodynamics of :class:`~dronelab.core.config.AeroConfig` (quadratic body drag,
           axial-inflow thrust change, blade-flapping moment). Default from stage 0b on.

A scenario adds disturbances on top of a world; a scenario value (e.g. ``motor_tau``)
replaces the world value.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class World:
    name: str
    physics: str  # gym-pybullet-drones physics mode
    motor_tau: float  # [s]
    aero: dict = field(default_factory=dict)  # AeroConfig fields

    def disturbance_fields(self) -> dict:
        return {"motor_tau": self.motor_tau, "aero": dict(self.aero)}


WORLDS: dict[str, World] = {
    "ideal": World("ideal", "pyb", 0.0, {"enabled": False}),
    "real": World("real", "pyb_gnd_drag_dw", 0.02, {"enabled": True}),
}
DEFAULT_WORLD = "real"


def resolve(world: str, disturbances: dict | None = None) -> tuple[str, dict]:
    """Physics mode and DisturbanceConfig kwargs of ``world`` + scenario ``disturbances``."""
    if world not in WORLDS:
        raise ValueError(f"unknown world {world!r}, choose from {tuple(WORLDS)}")
    w = WORLDS[world]
    fields_ = w.disturbance_fields()
    for k, v in (disturbances or {}).items():
        if k == "aero":
            fields_["aero"] = {**fields_["aero"], **v}
        else:
            fields_[k] = v
    return w.physics, fields_
