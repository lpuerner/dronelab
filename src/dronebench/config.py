"""Experiment configuration (YAML -> validated pydantic models)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]


class SimConfig(BaseModel):
    pyb_freq: int = 240
    ctrl_freq: int = 48
    # Physics modes of gym-pybullet-drones
    physics: Literal["pyb", "pyb_drag", "pyb_gnd", "pyb_gnd_drag_dw"] = "pyb"
    gui: bool = False


class DisturbanceConfig(BaseModel):
    """All disturbances default to 'off'. Units: forces in N, times in s."""

    wind_force: Vec3 = (0.0, 0.0, 0.0)  # constant external force (Crazyflie weight ~0.26 N)
    gust_std: float = Field(0.0, ge=0)  # std of Ornstein-Uhlenbeck gust force [N]
    gust_tau: float = Field(1.0, gt=0)  # gust correlation time [s]
    mass_scale: float = Field(1.0, gt=0)  # true mass = nominal mass * scale
    motor_tau: float = Field(0.0, ge=0)  # first-order motor time constant [s]
    pos_noise_std: float = Field(0.0, ge=0)  # sensor noise [m]
    vel_noise_std: float = Field(0.0, ge=0)  # [m/s]
    att_noise_std: float = Field(0.0, ge=0)  # [rad]


class ArenaConfig(BaseModel):
    half_size: float = 5.0  # |x|,|y| bound [m]
    max_height: float = 5.0
    max_tilt_deg: float = 80.0
    ground_grace_time: float = 1.0  # ground contact before this time is not a crash


class ExperimentConfig(BaseModel):
    sim: SimConfig = SimConfig()
    disturbances: DisturbanceConfig = DisturbanceConfig()
    arena: ArenaConfig = ArenaConfig()
    course: str = "hover"
    controller: str = "dsl_pid"
    seed: int = 0


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(data)
