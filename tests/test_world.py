"""The "real" world: extra aerodynamics, world/scenario resolution and the new scenario set."""

import numpy as np
import pytest

from dronelab.core.config import AeroConfig, DisturbanceConfig
from dronelab.core.eval.scenarios import LEGACY_ORDER, SCENARIOS, scenario_order
from dronelab.core.world import WORLDS, resolve

pytest.importorskip("gym_pybullet_drones")

from dronelab.core.backends.pybullet_backend import PyBulletBackend, aero_wrench  # noqa: E402

KF, HOVER = 3.16e-10, 14468.0
ROTORS = [[0.028, -0.028], [-0.028, -0.028], [-0.028, 0.028], [0.028, 0.028]]
CFG = AeroConfig(enabled=True)
V_H = 4.0


def _wrench(vel=(0, 0, 0), ang_vel=(0, 0, 0), R=None, cfg=CFG):
    R = np.eye(3) if R is None else R
    return aero_wrench(cfg, np.full(4, HOVER), R, np.array(vel, float), np.array(ang_vel, float),
                       KF, ROTORS, V_H)


# In hover (no motion) the extra aerodynamics must vanish completely.
def test_aero_is_zero_in_hover():
    f, tau = _wrench()
    assert np.allclose(f, 0) and np.allclose(tau, 0)


# Forward flight: drag against the motion, the nose is pitched up (negative pitch moment).
def test_aero_forward_flight_signs():
    f, tau = _wrench(vel=(3, 0, 0))
    assert f[0] < 0 and abs(f[1]) < 1e-12
    assert tau[1] < 0 and abs(tau[0]) < 1e-12
    f, tau = _wrench(vel=(0, 3, 0))  # sideways to the left: roll away from the motion
    assert f[1] < 0 and tau[0] > 0


# Climbing costs thrust, descending gives thrust; size follows c * v / v_h.
def test_aero_axial_inflow():
    f_up, _ = _wrench(vel=(0, 0, 1))
    f_dn, _ = _wrench(vel=(0, 0, -1))
    thrust = 4 * KF * HOVER**2
    assert f_up[2] < 0 < f_dn[2]
    drag = 0.5 * CFG.air_density * CFG.body_drag_cda  # body drag at 1 m/s also acts along z
    assert -f_up[2] - drag == pytest.approx(CFG.inflow_coeff / V_H * thrust, rel=1e-6)


# Rolling: the rising side loses thrust, which damps the roll rate.
def test_aero_rotation_damping():
    _, tau = _wrench(ang_vel=(3, 0, 0))
    assert tau[0] < 0
    _, tau = _wrench(ang_vel=(0, -3, 0))
    assert tau[1] > 0


# Forces are computed in the world frame: a yawed drone gets the same world-frame drag.
def test_aero_frames():
    c, s = np.cos(0.7), np.sin(0.7)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    f0, tau0 = _wrench(vel=(2, 1, 0))
    f1, tau1 = _wrench(vel=(2, 1, 0), R=Rz)
    assert np.allclose(f0, f1)
    assert np.allclose(tau0, tau1)  # flapping moment depends on the velocity, not the heading


# Each effect can be switched off on its own.
def test_aero_effects_switch_off():
    off = AeroConfig(enabled=True, body_drag_cda=0, inflow_coeff=0, flapping_coeff=0)
    f, tau = _wrench(vel=(2, 1, 1), ang_vel=(1, 1, 0), cfg=off)
    assert np.allclose(f, 0) and np.allclose(tau, 0)


# A world sets physics, motor lag and aerodynamics; a scenario value replaces the world value.
def test_resolve_world_and_scenario():
    phys, d = resolve("real")
    assert phys == "pyb_gnd_drag_dw" and d["motor_tau"] == 0.02 and d["aero"]["enabled"]
    _, d = resolve("real", {"motor_tau": 0.1, "mass_scale": 1.2, "aero": {"inflow_coeff": 0}})
    assert d["motor_tau"] == 0.1 and d["mass_scale"] == 1.2
    assert d["aero"] == {"enabled": True, "inflow_coeff": 0}
    assert resolve("ideal") == ("pyb", {"motor_tau": 0.0, "aero": {"enabled": False}})
    DisturbanceConfig(**resolve("real")[1])  # valid config
    with pytest.raises(ValueError):
        resolve("moon")


# 20 scenarios in the real world; stage-1 CSVs keep their own order.
def test_scenario_set():
    assert len(SCENARIOS) == 20 and "aero" not in SCENARIOS and "motor_0.02" not in SCENARIOS
    assert SCENARIOS["ideal"].world == "ideal" and set(WORLDS) >= {"ideal", "real"}
    assert all(s.world is None for n, s in SCENARIOS.items() if n != "ideal")
    assert scenario_order({"aero", "nominal", "motor_0.02"}) == ["nominal", "motor_0.02", "aero"]
    assert scenario_order({"ideal", "nominal"}) == ["nominal", "ideal"]
    assert len(LEGACY_ORDER) == 21


# In PyBullet the flapping moment must turn the nose up in forward flight (torque frame check).
@pytest.mark.sim
def test_flapping_moment_direction_in_pybullet():
    pitch = {}
    for k in (0.0, 1e-6):
        aero = AeroConfig(enabled=True, body_drag_cda=0, inflow_coeff=0, flapping_coeff=k)
        with PyBulletBackend(disturbances=DisturbanceConfig(aero=aero)) as be:
            be.reset(np.array([0, 0, 1.0]), init_vel=np.array([2.0, 0, 0]))
            for _ in range(5):
                s = be.step(np.full(4, be.params.hover_rpm))
            pitch[k] = s.rpy[1]
    assert abs(pitch[0.0]) < 1e-6 and pitch[1e-6] < -0.05


# Switching the extra aerodynamics on changes nothing in hover (all effects vanish at rest).
@pytest.mark.sim
def test_aero_hover_unchanged():
    states = []
    for enabled in (False, True):
        with PyBulletBackend(disturbances=DisturbanceConfig(aero={"enabled": enabled})) as be:
            be.reset(np.array([0, 0, 1.0]))
            for _ in range(10):
                s = be.step(np.full(4, be.params.hover_rpm))
            states.append(s.pos)
    assert np.allclose(states[0], states[1], atol=1e-9)
