"""MPPI (model predictive path integral) control with a learned surrogate model.

Every control step: K perturbed sequences of (collective thrust, desired body rates) over H
steps are rolled out with a surrogate model from :mod:`dronelab.surrogate.models`, scored
against the reference preview, and averaged with weights exp(-J / lambda). The first command
is applied, the sequence is shifted and reused as the next nominal (warm start).

Inner loop: a fixed proportional rate controller, tau = J k (w_des - w) + w x J w, with the
rate gains of the tuned cascaded PID. It runs inside every rollout step on the predicted body
rates and once per control step on the measured ones. Sampling the torques directly was
tried first and tracked 5-10x worse than LQI (the sampler has to stabilise the attitude).

A disturbance estimate (offset-free MPC) is added to every predicted step: the one-step
prediction error of the last step, smoothed by an exponential moving average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from dronelab.control.mixer import Mixer
from dronelab.core.backends.base import DroneParams, DroneState
from dronelab.core.controller import Controller, Reference
from dronelab.core.models.rigid_body import mixer_matrix, quat_to_rot

MODEL_DIR = "projects/surrogate/results/models"
MODELS = {"urdf": None, "hybrid": "phys_mlp_full_s0.pt", "mlp": "mlp_full_s0.pt"}
WARMUP = 8  # history steps fed to the model (feature window + motor state)
HORIZON = 20  # 0.42 s, the horizon the surrogate was trained on
SAMPLES = 512
NOISE_CORR = 0.5  # low-pass of the sampling noise along the horizon (fixed, not tuned)
RATE_KP = (41.6, 41.6, 11.8)  # inner rate loop [1/s]: tuned PID (tuning_real/pid), P part only
RATE_SCALE = 5.0  # [rad/s] normalises rate commands in the command-change cost
CRASH_COST = 1e4
MIN_Z = 0.05


@dataclass
class MPPIGains:
    """Tuned parameters (search space: default / 5 ... default * 5, log-uniform).

    Defaults set by hand in the R4.2 smoke test (hover + figure8, see results/mppi/smoke.md).
    """

    lam: float = 0.1  # temperature, J normalised by (median - min)
    sigma_thrust: float = 0.003  # [N], ~1 % of hover thrust
    sigma_rate: float = 0.25  # [rad/s] roll/pitch rate
    sigma_yaw: float = 0.12  # [rad/s] yaw rate
    w_pos: float = 100.0  # [1/m^2]
    w_vel: float = 10.0  # [s^2/m^2]
    w_att: float = 2.0  # on 1 - cos(tilt)
    w_yaw: float = 1.0  # [1/rad^2]
    w_omega: float = 0.01  # [s^2/rad^2]
    w_du: float = 10.0  # on the command change, normalised by hover thrust / RATE_SCALE
    w_term: float = 5.0  # extra weight of the last predicted step
    tau_dob: float = 0.2  # [s] smoothing time of the disturbance estimate


def load_surrogate(kind: str):
    from dronelab.surrogate.models import load_model, urdf_physics

    if kind == "urdf":
        return urdf_physics()
    return load_model(f"{MODEL_DIR}/{MODELS[kind]}")[0]


class MPPIController(Controller):
    def __init__(self, params: DroneParams, ctrl_dt: float, gains: dict | None = None,
                 model: str = "hybrid", dob: bool = True, max_tilt_deg: float = 80.0):
        from dronelab.surrogate.data import DT

        if abs(ctrl_dt - DT) > 1e-9:
            raise ValueError(f"the surrogate runs at {DT:.5f} s, controller at {ctrl_dt:.5f} s")
        gains = dict(gains or {})
        self.samples = int(gains.pop("samples", SAMPLES))  # fixed per study, not tuned
        self.g = MPPIGains(**gains)
        self.name = f"mppi_{model}" + ("" if dob else "_nodob")
        self.p, self.dt, self.dob = params, ctrl_dt, dob
        torch.set_num_threads(1)  # benchmark/tuning run one process per core
        self.model = load_surrogate(model)
        self.mixer = Mixer(params)
        self.M_inv_T = torch.tensor(np.linalg.inv(mixer_matrix(params)).T, dtype=torch.float32)
        self.max_rpm2 = params.max_rpm**2
        self.J = torch.tensor(params.inertia, dtype=torch.float32)
        self.kp = torch.tensor(RATE_KP)
        hover = params.mass * params.gravity
        self.u_scale = torch.tensor([hover, RATE_SCALE, RATE_SCALE, RATE_SCALE])
        self.u_hover = torch.tensor([hover, 0.0, 0.0, 0.0])
        self.sigma = torch.tensor([self.g.sigma_thrust, self.g.sigma_rate, self.g.sigma_rate,
                                   self.g.sigma_yaw])
        self.cos_tilt = math.cos(math.radians(max_tilt_deg))
        self.ref_pos = self.ref_vel = None
        self.reset()

    # --------------------------------------------------------------- interface
    def reset(self) -> None:
        self.k = 0
        self.U = self.u_hover.repeat(HORIZON, 1)  # nominal (thrust, body rates)
        self.u_last = self.u_hover.clone()  # command applied in the previous step
        self.hist = None  # (pos, vel, rot, omega) each (WARMUP + 1, ...)
        self.u_hist = torch.full((WARMUP, 4), self.p.hover_rpm)  # rpm sent before
        self.dist = (torch.zeros(1, 3), torch.zeros(1, 3))
        self.gen = torch.Generator().manual_seed(0)  # same noise every episode -> reproducible

    def set_course(self, course) -> None:
        """Tabulate the reference for the preview (called by the runner before the flight)."""
        n = int(math.ceil(course.duration / self.dt)) + HORIZON + 1
        refs = [course.reference(k * self.dt) for k in range(n)]
        self.ref_pos = torch.tensor(np.array([r.pos for r in refs]), dtype=torch.float32)
        self.ref_vel = torch.tensor(np.array([r.vel for r in refs]), dtype=torch.float32)

    @torch.no_grad()
    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        omega = self._observe(state)
        if self.dob and self.k > WARMUP:  # buffer holds flown steps only from here on
            self._update_disturbance()
        ref_p, ref_v = self._preview(ref)
        eps = self._noise()
        U = self.U[None] + eps  # (K, H, 4)
        cost = self._rollout_cost(U, ref_p, ref_v)
        cost_n = (cost - cost.min()) / (cost.median() - cost.min() + 1e-9)
        w = torch.softmax(-cost_n / self.g.lam, 0)
        self.U = self.U + (w[:, None, None] * eps).sum(0)
        self.u_last = self.U[0].clone()
        tau = self._rate_loop(self.u_last[1:], omega).numpy().astype(float)
        rpm = self.mixer.wrench_to_rpm(float(self.u_last[0]), tau)
        self.U = torch.cat([self.U[1:], self.U[-1:]])
        self.u_hist = torch.cat([self.u_hist[1:], torch.tensor(rpm, dtype=torch.float32)[None]])
        self.k += 1
        return rpm

    # ----------------------------------------------------------------- pieces
    def _rate_loop(self, w_des, w):
        return self.J * self.kp * (w_des - w) + torch.cross(w, self.J * w, dim=-1)

    def _observe(self, s: DroneState) -> torch.Tensor:
        rot = torch.tensor(quat_to_rot(s.quat), dtype=torch.float32)
        omega = rot.T @ torch.tensor(s.ang_vel, dtype=torch.float32)  # body rates
        x = (torch.tensor(s.pos, dtype=torch.float32), torch.tensor(s.vel, dtype=torch.float32),
             rot, omega)
        if self.hist is None:
            self.hist = [xi[None].repeat(WARMUP + 1, *([1] * xi.dim())) for xi in x]
        else:
            self.hist = [torch.cat([h[1:], xi[None]]) for h, xi in zip(self.hist, x, strict=True)]
        return omega

    def _update_disturbance(self) -> None:
        from dronelab.surrogate.evaluate import estimate_disturbance

        h = [t[None] for t in self.hist]
        acc, alpha = estimate_disturbance(self.model, *h, self.u_hist[None], WARMUP, n=1)
        a = self.dt / self.g.tau_dob
        self.dist = (self.dist[0] + a * (acc - self.dist[0]),
                     self.dist[1] + a * (alpha - self.dist[1]))

    def _preview(self, ref: Reference):
        if self.ref_pos is None:  # no course given: hold the current reference
            p = torch.tensor(ref.pos, dtype=torch.float32)
            return p.expand(HORIZON, 3), torch.tensor(ref.vel, dtype=torch.float32).expand(
                HORIZON, 3)
        i = min(self.k + 1, len(self.ref_pos) - HORIZON)
        return self.ref_pos[i:i + HORIZON], self.ref_vel[i:i + HORIZON]

    def _noise(self) -> torch.Tensor:
        e = torch.randn(self.samples, HORIZON, 4, generator=self.gen)
        c = math.sqrt(1 - NOISE_CORR**2)
        for j in range(1, HORIZON):  # first-order low-pass, keeps the variance
            e[:, j] = NOISE_CORR * e[:, j - 1] + c * e[:, j]
        e = e * self.sigma
        e[0] = 0.0  # the unperturbed nominal is always a candidate
        return e

    def _rollout_cost(self, U, ref_p, ref_v) -> torch.Tensor:
        K = U.shape[0]

        def policy(j, pos, vel, rot, omega):  # inner rate loop on the predicted body rates
            wrench = torch.cat([U[:, j, :1], self._rate_loop(U[:, j, 1:], omega)], -1)
            return torch.sqrt((wrench @ self.M_inv_T).clamp(0.0, self.max_rpm2))

        h = [t[None].expand(K, *t.shape) for t in self.hist]
        dist = (self.dist[0].expand(K, 3), self.dist[1].expand(K, 3)) if self.dob else None
        pos, vel, rot, omega = self.model.rollout(*h, self.u_hist[None].expand(K, WARMUP, 4),
                                                  WARMUP, HORIZON, dist, policy)
        g = self.g
        r33 = rot[..., 2, 2]
        yaw = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
        un = U / self.u_scale
        du = torch.diff(un, dim=1, prepend=(self.u_last / self.u_scale).expand(K, 1, 4))
        stage = (g.w_pos * ((pos - ref_p) ** 2).sum(-1) + g.w_vel * ((vel - ref_v) ** 2).sum(-1)
                 + g.w_att * (1 - r33) + g.w_yaw * yaw**2 + g.w_omega * (omega**2).sum(-1)
                 + g.w_du * (du**2).sum(-1))
        crash = ((r33 < self.cos_tilt) | (pos[..., 2] < MIN_Z)).any(1)
        cost = stage.sum(1) + g.w_term * stage[:, -1] + CRASH_COST * crash
        return torch.nan_to_num(cost, nan=1e9, posinf=1e9)
