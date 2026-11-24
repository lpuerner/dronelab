"""Surrogate models of the quadrotor dynamics at the control rate (one step = 1/48 s).

All models share one interface, :meth:`Surrogate.rollout`, batched in torch, so a sampling
MPC can roll out many command sequences at once.

* ``phys``: rigid body, rotor thrust / torque ~ rpm^2, linear rotor drag ~ sum(Omega) * v,
  first-order motor lag. Learnable: k_F, k_M, k_D,xy, k_D,z, tau (mass, inertia and rotor
  positions are "weighed", i.e. taken from the URDF).
* ``phys+mlp``: as ``phys`` plus an MLP that predicts a residual force and torque (body
  frame), constant over the control step (NeuroBEM-style residual model).
* ``mlp``: an MLP predicts the change of the body-frame velocity and of the body rates;
  position and attitude are integrated kinematically.

The MLP sees the last ``window`` steps of features: height, body velocity, gravity direction
in the body frame, body rates and commanded rpm (``phys+mlp`` also the motor speeds estimated
by its motor model). Never the horizontal position or the yaw angle, never the applied rpm.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from dronelab.core.backends.base import DroneParams
from dronelab.surrogate.data import CF2X, DT, Segments

KINDS = ("phys", "phys+mlp", "mlp")
N_STATE_FEAT = 10  # z, v_body(3), gravity direction in body frame(3), omega(3)
RES_SCALE = (0.01, 1e-4)  # residual force [N] / torque [N m] per unit net output


def hat(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[..., 0])
    return torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0],
                        -w[..., 1], w[..., 0], z], -1).reshape(*w.shape, 3)


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, batched (..., 3) -> (..., 3, 3)."""
    t2 = (phi * phi).sum(-1, keepdim=True)[..., None]
    t = torch.sqrt(t2 + 1e-12)
    small = t2 < 1e-8
    a = torch.where(small, 1 - t2 / 6, torch.sin(t) / t)
    b = torch.where(small, 0.5 - t2 / 24, (1 - torch.cos(t)) / (t2 + 1e-12))
    K = hat(phi)
    eye = torch.eye(3, dtype=phi.dtype).expand_as(K)
    return eye + a * K + b * (K @ K)


def rot_error(r_hat: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Small-angle attitude error vector (body frame) between two rotations (..., 3)."""
    e = r.transpose(-1, -2) @ r_hat
    return 0.5 * torch.stack([e[..., 2, 1] - e[..., 1, 2], e[..., 0, 2] - e[..., 2, 0],
                              e[..., 1, 0] - e[..., 0, 1]], -1)


def rot_angle(r_hat: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Geodesic angle [rad] between two rotations."""
    tr = (r.transpose(-1, -2) @ r_hat).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0))


def state_features(pos, vel, rot, omega) -> torch.Tensor:
    v_b = (rot.transpose(-1, -2) @ vel[..., None])[..., 0]
    g_b = rot[..., 2, :]  # R^T e_z
    return torch.cat([pos[..., 2:3], v_b, g_b, omega], -1)


class Physics(nn.Module):
    """Differentiable grey-box model, integrated like PyBullet (semi-implicit Euler, 240 Hz)."""

    def __init__(self, params: DroneParams = CF2X, n_sub: int = 5, tau0: float = 0.03):
        super().__init__()
        self.n_sub = n_sub
        self.register_buffer("mass", torch.tensor(params.mass))
        self.register_buffer("gravity", torch.tensor(params.gravity))
        self.register_buffer("inertia", torch.tensor(params.inertia, dtype=torch.float32))
        self.register_buffer("rotor_xy", torch.tensor(params.rotor_xy, dtype=torch.float32))
        self.register_buffer("yaw_sign", torch.tensor(params.yaw_sign, dtype=torch.float32))
        self.kf0, self.km0 = params.kf, params.km
        # learnable, scaled to O(1): k = k0 * exp(theta), k_D = 1e-6 * theta, tau = exp(theta)
        self.log_kf = nn.Parameter(torch.zeros(()))
        self.log_km = nn.Parameter(torch.zeros(()))
        self.kd = nn.Parameter(torch.zeros(2))  # xy, z  [1e-6 N s / (m rad)]
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau0)))

    def identified(self) -> dict[str, float]:
        return {
            "kf": self.kf0 * self.log_kf.detach().exp().item(),
            "km": self.km0 * self.log_km.detach().exp().item(),
            "kd_xy": 1e-6 * self.kd[0].item(),
            "kd_z": 1e-6 * self.kd[1].item(),
            "tau": self.log_tau.detach().exp().item(),
        }

    def motor(self, m: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        a = torch.exp(-DT / self.log_tau.exp())
        return a * m + (1 - a) * u

    def step(self, pos, vel, rot, omega, rpm, f_res=None, t_res=None):
        rpm2 = rpm * rpm
        kf = self.kf0 * self.log_kf.exp()
        km = self.km0 * self.log_km.exp()
        thrust = kf * rpm2.sum(-1, keepdim=True)
        torque = torch.stack([
            kf * (rpm2 * self.rotor_xy[:, 1]).sum(-1),
            -kf * (rpm2 * self.rotor_xy[:, 0]).sum(-1),
            km * (rpm2 * self.yaw_sign).sum(-1),
        ], -1)
        if t_res is not None:
            torque = torque + t_res
        drag = 1e-6 * torch.cat([self.kd[:1], self.kd]) * (rpm.sum(-1, keepdim=True)
                                                           * (2 * math.pi / 60))
        zeros = torch.zeros_like(thrust)
        f_thrust = torch.cat([zeros, zeros, thrust], -1)
        if f_res is not None:
            f_thrust = f_thrust + f_res
        g = torch.stack([zeros[..., 0], zeros[..., 0], zeros[..., 0] + self.gravity], -1)
        h = DT / self.n_sub
        for _ in range(self.n_sub):
            v_b = (rot.transpose(-1, -2) @ vel[..., None])[..., 0]
            f_b = f_thrust - drag * v_b
            acc = (rot @ f_b[..., None])[..., 0] / self.mass - g
            w_dot = (torque - torch.cross(omega, self.inertia * omega, dim=-1)) / self.inertia
            vel = vel + h * acc
            omega = omega + h * w_dot
            pos = pos + h * vel
            rot = rot @ so3_exp(h * omega)
        return pos, vel, rot, omega


def mlp(n_in: int, n_out: int, hidden: int, layers: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    last = nn.Linear(d, n_out)
    nn.init.zeros_(last.weight)  # start as "no correction" (hybrid) / "no change" (pure)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


class Surrogate(nn.Module):
    def __init__(self, kind: str, window: int = 1, hidden: int = 128, layers: int = 2,
                 params: DroneParams = CF2X, n_sub: int = 5):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}, choose from {KINDS}")
        self.kind, self.window = kind, window
        self.config = {"kind": kind, "window": window, "hidden": hidden, "layers": layers,
                       "n_sub": n_sub}
        self.hover = params.hover_rpm
        self.phys = Physics(params, n_sub) if kind != "mlp" else None
        n_feat = N_STATE_FEAT + 4 + (4 if kind == "phys+mlp" else 0)
        self.net = mlp(window * n_feat, 6, hidden, layers) if kind != "phys" else None
        self.register_buffer("feat_mean", torch.zeros(n_feat))
        self.register_buffer("feat_std", torch.ones(n_feat))
        scale = [RES_SCALE[0]] * 3 + [RES_SCALE[1]] * 3
        self.register_buffer("out_scale", torch.tensor(scale))

    # ------------------------------------------------------------------ features
    def _features(self, pos, vel, rot, omega, u, m) -> torch.Tensor:
        parts = [state_features(pos, vel, rot, omega), u / self.hover - 1]
        if self.kind == "phys+mlp":
            parts.append(m / self.hover - 1)
        return (torch.cat(parts, -1) - self.feat_mean) / self.feat_std

    @torch.no_grad()
    def fit_normalizer(self, seg: Segments) -> None:
        """Feature statistics (and the output scale of the pure net) from training data."""
        raw = [state_features(seg.pos, seg.vel, seg.rot, seg.omega), seg.u / self.hover - 1]
        if self.kind == "phys+mlp":
            raw.append(seg.applied / self.hover - 1)  # the motor model's estimate tracks it
        f = torch.cat(raw, -1).reshape(-1, self.feat_mean.numel())
        self.feat_mean.copy_(f.mean(0))
        self.feat_std.copy_(f.std(0).clamp_min(1e-3))
        if self.kind == "mlp":
            dv = seg.vel[:, 1:] - seg.vel[:, :-1]
            dv_b = (seg.rot[:, :-1].transpose(-1, -2) @ dv[..., None])[..., 0]
            dw = seg.omega[:, 1:] - seg.omega[:, :-1]
            self.out_scale.copy_(torch.cat([dv_b, dw], -1).reshape(-1, 6).std(0).clamp_min(1e-4))

    # ---------------------------------------------------------------- dynamics
    def _step(self, pos, vel, rot, omega, m, u, win, dist=None):
        pos1, vel1, rot1, omega1, m1 = self._model_step(pos, vel, rot, omega, m, u, win)
        if dist is not None:  # constant extra acceleration (world) / angular acc. (body)
            acc, alpha = dist
            pos1 = pos1 + 0.5 * DT * DT * acc
            vel1 = vel1 + DT * acc
            omega1 = omega1 + DT * alpha
        return pos1, vel1, rot1, omega1, m1

    def _model_step(self, pos, vel, rot, omega, m, u, win):
        if self.kind == "mlp":
            out = self.net(win) * self.out_scale
            vel1 = vel + (rot @ out[..., :3, None])[..., 0]
            omega1 = omega + out[..., 3:]
            pos1 = pos + DT * 0.5 * (vel + vel1)
            rot1 = rot @ so3_exp(DT * 0.5 * (omega + omega1))
            return pos1, vel1, rot1, omega1, m
        m1 = self.phys.motor(m, u)
        f_res = t_res = None
        if self.net is not None:
            out = self.net(win) * self.out_scale
            f_res, t_res = out[..., :3], out[..., 3:]
        return (*self.phys.step(pos, vel, rot, omega, m1, f_res, t_res), m1)

    def rollout(self, pos, vel, rot, omega, u, warmup: int, horizon: int, dist=None):
        """Open-loop prediction from recorded commands.

        ``pos, vel, rot, omega``: true states of shape (B, >= warmup + 1, ...); the first
        ``warmup`` steps only fill the feature window and settle the motor model, the
        prediction starts from the true state at index ``warmup``. ``u``: commanded rpm
        (B, warmup + horizon, 4). Returns predicted (pos, vel, rot, omega) for the indices
        ``warmup + 1 ... warmup + horizon``, each (B, horizon, ...). ``dist``: optional
        (acc (B, 3) world frame, alpha (B, 3) body frame) added in every predicted step.
        """
        if self.net is not None and warmup < self.window - 1:
            raise ValueError("warmup must be >= window - 1")
        m = u[:, 0]
        feats: list[torch.Tensor] = []
        for k in range(warmup):
            if self.net is not None:
                feats.append(self._features(pos[:, k], vel[:, k], rot[:, k], omega[:, k],
                                            u[:, k], m))
            if self.phys is not None:
                m = self.phys.motor(m, u[:, k])
        x = (pos[:, warmup], vel[:, warmup], rot[:, warmup], omega[:, warmup])
        feats = feats[len(feats) - (self.window - 1):] if self.window > 1 else []
        out = []
        for j in range(horizon):
            uk = u[:, warmup + j]
            win = None
            if self.net is not None:
                feats.append(self._features(*x, uk, m))
                feats = feats[-self.window:]
                win = torch.cat(feats, -1)
            *x, m = self._step(*x, m, uk, win, dist)
            out.append(x)
        return tuple(torch.stack([o[i] for o in out], 1) for i in range(4))


URDF_DRAG = (9.1785e-7, 10.311e-7)  # cf2x.urdf drag_coeff_xy, drag_coeff_z


def urdf_physics(tau: float = 0.02) -> Surrogate:
    """Physics with the data-sheet values (cf2x.urdf) and an assumed motor lag, not fitted.

    The baseline a model-based controller would use without flight data.
    """
    model = Surrogate("phys")
    with torch.no_grad():  # k_F, k_M are the URDF values already (log factor 0)
        model.phys.kd.copy_(torch.tensor(URDF_DRAG) / 1e-6)
        model.phys.log_tau.fill_(math.log(tau))
    return model.eval()


def save_model(model: Surrogate, path, extra: dict | None = None) -> None:
    torch.save({"config": model.config, "state": model.state_dict(), "extra": extra or {}}, path)


def load_model(path) -> tuple[Surrogate, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Surrogate(**ck["config"])
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["extra"]
