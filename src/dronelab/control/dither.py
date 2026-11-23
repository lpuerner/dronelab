"""Excitation for system identification: a controller wrapper that adds a known random
signal to the commanded wrench (collective thrust + body torques).

A well-tuned controller keeps the drone close to hover and makes the commands a function
of the state (u ~ K x), which hides the effect of u in closed-loop data. The added signal is
independent of the state: per channel 50 % multisine (4 frequencies in 0.3-6 Hz) and 50 %
PRBS (random sign, held 0.05-0.4 s), scaled to ``amp`` x a reference magnitude (hover thrust,
max roll/pitch/yaw torque). It is switched on after ``t_on`` seconds (take-off) and fully
determined by the seed.
"""

from __future__ import annotations

import numpy as np

from dronelab.control.mixer import Mixer
from dronelab.core.backends.base import DroneParams, DroneState
from dronelab.core.controller import Controller, Reference


class DitherSignal:
    def __init__(self, seed: int, ctrl_dt: float, n_freq: int = 4,
                 f_range: tuple[float, float] = (0.3, 6.0),
                 hold_range: tuple[float, float] = (0.05, 0.4)):
        rng = np.random.default_rng([seed, 31337])
        self.dt = ctrl_dt
        self.freq = rng.uniform(*f_range, (4, n_freq))
        self.phase = rng.uniform(0, 2 * np.pi, (4, n_freq))
        self._rng = rng
        self._hold_range = hold_range
        self.reset()

    def reset(self) -> None:
        self._prbs = np.ones(4)
        self._next_switch = np.zeros(4)
        self._k = 0

    def __call__(self) -> np.ndarray:
        """Next sample in [-1, 1] per channel (thrust, tau_x, tau_y, tau_z)."""
        t = self._k * self.dt
        self._k += 1
        sines = np.sin(2 * np.pi * self.freq * t + self.phase).mean(axis=1)
        switch = t >= self._next_switch
        if np.any(switch):
            n = int(switch.sum())
            self._prbs[switch] = self._rng.choice([-1.0, 1.0], n)
            self._next_switch[switch] = t + self._rng.uniform(*self._hold_range, n)
        return 0.5 * sines + 0.5 * self._prbs


class DitheredController(Controller):
    def __init__(self, base: Controller, params: DroneParams, ctrl_dt: float, seed: int,
                 amp: float = 0.1, t_on: float = 1.0):
        self.base, self.amp, self.t_on, self.dt = base, amp, t_on, ctrl_dt
        self.name = f"{base.name}+dither"
        self.mixer = Mixer(params)
        w_max = np.full(4, params.max_rpm**2)
        torque_max = np.abs(self.mixer.M[1:] @ np.diag(w_max)).sum(axis=1) / 2
        self.scale = np.concatenate([[params.mass * params.gravity], torque_max])
        self.signal = DitherSignal(seed, ctrl_dt)
        self._k = 0

    def reset(self) -> None:
        self.base.reset()
        self.signal.reset()
        self._k = 0

    def act(self, state: DroneState, ref: Reference) -> np.ndarray:
        rpm = self.base.act(state, ref)
        d = self.signal()
        t = self._k * self.dt
        self._k += 1
        if t < self.t_on:
            return rpm
        wrench = self.mixer.rpm_to_wrench(rpm) + self.amp * self.scale * d
        return self.mixer.wrench_to_rpm(float(wrench[0]), wrench[1:])
