from dronelab.core.controller import Controller, Reference

__all__ = ["CONTROLLERS", "Controller", "Reference", "make_controller"]

CONTROLLERS = ("dsl_pid", "pid", "lqr", "lqi", "indi", "indi_sg", "mppi_hybrid", "mppi_urdf",
               "ppo")


def make_controller(
    name: str, params, ctrl_dt: float, gains: dict | None = None, dither: dict | None = None
) -> Controller:
    """Build a controller by name; ``gains`` are tuned parameters (ignored by dsl_pid).

    ``dither`` (e.g. ``{"seed": 20001, "amp": 0.1}``) wraps the controller with an excitation
    signal on the commanded wrench, see :mod:`dronelab.control.dither`.
    """
    ctrl = _base(name, params, ctrl_dt, gains)
    if dither:
        from dronelab.control.dither import DitheredController

        ctrl = DitheredController(ctrl, params, ctrl_dt, **dither)
    return ctrl


def _base(name: str, params, ctrl_dt: float, gains: dict | None) -> Controller:
    if name == "dsl_pid":
        from dronelab.control.dsl_pid import DSLPIDController

        return DSLPIDController(ctrl_dt=ctrl_dt)
    if name == "pid":
        from dronelab.control.pid import CascadedPID

        return CascadedPID(params, ctrl_dt, gains)
    if name in ("lqr", "lqi"):
        from dronelab.control.lqr import LQRController

        return LQRController(params, ctrl_dt, gains, integral=name == "lqi")
    if name in ("indi", "indi_sg"):
        from dronelab.control.indi import ACC_WIN, INDIController

        return INDIController(params, ctrl_dt, gains,
                              acc_window=ACC_WIN if name == "indi_sg" else 0)
    if name.startswith("ppo"):  # ppo, ppo_dr, ppo_s1, ...: a trained policy, see rl_policy
        from dronelab.control.rl_policy import PPOController

        return PPOController(params, ctrl_dt, gains)
    base = name.removesuffix("_cand").removesuffix("_nodob")
    if base in ("mppi_hybrid", "mppi_urdf", "mppi_mlp"):  # suffixes: ablations (R4.5)
        from dronelab.control.mppi import MPPIController

        return MPPIController(params, ctrl_dt, gains, model=base.removeprefix("mppi_"),
                              dob="_nodob" not in name, cand_only=name.endswith("_cand"))
    raise ValueError(f"unknown controller {name!r}, choose from {CONTROLLERS}")
