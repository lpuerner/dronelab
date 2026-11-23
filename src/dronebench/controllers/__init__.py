from dronebench.controllers.base import Controller, Reference

__all__ = ["CONTROLLERS", "Controller", "Reference", "make_controller"]

CONTROLLERS = ("dsl_pid", "pid", "lqr", "lqi")


def make_controller(name: str, params, ctrl_dt: float, gains: dict | None = None) -> Controller:
    """Build a controller by name; ``gains`` are tuned parameters (ignored by dsl_pid)."""
    if name == "dsl_pid":
        from dronebench.controllers.dsl_pid import DSLPIDController

        return DSLPIDController(ctrl_dt=ctrl_dt)
    if name == "pid":
        from dronebench.controllers.pid import CascadedPID

        return CascadedPID(params, ctrl_dt, gains)
    if name in ("lqr", "lqi"):
        from dronebench.controllers.lqr import LQRController

        return LQRController(params, ctrl_dt, gains, integral=name == "lqi")
    raise ValueError(f"unknown controller {name!r}, choose from {CONTROLLERS}")
