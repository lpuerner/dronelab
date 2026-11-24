from dronelab.core.controller import Controller, Reference

__all__ = ["CONTROLLERS", "Controller", "Reference", "make_controller"]

CONTROLLERS = ("dsl_pid", "pid", "lqr", "lqi", "mppi_hybrid", "mppi_urdf")


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
    if name.removesuffix("_nodob") in ("mppi_hybrid", "mppi_urdf", "mppi_mlp"):
        from dronelab.control.mppi import MPPIController

        model = name.removesuffix("_nodob").removeprefix("mppi_")
        return MPPIController(params, ctrl_dt, gains, model=model,
                              dob=not name.endswith("_nodob"))
    raise ValueError(f"unknown controller {name!r}, choose from {CONTROLLERS}")
