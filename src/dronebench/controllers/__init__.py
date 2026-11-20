from dronebench.controllers.base import Controller, Reference

__all__ = ["Controller", "Reference", "make_controller"]


def make_controller(name: str, ctrl_dt: float, **kwargs) -> Controller:
    if name == "dsl_pid":
        from dronebench.controllers.dsl_pid import DSLPIDController

        return DSLPIDController(ctrl_dt=ctrl_dt, **kwargs)
    if name == "hover_rpm":
        from dronebench.controllers.base import ConstantRPMController

        return ConstantRPMController(**kwargs)
    raise ValueError(f"unknown controller {name!r}")
