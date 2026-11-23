from dronelab.core.backends.base import Backend, DroneParams, DroneState, ImuMeasurement

__all__ = ["Backend", "DroneParams", "DroneState", "ImuMeasurement", "make_backend"]


def make_backend(name: str = "pybullet", **kwargs) -> Backend:
    """Factory so that controllers/benchmarks never import a simulator directly."""
    if name == "pybullet":
        from dronelab.core.backends.pybullet_backend import PyBulletBackend

        return PyBulletBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")
