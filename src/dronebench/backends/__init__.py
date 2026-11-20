from dronebench.backends.base import Backend, DroneParams, DroneState

__all__ = ["Backend", "DroneParams", "DroneState", "make_backend"]


def make_backend(name: str = "pybullet", **kwargs) -> Backend:
    """Factory so that controllers/benchmarks never import a simulator directly."""
    if name == "pybullet":
        from dronebench.backends.pybullet_backend import PyBulletBackend

        return PyBulletBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")
