"""Kernel optimization interaction API. GPU use is cooperatively serialized."""
import importlib

__all__ = ["KernelEnvironment", "evaluate", "calibrate"]


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module(".environment", __name__), name)
    raise AttributeError(name)
