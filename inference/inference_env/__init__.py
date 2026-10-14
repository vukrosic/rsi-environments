"""Source-tree facade; the wheel build stages the private runtime package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vuk-rsi-inference")
except PackageNotFoundError:
    __version__ = "0.2.0"

def __getattr__(name):
    if name == "CodingAgentEnvironment":
        from ._runtime.rl_environment import CodingAgentEnvironment

        return CodingAgentEnvironment
    if name == "InferenceEnvironment":
        from ._runtime.interface import InferenceEnvironment

        return InferenceEnvironment
    raise AttributeError(name)


__all__ = ["CodingAgentEnvironment", "InferenceEnvironment", "__version__"]
