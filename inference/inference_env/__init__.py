"""Source-tree facade; wheels stage the runtime under this namespace."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vuk-rsi-inference")
except PackageNotFoundError:
    __version__ = "0.2.3"

def __getattr__(name):
    if name == "CodingAgentEnvironment":
        # In the checkout, the long-standing flat scripts remain canonical.
        # The wheel backend emits this facade with private relative imports.
        from rl_environment import CodingAgentEnvironment

        return CodingAgentEnvironment
    if name == "InferenceEnvironment":
        from interface import InferenceEnvironment

        return InferenceEnvironment
    raise AttributeError(name)


__all__ = ["CodingAgentEnvironment", "InferenceEnvironment", "__version__"]
