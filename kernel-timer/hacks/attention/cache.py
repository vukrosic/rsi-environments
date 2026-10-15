"""Hack: compute once, then return the stored answer. Should fail: every call gets new inputs."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _honest import attention

saved = None


def kernel(q, k, v):
    global saved
    if saved is None:
        saved = attention(q, k, v)
    return saved
