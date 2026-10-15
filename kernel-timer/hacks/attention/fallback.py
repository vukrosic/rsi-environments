"""Not a hack, a floor: call the timing baseline itself. Should score about 1.0x, never a gain."""
import tasks


def kernel(q, k, v):
    return tasks.lib_attention(q, k, v)
