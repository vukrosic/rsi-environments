"""Bounded runner protocol; candidate Python is NOT a security sandbox.

Only this process imports submitted code. The parent uses bytes/JSON/NPZ with
allow_pickle=False, independently checks outputs, times the handshake, and
computes the reward. Input/output copies occur outside that timed handshake.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import resource
import socket
import struct
import sys
import time

import numpy as np
import torch


def send(sock, data):
    sock.sendall(struct.pack("!Q", len(data)) + data)


def receive(sock, maximum=512 * 1024 * 1024, deadline=None):
    def read(n):
        chunks = bytearray()
        while len(chunks) < n:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("evaluation wall-clock limit reached during protocol read")
                sock.settimeout(remaining)
            part = sock.recv(min(n - len(chunks), 1024 * 1024))
            if not part:
                raise EOFError("worker connection closed")
            chunks.extend(part)
        return bytes(chunks)
    length = struct.unpack("!Q", read(8))[0]
    if length > maximum:
        raise ValueError("protocol message exceeds size limit")
    return read(length)


def pack_arrays(arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **{str(i): x for i, x in enumerate(arrays)})
    return buffer.getvalue()


def unpack_arrays(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return tuple(archive[str(i)] for i in range(len(archive.files)))


def fingerprint(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def main():
    fd, device, mode, task, source = sys.argv[1:]
    sock = socket.socket(fileno=int(fd))
    # Bound per-file logs/artifacts from a runaway candidate. This is only a
    # resource guard; there is no total-disk or memory security boundary.
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # CUDA TF32 changes this task's specified numerical semantics.
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    if mode == "baseline":
        from .tasks import baseline
        fn = lambda *xs: baseline(task, *xs)
    else:
        scope = {"__name__": "candidate", "__file__": source}
        exec(compile(open(source).read(), source, "exec"), scope)
        fn = scope.get("kernel")
        if not callable(fn):
            raise ValueError("candidate must define kernel(*inputs)")
    send(sock, b"hello")
    with torch.no_grad():
        while True:
            payload = receive(sock)
            if payload == b"quit":
                break
            arrays = unpack_arrays(payload)
            xs = tuple(torch.from_numpy(x.copy()).to(device) for x in arrays)
            sync()
            send(sock, b"ready")
            if receive(sock, 16) != b"run":
                raise ValueError("expected run")
            out = fn(*xs)
            sync()
            send(sock, b"done")
            if not isinstance(out, torch.Tensor):
                send(sock, json.dumps({"error": "output must be a Tensor"}).encode())
                continue
            metadata = {
                "device": out.device.type,
                "dtype": str(out.dtype),
                "input_hashes": [fingerprint(x.detach().cpu().numpy()) for x in xs],
            }
            send(sock, json.dumps(metadata).encode())
            send(sock, pack_arrays((out.detach().cpu().numpy(),)))


if __name__ == "__main__":
    main()
