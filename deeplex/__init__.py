from warnings import warn
import numpy as np


GPU_AVAIL = True
try:
    import cupy as cp
except:
    GPU_AVAIL = False
    warn("GPU (cupy) not available. Falling back to CPU (numpy) computations.")


def get_d__(device):
    if device == "cpu":
        return np, device
    elif device == "cuda":
        if GPU_AVAIL:
            return cp, device
        else:
            raise RuntimeError("GPU (cupy) not available.")
    else:
        raise ValueError("Unknown value passed as device")

def dLex():
    return cp if GPU_AVAIL else np
