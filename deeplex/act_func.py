import numpy as np
from .engine import Scaler, Tensor


def _raise_val_error():
    raise ValueError("val should be Scaler or Tensor")


def relu(val: Scaler | Tensor):
    def for_scaler(scaler):
        res = Scaler(0 if scaler.data < 0 else scaler.data, (scaler,), "ReLU")

        def backward():
            scaler.grad += (res.data > 0) * res.grad

        res._backward = backward

        return res

    if type(val) == Scaler:
        return for_scaler(val)

    elif type(val) == Tensor:
        return np.vectorize(for_scaler)(val)

    else:
        _raise_val_error()


def tanh(val: Scaler | Tensor):
    def for_scaler(scaler):
        t = (np.exp(2 * scaler.data) - 1) / (np.exp(2 * scaler.data) + 1)
        res = Scaler(t, (scaler,), "TanH")

        def backward():
            scaler.grad += (1 - t**2) * res.grad

        res._backward = backward
        return res

    if type(val) == Scaler:
        return for_scaler(val)

    elif type(val) == Tensor:
        return np.vectorize(for_scaler)(val)

    else:
        _raise_val_error()
