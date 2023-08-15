import numpy as np
from .engine import Scaler, Tensor


def _apply_on_val(val, for_scaler):
    """for scaler activations"""
    if type(val) == Scaler:
        return for_scaler(val)

    elif type(val) == Tensor:
        return np.vectorize(for_scaler)(val)

    else:
        raise ValueError("val should be Scaler or Tensor")


def relu(val: Scaler | Tensor):
    def for_scaler(scaler):
        res = Scaler(0 if scaler.data < 0 else scaler.data, (scaler,), "ReLU")

        def backward():
            scaler.grad += (res.data > 0) * res.grad

        res._backward = backward

        return res

    return _apply_on_val(val, for_scaler)


def tanh(val: Scaler | Tensor):
    def for_scaler(scaler):
        exp_2x = np.exp(2 * scaler.data)
        t = (exp_2x - 1) / (exp_2x + 1)
        res = Scaler(t, (scaler,), "TanH")

        def backward():
            scaler.grad += (1 - t**2) * res.grad

        res._backward = backward

        return res

    return _apply_on_val(val, for_scaler)


def sigmoid(val: Scaler | Tensor):
    def for_scaler(scaler):
        exp_x = np.exp(-scaler.data)
        res = Scaler(1 / (1 + exp_x), (scaler,), "Sigmoid")

        def backward():
            scaler.grad += (exp_x / (1 + exp_x) ** 2) * res.grad

        res._backward = backward

        return res

    return _apply_on_val(val, for_scaler)


# in the below softmax implementation; i'm taking advantage of the autograd engine,
# since it's an obvious function; we actually have to implement the backward pass by manually defining it,
# rather relaying on the autograd (which may not be the performant way)
def softmax(val: Tensor, axis=1):
    exp_x = np.vectorize(lambda x: x.exp())(val)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    res = exp_x / sum_exp_x
    return res
