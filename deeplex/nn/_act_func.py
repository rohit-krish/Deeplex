import numpy as np
from ..engine import Tensor


def relu(val: Tensor):
    res = Tensor(np.maximum(0, val), (val,), "ReLU")

    def backward():
        val.grad += (val.data > 0) * res.grad

    res._backward = backward
    return res


def tanh(val: Tensor):
    res = Tensor(np.tanh(val), (val,), "TanH")

    def backward():
        val.grad += (1 - res.data**2) * res.grad

    res._backward = backward
    return res


def sigmoid(val: Tensor):
    exp_x = np.exp(-val.data)
    res = Tensor(1 / (1 + exp_x), (val,), "Sigmoid")

    def backward():
        val.grad += (exp_x / (1 + exp_x) ** 2) * res.grad

    res._backward = backward
    return res


# the below softmax implementation is taking advantage of the autograd engine.
def softmax(val: Tensor, axis):
    exp_x = val.exp()
    sum_exp_x = exp_x.sum(axis=axis, keepdims=True, dtype=None)
    res = exp_x / sum_exp_x
    return res
