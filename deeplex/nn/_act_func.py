from ..engine import Tensor


def relu(val: Tensor):
    res = Tensor(val.d.maximum(0, val.data), val.device, val.dtype, (val,))

    if val.requires_grad and Tensor.grad_enabled:

        def backward():
            val.grad += (val.data > 0) * res.grad

        res._backward = backward
        res.requires_grad = True

    return res


def tanh(val: Tensor):
    res = Tensor(val.d.tanh(val.data), val.device, val.dtype, (val,))

    if val.requires_grad and Tensor.grad_enabled:

        def backward():
            val.grad += (1 - res.data**2) * res.grad

        res._backward = backward
        res.requires_grad = True

    return res


def sigmoid(val: Tensor):
    exp_x = val.d.exp(-val.data)
    res = Tensor(1 / (1 + exp_x), val.device, val.dtype, (val,))

    if val.requires_grad and Tensor.grad_enabled:

        def backward():
            val.grad += (exp_x / (1 + exp_x) ** 2) * res.grad

        res._backward = backward
        res.requires_grad = True

    return res


# the below softmax implementation is taking advantage of the autograd engine.
def softmax(val: Tensor, axis=-1):
    # applying "log-sum-exp"(subtracting the max from the input) trick to avoid numerical instability when taking exp
    max_val = Tensor(val.data.max(axis=axis, keepdims=True), val.device, val.dtype)
    exp_shifted_x = (val - max_val).exp()

    sum_exp_shifted_x = exp_shifted_x.sum(axis=axis, keepdims=True)
    res = exp_shifted_x / sum_exp_shifted_x
    return res
