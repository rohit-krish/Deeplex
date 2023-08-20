from ..engine import Tensor


def relu(val: Tensor) -> Tensor:
    """
    Apply the ReLU (Rectified Linear Unit) activation function element-wise.

    Args:
        val (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with ReLU applied element-wise.
    """

    res = Tensor(val.d.maximum(0, val.data), val.device, val.dtype, (val,))

    if val.requires_grad and Tensor.grad_enabled:

        def backward():
            val.grad += (val.data > 0) * res.grad

        res._backward = backward
        res.requires_grad = True

    return res


def tanh(val: Tensor) -> Tensor:
    """
    Apply the hyperbolic tangent (tanh) activation function element-wise.

    Args:
        val (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with tanh applied element-wise.
    """

    res = Tensor(val.d.tanh(val.data), val.device, val.dtype, (val,))

    if val.requires_grad and Tensor.grad_enabled:

        def backward():
            val.grad += (1 - res.data**2) * res.grad

        res._backward = backward
        res.requires_grad = True

    return res


def sigmoid(val: Tensor) -> Tensor:
    """
    Apply the sigmoid activation function element-wise.

    Args:
        val (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with sigmoid applied element-wise.
    """

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
    """
    Apply the softmax activation function along the specified axis.

    Args:
        val (Tensor): The input tensor.
        axis (int, optional): The axis along which to apply softmax. Default is -1.

    Returns:
        Tensor: The tensor with softmax applied along the specified axis.
    """

    # applying "log-sum-exp"(subtracting the max from the input) trick to avoid numerical instability when taking exp
    max_val = Tensor(val.data.max(axis=axis, keepdims=True), val.device, val.dtype)
    exp_shifted_x = (val - max_val).exp()

    sum_exp_shifted_x = exp_shifted_x.sum(axis=axis, keepdims=True)
    res = exp_shifted_x / sum_exp_shifted_x
    return res
