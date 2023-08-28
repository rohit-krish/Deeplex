from ._act_func import relu, sigmoid, tanh, softmax
from ._loss_func import BCELoss, MSELoss, NLLLoss
from ..engine import Tensor


def exp(input: Tensor):
    return input.exp()


def log(input: Tensor):
    return input.log()


def sum(input: Tensor, axis=None, keepdims=False, dtype=None):
    return input.sum(axis, keepdims, dtype)


def reshape(input: Tensor, shape: tuple | list):
    return input.reshape(*shape)


def pad(input: Tensor, thickness: int, const_val=0):
    """
    Pad the tensor with the specified thickness and constant value.

    Args:
        thickness (int): The padding thickness for each dimension.
        const_val (scalar, optional): The constant value used for padding. Default is 0.

    Returns:
        Tensor: The padded tensor.
    """

    pad_width = input.d.zeros((len(input.shape), 2), "int")
    pad_width[-2:] = thickness

    res = Tensor(
        input.d.pad(input.data, pad_width, "constant", constant_values=const_val),
        input.device,
        input.dtype,
        (input,),
        input.requires_grad,
    )

    if input.requires_grad and input.grad_enabled:
        slices = tuple(
            [
                slice(int(pad[0]), -int(pad[1]) if pad[0] != 0 else None)
                for pad in pad_width
            ]
        )

        def backward():
            input.grad += res.grad[slices]

        res._backward = backward
        res.requires_grad = True

    return res
