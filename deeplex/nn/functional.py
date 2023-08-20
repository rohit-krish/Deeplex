from ._act_func import relu, sigmoid, tanh, softmax
from ._loss_func import BCELoss, MSELoss, NLLLoss


def exp(input):
    return input.exp()


def log(input):
    return input.log()


def sum(input, axis=None, keepdims=False, dtype=None):
    return input.sum(axis, keepdims, dtype)


def reshape(input, shape: tuple | list):
    return input.reshape(*shape)
