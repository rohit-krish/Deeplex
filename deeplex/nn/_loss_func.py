from ..engine import Tensor

_EPS = 1e-8


def _check_if_same_device(y1: Tensor, y2: Tensor):
    if y1.device != y2.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two devices, cuda and cpu!"
        )


def MSELoss(y1: Tensor, y2: Tensor):
    _check_if_same_device(y1, y2)
    return ((y1 - y2) ** 2).sum() / len(y2)


def BCELoss(input: Tensor, pred: Tensor):
    _check_if_same_device(input, pred)

    # clipping to avoid numerical instability
    input.data = input.data.clip(_EPS, 1 - _EPS)
    pred.data = pred.data.clip(_EPS, 1 - _EPS)

    return -(pred * input.log() + (1 - pred) * (1 - input).log()).sum() / len(pred)


def NLLLoss(input: Tensor, pred: Tensor):
    _check_if_same_device(input, pred)

    # clipping to avoid numerical instability
    input.data = input.data.clip(_EPS, 1 - _EPS)
    pred.data = pred.data.clip(_EPS, 1 - _EPS)

    return -(input * pred.log()).sum() / len(pred)
