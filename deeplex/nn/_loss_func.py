from ..engine import Tensor


def _check_if_same_device(y1: Tensor, y2: Tensor):
    if y1.device != y2.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two devices, cuda and cpu!"
        )


def MSELoss(y1: Tensor, y2: Tensor):
    _check_if_same_device(y1, y2)
    return ((y1 - y2) ** 2).sum() / y2.shape[0]


def BCELoss(y1: Tensor, y2: Tensor):
    _check_if_same_device(y1, y2)
    return -(y2 * y1.log() + (1 - y2) * (1 - y1).log()).sum() / y2.shape[0]
