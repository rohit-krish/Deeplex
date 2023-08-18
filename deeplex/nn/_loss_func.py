from ..engine import Tensor


def MSELoss(y1: Tensor, y2: Tensor):
    return ((y1 - y2) ** 2).sum() / y2.shape[0]


def BCELoss(y1: Tensor, y2: Tensor):
    return -(y2 * y1.log() + (1 - y2) * (1 - y1).log()).sum() / y2.shape[0]
