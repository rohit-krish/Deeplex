import numpy as np
from engine import Tensor, Scaler


def _asserts(y1, y1_str, y2, y2_str):
    assert type(y1) == type(y2), f"Type of {y1_str} & {y2_str} must match"
    assert y1.shape == y2.shape, f"Shapes of {y1_str} & {y2_str} must match"


def MSELoss(y1: Tensor | Scaler, y2: Tensor | Scaler):
    _asserts(y1, "y1", y2, "y2")

    return np.mean((y1 - y2) ** 2)


# a = Tensor(np.random.randn(2))
# b = Tensor(np.random.randn(2))
# c = MSELoss(a, b)
# c.backward()
# print(type(c))
# print(c)
