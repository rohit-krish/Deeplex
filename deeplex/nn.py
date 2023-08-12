import numpy as np
from .engine import Tensor


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters():
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.bias = bias
        self.W = Tensor(np.random.uniform(-1, 1, (in_features, out_features)))
        if self.bias:
            self.b = Tensor(np.random.uniform(-1, 1, (1, out_features)))

    def __call__(self, X: Tensor):
        out = X @ self.W + self.b
        return out

    def parameters(self):
        return [self.W, self.b] if self.bias else [self.W]
