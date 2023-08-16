import numpy as np
from .engine import Tensor
from .act_func import tanh


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
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


class RNN(Module):
    def __init__(self, input_size, hidden_size, n_layers):
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.i2h_layers = [
            Linear(input_size, hidden_size)
            if layer_i == 0
            else Linear(hidden_size, hidden_size)
            for layer_i in range(n_layers)
        ]

        self.h2h_layers = [Linear(hidden_size, hidden_size) for _ in range(n_layers)]

    def __call__(self, x: Tensor, h0=None):
        batch_size, seq_len, input_size = x.shape

        if h0 is None:
            h0 = Tensor(
                np.zeros((self.n_layers, batch_size, self.hidden_size), dtype=object)
            )

        h_t = h0

        outputs = Tensor(np.zeros((seq_len, batch_size, self.hidden_size)))

        for t in range(seq_len):
            x_t = x[:, t, :]

            for layer_i in range(self.n_layers):
                i2h_res = self.i2h_layers[layer_i](x_t)
                h2h_res = self.h2h_layers[layer_i](h_t[layer_i])

                combined = tanh(i2h_res + h2h_res)

                h_t[layer_i] = combined
                x_t = combined

            outputs[t] = h_t[-1]

        return outputs, h_t

    def parameters(self):
        return self.i2h_layers + self.h2h_layers
