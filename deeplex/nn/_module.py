from .. import get_d__
from ..engine import Tensor
from ._act_func import tanh


class ModuleList:
    def __init__(self, modules):
        self.modules = modules

    def __getitem__(self, indices):
        return self.modules[indices]


class Module:
    def __init__(self, device: str):
        self.d, self.device = get_d__(device)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        TODO: distinguish between a tensor which has to learn(original parameter, (you can do that by creating another method like nn.Parameter)) and ones which doesn't has to learn.
        """
        parameters = []

        for _, val in self.__dict__.items():
            if isinstance(val, Tensor):
                parameters.append(val)
            elif isinstance(val, ModuleList):
                for v in val:
                    parameters += v.parameters()
            elif isinstance(val, (Linear, RNN)):
                parameters += val.parameters()

        return parameters

    def to(self, device: str):
        if device == self.device:
            return self

        self.d, self.device = get_d__(device)

        for tensor in self.parameters():
            tensor.to(device)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device="cpu"):
        super().__init__(device)
        self.bias = bias
        self.W = Tensor(
            self.d.random.uniform(-1, 1, (in_features, out_features)), device=device
        )
        if self.bias:
            self.b = Tensor(
                self.d.random.uniform(-1, 1, (1, out_features)), device=device
            )

    def __call__(self, X: Tensor):
        out = X @ self.W + self.b
        return out


class RNN(Module):
    def __init__(self, input_size, hidden_size, n_layers, device="cpu"):
        super().__init__(device)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.i2h_layers = ModuleList(
            [
                Linear(input_size, hidden_size, device=device)
                if layer_i == 0
                else Linear(hidden_size, hidden_size, device=device)
                for layer_i in range(n_layers)
            ]
        )

        self.h2h_layers = ModuleList(
            [Linear(hidden_size, hidden_size, device=device) for _ in range(n_layers)]
        )

    def __call__(self, x: Tensor, h0=None):
        batch_size, seq_len, input_size = x.shape

        if h0 is None:
            h0 = Tensor(
                self.d.zeros((self.n_layers, batch_size, self.hidden_size)),
                device=self.device,
            )

        h_t = h0

        outputs = Tensor(
            self.d.zeros((seq_len, batch_size, self.hidden_size)), device=self.device
        )

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
