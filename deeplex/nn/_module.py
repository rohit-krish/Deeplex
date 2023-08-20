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
        return [t for t in self._get_tensors() if t.requires_grad]

    def _get_tensors(self):
        tensors = []

        for _, val in self.__dict__.items():
            if isinstance(val, Tensor):
                tensors.append(val)
            elif isinstance(val, ModuleList):
                for v in val:
                    tensors += v._get_tensors()
            elif isinstance(val, (Linear, RNN)):
                tensors += val._get_tensors()

        return tensors

    def to(self, device: str):
        if device == self.device:
            return self

        self.d, self.device = get_d__(device)

        for tensor in self._get_tensors():
            tensor.to(device)

        return self


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device="cpu", dtype="float32"
    ):
        super().__init__(device)
        self.bias = bias
        self.W = Tensor(
            self.d.random.uniform(-1, 1, (in_features, out_features)),
            device,
            dtype,
            requires_grad=True,
        )
        if self.bias:
            self.b = Tensor(
                self.d.random.uniform(-1, 1, (1, out_features)),
                device,
                dtype,
                requires_grad=True,
            )

    def __call__(self, X: Tensor):
        out = X @ self.W + self.b
        out.requires_grad = False
        return out


class RNN(Module):
    def __init__(
        self, input_size, hidden_size, n_layers, device="cpu", dtype="float32"
    ):
        super().__init__(device)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dtype = dtype

        self.i2h_layers = ModuleList(
            [
                Linear(input_size, hidden_size, device=device, dtype=dtype)
                if layer_i == 0
                else Linear(hidden_size, hidden_size, device=device, dtype=dtype)
                for layer_i in range(n_layers)
            ]
        )

        self.h2h_layers = ModuleList(
            [
                Linear(hidden_size, hidden_size, device=device, dtype=dtype)
                for _ in range(n_layers)
            ]
        )

    def __call__(self, x: Tensor, h0=None):
        batch_size, seq_len, input_size = x.shape

        if h0 is None:
            h0 = Tensor(
                self.d.zeros((self.n_layers, batch_size, self.hidden_size)),
                self.device,
                self.dtype,
            )

        h_t = h0

        outputs = Tensor(
            self.d.zeros((seq_len, batch_size, self.hidden_size)),
            self.device,
            self.dtype,
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
