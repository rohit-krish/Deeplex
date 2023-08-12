import numpy as np


class Scaler:
    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label
        self._prev = set(_prev)
        self._op = _op

    def _convert(self, other):
        if isinstance(other, Scaler):
            return other
        elif isinstance(other, (int, float)):
            return Scaler(other)
        else:
            raise Exception("Unsupported datatype!")

    def __add__(self, other):
        other = self._convert(other)
        res = Scaler(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += res.grad
            other.grad += res.grad

        res._backward = backward

        return res

    def __mul__(self, other):
        other = self._convert(other)
        res = Scaler(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad

        res._backward = backward

        return res

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            res = Scaler(self.data**other, (self,), f"**{other}")

            def backward():
                self.grad += other * (self.data ** (other - 1)) * res.grad

            res._backward = backward
            return res
        elif isinstance(other, Scaler):
            res = Scaler(self.data**other.data, (self, other), f"^{other.data}")

            def backward():
                self.grad += other.data * (self.data ** (other.data - 1)) * res.grad
                other.grad += self.data**other.data * np.log(self.data)

            res._backward = backward
            return res
        else:
            raise Exception("Only int, float & Scaler are alllowed for power op.")

    def relu(self):
        res = Scaler(0 if self.data < 0 else self.data, (self,), "ReLU")

        def backward():
            self.grad += (res.data > 0) * res.grad

        res._backward = backward

        return res

    def tanh(self):
        x = self.data
        t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        res = Scaler(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t**2) * res.grad

        res._backward = backward
        return res

    def backward(self):
        # build the topological graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_topo(prev)
                topo.append(v)

        build_topo(self)

        # set the gradient of the current node to 1
        self.grad = 1.0

        # run the _backward for all nodes
        for node in reversed(topo):
            node._backward()

    def __truediv__(self, other):  # self / other
        return self * (other**-1)

    def __rtruediv__(self, other):  # other / self
        return self._convert(other) * other**-1

    def __rmul__(self, other):  # other * self
        return self * self._convert(other)

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return self + (-self._convert(other))

    def __radd__(self, other):  # other + self
        return self + self._convert(other)

    def __repr__(self):
        return f"Scaler(data={self.data})"


class Neuron:
    def __init__(self, n_in):
        self.w = [Scaler(np.random.uniform(-1, 1, 1)) for _ in range(n_in)]
        self.b = Scaler(np.random.uniform(-1, 1, 1))

    def __call__(self, x):
        res = sum([wi * xi for wi, xi in zip(self.w, x)], start=self.b)
        return res.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multi Layered Perceptron"""

    def __init__(self, n_in, n_out):
        sz = [n_in] + n_out
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_out))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
