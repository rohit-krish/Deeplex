import numpy as np


class Tensor(np.ndarray):
    def __init__(self, *args, **kwargs):
        super(np.ndarray, self).__init__()

    def __new__(cls, tensor):
        return np.vectorize(lambda x: Scaler(x))(np.asarray(tensor)).view(cls)

    def backward(self):
        # np.vectorize(lambda x: x.backward())(self) # this line of statement doesn't seems to work :( don't know why?
        for s in self.flatten():
            s.backward()

    @property
    def grad(self):
        return np.vectorize(lambda x: x.grad)(self)

    @grad.setter
    def grad(self, new_grad: int | float):
        for s in self.flatten():
            s.grad = new_grad

    @property
    def data(self):
        return np.vectorize(lambda x: x.data)(self)

    @data.setter
    def data(self, new_data):
        new_data = new_data.flatten()
        for i, s in enumerate(self.flatten()):
            s.data = new_data[i]

    def to_numpy(self):
        return np.array([s.data for s in self.flatten()]).reshape(self.shape)


class Scaler:
    def __init__(self, data, _prev=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
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
        return self._convert(other) * self**-1

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
        return f"Scaler({self.data})"
