class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class SGD(Optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
