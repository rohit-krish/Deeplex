import numpy as np


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0


class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self.velocities):
            v[:] = ((1 - self.momentum) * p.grad) + (self.momentum * v)
            p.data -= self.lr * v


class RMSProp(Optimizer):
    def __init__(self, parameters, lr, decay=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        self.decay = decay
        self.eps = eps
        self.accumulators = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, acc in zip(self.parameters, self.accumulators):
            acc[:] = ((1 - self.decay) * p.grad**2) + (self.decay * acc)
            p.data -= (self.lr / (np.sqrt(acc) + self.eps)) * p.grad


class Adam(Optimizer):
    def __init__(self, parameters, lr, beta1=0.9, beta2=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]
        self.accumulators = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0  # time step counter

    def step(self):
        self.t += 1
        for p, v, a in zip(self.parameters, self.velocities, self.accumulators):
            v[:] = ((1 - self.beta1) * p.grad) + (self.beta1 * v)
            a[:] = ((1 - self.beta2) * p.grad**2) + (self.beta2 * a)

            # bias correction factors
            v_hat = v / (1 - self.beta1**self.t)
            a_hat = a / (1 - self.beta2**self.t)

            p.data -= (self.lr / (np.sqrt(a_hat) + self.eps)) * v_hat
