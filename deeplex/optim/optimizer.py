from .. import get_d__
# from ..engine import Tensor


class Optimizer:
    def __init__(self, parameters, lr, device: str):
        """
        Base class for optimizers.

        Args:
            parameters (list): List of tensors representing the trainable parameters.
            lr (float): Learning rate for optimization.
            device (str): Device on which the optimizer's computations should be performed.
        """
        self.d, self.device = get_d__(device)
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Update parameters"""
        raise NotImplementedError

    def zero_grad(self):
        """
        Reset gradients of all parameters to zero.
        """
        for p in self.parameters:
            p.grad = 0

    # def to(self, device: str):
    #     if device == self.device:
    #         return self

    #     self.d, self.device = get_d__(device)

    #     for _, val in self.__dict__.items():
    #         if isinstance(val, Tensor):
    #             val.to(device)


class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0.9, device="cpu"):
        """
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters (list): List of tensors representing the trainable parameters.
            lr (float): Learning rate for optimization.
            momentum (float, optional): Momentum factor. Default is 0.9.
            device (str, optional): Device on which the optimizer's computations should be performed. Default is "cpu".
        """
        super().__init__(parameters, lr, device)
        self.momentum = momentum
        self.velocities = [self.d.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self.velocities):
            v[:] = ((1 - self.momentum) * p.grad) + (self.momentum * v)
            p.data -= self.lr * v


class RMSProp(Optimizer):
    def __init__(self, parameters, lr, decay=0.99, eps=1e-8, device="cpu"):
        """
        RMSProp optimizer.

        Args:
            parameters (list): List of tensors representing the trainable parameters.
            lr (float): Learning rate for optimization.
            decay (float, optional): Decay factor for accumulating squared gradients. Default is 0.99.
            eps (float, optional): Small value to prevent division by zero. Default is 1e-8.
            device (str, optional): Device on which the optimizer's computations should be performed. Default is "cpu".
        """
        super().__init__(parameters, lr, device)
        self.decay = decay
        self.eps = eps
        self.accumulators = [self.d.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, acc in zip(self.parameters, self.accumulators):
            acc[:] = ((1 - self.decay) * p.grad**2) + (self.decay * acc)
            p.data -= (self.lr / (self.d.sqrt(acc) + self.eps)) * p.grad


class Adam(Optimizer):
    def __init__(self, parameters, lr, beta1=0.9, beta2=0.99, eps=1e-8, device="cpu"):
        """
        Adam optimizer.

        Args:
            parameters (list): List of tensors representing the trainable parameters.
            lr (float): Learning rate for optimization.
            beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.99.
            eps (float, optional): Small value to prevent division by zero. Default is 1e-8.
            device (str, optional): Device on which the optimizer's computations should be performed. Default is "cpu".
        """
        super().__init__(parameters, lr, device)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.velocities = [self.d.zeros_like(p.data) for p in self.parameters]
        self.accumulators = [self.d.zeros_like(p.data) for p in self.parameters]
        self.t = 0  # time step counter

    def step(self):
        self.t += 1
        for p, v, a in zip(self.parameters, self.velocities, self.accumulators):
            v[:] = ((1 - self.beta1) * p.grad) + (self.beta1 * v)
            a[:] = ((1 - self.beta2) * p.grad**2) + (self.beta2 * a)

            # bias correction factors
            v_hat = v / (1 - self.beta1**self.t)
            a_hat = a / (1 - self.beta2**self.t)

            p.data -= (self.lr / (self.d.sqrt(a_hat) + self.eps)) * v_hat
