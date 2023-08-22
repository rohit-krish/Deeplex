<img src="images/banner.png">

Deeplex is a PyTorch inspired, basic Deep Learning Framework; built on top of an Reverse-Mode AutoGrad Engine.
<br>
<i>Computations are handled using NumPy (CPU) & CuPy (GPU).</i>

### Autograd Demo

```python
from deeplex.engine import Tensor, no_grad
from deeplex.nn import functional as F
from deeplex import dLex

dlex = dLex()

with no_grad():
    r1 = -Tensor(dlex.random.randn(2, 3))
    r2 = F.tanh(r1.sum(axis=0).exp())

r3 = F.softmax(Tensor(dlex.ones_like(r2), requires_grad=True))
r4 = (r1 * r3/r2).T
r5 = Tensor(dlex.random.randint(0, 10, (3, 2))).T
r6 = r4 @ r5   # dot product

r6.backward()  # reverse-mode autograd

print(r1.grad) # ∂r6 / ∂r1 -> None (requires_grad = False)
print(r2.grad) # ∂r6 / ∂r2 -> None (requires_grad = False)
print(r3.grad) # ∂r6 / ∂r3
print(r4.grad) # ∂r6 / ∂r4
print(r5.grad) # ∂r6 / ∂r5 -> None (requires_grad = False)
print(r6.grad) # ∂r6 / ∂r6 -> ones(3, 3)

```

### Requirements

```
numpy # tested version 1.25.2
cupy  # tested version 11.0.0
```

### TODO

- [x] computation on GPU aswell
- [x] add .requires_grad in Tensor
- [x] add the 'with block' of no_grad
- [x] automatically list the parameters
- [ ] implement more loss functions
- [x] Adam, RMSProp, SGD+Momentum
- [ ] learning rate schedulers
- [ ] weight initializations
- [ ] Conv2d, RNN, LSTM, GRU
- [ ] dataloader
- [ ] testes
- [ ] update README.md
- [ ] add setup.py
- [ ] etc...

### Resources

- https://youtu.be/wG_nF1awSSY (What is Automatic Differentiation?)
- https://youtu.be/VMj-3S1tku0 (Micrograd - Andrej Karpathy)
- https://youtu.be/pauPCy_s0Ok (Neural Network from Scratch | Mathematics & Python Code)

### Related Works

- https://github.com/conscell/ugrad
- https://github.com/karpathy/micrograd

### License

MIT
