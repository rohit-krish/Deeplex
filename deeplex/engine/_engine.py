from typing import Tuple
from .. import get_d__
from ..utils import broadcast_axis__


class Tensor:
    def __init__(
        self,
        data,
        device="cpu",
        dtype="float32",
        _prev: Tuple["Tensor"] = (),
        requires_grad=False,
    ):
        self.d, self.device = get_d__(device)

        self._r_grad = requires_grad
        self.data = self.d.asarray(data, dtype=dtype)
        self.grad = self.d.zeros_like(data, dtype=dtype) if requires_grad else None
        self._backward = lambda: None
        self._prev = set([p for p in _prev if p.requires_grad])

        self.shape = self.data.shape
        self.dtype = self.data.dtype

    @property
    def requires_grad(self):
        return self._r_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        if type(val) != bool:
            raise ValueError("Invalid assignment in requires_grad (only bool is allowed)")

        if (self._r_grad == False) and (self.grad is None):
            self.grad = self.d.zeros_like(self.data, dtype=self.dtype)
        self._r_grad = val

    def to(self, device: str):
        if device == self.device:
            return self

        self.d, self.device = get_d__(device)

        if self.device == "cpu":
            # gpu(cupy) -> cpu(numpy)
            self.data = self.data.get()
            self.grad = self.grad.get() if self.requires_grad else None
        else:
            # cpu(numpy) -> gpu(cupy)
            # now self.d is cupy
            self.data = self.d.asarray(self.data)
            self.grad = self.d.asarray(self.grad) if self.requires_grad else None

        return self

    def backward(self):
        if self._r_grad == False:
            raise RuntimeError("Doesn't have a gradient (self.requires_grad is False)")

        # build the topological graph
        topo = []
        visited = set()

        def build_top(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_top(prev)
                topo.append(v)

        build_top(self)

        # set the gradient of the current node to ones
        self.grad = self.d.ones_like(self.grad)

        # run the _backward for all nodes
        for node in reversed(topo):
            node._backward()

    def reshape(self, *shape):
        res = Tensor(self.data.reshape(shape), self.device, self.dtype, (self,))

        if self.requires_grad:

            def backward():
                self.grad += res.grad.reshape(self.shape)

            res._backward = backward
            res.requires_grad = True

        return res

    @property
    def T(self):
        res = Tensor(self.data.T, self.device, self.dtype, (self,))

        if self.requires_grad:

            def backward():
                self.grad += res.grad.T

            res._backward = backward
            res.requires_grad = True

        return res

    def exp(self):
        res = Tensor(self.d.exp(self.data), self.device, self.dtype, (self,))

        if self.requires_grad:

            def backward():
                self.grad = res.data * res.grad

            res._backward = backward
            res.requires_grad = True

        return res

    def log(self):
        """natural logarithm"""
        res = Tensor(self.d.log(self.data), self.device, self.dtype, (self,))

        if self.requires_grad:

            def backward():
                self.grad += res.grad / self.data

            res._backward = backward
            res.requires_grad = True

        return res

    def sum(self, axis=None, keepdims=False, dtype=None):
        sum_val = self.d.sum(self.data, axis=axis, keepdims=keepdims, dtype=dtype)
        res = Tensor(sum_val, self.device, self.dtype, (self,))

        if self.requires_grad:
            expand_axis = axis if axis and not keepdims else ()

            def backward():
                self.grad += self.d.ones_like(self.grad) * self.d.expand_dims(
                    res.grad, axis=expand_axis
                )

            res._backward = backward
            res.requires_grad = True

        return res

    def __len__(self):
        return self.shape[0]

    def __setitem__(self, indices, other):
        self._check(other, if_tensor=True)
        self.data[indices] = other.data.astype(self.data.dtype).copy()
        self.grad[indices] = other.grad.astype(self.grad.dtype).copy()

    def __getitem__(self, indices):
        res = Tensor(self.data[indices], self.device, self.dtype, (self,))

        if self.requires_grad:

            def backward():
                self.grad[indices] += res.grad

            res._backward = backward
            res.requires_grad = True

        return res

    def __matmul__(self, other):  # self @ other
        self._check(other, if_tensor=True)
        self._check(other, if_same_device=True)
        res = Tensor(self.data @ other.data, self.device, self.dtype, (self, other))

        if (self.requires_grad or other.requires_grad) == False:
            return res

        if self.data.ndim == other.data.ndim == 2:
            # backward for matmul of 2D tensors
            def backward():
                if self.requires_grad:
                    self.grad += res.grad @ other.data.T
                if other.requires_grad:
                    other.grad += self.data.T @ res.grad

        else:
            # Other cases
            if self.data.ndim == 1:
                self_expand_axis = (0,)
                self_expanded_shape = (1,) + self.shape
            else:
                self_expand_axis = ()
                self_expanded_shape = self.shape

            if other.data.ndim == 1:
                other_expand_axis = (-1,)
                other_expanded_shape = (1,) + other.shape
            else:
                other_expand_axis = ()
                other_expanded_shape = other.shape

            # Determine the axes for broadcasting and reduction
            result_expand_axis = self_expand_axis + other_expand_axis
            axis_self, axis_other = broadcast_axis__(
                self_expanded_shape[:-2], other_expanded_shape[:-2]
            )

            def backward():
                if self.requires_grad:
                    self.grad += self.d.reshape(
                        self.d.sum(
                            self.d.squeeze(
                                self.d.expand_dims(res.grad, axis=result_expand_axis)
                                @ self.d.expand_dims(
                                    other.data, axis=other_expand_axis
                                ).swapaxes(-1, -2),
                                axis=self_expand_axis,
                            ),
                            axis=axis_self,
                        ),
                        self.shape,
                    )

                if other.requires_grad:
                    other.grad += self.d.reshape(
                        self.d.sum(
                            self.d.squeeze(
                                self.d.expand_dims(
                                    self.grad, axis=self_expand_axis
                                ).swapaxes(-1, -2)
                                @ self.d.expand_dims(res.grad, axis=result_expand_axis),
                                axis=other_expand_axis,
                            ),
                            axis=axis_other,
                        ),
                        other.shape,
                    )

        res._backward = backward
        res.requires_grad = True
        return res

    def __add__(self, other):
        if isinstance(other, (int, float)):  # element wise addition
            res = Tensor(self.data + other, self.device, self.dtype, (self,))

            if self.requires_grad:

                def backward():
                    self.grad += res.grad

                res._backward = backward
                res.requires_grad = True

            return res

        elif isinstance(other, Tensor):
            self._check(other, if_same_device=True)
            res = Tensor(self.data + other.data, self.device, self.dtype, (self, other))

            if (self.requires_grad or other.requires_grad) == False:
                return res

            self_shape, other_shape = self.shape, other.shape

            if self_shape == other_shape:
                # backward for element-wise addition of tensors with same shape
                def backward():
                    if self.requires_grad:
                        self.grad += res.grad
                    if other.requires_grad:
                        other.grad += res.grad

            else:
                # determine the axes along the broadcasting occurs
                axis_self, axis_other = broadcast_axis__(self_shape, other_shape)

                # backward for addition of tensors of unequal shapes (basically means -> addition with broadcasting)
                def backward():
                    if self.requires_grad:
                        self.grad += self.d.reshape(
                            self.d.sum(res.grad, axis=axis_self), self_shape
                        )
                    if other.requires_grad:
                        other.grad += self.d.reshape(
                            self.d.sum(res.grad, axis=axis_other), other_shape
                        )

            res._backward = backward
            res.requires_grad = True
            return res
        else:
            self._check(None, raise_error_right_away=True)

    def __mul__(self, other):
        if isinstance(other, (int, float)):  # element wise multiplication
            res = Tensor(self.data * other, self.device, self.dtype, (self,))

            if self.requires_grad:

                def backward():
                    self.grad += other * res.grad

                res._backward = backward
                res.requires_grad = True

            return res

        elif isinstance(other, Tensor):
            self._check(other, if_same_device=True)
            res = Tensor(self.data * other.data, self.device, self.dtype, (self, other))

            if (self.requires_grad or other.requires_grad) == False:
                return res

            self_shape, other_shape = self.shape, other.shape

            if self_shape == other_shape:
                # backward for element-wise multiplication of tensors with same shape
                def backward():
                    if self.requires_grad:
                        self.grad += other.data * res.grad
                    if other.requires_grad:
                        other.grad += self.data * res.grad

            else:
                # determine the axes along the broadcasting occurs
                axis_self, axis_other = broadcast_axis__(self_shape, other_shape)

                # backward for multiplication of tensors of unequal shapes (basically means -> addition with broadcasting)
                def backward():
                    if self.requires_grad:
                        self.grad += self.d.reshape(
                            self.d.sum(other.data * res.grad, axis=axis_self),
                            self_shape,
                        )
                    if other.requires_grad:
                        other.grad += self.d.reshape(
                            self.d.sum(self.data * res.grad, axis=axis_other),
                            other_shape,
                        )

            res._backward = backward
            res.requires_grad = True
            return res
        else:
            self._check(None, raise_error_right_away=True)

    def __pow__(self, other):
        # Numpy or Cupy doesn't support powering by -ve values so taking reciprocal of variable powered the abs of the -ve powering term
        # variable ** negative_value = 1 / (variable ** abs(negative_value))

        def _neg_pow(a, b):
            return 1 / (a ** self.d.abs(b)) if b < 0 else a**b

        if isinstance(other, (int, float)):
            res = Tensor(_neg_pow(self.data, other), self.device, self.dtype, (self,))

            if self.requires_grad:

                def backward():
                    self.grad += other * _neg_pow(self.data, other - 1) * res.grad

                res._backward = backward
                res.requires_grad = True

            return res

        elif isinstance(other, Tensor):
            self._check(other, if_same_device=True)

            # for cupy to work here the inputs should be float64
            other_data = other.data.astype("float64")
            self_data = self.data.astype("float64")

            res_data = self.d.vectorize(_neg_pow)(self_data, other_data)
            res = Tensor(res_data, self.device, "float64", (self, other))

            if (self.requires_grad or other.requires_grad) == False:
                return res

            data_pow_other_min_1 = self.d.vectorize(_neg_pow)(self_data, other_data - 1)
            data_pow_other = self.d.vectorize(_neg_pow)(self_data, other_data)

            if self.shape == other.shape:
                # backward for element-wise powering of tensors with same shape
                def backward():
                    if self.requires_grad:
                        self.grad += other_data * data_pow_other_min_1 * res.grad
                    if other.requires_grad:
                        other.grad += data_pow_other * self.d.log(self_data)

            else:
                # determine the axes along the broadcasting occurs
                axis_self, axis_other = broadcast_axis__(self.shape, other.shape)

                # backward for multiplication of tensors of unequal shapes (basically means -> addition with broadcasting)
                def backward():
                    if self.requires_grad:
                        self.grad += self.d.reshape(
                            self.d.sum(
                                other_data * data_pow_other_min_1 * res.grad,
                                axis=axis_self,
                            ),
                            self.shape,
                        )
                    if other.requires_grad:
                        other.grad += self.d.reshape(
                            self.d.sum(
                                data_pow_other * self.d.log(self_data), axis=axis_other
                            ),
                            other.shape,
                        )

            res._backward = backward
            res.requires_grad = True
            return res
        else:
            self._check(None, raise_error_right_away=True)

    def __radd__(self, other):  # other(dtype -> not Tensor) + self
        self._check(other, if_int_or_float=True)
        return self + other

    def __rmul__(self, other):  # other(dtype -> not Tensor) * self
        self._check(other, if_int_or_float=True)
        return self * other

    def __truediv__(self, other):  # self / other
        self._check(other, if_int_or_float_or_tensor=True)
        return self * (other**-1)

    def __rtruediv__(self, other):  # other(dtype -> not Tensor) / self
        self._check(other, if_int_or_float=True)
        return (self**-1) * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        self._check(other, if_int_or_float_or_tensor=True)
        return self + (-other)

    def __rsub__(self, other):  # other(dtype -> not Tensor) - self
        self._check(other, if_int_or_float=True)
        return -self + other

    def __str__(self):
        device_str = f", device={self.device}" if self.device == "cuda" else ""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor{repr(self.data)[5:-1]}{device_str}{grad_str})"

    def __repr__(self):
        return str(self)

    def _check(
        self,
        other,
        raise_error_right_away=False,
        if_tensor=False,
        if_int_or_float=False,
        if_int_or_float_or_tensor=False,
        if_same_device=False,
    ):
        error_str = "Unsupported datatype :("

        if if_int_or_float_or_tensor and (not isinstance(other, (int, float, Tensor))):
            raise_error_right_away = True

        elif if_int_or_float and (not isinstance(other, (int, float))):
            raise_error_right_away = True

        elif if_tensor and (not isinstance(other, Tensor)):
            raise_error_right_away = True

        elif if_same_device and (self.device != other.device):
            raise_error_right_away = True
            error_str = "Expected all tensors to be on the same device, but found at least two devices, cuda and cpu!"

        if raise_error_right_away:
            raise RuntimeError(error_str)
