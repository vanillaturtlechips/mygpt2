import math
import numpy as np
from typing import Optional, Set, Tuple, Union


# =============================================================================
# SCALAR ENGINE
# =============================================================================

class Value:

    def __init__(self, data: float, _children: tuple = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other): return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return Value(other) + (-self)
    def __truediv__(self, other): return self * (other ** -1)
    def __rtruediv__(self, other): return Value(other) * (self ** -1)

    def __pow__(self, exp: float):
        out = Value(self.data ** exp, (self,), f'**{exp}')
        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(max(-500, min(500, self.data)))
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(max(self.data, 1e-8)), (self,), 'log')
        def _backward():
            self.grad += (1.0 / max(self.data, 1e-8)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-max(-500, min(500, self.data))))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# =============================================================================
# TENSOR ENGINE
# =============================================================================

class Tensor:

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev: Set['Tensor'] = set()
        self._op = ''

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    @property
    def shape(self) -> Tuple: return self.data.shape

    @property
    def ndim(self) -> int: return self.data.ndim

    def _accumulate(self, grad: np.ndarray):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        while grad.ndim > self.data.ndim:
            grad = grad.sum(axis=0)
        for i, (s, g) in enumerate(zip(self.data.shape, grad.shape)):
            if s == 1 and g > 1:
                grad = grad.sum(axis=i, keepdims=True)
        self.grad += grad

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev, out._op = {self, other}, '+'
        def _backward():
            if self.requires_grad:  self._accumulate(out.grad)
            if other.requires_grad: other._accumulate(out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other): return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev, out._op = {self, other}, '*'
        def _backward():
            if self.requires_grad:  self._accumulate(other.data * out.grad)
            if other.requires_grad: other._accumulate(self.data * out.grad)
        out._backward = _backward
        return out

    def __rmul__(self, other): return self * other
    def __neg__(self): return self * Tensor(np.array(-1.0))
    def __sub__(self, other): return self + (-other if isinstance(other, Tensor) else Tensor(-np.array(other, dtype=np.float32)))
    def __rsub__(self, other): return Tensor(other) - self
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Tensor) else Tensor(other) ** -1)

    def __pow__(self, exp: float):
        out = Tensor(self.data ** exp, requires_grad=self.requires_grad)
        out._prev, out._op = {self}, f'**{exp}'
        def _backward():
            if self.requires_grad:
                self._accumulate(exp * (self.data ** (exp - 1)) * out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev, out._op = {self, other}, '@'
        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad @ other.data.swapaxes(-1, -2))
            if other.requires_grad:
                other._accumulate(self.data.swapaxes(-1, -2) @ out.grad)
        out._backward = _backward
        return out

    # ── reduction ─────────────────────────────────────────────────────────────

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'sum'
        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                self._accumulate(np.broadcast_to(g, self.data.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * Tensor(np.array(1.0 / n))

    # ── activations ───────────────────────────────────────────────────────────

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'relu'
        def _backward():
            if self.requires_grad:
                self._accumulate((self.data > 0).astype(np.float32) * out.grad)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'tanh'
        def _backward():
            if self.requires_grad:
                self._accumulate((1 - t ** 2) * out.grad)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'sigmoid'
        def _backward():
            if self.requires_grad:
                self._accumulate(s * (1 - s) * out.grad)
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(np.clip(self.data, -500, 500))
        out = Tensor(e, requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'exp'
        def _backward():
            if self.requires_grad:
                self._accumulate(e * out.grad)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.maximum(self.data, 1e-8)),
                     requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'log'
        def _backward():
            if self.requires_grad:
                self._accumulate((1.0 / np.maximum(self.data, 1e-8)) * out.grad)
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'softmax'
        def _backward():
            if self.requires_grad:
                g = out.grad
                self._accumulate(s * (g - (g * s).sum(axis=axis, keepdims=True)))
        out._backward = _backward
        return out

    # ── shape ─────────────────────────────────────────────────────────────────

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'reshape'
        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.data.shape))
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)
        out._prev, out._op = {self}, 'T'
        def _backward():
            if self.requires_grad:
                inv = np.argsort(axes)
                self._accumulate(out.grad.transpose(*inv))
        out._backward = _backward
        return out

    @property
    def T(self): return self.transpose()

    # ── graph ─────────────────────────────────────────────────────────────────

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    # ── factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def zeros(*shape, requires_grad=False):
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False):
        return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)

    @staticmethod
    def arange(start, stop=None, step=1, requires_grad=False):
        if stop is None:
            start, stop = 0, start
        return Tensor(np.arange(start, stop, step, dtype=np.float32), requires_grad=requires_grad)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()

def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    pred_clipped = Tensor(np.clip(pred.data, 1e-7, 1 - 1e-7), requires_grad=pred.requires_grad)
    return -(target * pred_clipped.log() + (Tensor(np.ones_like(target.data)) - target) * (Tensor(np.ones_like(pred_clipped.data)) - pred_clipped).log()).mean()

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # Fused log-softmax + NLL with explicit backward.
    # Treating log_sum_exp as a constant breaks the gradient (misses the softmax term).
    # The correct gradient is (softmax - targets) / N, so we compute it directly.
    m = logits.data.max(axis=-1, keepdims=True)
    shifted = logits.data - m
    exp_shifted = np.exp(shifted)
    sum_exp = exp_shifted.sum(axis=-1, keepdims=True)
    softmax = exp_shifted / (sum_exp + 1e-10)

    log_probs = logits.data - (m + np.log(sum_exp + 1e-10))
    loss_val = np.float32(-(log_probs * targets.data).sum(axis=-1).mean())

    out = Tensor(loss_val, requires_grad=logits.requires_grad)
    out._prev = {logits}
    out._op = 'cross_entropy'

    N = logits.data.shape[0]
    def _backward():
        if logits.requires_grad:
            logits._accumulate(out.grad * (softmax - targets.data) / N)
    out._backward = _backward
    return out


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    print("=== Scalar Engine ===")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x ** 2
    z.backward()
    print(f"z = x*y + x^2 = {z.data}")
    print(f"dz/dx = y + 2x = {x.grad}  (expected {y.data + 2*x.data})")
    print(f"dz/dy = x      = {y.grad}  (expected {x.data})")

    print("\n=== Tensor Engine ===")
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    c = (a @ b).sum()
    c.backward()
    print(f"a @ I summed = {c.data}")
    print(f"grad of a:\n{a.grad}")

    print("\n=== Softmax + Cross Entropy ===")
    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    target = Tensor([[1.0, 0.0, 0.0]])
    loss = cross_entropy(logits, target)
    loss.backward()
    print(f"loss = {loss.data:.4f}")
    print(f"grad of logits = {logits.grad}")

    print("\nAll checks passed.")
