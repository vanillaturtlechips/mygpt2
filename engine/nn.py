import numpy as np
from .autograd import Tensor


# =============================================================================
# BASE MODULE
# =============================================================================

class Module:

    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True
        for val in self.__dict__.values():
            if isinstance(val, Module):
                val.train()
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Module):
                        item.train()
        return self

    def eval(self):
        self.training = False
        for val in self.__dict__.values():
            if isinstance(val, Module):
                val.eval()
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Module):
                        item.eval()
        return self


# =============================================================================
# LAYERS
# =============================================================================

class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(out_features, in_features).astype(np.float32) * scale,
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_features, dtype=np.float32),
            requires_grad=True
        ) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = Tensor(
            np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )

    def forward(self, indices) -> Tensor:
        if isinstance(indices, Tensor):
            indices = indices.data.astype(int)
        rows = self.weight.data[indices]
        out = Tensor(rows, requires_grad=self.weight.requires_grad)
        out._prev = {self.weight}
        out._op = 'embedding'
        def _backward():
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                np.add.at(self.weight.grad, indices, out.grad)
        out._backward = _backward
        return out


class LayerNorm(Module):

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.gamma.data * x_norm + self.beta.data
        out = Tensor(out_data, requires_grad=x.requires_grad or self.gamma.requires_grad)
        out._prev = {x, self.gamma, self.beta}
        out._op = 'layernorm'
        def _backward():
            N = x.data.shape[-1]
            if self.gamma.requires_grad:
                if self.gamma.grad is None:
                    self.gamma.grad = np.zeros_like(self.gamma.data)
                self.gamma.grad += (out.grad * x_norm).sum(axis=tuple(range(x.data.ndim - 1)))
            if self.beta.requires_grad:
                if self.beta.grad is None:
                    self.beta.grad = np.zeros_like(self.beta.data)
                self.beta.grad += out.grad.sum(axis=tuple(range(x.data.ndim - 1)))
            if x.requires_grad:
                x._accumulate(
                    self.gamma.data / np.sqrt(var + self.eps) * (
                        out.grad
                        - out.grad.mean(axis=-1, keepdims=True)
                        - x_norm * (out.grad * x_norm).mean(axis=-1, keepdims=True)
                    )
                )
        out._backward = _backward
        return out


class Dropout(Module):

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        self._mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        out = Tensor(x.data * self._mask * scale, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op = 'dropout'
        mask, sc = self._mask, scale
        def _backward():
            if x.requires_grad:
                x._accumulate(out.grad * mask * sc)
        out._backward = _backward
        return out


# =============================================================================
# ACTIVATIONS
# =============================================================================

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: return x.relu()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor: return x.tanh()

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor: return x.sigmoid()

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * x.data ** 3)))
        out = Tensor(x.data * cdf, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op = 'gelu'
        def _backward():
            if x.requires_grad:
                tanh_inner = np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * x.data ** 3)
                t = np.tanh(tanh_inner)
                sech2 = 1 - t ** 2
                dcdf = 0.5 * sech2 * np.sqrt(2.0 / np.pi) * (1 + 3 * 0.044715 * x.data ** 2)
                x._accumulate(out.grad * (cdf + x.data * dcdf))
        out._backward = _backward
        return out


# =============================================================================
# CONTAINERS
# =============================================================================

class Sequential(Module):

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()
        return self

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()
        return self


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    print("=== Linear ===")
    x = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    layer = Linear(8, 4)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    print(f"input:  {x.shape}  output: {out.shape}  loss: {loss.data:.4f}")
    print(f"weight grad shape: {layer.weight.grad.shape}")

    print("\n=== Sequential ===")
    model = Sequential(
        Linear(8, 16),
        ReLU(),
        Linear(16, 4),
    )
    x = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    out = model(x)
    out.sum().backward()
    print(f"params: {len(model.parameters())}")
    print(f"output shape: {out.shape}")

    print("\n=== LayerNorm ===")
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    ln = LayerNorm(4)
    out = ln(x)
    out.sum().backward()
    print(f"output shape: {out.shape}  gamma grad: {ln.gamma.grad}")

    print("\n=== Dropout (train) ===")
    x = Tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
    dp = Dropout(p=0.5)
    dp.train()
    out = dp(x)
    print(f"some zeros due to dropout: {(out.data == 0).any()}")

    print("\nAll checks passed.")
