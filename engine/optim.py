import numpy as np
from typing import List
from .autograd import Tensor


# =============================================================================
# BASE OPTIMIZER
# =============================================================================

class Optimizer:

    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


# =============================================================================
# OPTIMIZERS
# =============================================================================

class SGD(Optimizer):

    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            p.data -= self.lr * self.velocity[i]


class Adam(Optimizer):

    def __init__(self, params: List[Tensor], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * p.grad ** 2
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):

    def __init__(self, params: List[Tensor], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            p.data -= self.lr * self.weight_decay * p.data
            self.m[i] = b1 * self.m[i] + (1 - b1) * p.grad
            self.v[i] = b2 * self.v[i] + (1 - b2) * p.grad ** 2
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from engine.autograd import Tensor, mse_loss
    from engine.nn import Linear, ReLU, Sequential

    np.random.seed(0)

    X = Tensor(np.random.randn(64, 4).astype(np.float32))
    y = Tensor((X.data[:, 0] + X.data[:, 1] > 0).astype(np.float32).reshape(-1, 1))

    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 1))
    opt = AdamW(model.parameters(), lr=1e-2)

    for step in range(200):
        opt.zero_grad()
        pred = model(X)
        loss = mse_loss(pred, y)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss.data:.4f}")

    print("AdamW training complete.")
