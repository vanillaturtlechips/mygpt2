import numpy as np
from .autograd import Tensor
from .nn import Module, Linear


class LoRALinear(Module):
    """Linear + low-rank adapter. W frozen; only A, B are trained.

    forward: y = Wx + (alpha/r) * B @ A @ x
    """

    def __init__(self, base: Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f = base.weight.data.shape[1]
        out_f = base.weight.data.shape[0]

        base.weight.requires_grad = False
        if base.bias is not None:
            base.bias.requires_grad = False

        # A: randn init so adapter fires immediately; B: zeros so delta=0 at step 0
        self.lora_A = Tensor(
            (np.random.randn(rank, in_f) / np.sqrt(rank)).astype(np.float32),
            requires_grad=True,
        )
        self.lora_B = Tensor(
            np.zeros((out_f, rank), dtype=np.float32),
            requires_grad=True,
        )
        self._scaling_t = Tensor(
            np.full((1,), self.scaling, dtype=np.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.transpose()) @ self.lora_B.transpose()
        return base_out + lora_out * self._scaling_t

    def parameters(self):
        return [self.lora_A, self.lora_B]

    def merge(self) -> Linear:
        """Return a plain Linear with W' = W + scaling * B @ A (no runtime overhead)."""
        in_f = self.base.weight.data.shape[1]
        out_f = self.base.weight.data.shape[0]
        has_bias = self.base.bias is not None

        merged = Linear(in_f, out_f, bias=has_bias)
        delta = self.lora_B.data @ self.lora_A.data  # [out_f, in_f]
        merged.weight = Tensor(
            (self.base.weight.data + self.scaling * delta).astype(np.float32),
            requires_grad=True,
        )
        if has_bias:
            merged.bias = self.base.bias
        else:
            merged.bias = None
        return merged


# =============================================================================
# INJECT / MERGE HELPERS
# =============================================================================

def inject_lora(model, rank: int = 8, alpha: float = 16.0,
                targets: tuple = ("q_proj", "v_proj")) -> None:
    """Replace target Linear layers in every attention block with LoRALinear."""
    for block in model.blocks:
        attn = block.attn
        for name in targets:
            original = getattr(attn, name)
            setattr(attn, name, LoRALinear(original, rank, alpha))


def merge_lora(model) -> None:
    """Merge all LoRALinear adapters back into plain Linear in-place."""
    for block in model.blocks:
        attn = block.attn
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            layer = getattr(attn, name)
            if isinstance(layer, LoRALinear):
                setattr(attn, name, layer.merge())


def lora_parameters(model) -> list:
    """Return only the LoRA adapter parameters (A, B matrices)."""
    params = []
    for block in model.blocks:
        attn = block.attn
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            layer = getattr(attn, name)
            if isinstance(layer, LoRALinear):
                params.extend(layer.parameters())
    return params
