import numpy as np
from .autograd import Tensor
from .nn import Module, Linear, Embedding, LayerNorm, Dropout, GELU, Sequential


# =============================================================================
# CAUSAL SELF-ATTENTION
# =============================================================================

class CausalSelfAttention(Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)

    def _split_heads(self, x: Tensor, B: int, T: int) -> Tensor:
        H, D = self.num_heads, self.head_dim
        data = x.data.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        out = Tensor(data, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op = 'split_heads'
        def _backward():
            if x.requires_grad:
                x._accumulate(out.grad.transpose(0, 2, 1, 3).reshape(B, T, H * D))
        out._backward = _backward
        return out

    def _merge_heads(self, x: Tensor, B: int, T: int) -> Tensor:
        H, D = self.num_heads, self.head_dim
        data = x.data.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        out = Tensor(data, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op = 'merge_heads'
        def _backward():
            if x.requires_grad:
                x._accumulate(out.grad.reshape(B, T, H, D).transpose(0, 2, 1, 3))
        out._backward = _backward
        return out

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        q = self._split_heads(self.q_proj(x), B, T)
        k = self._split_heads(self.k_proj(x), B, T)
        v = self._split_heads(self.v_proj(x), B, T)

        scores = (q @ k.transpose(0, 1, 3, 2)) * Tensor(
            np.full((1,), self.scale, dtype=np.float32)
        )

        mask = Tensor(np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1))
        scores = scores + mask

        attn = self.attn_drop(scores.softmax(axis=-1))
        out = self._merge_heads(attn @ v, B, T)

        return self.resid_drop(self.out_proj(out))

    def parameters(self):
        return (self.q_proj.parameters() + self.k_proj.parameters() +
                self.v_proj.parameters() + self.out_proj.parameters())


# =============================================================================
# FEED FORWARD
# =============================================================================

class FeedForward(Module):

    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = Sequential(
            Linear(embed_dim, 4 * embed_dim),
            GELU(),
            Linear(4 * embed_dim, embed_dim),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters() +
                self.ln2.parameters() + self.ff.parameters())

    def train(self):
        self.training = True
        for m in [self.ln1, self.attn, self.ln2, self.ff]:
            m.train()
        return self

    def eval(self):
        self.training = False
        for m in [self.ln1, self.attn, self.ln2, self.ff]:
            m.eval()
        return self


# =============================================================================
# GPT
# =============================================================================

class GPT(Module):

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout

        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_seq_len, embed_dim)
        self.drop = Dropout(dropout)
        self.blocks = [TransformerBlock(embed_dim, num_heads, dropout)
                       for _ in range(num_layers)]
        self.ln_f = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx) -> Tensor:
        if isinstance(idx, Tensor):
            idx = idx.data.astype(int)
        B, T = idx.shape
        assert T <= self.max_seq_len

        tok = self.token_emb(idx)
        pos = self.pos_emb(np.arange(T))
        pos_b = Tensor(np.broadcast_to(pos.data, (B, T, self.embed_dim)).copy(),
                       requires_grad=pos.requires_grad)
        pos_b._prev = {pos}
        def _pos_bwd():
            if pos.requires_grad:
                pos._accumulate(pos_b.grad.sum(axis=0))
        pos_b._backward = _pos_bwd

        x = tok + pos_b
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def parameters(self):
        params = []
        for m in [self.token_emb, self.pos_emb, self.ln_f, self.head]:
            params.extend(m.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        return params

    def train(self):
        self.training = True
        self.drop.train()
        for block in self.blocks:
            block.train()
        return self

    def eval(self):
        self.training = False
        self.drop.eval()
        for block in self.blocks:
            block.eval()
        return self

    def num_params(self) -> int:
        return sum(p.data.size for p in self.parameters())

    @staticmethod
    def nano(vocab_size: int) -> 'GPT':
        return GPT(vocab_size=vocab_size, embed_dim=64, num_heads=4,
                   num_layers=2, max_seq_len=128, dropout=0.1)

    @staticmethod
    def small(vocab_size: int) -> 'GPT':
        return GPT(vocab_size=vocab_size, embed_dim=256, num_heads=8,
                   num_layers=6, max_seq_len=512, dropout=0.1)

    @staticmethod
    def gpt2(vocab_size: int) -> 'GPT':
        return GPT(vocab_size=vocab_size, embed_dim=768, num_heads=12,
                   num_layers=12, max_seq_len=1024, dropout=0.1)

    def generate(self, idx, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = None) -> list:
        self.eval()
        generated = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        for _ in range(max_new_tokens):
            ctx = np.array([generated[-self.max_seq_len:]])
            logits = self.forward(ctx)
            last = logits.data[0, -1] / temperature
            if top_k:
                idx_topk = np.argpartition(last, -top_k)[-top_k:]
                mask = np.full_like(last, -1e9)
                mask[idx_topk] = 0
                last = last + mask
            probs = np.exp(last - last.max())
            probs /= probs.sum()
            next_tok = int(np.random.choice(len(probs), p=probs))
            generated.append(next_tok)
        return generated
