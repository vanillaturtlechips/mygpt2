import time
import numpy as np
from typing import Optional, Tuple
from .autograd import Tensor
from .nn import Module, Linear
from .transformer import GPT


# =============================================================================
# INT4 PACK / UNPACK
# =============================================================================
# Two int4 values (range [-8, 7]) stored in one uint8 byte.
# Byte layout: bits[0:3] = first value + 8,  bits[4:7] = second value + 8

def pack_int4(w: np.ndarray) -> Tuple[np.ndarray, tuple]:
    shape = w.shape
    flat  = np.clip(w.flatten().astype(np.int8), -8, 7)
    u     = (flat + 8).astype(np.uint8)          # [0, 15]
    if len(u) % 2:
        u = np.append(u, np.uint8(0))            # pad to even
    packed = (u[0::2] & 0x0F) | ((u[1::2] & 0x0F) << 4)
    return packed.astype(np.uint8), shape


def unpack_int4(packed: np.ndarray, shape: tuple) -> np.ndarray:
    lo = (packed        & 0x0F).astype(np.uint8)
    hi = ((packed >> 4) & 0x0F).astype(np.uint8)
    interleaved          = np.empty(lo.size + hi.size, dtype=np.uint8)
    interleaved[0::2]    = lo
    interleaved[1::2]    = hi
    n      = int(np.prod(shape))
    result = interleaved[:n].astype(np.int32) - 8  # [0,15] → [-8,7]
    return result.astype(np.int8).reshape(shape)


# =============================================================================
# QUANTIZED LINEAR
# =============================================================================

class QuantizedLinear(Module):
    """
    Drop-in replacement for Linear.
    Weights are stored as int8 or packed int4 + a float32 per-channel scale.
    Bias stays float32. Dequantization happens on the fly in forward().
    """

    def __init__(self, w_quant: np.ndarray, scale: np.ndarray,
                 bias: Optional[Tensor], bits: int, orig_shape: tuple):
        super().__init__()
        self.w_quant   = w_quant    # int8 array  (bits=8)  OR  packed uint8 (bits=4)
        self.scale     = scale      # float32  shape (out_features, 1)  per-channel
        self.bias      = bias       # Tensor | None
        self.bits      = bits
        self.orig_shape = orig_shape

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_linear(cls, linear: Linear, bits: int = 8) -> 'QuantizedLinear':
        w     = linear.weight.data                          # (out, in)  float32
        q_max = 127.0 if bits == 8 else 7.0

        # per-channel scale: one scale per output neuron
        scale = np.abs(w).max(axis=1, keepdims=True) / q_max  # (out, 1)
        scale = np.where(scale == 0, 1e-8, scale).astype(np.float32)

        w_int = np.round(w / scale).clip(-q_max, q_max).astype(np.int8)

        if bits == 4:
            w_stored, orig_shape = pack_int4(w_int)
        else:
            w_stored, orig_shape = w_int, w.shape

        return cls(w_stored, scale, linear.bias, bits, orig_shape)

    # ── dequantize ────────────────────────────────────────────────────────────

    def _dequantize(self) -> np.ndarray:
        if self.bits == 4:
            w_int = unpack_int4(self.w_quant, self.orig_shape).astype(np.float32)
        else:
            w_int = self.w_quant.astype(np.float32)
        return w_int * self.scale   # (out, in)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        w        = self._dequantize()           # (out, in)  float32
        out_data = x.data @ w.T                 # (..., out)
        if self.bias is not None:
            out_data = out_data + self.bias.data

        out       = Tensor(out_data, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op   = f'qlinear{self.bits}'

        def _backward():
            if x.requires_grad:
                x._accumulate(out.grad @ w)     # grad w.r.t. activations

        out._backward = _backward
        return out

    # ── housekeeping ──────────────────────────────────────────────────────────

    def parameters(self):
        # Quantized weights are frozen integers — only bias is a live Tensor
        return [self.bias] if self.bias is not None else []

    @property
    def weight_bytes(self) -> int:
        return self.w_quant.nbytes

    @property
    def float32_bytes(self) -> int:
        return int(np.prod(self.orig_shape)) * 4


# =============================================================================
# MODEL-LEVEL QUANTIZATION
# =============================================================================

def quantize_model(model: GPT, bits: int = 8) -> GPT:
    """
    Replace every Linear layer in model with QuantizedLinear in-place.
    Embeddings and LayerNorm parameters stay float32.
    """
    assert bits in (4, 8), "bits must be 4 or 8"

    def ql(linear: Linear) -> QuantizedLinear:
        return QuantizedLinear.from_linear(linear, bits)

    model.head = ql(model.head)

    for block in model.blocks:
        a           = block.attn
        a.q_proj    = ql(a.q_proj)
        a.k_proj    = ql(a.k_proj)
        a.v_proj    = ql(a.v_proj)
        a.out_proj  = ql(a.out_proj)

        layers      = block.ff.net.layers
        layers[0]   = ql(layers[0])
        layers[2]   = ql(layers[2])

    return model


# =============================================================================
# MEMORY REPORT
# =============================================================================

def memory_report(model: GPT) -> dict:
    """
    Returns bytes used by:
      float_bytes  — embeddings, LayerNorm, biases  (always float32)
      weight_bytes — linear weights (float32 or quantized)
      total_bytes  — sum
    """
    float_bytes  = sum(p.data.nbytes for p in model.parameters())
    weight_bytes = 0

    def tally(layer):
        nonlocal weight_bytes
        if isinstance(layer, QuantizedLinear):
            weight_bytes += layer.weight_bytes
        elif isinstance(layer, Linear):
            weight_bytes += layer.weight.data.nbytes

    tally(model.head)
    for block in model.blocks:
        for lyr in [block.attn.q_proj, block.attn.k_proj,
                    block.attn.v_proj, block.attn.out_proj]:
            tally(lyr)
        for lyr in block.ff.net.layers:
            if isinstance(lyr, (Linear, QuantizedLinear)):
                tally(lyr)

    return {
        'float_bytes':  float_bytes,
        'weight_bytes': weight_bytes,
        'total_bytes':  float_bytes + weight_bytes,
    }


# =============================================================================
# QUANTIZATION ERROR
# =============================================================================

def _iter_linears(model: GPT):
    """Yield all Linear / QuantizedLinear layers in order."""
    yield model.head
    for block in model.blocks:
        yield block.attn.q_proj
        yield block.attn.k_proj
        yield block.attn.v_proj
        yield block.attn.out_proj
        for lyr in block.ff.net.layers:
            if isinstance(lyr, (Linear, QuantizedLinear)):
                yield lyr


def quantization_error(original: GPT, quantized: GPT) -> dict:
    """Max and mean absolute error between original and dequantized weights."""
    max_err, mean_err, n = 0.0, 0.0, 0
    for orig_lyr, quant_lyr in zip(_iter_linears(original), _iter_linears(quantized)):
        w_orig  = orig_lyr.weight.data          # float32
        w_quant = quant_lyr._dequantize()       # float32 (reconstructed)
        diff    = np.abs(w_orig - w_quant)
        max_err  = max(max_err, float(diff.max()))
        mean_err += float(diff.mean())
        n += 1
    return {'max_abs_error': max_err, 'mean_abs_error': mean_err / max(n, 1)}


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import copy
    from .tokenizer import BPETokenizer

    np.random.seed(42)

    # ── corpus + tokenizer ────────────────────────────────────────────────────
    corpus = """
인공지능은 인간의 지능을 모방하는 기술입니다.
머신러닝은 데이터로부터 패턴을 학습합니다.
딥러닝은 신경망을 사용하는 머신러닝 방법입니다.
트랜스포머는 어텐션 메커니즘을 기반으로 합니다.
GPT는 트랜스포머 기반의 언어 모델입니다.
""" * 50

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=400)

    # ── train small model ─────────────────────────────────────────────────────
    from .trainer import TextDataset, Trainer
    model_fp32 = GPT.nano(tok.vocab_size)
    token_ids  = tok.encode(corpus)
    dataset    = TextDataset(token_ids, seq_len=32)
    trainer    = Trainer(model_fp32, tok, lr=3e-4)
    print("학습 중 (3 에폭)...")
    trainer.train(dataset, batch_size=4, epochs=3, log_every=99999)

    # ── float32 baseline ──────────────────────────────────────────────────────
    prompt_ids = tok.encode("인공지능은")
    ctx        = np.array([prompt_ids])

    t0      = time.perf_counter()
    for _   in range(20):
        model_fp32.forward(ctx)
    fp32_ms = (time.perf_counter() - t0) / 20 * 1000

    mem_fp32 = memory_report(model_fp32)

    # ── INT8 ──────────────────────────────────────────────────────────────────
    model_int8 = copy.deepcopy(model_fp32)
    quantize_model(model_int8, bits=8)

    t0      = time.perf_counter()
    for _   in range(20):
        model_int8.forward(ctx)
    int8_ms = (time.perf_counter() - t0) / 20 * 1000

    mem_int8 = memory_report(model_int8)

    # ── INT4 ──────────────────────────────────────────────────────────────────
    model_int4 = copy.deepcopy(model_fp32)
    quantize_model(model_int4, bits=4)

    t0      = time.perf_counter()
    for _   in range(20):
        model_int4.forward(ctx)
    int4_ms = (time.perf_counter() - t0) / 20 * 1000

    mem_int4 = memory_report(model_int4)

    # ── results ───────────────────────────────────────────────────────────────
    def mb(b): return b / 1024 / 1024

    print("\n=== 메모리 비교 ===")
    print(f"{'':10} {'weight':>10} {'float32':>10} {'total':>10}")
    print(f"{'fp32':10} {mb(mem_fp32['weight_bytes']):>9.2f}MB"
          f" {mb(mem_fp32['float_bytes']):>9.2f}MB"
          f" {mb(mem_fp32['total_bytes']):>9.2f}MB")
    print(f"{'int8':10} {mb(mem_int8['weight_bytes']):>9.2f}MB"
          f" {mb(mem_int8['float_bytes']):>9.2f}MB"
          f" {mb(mem_int8['total_bytes']):>9.2f}MB"
          f"  ({mem_fp32['total_bytes']/mem_int8['total_bytes']:.1f}x 압축)")
    print(f"{'int4':10} {mb(mem_int4['weight_bytes']):>9.2f}MB"
          f" {mb(mem_int4['float_bytes']):>9.2f}MB"
          f" {mb(mem_int4['total_bytes']):>9.2f}MB"
          f"  ({mem_fp32['total_bytes']/mem_int4['total_bytes']:.1f}x 압축)")

    print("\n=== 추론 속도 비교 (forward 1회) ===")
    print(f"fp32  : {fp32_ms:.2f} ms")
    print(f"int8  : {int8_ms:.2f} ms  ({fp32_ms/int8_ms:.1f}x)")
    print(f"int4  : {int4_ms:.2f} ms  ({fp32_ms/int4_ms:.1f}x)")

    print("\n=== 양자화 오차 ===")
    err8 = quantization_error(model_fp32, model_int8)
    err4 = quantization_error(model_fp32, model_int4)
    print(f"int8  max={err8['max_abs_error']:.6f}  mean={err8['mean_abs_error']:.6f}")
    print(f"int4  max={err4['max_abs_error']:.6f}  mean={err4['mean_abs_error']:.6f}")

    print("\n=== 생성 비교 ===")
    for label, m in [('fp32', model_fp32), ('int8', model_int8), ('int4', model_int4)]:
        np.random.seed(0)
        ids  = m.generate(prompt_ids, max_new_tokens=20, temperature=0.8, top_k=10)
        text = tok.decode(ids)
        print(f"[{label}] {text}")
