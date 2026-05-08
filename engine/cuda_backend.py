"""
CUDA backend for transformer operations.

GPU 메모리를 직접 관리하고 libengine.so 커널을 호출한다.
engine/server.py의 numpy 연산을 이 모듈로 교체하면
GTX 1060 / A100 등 CUDA GPU에서 돌아간다.

사용법:
    from engine.cuda_backend import CUDABackend
    gpu = CUDABackend()          # libengine.so 로드
    d_x = gpu.to_gpu(np_array)  # numpy → GPU
    d_y = gpu.matmul(d_x, d_w, M, K, N)
    result = gpu.to_cpu(d_y, shape)
    gpu.free(d_x, d_y)
"""

import ctypes
import os
import numpy as np
from typing import Tuple


# =============================================================================
# 라이브러리 로드
# =============================================================================

def _load_lib() -> ctypes.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))
    so   = os.path.join(here, '..', 'cuda', 'libengine.so')
    so   = os.path.normpath(so)
    if not os.path.exists(so):
        raise FileNotFoundError(
            f"libengine.so not found at {so}\n"
            "Run: cd cuda && nvcc -shared -Xcompiler -fPIC -o libengine.so libengine.cu -lm")
    lib = ctypes.CDLL(so)

    # ── 함수 시그니처 등록 ──────────────────────────────────────────────────
    _p = ctypes.c_void_p
    _f = ctypes.POINTER(ctypes.c_float)
    _i = ctypes.c_int
    _ip = ctypes.POINTER(ctypes.c_int)

    lib.engine_malloc.restype  = _p
    lib.engine_malloc.argtypes = [_i]
    lib.engine_free.argtypes   = [_p]
    lib.engine_h2d.argtypes    = [_p, _p, _i]
    lib.engine_d2h.argtypes    = [_p, _p, _i]
    lib.engine_sync.argtypes   = []

    lib.engine_matmul.argtypes   = [_f, _f, _f, _i, _i, _i]
    lib.engine_softmax.argtypes  = [_f, _f, _i, _i]
    lib.engine_layernorm.argtypes= [_f, _f, _f, _f, _i, _i]
    lib.engine_gelu.argtypes     = [_f, _f, _i]
    lib.engine_embedding.argtypes= [_f, _ip, _f, _i, _i]
    lib.engine_add.argtypes      = [_f, _f, _f, _i]
    lib.engine_bias_add.argtypes = [_f, _f, _i, _i]
    lib.engine_attention.argtypes= [_f, _f, _f, _f, _f, _i, _i]

    return lib


# =============================================================================
# GPU 포인터 래퍼
# =============================================================================

class GPUArray:
    """GPU 메모리 포인터 + 크기 정보."""

    def __init__(self, ptr: ctypes.c_void_p, shape: tuple, dtype=np.float32):
        self.ptr   = ptr
        self.shape = shape
        self.dtype = dtype
        self.size  = int(np.prod(shape))

    @property
    def float_ptr(self) -> ctypes.POINTER(ctypes.c_float):
        return ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_float))

    @property
    def int_ptr(self) -> ctypes.POINTER(ctypes.c_int):
        return ctypes.cast(self.ptr, ctypes.POINTER(ctypes.c_int))

    def __repr__(self):
        return f"GPUArray({self.shape}, {self.dtype.__name__})"


# =============================================================================
# CUDA Backend
# =============================================================================

class CUDABackend:

    def __init__(self):
        self.lib = _load_lib()

    # ── 메모리 ────────────────────────────────────────────────────────────────

    def to_gpu(self, arr: np.ndarray) -> GPUArray:
        """numpy → GPU."""
        arr  = np.ascontiguousarray(arr)
        ptr  = self.lib.engine_malloc(arr.nbytes)
        self.lib.engine_h2d(ptr,
                            arr.ctypes.data_as(ctypes.c_void_p),
                            arr.nbytes)
        return GPUArray(ptr, arr.shape, arr.dtype)

    def to_cpu(self, g: GPUArray) -> np.ndarray:
        """GPU → numpy."""
        arr = np.empty(g.shape, dtype=g.dtype)
        self.lib.engine_d2h(arr.ctypes.data_as(ctypes.c_void_p),
                            g.ptr, arr.nbytes)
        return arr

    def alloc(self, shape: tuple, dtype=np.float32) -> GPUArray:
        """GPU 메모리만 할당 (초기화 없음)."""
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        ptr    = self.lib.engine_malloc(nbytes)
        return GPUArray(ptr, shape, dtype)

    def free(self, *arrays: GPUArray):
        for g in arrays:
            if g is not None:
                self.lib.engine_free(g.ptr)

    def sync(self):
        self.lib.engine_sync()

    # ── 연산 ──────────────────────────────────────────────────────────────────

    def matmul(self, A: GPUArray, B: GPUArray,
               M: int, K: int, N: int) -> GPUArray:
        """C = A @ B  (M×K @ K×N → M×N)."""
        C = self.alloc((M, N))
        self.lib.engine_matmul(A.float_ptr, B.float_ptr, C.float_ptr,
                               M, K, N)
        return C

    def softmax(self, X: GPUArray, rows: int, cols: int) -> GPUArray:
        Y = self.alloc((rows, cols))
        self.lib.engine_softmax(X.float_ptr, Y.float_ptr, rows, cols)
        return Y

    def layernorm(self, X: GPUArray, gamma: GPUArray, beta: GPUArray,
                  rows: int, cols: int) -> GPUArray:
        Y = self.alloc((rows, cols))
        self.lib.engine_layernorm(X.float_ptr, Y.float_ptr,
                                  gamma.float_ptr, beta.float_ptr,
                                  rows, cols)
        return Y

    def gelu(self, X: GPUArray) -> GPUArray:
        Y = self.alloc(X.shape)
        self.lib.engine_gelu(X.float_ptr, Y.float_ptr, X.size)
        return Y

    def embedding(self, weight: GPUArray, ids: GPUArray,
                  seq_len: int, d_model: int) -> GPUArray:
        out = self.alloc((seq_len, d_model))
        self.lib.engine_embedding(weight.float_ptr, ids.int_ptr,
                                  out.float_ptr, seq_len, d_model)
        return out

    def add(self, A: GPUArray, B: GPUArray) -> GPUArray:
        out = self.alloc(A.shape)
        self.lib.engine_add(A.float_ptr, B.float_ptr, out.float_ptr, A.size)
        return out

    def bias_add(self, X: GPUArray, bias: GPUArray,
                 rows: int, cols: int) -> GPUArray:
        # in-place: X가 수정됨
        self.lib.engine_bias_add(X.float_ptr, bias.float_ptr, rows, cols)
        return X

    def attention(self, Q: GPUArray, K: GPUArray, V: GPUArray,
                  seq_len: int, d_k: int) -> GPUArray:
        out    = self.alloc((seq_len, d_k))
        scores = self.alloc((seq_len, seq_len))
        self.lib.engine_attention(Q.float_ptr, K.float_ptr, V.float_ptr,
                                  out.float_ptr, scores.float_ptr,
                                  seq_len, d_k)
        self.free(scores)
        return out


# =============================================================================
# 검증 & 벤치마크
# =============================================================================

if __name__ == '__main__':
    import time

    print("CUDABackend 로드 중...")
    gpu = CUDABackend()
    print("로드 완료\n")

    # ── 행렬곱 ────────────────────────────────────────────────────────────────
    M, K, N = 512, 512, 512
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    ref = A @ B

    dA = gpu.to_gpu(A)
    dB = gpu.to_gpu(B)

    t0 = time.perf_counter()
    dC = gpu.matmul(dA, dB, M, K, N)
    gpu.sync()
    t1 = time.perf_counter()

    C = gpu.to_cpu(dC)
    err = np.abs(C - ref).max()
    print(f"[matmul]    {M}×{K}×{N}  오차={err:.6f}  시간={1000*(t1-t0):.2f}ms")
    gpu.free(dA, dB, dC)

    # ── LayerNorm ─────────────────────────────────────────────────────────────
    rows, cols = 512, 768
    X     = np.random.randn(rows, cols).astype(np.float32)
    gamma = np.ones(cols, dtype=np.float32)
    beta  = np.zeros(cols, dtype=np.float32)

    # CPU 기준
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-5
    ref  = (X - mean) / std

    dX = gpu.to_gpu(X)
    dg = gpu.to_gpu(gamma)
    db = gpu.to_gpu(beta)

    t0 = time.perf_counter()
    dY = gpu.layernorm(dX, dg, db, rows, cols)
    gpu.sync()
    t1 = time.perf_counter()

    Y   = gpu.to_cpu(dY)
    err = np.abs(Y - ref).max()
    print(f"[layernorm] {rows}×{cols}  오차={err:.6f}  시간={1000*(t1-t0):.2f}ms")
    gpu.free(dX, dg, db, dY)

    # ── Attention ─────────────────────────────────────────────────────────────
    seq_len, d_k = 128, 64
    Q = np.random.randn(seq_len, d_k).astype(np.float32) * 0.1
    K_ = np.random.randn(seq_len, d_k).astype(np.float32) * 0.1
    V = np.random.randn(seq_len, d_k).astype(np.float32) * 0.1

    dQ = gpu.to_gpu(Q)
    dK = gpu.to_gpu(K_)
    dV = gpu.to_gpu(V)

    t0 = time.perf_counter()
    dOut = gpu.attention(dQ, dK, dV, seq_len, d_k)
    gpu.sync()
    t1 = time.perf_counter()

    out = gpu.to_cpu(dOut)
    print(f"[attention] seq={seq_len} d_k={d_k}  "
          f"출력shape={out.shape}  시간={1000*(t1-t0):.2f}ms")
    gpu.free(dQ, dK, dV, dOut)

    print("\n모든 커널 Python에서 정상 호출됨.")
