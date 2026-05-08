"""
Inference server for our GPT engine.

InferenceEngine  — pure-numpy forward + KV cache (no autograd overhead)
GenerateServer   — HTTP server with /generate and /health endpoints
"""

import json
import time
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Iterator, Optional

from .transformer import GPT
from .tokenizer import BPETokenizer
try:
    from .quantize import QuantizedLinear
    _has_quant = True
except ImportError:
    _has_quant = False


# =============================================================================
# WEIGHT EXTRACTION
# =============================================================================

def _wb(layer):
    """Return (weight: ndarray, bias: ndarray|None) for Linear or QuantizedLinear."""
    if _has_quant and isinstance(layer, QuantizedLinear):
        w = layer._dequantize()
        b = layer.bias.data if layer.bias is not None else None
    else:
        w = layer.weight.data
        b = layer.bias.data if layer.bias is not None else None
    return w, b


# =============================================================================
# NUMPY MODULES
# =============================================================================

class _LayerNorm:
    def __init__(self, ln):
        self.gamma = ln.gamma.data          # (C,)
        self.beta  = ln.beta.data           # (C,)
        self.eps   = getattr(ln, 'eps', 1e-5)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class _Attention:
    def __init__(self, attn, embed_dim: int, num_heads: int):
        self.Wq, _   = _wb(attn.q_proj)    # (C, C)
        self.Wk, _   = _wb(attn.k_proj)
        self.Wv, _   = _wb(attn.v_proj)
        self.Wo, self.bo = _wb(attn.out_proj)
        self.H       = num_heads
        self.D       = embed_dim // num_heads
        self.scale   = self.D ** -0.5

    def __call__(self, x: np.ndarray,
                 cache: Optional[dict] = None) -> np.ndarray:
        B, T, C = x.shape
        H, D    = self.H, self.D

        q = (x @ self.Wq.T).reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B,H,T,D)
        k = (x @ self.Wk.T).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = (x @ self.Wv.T).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        if cache is not None:
            # Append new K,V and keep full history
            cache['k'] = np.concatenate([cache['k'], k], axis=2)
            cache['v'] = np.concatenate([cache['v'], v], axis=2)
            k, v = cache['k'], cache['v']       # (B, H, T_total, D)

        T_total = k.shape[2]
        scores  = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B,H,T,T_total)

        # Causal mask only needed during prefill (T > 1)
        if T > 1:
            mask   = np.triu(np.full((T, T_total), -1e9, dtype=np.float32), k=1)
            scores = scores + mask

        scores -= scores.max(axis=-1, keepdims=True)
        attn    = np.exp(scores)
        attn   /= attn.sum(axis=-1, keepdims=True) + 1e-10

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        out = out @ self.Wo.T
        if self.bo is not None:
            out += self.bo
        return out


class _FFN:
    def __init__(self, ff):
        self.W1, self.b1 = _wb(ff.net.layers[0])
        self.W2, self.b2 = _wb(ff.net.layers[2])

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.W1.T + (self.b1 if self.b1 is not None else 0)
        h = self._gelu(h)
        return h @ self.W2.T + (self.b2 if self.b2 is not None else 0)


# =============================================================================
# INFERENCE ENGINE  (pure numpy + KV cache)
# =============================================================================

class InferenceEngine:
    """
    Fast inference engine extracted from a trained GPT model.
    Uses pure numpy (no autograd overhead) and KV cache for O(T) decode steps.

    Without KV cache: generating N tokens = N × T matmuls  (O(N²T))
    With    KV cache: generating N tokens = 1 prefill + N × 1 matmuls  (O(NT))
    """

    def __init__(self, model: GPT):
        model.eval()
        self.embed_dim   = model.embed_dim
        self.num_heads   = model.num_heads
        self.num_layers  = len(model.blocks)
        self.max_seq_len = model.max_seq_len

        # Embeddings (always float32)
        self.tok_emb = model.token_emb.weight.data.copy()   # (V, C)
        self.pos_emb = model.pos_emb.weight.data.copy()     # (L, C)

        # Final LayerNorm + LM head
        self.ln_f   = _LayerNorm(model.ln_f)
        self.Wh, self.bh = _wb(model.head)

        # Transformer blocks
        self.blocks = [
            {
                'ln1':  _LayerNorm(b.ln1),
                'attn': _Attention(b.attn, model.embed_dim, model.num_heads),
                'ln2':  _LayerNorm(b.ln2),
                'ff':   _FFN(b.ff),
            }
            for b in model.blocks
        ]

    # ── internal forward ──────────────────────────────────────────────────────

    def _init_cache(self) -> list:
        D = self.embed_dim // self.num_heads
        empty = np.zeros((1, self.num_heads, 0, D), dtype=np.float32)
        return [{'k': empty.copy(), 'v': empty.copy()}
                for _ in range(self.num_layers)]

    def _forward(self, token_ids: np.ndarray,
                 cache: Optional[list] = None) -> np.ndarray:
        """
        token_ids : (1, T)
        cache     : list of per-layer dicts, modified in-place
        returns   : logits (1, T, vocab_size)
        """
        T        = token_ids.shape[1]
        cache_len = cache[0]['k'].shape[2] if cache is not None else 0
        positions = np.arange(cache_len, cache_len + T)

        x = (self.tok_emb[token_ids[0]] +
             self.pos_emb[positions])[np.newaxis, :]         # (1, T, C)

        for i, blk in enumerate(self.blocks):
            c  = cache[i] if cache is not None else None
            x  = x + blk['attn'](blk['ln1'](x), c)
            x  = x + blk['ff'](blk['ln2'](x))

        x = self.ln_f(x)
        logits = x @ self.Wh.T
        if self.bh is not None:
            logits += self.bh
        return logits                                        # (1, T, V)

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float,
                top_k: Optional[int]) -> int:
        logits = logits / max(temperature, 1e-8)
        if top_k:
            idx  = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -1e9)
            mask[idx] = 0
            logits = logits + mask
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    # ── public API ────────────────────────────────────────────────────────────

    def generate_stream(self, prompt_ids: list, max_tokens: int = 50,
                        temperature: float = 0.8,
                        top_k: Optional[int] = 20) -> Iterator[int]:
        """Yield token IDs one at a time as they are generated."""
        cache   = self._init_cache()
        ctx     = np.array([prompt_ids], dtype=np.int32)
        logits  = self._forward(ctx, cache)                  # prefill
        next_id = self._sample(logits[0, -1], temperature, top_k)

        for _ in range(max_tokens):
            yield next_id
            logits  = self._forward(np.array([[next_id]], dtype=np.int32), cache)
            next_id = self._sample(logits[0, 0], temperature, top_k)

    def generate(self, prompt_ids: list, max_tokens: int = 50,
                 temperature: float = 0.8,
                 top_k: Optional[int] = 20) -> list:
        result = list(prompt_ids)
        for tok in self.generate_stream(prompt_ids, max_tokens, temperature, top_k):
            result.append(tok)
        return result

    def encode(self, token_ids: list) -> np.ndarray:
        """Mean-pooled hidden state — used as a text embedding vector."""
        T         = len(token_ids)
        positions = np.arange(T)
        x = (self.tok_emb[token_ids] + self.pos_emb[positions])[np.newaxis, :]
        for blk in self.blocks:
            x = x + blk['attn'](blk['ln1'](x))   # no KV cache needed
            x = x + blk['ff'](blk['ln2'](x))
        x = self.ln_f(x)           # (1, T, C)
        return x[0].mean(axis=0)   # (C,)  mean-pool over token positions


# =============================================================================
# HTTP SERVER
# =============================================================================

class _Handler(BaseHTTPRequestHandler):
    engine: InferenceEngine = None
    tokenizer: BPETokenizer = None

    # ── routing ───────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == '/health':
            self._json(200, {'status': 'ok'})
        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self):
        if self.path == '/generate':
            length = int(self.headers.get('Content-Length', 0))
            try:
                req = json.loads(self.rfile.read(length))
                self._generate(req)
            except Exception as e:
                self._json(500, {'error': str(e)})
        else:
            self._json(404, {'error': 'not found'})

    # ── /generate ─────────────────────────────────────────────────────────────

    def _generate(self, req: dict):
        prompt      = req.get('prompt', '')
        max_tokens  = int(req.get('max_tokens', 50))
        temperature = float(req.get('temperature', 0.8))
        top_k       = req.get('top_k', 20)
        stream      = bool(req.get('stream', False))

        prompt_ids = self.tokenizer.encode(prompt)

        if stream:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            gen = self.engine.generate_stream(
                prompt_ids, max_tokens, temperature, top_k)
            for tok_id in gen:
                token_text = self.tokenizer.decode([tok_id])
                data = json.dumps({'token': token_text}, ensure_ascii=False)
                self.wfile.write(f'data: {data}\n\n'.encode('utf-8'))
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        else:
            t0  = time.perf_counter()
            ids = self.engine.generate(
                prompt_ids, max_tokens, temperature, top_k)
            ms  = (time.perf_counter() - t0) * 1000
            self._json(200, {
                'text':       self.tokenizer.decode(ids),
                'tokens':     len(ids) - len(prompt_ids),
                'latency_ms': round(ms, 1),
            })

    # ── helpers ───────────────────────────────────────────────────────────────

    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f'[server] {self.address_string()} {fmt % args}')


class GenerateServer:

    def __init__(self, engine: InferenceEngine, tokenizer: BPETokenizer,
                 host: str = '127.0.0.1', port: int = 8080):
        # Bind engine and tokenizer to the handler class
        _Handler.engine    = engine
        _Handler.tokenizer = tokenizer
        self._server = HTTPServer((host, port), _Handler)
        self.host    = host
        self.port    = port

    def start(self):
        """Block and serve forever."""
        print(f'서버 시작: http://{self.host}:{self.port}')
        print('  POST /generate  {"prompt":"...", "max_tokens":50, "stream":false}')
        print('  GET  /health')
        self._server.serve_forever()

    def start_background(self) -> threading.Thread:
        """Start in a background daemon thread."""
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        return t

    def stop(self):
        self._server.shutdown()


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import urllib.request
    from .tokenizer import BPETokenizer
    from .trainer import TextDataset, Trainer

    np.random.seed(42)

    corpus = """
인공지능은 인간의 지능을 모방하는 기술입니다.
머신러닝은 데이터로부터 패턴을 학습합니다.
딥러닝은 신경망을 사용하는 머신러닝 방법입니다.
트랜스포머는 어텐션 메커니즘을 기반으로 합니다.
GPT는 트랜스포머 기반의 언어 모델입니다.
언어 모델은 다음 토큰을 예측하는 방식으로 학습합니다.
""" * 60

    # ── train ─────────────────────────────────────────────────────────────────
    print("=== 모델 학습 ===")
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=400)
    model   = GPT.nano(tok.vocab_size)
    dataset = TextDataset(tok.encode(corpus), seq_len=32)
    trainer = Trainer(model, tok, lr=3e-4)
    trainer.train(dataset, batch_size=4, epochs=3, log_every=99999)

    prompt_ids = tok.encode("인공지능은")
    engine     = InferenceEngine(model)

    # ── KV cache speedup benchmark ────────────────────────────────────────────
    print("\n=== KV Cache 속도 비교 ===")
    N = 30  # tokens to generate

    # Without cache: use the original GPT.generate() (recomputes full context each step)
    np.random.seed(1)
    t0 = time.perf_counter()
    model.generate(prompt_ids, max_new_tokens=N, temperature=0.8, top_k=10)
    no_cache_ms = (time.perf_counter() - t0) * 1000

    # With KV cache: InferenceEngine
    np.random.seed(1)
    t0 = time.perf_counter()
    engine.generate(prompt_ids, max_tokens=N, temperature=0.8, top_k=10)
    cache_ms = (time.perf_counter() - t0) * 1000

    print(f"KV cache 없음 (autograd GPT.generate) : {no_cache_ms:.1f} ms")
    print(f"KV cache 있음 (InferenceEngine)        : {cache_ms:.1f} ms")
    print(f"속도 향상: {no_cache_ms/cache_ms:.1f}x")

    # ── generation quality ────────────────────────────────────────────────────
    print("\n=== 생성 품질 ===")
    np.random.seed(0)
    ids  = engine.generate(prompt_ids, max_tokens=25, temperature=0.7, top_k=15)
    print(f"결과: {tok.decode(ids)}")

    # ── HTTP server test ──────────────────────────────────────────────────────
    print("\n=== HTTP 서버 테스트 ===")
    srv = GenerateServer(engine, tok, host='127.0.0.1', port=18080)
    srv.start_background()
    time.sleep(0.2)

    # Health check
    resp = urllib.request.urlopen('http://127.0.0.1:18080/health')
    print(f"GET /health  → {resp.read().decode()}")

    # Generate
    body = json.dumps({'prompt': '인공지능은', 'max_tokens': 20,
                       'temperature': 0.7, 'top_k': 10}).encode()
    req  = urllib.request.Request(
        'http://127.0.0.1:18080/generate',
        data=body, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read().decode())
    print(f"POST /generate → {result}")

    srv.stop()
    print("\n서버 종료.")
