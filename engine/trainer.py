import os
import time
import math
import numpy as np
from typing import Optional
from .autograd import Tensor, cross_entropy
from .transformer import GPT
from .tokenizer import BPETokenizer
from .optim import AdamW
from .data import DataLoader, Dataset


# =============================================================================
# LANGUAGE MODEL DATASET
# =============================================================================

class TextDataset(Dataset):

    def __init__(self, token_ids: list, seq_len: int):
        self.data = np.array(token_ids, dtype=np.int32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        if hasattr(idx, '__len__'):
            xs = np.stack([self.data[i: i + self.seq_len] for i in idx])
            ys = np.stack([self.data[i + 1: i + self.seq_len + 1] for i in idx])
            return Tensor(xs.astype(np.float32)), Tensor(ys.astype(np.int32))
        chunk = self.data[idx: idx + self.seq_len + 1]
        x = Tensor(chunk[:-1].astype(np.float32))
        y = Tensor(chunk[1:].astype(np.int32))
        return x, y


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:

    def __init__(self, model: GPT, tokenizer: BPETokenizer,
                 lr: float = 3e-4, weight_decay: float = 0.1,
                 grad_clip: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.grad_clip = grad_clip
        self._base_lr = lr
        self.optimizer = AdamW(model.parameters(), lr=lr,
                               betas=(0.9, 0.95), weight_decay=weight_decay)
        self.step = 0
        self.best_loss = float('inf')

    # ── loss ─────────────────────────────────────────────────────────────────

    def _compute_loss(self, x: np.ndarray, y: np.ndarray) -> Tensor:
        logits = self.model.forward(x)
        B, T, V = logits.data.shape

        targets_flat = y.reshape(B * T)
        one_hot = np.zeros((B * T, V), dtype=np.float32)
        one_hot[np.arange(B * T), targets_flat] = 1.0

        logits_flat = logits.reshape(B * T, V)
        target_t = Tensor(one_hot)
        return cross_entropy(logits_flat, target_t)

    # ── gradient clip ─────────────────────────────────────────────────────────

    def _clip_gradients(self):
        total_norm = 0.0
        params = self.model.parameters()
        for p in params:
            if p.grad is not None:
                total_norm += np.sum(p.grad ** 2)
        total_norm = math.sqrt(total_norm)
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-8)
            for p in params:
                if p.grad is not None:
                    p.grad *= scale
        return total_norm

    # ── lr schedule (cosine warmup) ───────────────────────────────────────────

    def _get_lr(self, total_steps: int, warmup_steps: int) -> float:
        if self.step < warmup_steps:
            return self._base_lr * self.step / max(1, warmup_steps)
        progress = (self.step - warmup_steps) / max(1, total_steps - warmup_steps)
        return self._base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        self.optimizer.lr = lr

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, dataset: TextDataset, batch_size: int = 8,
              epochs: int = 5, log_every: int = 10,
              save_path: Optional[str] = None):

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(loader) * epochs
        warmup_steps = total_steps // 10

        print(f"총 토큰    : {len(dataset.data):,}")
        print(f"배치 크기  : {batch_size}")
        print(f"배치 수    : {len(loader)}")
        print(f"에폭       : {epochs}")
        print(f"총 스텝    : {total_steps:,}")
        print(f"파라미터   : {self.model.num_params():,}")
        print("-" * 50)

        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_start = time.time()

            norm = 0.0
            for batch_idx, (xb, yb) in enumerate(loader):
                lr = self._get_lr(total_steps, warmup_steps)
                self._set_lr(lr)

                x = xb.data.astype(int)
                y = yb.data.astype(int)

                self.optimizer.zero_grad()
                loss = self._compute_loss(x, y)

                if not np.isfinite(loss.data):
                    self.step += 1
                    continue

                loss.backward()
                norm = self._clip_gradients()
                self.optimizer.step()

                epoch_loss += loss.data
                self.step += 1

                if self.step % log_every == 0:
                    ppl = math.exp(min(loss.data, 20))
                    print(f"epoch {epoch} | step {self.step:5d} | "
                          f"loss {loss.data:.4f} | ppl {ppl:.2f} | "
                          f"lr {lr:.2e} | norm {norm:.2f}")

            avg_loss = epoch_loss / len(loader)
            elapsed = time.time() - epoch_start
            print(f"\nepoch {epoch} 완료 | avg loss {avg_loss:.4f} | "
                  f"ppl {math.exp(min(avg_loss, 20)):.2f} | {elapsed:.1f}s\n")

            if save_path and avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save(save_path)
                print(f"모델 저장 → {save_path}")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        import json
        weights = {}
        for i, p in enumerate(self.model.parameters()):
            weights[str(i)] = p.data.tolist()
        config = {
            'vocab_size': self.model.vocab_size,
            'embed_dim': self.model.embed_dim,
            'max_seq_len': self.model.max_seq_len,
        }
        with open(path, 'w') as f:
            json.dump({'config': config, 'weights': weights}, f)

    def load(self, path: str):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        params = self.model.parameters()
        for i, p in enumerate(params):
            p.data = np.array(data['weights'][str(i)], dtype=np.float32)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    corpus = """
인공지능은 인간의 지능을 모방하는 기술입니다.
머신러닝은 인공지능의 한 분야로 데이터로부터 학습합니다.
딥러닝은 머신러닝의 한 방법으로 신경망을 사용합니다.
트랜스포머는 딥러닝 모델의 핵심 아키텍처입니다.
자연어 처리는 인공지능이 언어를 이해하는 분야입니다.
GPT는 트랜스포머 기반의 언어 모델입니다.
언어 모델은 다음 토큰을 예측하는 방식으로 학습합니다.
어텐션 메커니즘은 중요한 정보에 집중하게 합니다.
사전학습은 대규모 텍스트로 기본 언어 능력을 익힙니다.
파인튜닝은 특정 목적에 맞게 모델을 조정합니다.
""" * 100

    print("=== 토크나이저 학습 ===")
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=400)
    token_ids = tok.encode(corpus)
    print(f"vocab size : {tok.vocab_size}")
    print(f"토큰 수    : {len(token_ids):,}")

    print("\n=== 모델 생성 ===")
    model = GPT.nano(tok.vocab_size)
    print(f"파라미터   : {model.num_params():,}")

    print("\n=== 학습 시작 ===")
    dataset = TextDataset(token_ids, seq_len=32)
    trainer = Trainer(model, tok, lr=3e-4)
    trainer.train(dataset, batch_size=4, epochs=3, log_every=20)

    print("\n=== 텍스트 생성 ===")
    prompt = "인공지능은"
    prompt_ids = tok.encode(prompt)
    generated_ids = model.generate(prompt_ids, max_new_tokens=30,
                                   temperature=0.8, top_k=20)
    print(f"프롬프트 : {prompt}")
    print(f"생성     : {tok.decode(generated_ids)}")
