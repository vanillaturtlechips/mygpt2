"""
A100 최적화 GPT-2 124M 한국어 학습 스크립트

실행:
    pip install sentencepiece
    python3 train_a100.py

설정값은 아래 CONFIG에서 변경.
"""

import os
import time
import math
import json
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # 모델
    'vocab_size':   32000,
    'd_model':      768,
    'n_heads':      12,
    'n_layers':     12,
    'max_seq_len':  1024,
    'dropout':      0.1,

    # 학습
    'batch_size':   16,       # 배치당 시퀀스 수
    'grad_accum':   32,       # 유효 배치 = 16 × 32 = 512
    'lr':           3e-4,
    'warmup_steps': 2000,
    'max_steps':    200000,
    'grad_clip':    1.0,

    # 경로
    'data_path':    '/workspace/train.txt',
    'save_dir':     '/workspace/checkpoints',
    'sp_model':     '/workspace/tokenizer.model',

    # 체크포인트
    'save_interval_sec': 1800,   # 30분마다 저장
    'log_interval':      100,
}


# =============================================================================
# 토크나이저 (SentencePiece BPE)
# =============================================================================

def train_tokenizer(data_path: str, model_path: str, vocab_size: int):
    import sentencepiece as spm

    print("토크나이저 학습 중...")
    spm.SentencePieceTrainer.train(
        input=data_path,
        model_prefix=model_path.replace('.model', ''),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        input_sentence_size=5000000,
        shuffle_input_sentence=True,
    )
    print(f"토크나이저 저장: {model_path}")


def load_tokenizer(model_path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


# =============================================================================
# 데이터셋
# =============================================================================

class TextDataset(Dataset):

    def __init__(self, data_path: str, tokenizer, seq_len: int,
                 max_tokens: int = None):
        print("데이터 토크나이징 중... (시간 걸림)")
        tokens = []
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    tokens.extend(tokenizer.encode(line))
                if max_tokens and len(tokens) >= max_tokens:
                    break
                if i % 100000 == 0:
                    print(f"  {i:,}줄 처리... ({len(tokens):,} 토큰)")

        self.seq_len = seq_len
        self.data    = torch.tensor(tokens, dtype=torch.long)
        n_chunks     = (len(self.data) - 1) // seq_len
        print(f"총 토큰: {len(self.data):,}  청크: {n_chunks:,}")

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


# =============================================================================
# GPT-2 모델 (PyTorch)
# =============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj    = nn.Linear(d_model, d_model, bias=True)
        self.drop    = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len))
               .view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scale  = 1.0 / math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        scores = self.drop(scores)

        out = (scores @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FFN(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1  = nn.Linear(d_model, 4 * d_model)
        self.fc2  = nn.Linear(4 * d_model, d_model)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.gelu(self.fc1(x))))


class Block(nn.Module):

    def __init__(self, d_model, n_heads, dropout, max_seq_len):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FFN(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT2(nn.Module):

    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 max_seq_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            Block(d_model, n_heads, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight   # weight tying

        # 그래디언트 체크포인팅 (메모리 절약)
        for block in self.blocks:
            block.attn.drop  = nn.Dropout(dropout)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"파라미터: {n_params/1e6:.1f}M")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T  = idx.shape
        pos   = torch.arange(T, device=idx.device)
        x     = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)

        x    = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
        return logits, loss


# =============================================================================
# LR 스케줄 (코사인 + 워밍업)
# =============================================================================

def get_lr(step: int, cfg: dict) -> float:
    warmup = cfg['warmup_steps']
    max_s  = cfg['max_steps']
    lr     = cfg['lr']

    if step < warmup:
        return lr * step / warmup
    if step > max_s:
        return lr * 0.1

    ratio = (step - warmup) / (max_s - warmup)
    return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * ratio))


# =============================================================================
# 체크포인트
# =============================================================================

def save_checkpoint(model, optimizer, scaler, step, loss, cfg):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    path = os.path.join(cfg['save_dir'], f'ckpt_step{step}.pt')
    torch.save({
        'step':      step,
        'loss':      loss,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler':    scaler.state_dict(),
        'config':    cfg,
    }, path)

    # 최신 2개만 유지
    ckpts = sorted(glob.glob(os.path.join(cfg['save_dir'], 'ckpt_step*.pt')))
    for old in ckpts[:-2]:
        os.remove(old)
    print(f"\n체크포인트 저장: {path}  (loss={loss:.4f})")


def load_checkpoint(model, optimizer, scaler, cfg):
    ckpts = sorted(glob.glob(os.path.join(cfg['save_dir'], 'ckpt_step*.pt')))
    if not ckpts:
        return 0
    ckpt = torch.load(ckpts[-1])
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    print(f"체크포인트 로드: {ckpts[-1]}  (step={ckpt['step']})")
    return ckpt['step']


# =============================================================================
# 메인 학습 루프
# =============================================================================

def train():
    cfg    = CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 토크나이저 ────────────────────────────────────────────────────────────
    import sentencepiece as spm
    if not os.path.exists(cfg['sp_model']):
        train_tokenizer(cfg['data_path'], cfg['sp_model'], cfg['vocab_size'])
    tokenizer = load_tokenizer(cfg['sp_model'])

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    dataset = TextDataset(cfg['data_path'], tokenizer, cfg['max_seq_len'])
    loader  = DataLoader(dataset, batch_size=cfg['batch_size'],
                         shuffle=True, num_workers=4, pin_memory=True)

    # ── 모델 ──────────────────────────────────────────────────────────────────
    model = GPT2(
        vocab_size   = cfg['vocab_size'],
        d_model      = cfg['d_model'],
        n_heads      = cfg['n_heads'],
        n_layers     = cfg['n_layers'],
        max_seq_len  = cfg['max_seq_len'],
        dropout      = cfg['dropout'],
    ).to(device)

    # bf16 컴파일 (A100 최적화)
    if device.type == 'cuda':
        model = torch.compile(model)

    # ── 옵티마이저 ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'],
        betas=(0.9, 0.95), weight_decay=0.1,
    )
    scaler = GradScaler()

    # ── 체크포인트 이어서 학습 ────────────────────────────────────────────────
    start_step = load_checkpoint(model, optimizer, scaler, cfg)

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    model.train()
    step         = start_step
    opt_step     = 0
    last_save    = time.time()
    accum_loss   = 0.0
    t_start      = time.time()

    print(f"\n학습 시작 (step {step} → {cfg['max_steps']})\n")

    while step < cfg['max_steps']:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss    = loss / cfg['grad_accum']

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (step + 1) % cfg['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg['grad_clip'])

                lr = get_lr(opt_step, cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % cfg['log_interval'] == 0:
                    elapsed = time.time() - t_start
                    print(f"opt {opt_step:6d} | loss {accum_loss:.4f} | "
                          f"lr {lr:.2e} | {elapsed:.0f}s", flush=True)
                    accum_loss = 0.0

                if time.time() - last_save > cfg['save_interval_sec']:
                    save_checkpoint(model, optimizer, scaler,
                                    opt_step, accum_loss, cfg)
                    last_save = time.time()

            step += 1
            if step >= cfg['max_steps']:
                break

    # 최종 저장
    save_checkpoint(model, optimizer, scaler, step, accum_loss, cfg)
    print("\n학습 완료!")


if __name__ == '__main__':
    train()
