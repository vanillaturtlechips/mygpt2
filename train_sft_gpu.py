"""
GPU SFT trainer: ckpt_step200000.pt → finetuned_gpu.json

RTX 2070 등 로컬 GPU에서 SFT 학습 후 numpy 엔진 호환 JSON으로 저장.
eval.py / compare.py는 변경 없이 그대로 사용 가능.

사용법:
    python train_sft_gpu.py ^
        --ckpt checkpoints/ckpt_step200000.pt ^
        --tok  tokenizer.model ^
        --sft  data/sft_data.json ^
        --out  checkpoints/finetuned_gpu.json
"""

import argparse
import json
import math
import os

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_a100 import GPT2


# =============================================================================
# 토크나이저
# =============================================================================

class SPChatTokenizer:
    def __init__(self, model_path: str):
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        base         = self._sp.get_piece_size()  # 32000
        self.user_id = base                        # 32000
        self.asst_id = base + 1                    # 32001
        self.eos_id  = self._sp.eos_id()           # 3
        self.vocab_size = base + 2                 # 32002

    def encode(self, text: str) -> list:
        return self._sp.encode(text, out_type=int)


# =============================================================================
# SFT 데이터셋
# =============================================================================

class SFTDataset(Dataset):
    def __init__(self, data: list, tok: SPChatTokenizer, seq_len: int = 512):
        self.samples = []
        for conv in data:
            user_ids = [tok.user_id] + tok.encode(conv['user'])
            asst_ids = [tok.asst_id] + tok.encode(conv['assistant']) + [tok.eos_id]
            ids  = user_ids + asst_ids
            mask = [0] * len(user_ids) + [1] * len(asst_ids)
            if len(ids) > seq_len + 1:
                ids, mask = ids[:seq_len + 1], mask[:seq_len + 1]
            self.samples.append((
                torch.tensor(ids,  dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, mask = self.samples[idx]
        return ids[:-1], ids[1:], mask[1:]


def collate_fn(batch):
    xs, ys, masks = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xs_p  = torch.zeros(len(xs),    max_len, dtype=torch.long)
    ys_p  = torch.zeros(len(ys),    max_len, dtype=torch.long)
    ms_p  = torch.zeros(len(masks), max_len, dtype=torch.float32)
    for i, (x, y, m) in enumerate(zip(xs, ys, masks)):
        xs_p[i, :x.size(0)] = x
        ys_p[i, :y.size(0)] = y
        ms_p[i, :m.size(0)] = m
    return xs_p, ys_p, ms_p


# =============================================================================
# 모델 로드 + vocab 확장
# =============================================================================

def load_and_expand(ckpt_path: str, new_vocab_size: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd   = ckpt['model'] if 'model' in ckpt else ckpt
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    cfg = ckpt.get('config', {})
    vocab_size  = cfg.get('vocab_size',  32000)
    d_model     = cfg.get('d_model',     768)
    n_heads     = cfg.get('n_heads',     12)
    n_layers    = cfg.get('n_layers',    12)
    max_seq_len = cfg.get('max_seq_len', 1024)

    model = GPT2(vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.0)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys (무시 가능): {missing[:3]}")

    # vocab 확장: 32000 → 32002
    if vocab_size < new_vocab_size:
        n_new   = new_vocab_size - vocab_size
        old_w   = model.tok_emb.weight.data.clone()           # (V_old, D)
        new_rows = torch.randn(n_new, d_model) * 0.02
        new_w    = torch.cat([old_w, new_rows], dim=0)        # (V_new, D)

        model.tok_emb = nn.Embedding(new_vocab_size, d_model)
        model.tok_emb.weight.data = new_w
        model.head = nn.Linear(d_model, new_vocab_size, bias=False)
        model.head.weight = model.tok_emb.weight               # re-tie
        print(f"  vocab 확장: {vocab_size} → {new_vocab_size}")

    return model.to(device), cfg


# =============================================================================
# SFT 학습
# =============================================================================

def train_sft(model, dataset, device, lr, epochs, batch_size):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    optim  = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"{'='*55}")
    print(f"SFT  lr={lr}  epochs={epochs}  batch={batch_size}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_steps = 0.0, 0

        for step, (x, y, mask) in enumerate(loader, 1):
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits, _ = model(x)                           # (B, T, V)
                B, T, V   = logits.shape
                loss_all  = nn.functional.cross_entropy(
                    logits.reshape(B * T, V),
                    y.reshape(B * T),
                    reduction='none',
                )
                loss = (loss_all * mask.reshape(B * T)).sum() / mask.sum().clamp(min=1)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            epoch_loss  += loss.item()
            epoch_steps += 1

            if step % 20 == 0 or step == len(loader):
                ppl = math.exp(min(loss.item(), 20))
                print(f"  epoch {epoch} | step {step:4d}/{len(loader)} | "
                      f"loss {loss.item():.4f} | ppl {ppl:.2f}")

        avg = epoch_loss / max(epoch_steps, 1)
        print(f"\nepoch {epoch} 완료 | avg loss {avg:.4f} | "
              f"ppl {math.exp(min(avg, 20)):.2f}\n")


# =============================================================================
# PyTorch 가중치 → numpy 엔진 JSON 저장
# =============================================================================

def save_as_numpy_json(pt_model, cfg, vocab_size, out_path):
    """
    PyTorch state_dict → engine/transformer.py GPT → JSON.
    eval.py / compare.py가 로드하는 포맷과 동일.
    """
    from engine.transformer import GPT

    d_model     = cfg.get('d_model',     768)
    n_heads     = cfg.get('n_heads',     12)
    n_layers    = cfg.get('n_layers',    12)
    max_seq_len = cfg.get('max_seq_len', 1024)

    np_model = GPT(
        vocab_size  = vocab_size,
        embed_dim   = d_model,
        num_heads   = n_heads,
        num_layers  = n_layers,
        max_seq_len = max_seq_len,
        dropout     = 0.0,
    )

    pt_model.cpu().eval()
    sd = {k: v.float().numpy() for k, v in pt_model.state_dict().items()}

    # ── 임베딩 ──────────────────────────────────────────────────────────────────
    np_model.token_emb.weight.data = sd['tok_emb.weight']
    np_model.pos_emb.weight.data   = sd['pos_emb.weight']
    np_model.ln_f.gamma.data       = sd['ln_f.weight']
    np_model.ln_f.beta.data        = sd['ln_f.bias']
    np_model.head.weight.data      = sd['tok_emb.weight']   # weight tying

    # ── 블록 ────────────────────────────────────────────────────────────────────
    E = d_model
    for i, block in enumerate(np_model.blocks):
        p = f'blocks.{i}'

        block.ln1.gamma.data = sd[f'{p}.ln1.weight']
        block.ln1.beta.data  = sd[f'{p}.ln1.bias']

        # qkv (3E, E) → q / k / v 분리 (bias 무시 — numpy engine은 bias=False)
        qkv_w = sd[f'{p}.attn.qkv.weight']
        block.attn.q_proj.weight.data = qkv_w[0*E : 1*E]
        block.attn.k_proj.weight.data = qkv_w[1*E : 2*E]
        block.attn.v_proj.weight.data = qkv_w[2*E : 3*E]

        block.attn.out_proj.weight.data = sd[f'{p}.attn.proj.weight']
        block.attn.out_proj.bias.data   = sd[f'{p}.attn.proj.bias']

        block.ln2.gamma.data = sd[f'{p}.ln2.weight']
        block.ln2.beta.data  = sd[f'{p}.ln2.bias']

        block.ff.net.layers[0].weight.data = sd[f'{p}.ffn.fc1.weight']
        block.ff.net.layers[0].bias.data   = sd[f'{p}.ffn.fc1.bias']
        block.ff.net.layers[2].weight.data = sd[f'{p}.ffn.fc2.weight']
        block.ff.net.layers[2].bias.data   = sd[f'{p}.ffn.fc2.bias']

    # ── 저장 ────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    weights    = {str(i): p.data.tolist() for i, p in enumerate(np_model.parameters())}
    config_out = {
        'vocab_size':  vocab_size,
        'embed_dim':   d_model,
        'num_heads':   n_heads,
        'num_layers':  n_layers,
        'max_seq_len': max_seq_len,
        'dropout':     0.0,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'config': config_out, 'weights': weights}, f)
    print(f"저장 완료: {out_path}")
    print(f"  파라미터 수: {sum(p.data.size for p in np_model.parameters()):,}")


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPU SFT: .pt → finetuned_gpu.json")
    parser.add_argument('--ckpt',    required=True, help='.pt 체크포인트 경로')
    parser.add_argument('--tok',     required=True, help='tokenizer.model 경로')
    parser.add_argument('--sft',     required=True, help='SFT 데이터 JSON 경로')
    parser.add_argument('--out',     default='checkpoints/finetuned_gpu.json')
    parser.add_argument('--lr',      type=float, default=2e-5)
    parser.add_argument('--epochs',  type=int,   default=3)
    parser.add_argument('--batch',   type=int,   default=4)
    parser.add_argument('--seq-len', type=int,   default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 토크나이저
    tok = SPChatTokenizer(args.tok)
    print(f"vocab  : {tok.vocab_size}  (user={tok.user_id}, asst={tok.asst_id})\n")

    # 모델
    print("체크포인트 로드 중...")
    model, cfg = load_and_expand(args.ckpt, tok.vocab_size, device)

    # SFT 데이터
    with open(args.sft, encoding='utf-8') as f:
        raw = f.read().strip()
    data = json.loads(raw) if raw.startswith('[') else \
           [json.loads(line) for line in raw.splitlines() if line.strip()]
    dataset = SFTDataset(data, tok, seq_len=args.seq_len)
    print(f"SFT 데이터: {len(dataset)}쌍\n")

    # 학습
    train_sft(model, dataset, device, lr=args.lr,
              epochs=args.epochs, batch_size=args.batch)

    # JSON 저장
    print(f"{'='*55}")
    print("numpy 엔진 JSON 변환 중...")
    save_as_numpy_json(model, cfg, tok.vocab_size, args.out)
    print(f"\n완료. eval.py에서 --after {args.out} 로 평가 가능.")


if __name__ == '__main__':
    main()
