"""
Fine-tuning pipeline: LoRA-SFT → DPO → merge → save.

사용법:
    python finetune.py \\
        --ckpt  checkpoints/ckpt_step5486.pt \\
        --tok   data/tokenizer.json \\
        --sft   data/sft.jsonl \\
        [--dpo  data/dpo.jsonl] \\
        --out   checkpoints/finetuned.json

데이터 형식:
    sft.jsonl  : {"user": "질문", "assistant": "답변"}
    dpo.jsonl  : {"prompt": "질문", "chosen": "좋은 답변", "rejected": "나쁜 답변"}
"""

import argparse
import json
import os
import numpy as np

from engine.load_checkpoint import load_from_pytorch_ckpt, expand_vocab
from engine.transformer import GPT
from engine.alignment import (
    ChatTokenizer,
    SFTDataset, SFTTrainer,
    PreferenceDataset, DPOTrainer,
    clone_gpt,
)
from engine.lora import inject_lora, merge_lora, lora_parameters


# =============================================================================
# HELPERS
# =============================================================================

def _load_jsonl(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        data = f.read().strip()
    # JSON 배열([...]) 과 JSONL 둘 다 지원
    if data.startswith('['):
        return json.loads(data)
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def _load_model_json(path: str) -> GPT:
    """JSON 형식(SFTTrainer.save 출력)에서 GPT 로드."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    cfg = data['config']
    model = GPT(**cfg)
    for i, p in enumerate(model.parameters()):
        p.data = np.array(data['weights'][str(i)], dtype=np.float32)
    print(f"JSON 체크포인트 로드: {path}")
    print(f"  파라미터: {model.num_params():,}")
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LoRA SFT→DPO fine-tuning pipeline")
    parser.add_argument('--ckpt',       required=True,
                        help='.pt (PyTorch) 또는 .json (이전 SFTTrainer.save 출력)')
    parser.add_argument('--tok',        required=True,
                        help='ChatTokenizer JSON 경로')
    parser.add_argument('--sft',        required=True,
                        help='SFT 데이터 JSONL ({user, assistant})')
    parser.add_argument('--dpo',        default=None,
                        help='DPO 데이터 JSONL ({prompt, chosen, rejected}) — 생략 시 SFT만')
    parser.add_argument('--out',        default='checkpoints/finetuned.json',
                        help='저장 경로')
    parser.add_argument('--lora-rank',  type=int,   default=8)
    parser.add_argument('--lora-alpha', type=float, default=16.0)
    parser.add_argument('--sft-lr',     type=float, default=2e-4)
    parser.add_argument('--sft-epochs', type=int,   default=3)
    parser.add_argument('--dpo-lr',     type=float, default=1e-5)
    parser.add_argument('--dpo-epochs', type=int,   default=2)
    parser.add_argument('--dpo-beta',   type=float, default=0.1)
    parser.add_argument('--seq-len',    type=int,   default=512)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── 1. 토크나이저 ────────────────────────────────────────────────────────
    # .model → SentencePiece(사전학습용), .json → ChatTokenizer(커스텀 BPE)
    if args.tok.endswith('.model'):
        from engine.sp_tokenizer import SPChatTokenizer
        tok = SPChatTokenizer(args.tok)
    else:
        tok = ChatTokenizer()
        tok.load(args.tok)
    print(f"토크나이저 로드: vocab={tok.vocab_size}")

    # ── 2. 체크포인트 ────────────────────────────────────────────────────────
    if args.ckpt.endswith('.pt'):
        model = load_from_pytorch_ckpt(args.ckpt)
    else:
        model = _load_model_json(args.ckpt)

    # SentencePiece 토크나이저 사용 시 vocab 확장 (32000 → 32002)
    if model.vocab_size < tok.vocab_size:
        expand_vocab(model, tok.vocab_size)

    # ── 3. LoRA 주입 ─────────────────────────────────────────────────────────
    inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    n_lora  = sum(p.data.size for p in lora_parameters(model))
    n_total = sum(p.data.size for p in model.parameters())
    print(f"\nLoRA 주입 완료: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  학습 파라미터: {n_total:,}개 중 LoRA {n_lora:,}개 ({100*n_lora/n_total:.1f}%)")

    # ── 4. SFT ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"SFT  lr={args.sft_lr}  epochs={args.sft_epochs}")
    print(f"{'='*55}")
    sft_data    = _load_jsonl(args.sft)
    sft_dataset = SFTDataset(sft_data, tok, seq_len=args.seq_len)
    sft_trainer = SFTTrainer(model, tok, lr=args.sft_lr)
    sft_trainer.train(sft_dataset, epochs=args.sft_epochs)

    # SFT LoRA를 먼저 병합해야 clone_gpt()가 올바른 weight를 복사한다.
    # (LoRA 주입 상태에서 clone 시 parameter 순서/shape가 달라져 불일치 발생)
    merge_lora(model)

    # ── 5. DPO (선택) ────────────────────────────────────────────────────────
    if args.dpo:
        print(f"\n{'='*55}")
        print(f"DPO  lr={args.dpo_lr}  beta={args.dpo_beta}  epochs={args.dpo_epochs}")
        print(f"{'='*55}")
        ref_model = clone_gpt(model)            # SFT 병합 가중치의 frozen 스냅샷
        inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)  # DPO용 재주입
        dpo_data    = _load_jsonl(args.dpo)
        dpo_dataset = PreferenceDataset(dpo_data, tok, seq_len=args.seq_len)
        dpo_trainer = DPOTrainer(
            model, ref_model, tok,
            lr=args.dpo_lr, beta=args.dpo_beta,
        )
        dpo_trainer.train(dpo_dataset, epochs=args.dpo_epochs)
        merge_lora(model)                       # DPO LoRA 최종 병합
    else:
        print("\nDPO 데이터 없음 — SFT 단계만 적용")

    # ── 6. 저장 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("저장")
    print(f"{'='*55}")
    print(f"LoRA 병합 완료: 추론 오버헤드 없음")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    sft_trainer.save(args.out)
    print(f"저장 완료: {args.out}")


if __name__ == '__main__':
    main()
