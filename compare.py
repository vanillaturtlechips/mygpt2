"""
파인튜닝 전/후 생성 비교 스크립트.

사용법:
    python compare.py \
        --before checkpoints/ckpt_step200000.pt \
        --after  checkpoints/finetuned.json \
        --tok    tokenizer.model
"""

import argparse
import json
import numpy as np

from engine.load_checkpoint import load_from_pytorch_ckpt, expand_vocab
from engine.sp_tokenizer import SPChatTokenizer
from engine.transformer import GPT


PROMPTS = [
    "SFT가 뭐야?",
    "트랜스포머 어텐션 메커니즘을 설명해줘.",
    "LoRA가 일반 파인튜닝과 다른 점이 뭐야?",
    "RAG란 무엇인가?",
    "DPO와 RLHF의 차이점을 설명해줘.",
]


def load_before(ckpt_path, tok):
    model = load_from_pytorch_ckpt(ckpt_path)
    expand_vocab(model, tok.vocab_size)
    model.eval()
    return model


def load_after(ckpt_path, tok):
    with open(ckpt_path, encoding='utf-8') as f:
        data = json.load(f)
    cfg = data['config']
    model = GPT(**cfg)
    for i, p in enumerate(model.parameters()):
        p.data = np.array(data['weights'][str(i)], dtype=np.float32)
    model.eval()
    return model


def generate(model, tok, prompt, max_new_tokens=100, temperature=0.7, top_k=50,
             top_p=0.9, repetition_penalty=1.3):
    ids = [tok.user_id] + tok.encode(prompt) + [tok.asst_id]
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k,
                         top_p=top_p, repetition_penalty=repetition_penalty)
    return tok.decode(out[len(ids):])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True, help='사전학습 체크포인트 (.pt)')
    parser.add_argument('--after',  required=True, help='파인튜닝 결과 (.json)')
    parser.add_argument('--tok',    required=True, help='tokenizer.model')
    parser.add_argument('--tokens', type=int, default=100)
    parser.add_argument('--temp',   type=float, default=0.7)
    parser.add_argument('--topk',   type=int, default=50)
    args = parser.parse_args()

    np.random.seed(42)

    tok = SPChatTokenizer(args.tok)

    print("모델 로드 중...")
    before = load_before(args.before, tok)
    after  = load_after(args.after,   tok)

    print("\n" + "=" * 70)
    print("BEFORE (사전학습)  vs  AFTER (파인튜닝)")
    print("=" * 70)

    for prompt in PROMPTS:
        b = generate(before, tok, prompt, args.tokens, args.temp, args.topk)
        a = generate(after,  tok, prompt, args.tokens, args.temp, args.topk)

        print(f"\n Q: {prompt}")
        print(f"[BEFORE] {b.strip()}")
        print(f"[AFTER]  {a.strip()}")
        print("-" * 70)


if __name__ == '__main__':
    main()
