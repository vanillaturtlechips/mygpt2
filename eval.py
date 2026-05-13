"""
LLM-as-judge 평가 스크립트.

compare.py의 before/after 생성 결과를 Claude Haiku로 채점한다.

사용법:
    python eval.py \
        --before checkpoints/ckpt_step200000.pt \
        --after  checkpoints/finetuned.json \
        --tok    tokenizer.model \
        --api-key sk-ant-...

환경변수로도 가능:
    set ANTHROPIC_API_KEY=sk-ant-...
    python eval.py --before ... --after ... --tok ...
"""

import argparse
import json
import os
import re
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

JUDGE_SYSTEM = """당신은 AI/ML 교육 답변을 평가하는 전문 평가자입니다.
주어진 질문과 두 모델의 답변을 보고 각각을 1~5점으로 채점하세요.

채점 기준:
5점: 정확하고 핵심을 잘 설명, 한국어 자연스러움
4점: 대체로 정확하나 일부 누락
3점: 부분적으로 맞지만 불완전하거나 어색함
2점: 거의 관련 없거나 반복적
1점: 완전히 틀리거나 무의미한 반복

반드시 아래 JSON 형식으로만 응답하세요:
{"before": <1-5>, "after": <1-5>, "reason": "<한 줄 이유>"}"""

JUDGE_USER = """질문: {question}

[모델 A 답변 (BEFORE)]:
{before}

[모델 B 답변 (AFTER)]:
{after}

위 두 답변을 채점하세요."""


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


def generate(model, tok, prompt, max_new_tokens=120, temperature=0.7,
             top_k=50, top_p=0.9, repetition_penalty=1.3):
    ids = [tok.user_id] + tok.encode(prompt) + [tok.asst_id]
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k,
                         top_p=top_p, repetition_penalty=repetition_penalty)
    return tok.decode(out[len(ids):]).strip()


def repetition_rate(text: str, n: int = 4) -> float:
    """n-gram 반복률 (높을수록 반복이 많음)."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def judge(client, question: str, before: str, after: str) -> dict:
    """Claude Haiku로 before/after 채점."""
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": JUDGE_USER.format(
                question=question, before=before, after=after
            )
        }]
    )
    text = resp.content[0].text.strip()
    # JSON 파싱
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return {"before": 0, "after": 0, "reason": f"파싱 실패: {text}"}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"before": 0, "after": 0, "reason": f"JSON 오류: {text}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before',  required=True)
    parser.add_argument('--after',   required=True)
    parser.add_argument('--tok',     required=True)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--tokens',  type=int,   default=120)
    parser.add_argument('--temp',    type=float, default=0.7)
    parser.add_argument('--seed',    type=int,   default=42)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("--api-key 또는 ANTHROPIC_API_KEY 환경변수가 필요합니다.")

    try:
        import anthropic
    except ImportError:
        raise ImportError("uv add anthropic 을 먼저 실행하세요.")

    client = anthropic.Anthropic(api_key=api_key)

    np.random.seed(args.seed)
    tok = SPChatTokenizer(args.tok)

    print("모델 로드 중...")
    before_model = load_before(args.before, tok)
    after_model  = load_after(args.after,   tok)

    results = []
    print("\n" + "=" * 70)
    print("LLM-as-judge 평가 (Claude Haiku)")
    print("=" * 70)

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] Q: {prompt}")

        b = generate(before_model, tok, prompt, args.tokens, args.temp)
        a = generate(after_model,  tok, prompt, args.tokens, args.temp)

        rep_b = repetition_rate(b)
        rep_a = repetition_rate(a)

        print(f"  BEFORE ({len(b)}자, 반복률 {rep_b:.0%}): {b[:80]}{'...' if len(b)>80 else ''}")
        print(f"  AFTER  ({len(a)}자, 반복률 {rep_a:.0%}): {a[:80]}{'...' if len(a)>80 else ''}")

        verdict = judge(client, prompt, b, a)
        print(f"  판정 → BEFORE {verdict['before']}점 / AFTER {verdict['after']}점")
        print(f"  이유: {verdict['reason']}")

        results.append({
            "prompt": prompt,
            "before": b, "after": a,
            "rep_before": rep_b, "rep_after": rep_a,
            "score_before": verdict["before"],
            "score_after":  verdict["after"],
            "reason": verdict["reason"],
        })

    # ── 최종 요약 ──────────────────────────────────────────────────────────────
    valid = [r for r in results if r["score_before"] > 0]
    if valid:
        avg_b = sum(r["score_before"] for r in valid) / len(valid)
        avg_a = sum(r["score_after"]  for r in valid) / len(valid)
        wins  = sum(1 for r in valid if r["score_after"] > r["score_before"])
        ties  = sum(1 for r in valid if r["score_after"] == r["score_before"])
        losses= sum(1 for r in valid if r["score_after"] < r["score_before"])

        print("\n" + "=" * 70)
        print("최종 결과")
        print("=" * 70)
        print(f"  BEFORE 평균: {avg_b:.2f}점")
        print(f"  AFTER  평균: {avg_a:.2f}점  (Δ {avg_a-avg_b:+.2f})")
        print(f"  Win/Tie/Loss: {wins}/{ties}/{losses}  "
              f"(Win rate {wins/len(valid):.0%})")
        rep_b_avg = sum(r["rep_before"] for r in results) / len(results)
        rep_a_avg = sum(r["rep_after"]  for r in results) / len(results)
        print(f"  반복률 BEFORE {rep_b_avg:.0%} → AFTER {rep_a_avg:.0%}")

    # JSON 저장
    out_path = "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == '__main__':
    main()
