# Sovereign AI — From Scratch

외부 API 없이 직접 만든 AI 에이전트 스택.
numpy 기반 자동미분부터 CUDA 커널, 멀티 에이전트 프로토콜까지 전 레이어를 구현한다.

## 구조

```
engine/
  autograd.py         — Tensor, 역전파, log-sum-exp
  tokenizer.py        — BPE 토크나이저
  transformer.py      — GPT (Self-Attention, MHA, FFN, LayerNorm)
  trainer.py          — 사전학습 루프 (cosine LR, grad clip)
  alignment.py        — SFT + DPO (정렬 파이프라인)
  quantize.py         — INT8 / INT4 양자화 (4~5x 압축)
  server.py           — KV-cache 추론 서버 (3.5x 속도향상)
  memory.py           — 3계층 메모리 (Core / Recall / Archive)
  tools.py            — ToolRegistry + 티켓팅 도구
  agent.py            — ReAct 에이전트 루프
  protocol.py         — ANP (DID) + A2A (HTTP) + x402 (마이크로페이먼트)
  ticketing_agent.py  — 전체 스택 통합
  cuda_backend.py     — CUDA 커널 Python 바인딩 (ctypes)

cuda/
  01_vector_add.cu    — Thread / Block 개념
  02_matmul.cu        — 타일 행렬곱 (공유 메모리, 28x 속도향상)
  03_softmax.cu       — 병렬 리덕션 Softmax
  04_layernorm.cu     — LayerNorm
  05_gelu.cu          — GELU 활성화
  06_attention.cu     — Scaled Dot-Product Attention (인과 마스크 포함)
  07_embedding.cu     — Embedding lookup + Bias add + Residual add
  libengine.cu        — 전체 커널 공유 라이브러리 (.so)
```

## 레이어 맵

```
[12] 프로토콜       ANP (신원) ↔ A2A (통신) ↔ x402 (결제)
[11] 툴 레지스트리  search_ktx / book_concert / calculator ...
[10] 에이전트 루프  Thought → Action → Observation (ReAct)
 [9] 메모리         Core | Recall | Archive
 [8] 추론 서버      KV-cache, 스트리밍 생성
 [7] 양자화         FP32 → INT8 / INT4
 [6] 정렬           SFT (마스킹 손실) → DPO (레퍼런스 모델)
 [5] 사전학습       raw text → GPT 학습 루프 → 가중치
 [4] 트랜스포머     Self-Attention / MHA / FFN
 [3] 토크나이저     "안녕" → [token_ids]
 [2] 딥러닝 프레임  Module / Optimizer / DataLoader
 [1] 자동미분       행렬 연산 / 역전파 / 기울기
```

## 실행

```bash
# 각 모듈 스모크 테스트
python3 -m engine.autograd
python3 -m engine.trainer
python3 -m engine.alignment
python3 -m engine.quantize
python3 -m engine.server
python3 -m engine.memory
python3 -m engine.agent
python3 -m engine.protocol
python3 -m engine.ticketing_agent

# CUDA 커널 빌드 및 테스트
cd cuda
nvcc -shared -Xcompiler -fPIC -o libengine.so libengine.cu -lm
python3 -m engine.cuda_backend
```

## 다음 단계

- 한국어 GPT-2 (124M) 학습 → `A100_TRAINING_PLAN.md` 참고
- `engine/server.py` numpy → `cuda_backend.py` 교체
- Ollama 제거 → 직접 학습한 가중치로 완전한 Sovereign AI 구현
