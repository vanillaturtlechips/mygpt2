# DevMind — AI 개발자 플랫폼 설계

> 4개 포트폴리오 프로젝트를 하나의 플랫폼으로 통합하는 설계 문서  
> 철학: 당근마켓식 플랫폼 엔지니어링 — 내부 개발자를 고객으로, engine/을 플랫폼으로

---

## 1. 핵심 철학

### 당근마켓 플랫폼 엔지니어링 → DevMind 적용

| 당근마켓 원칙 | DevMind 적용 |
|-------------|-------------|
| Platform as a Product | `engine/`이 내부 AI 플랫폼. 4개 프로젝트는 플랫폼 소비자 |
| Golden Path | `ToolRegistry`(engine/tools.py)가 표준 에이전트 개발 경로 |
| Self-service | 새 에이전트는 engine API만 호출 — 인프라 몰라도 됨 |
| Paved Road + Escape Hatch | 기본은 engine 표준, 필요하면 직접 확장 가능 |
| Observability First | 모든 에이전트 호출 → engine/memory.py에 trace 저장 |

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   DevMind Platform                       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              engine/ (내부 플랫폼 레이어)           │   │
│  │                                                  │   │
│  │  server.py    memory.py    agent.py    tools.py  │   │
│  │  (추론서버)    (벡터/그래프)  (ReAct루프)  (도구등록) │   │
│  │                                                  │   │
│  │  alignment.py           protocol.py             │   │
│  │  (SFT/DPO/CAI)          (ANP/A2A/x402)          │   │
│  └──────────────────────────────────────────────────┘   │
│          ↑              ↑              ↑              ↑  │
│  ┌───────┴──┐  ┌────────┴─┐  ┌────────┴─┐  ┌───────┴─┐ │
│  │프로젝트①  │  │프로젝트② │  │프로젝트③ │  │프로젝트④│ │
│  │파인튜닝   │  │RAG-코드  │  │RAG-챗봇  │  │Issue→PR │ │
│  │파이프라인 │  │베이스    │  │(배포)    │  │에이전트  │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 프로젝트 간 데이터 흐름

```
[한국어 GPT-2 124M 가중치]
         │
         ▼
  ① 파인튜닝 파이프라인
    SFT → DPO → Constitutional AI
    산출물: instruction-following 한국어 모델
         │
         ▼ (모델 가중치 → engine/server.py에 로드)
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
  ② RAG over 코드베이스               ③ 프로덕션 RAG 챗봇
    GraphRAG 인덱스 구축                  CRAG + Prompt Caching
    AST 파싱 + HyDE + Reranker            GraphRAG 검색 레이어
    산출물: 코드 지식 그래프               산출물: 라이브 챗봇 URL
         │
         ▼ (GraphRAG 인덱스 재활용)
  ④ GitHub Issue-to-PR 에이전트
    SWE-agent + Reflexion 자기검토
    산출물: 실제 동작하는 GitHub Action
```

---

## 4. engine/ 레이어 상세

### 각 모듈의 플랫폼 역할

| 파일 | 역할 | 소비 프로젝트 |
|------|------|-------------|
| `server.py` | KV-cache 추론 서버. 모든 모델 호출의 단일 진입점 | ① ② ③ ④ |
| `memory.py` | 벡터 DB + GraphRAG 인덱스 관리 | ② ③ ④ |
| `agent.py` | ReAct 루프. 도구 선택 → 실행 → 관찰 사이클 | ① ③ ④ |
| `tools.py` | ToolRegistry. 에이전트가 쓸 수 있는 도구 카탈로그 | ④ |
| `alignment.py` | SFT/DPO/Constitutional AI 학습 파이프라인 | ① |
| `protocol.py` | ANP(신원) + A2A(통신) + x402(결제) 에이전트 프로토콜 | ① ④ |

### 플랫폼 API 계약 (인터페이스)

```python
# 모든 프로젝트가 이 인터페이스만 알면 됨

# 추론
response = engine.server.generate(prompt, max_tokens=512)

# 검색
results = engine.memory.search(query, top_k=5, mode="hybrid")  # BM25 + 벡터

# 에이전트 실행
output = engine.agent.run(task, tools=["read_file", "write_file", "search"])

# 에이전트 간 통신
engine.protocol.a2a_call(target_agent="evaluator", payload={"model": model_path})
```

---

## 5. 프로젝트별 통합 포인트

### ① 파인튜닝 파이프라인 → 플랫폼 "모델 공급자"

```
역할: engine/server.py에 올릴 모델을 생산하는 파이프라인

[데이터 수집 에이전트] ──A2A──> [학습 에이전트] ──A2A──> [평가 에이전트]
  ANP 신원 + x402 결제             SFT → DPO               CAI 검증
        ↓
  engine/server.py에 모델 가중치 등록
        ↓
  나머지 ② ③ ④ 프로젝트가 즉시 사용 가능
```

**플랫폼 기여:** 학습된 모델이 플랫폼 공유 자산이 됨

---

### ② RAG over 코드베이스 → 플랫폼 "지식 레이어"

```
역할: engine/memory.py에 코드 지식 그래프를 구축

AST 파서 → 함수/클래스 단위 청킹
        ↓
GraphRAG 인덱스 → engine/memory.py 저장
        ↓
HyDE + Hybrid Search + Reranker → 검색 품질 향상
        ↓
③ RAG 챗봇, ④ Issue-to-PR이 동일 인덱스를 재사용
```

**플랫폼 기여:** 한 번 인덱싱 → 모든 소비자가 셀프서비스 검색

---

### ③ 프로덕션 RAG 챗봇 → 플랫폼 "Golden Path 데모"

```
역할: 플랫폼을 가장 잘 보여주는 레퍼런스 구현 + 실제 배포

사용자 질문
  ↓
engine/memory.py (GraphRAG 검색)
  ↓
CRAG 신뢰도 평가 → 낮으면 웹 검색 보완
  ↓
engine/server.py (Prompt Caching으로 비용 70% 절감)
  ↓
HuggingFace Spaces or Vercel 배포 → 라이브 URL
```

**플랫폼 기여:** "이렇게 만들면 됩니다" 표준 예제 + 외부 접근 가능한 데모

---

### ④ GitHub Issue-to-PR 에이전트 → 플랫폼 "자동화 소비자"

```
역할: engine/ 전체를 조합해 실제 가치를 만드는 최종 소비자

GitHub Issue 등록
  ↓
engine/memory.py (GraphRAG로 관련 코드 탐색)
  ↓
engine/agent.py (SWE-agent 패턴 — 파일 읽기/수정 도구)
  ↓
engine/protocol.py (A2A로 검토 에이전트에 위임)
  ↓
Reflexion: "이 PR이 Issue를 해결하는가?" 자기검토
  ↓ (통과)
GitHub PR 자동 생성
```

**플랫폼 기여:** 플랫폼이 실제 개발 워크플로우를 자동화함을 증명

---

## 6. 배포 전략

### 당근마켓식 독립 배포 원칙

```
각 프로젝트는 독립적으로 배포 가능
공유 상태는 engine/ API를 통해서만 접근

┌─────────────────────────────────────┐
│  인프라                              │
│  ├── HuggingFace Spaces (챗봇 UI)   │
│  ├── GitHub Actions (Issue-to-PR)   │
│  ├── Vercel (API gateway, optional) │
│  └── RunPod/Lambda (학습 전용)       │
└─────────────────────────────────────┘
```

### 단계별 배포

```
Step 1: engine/ 로컬 테스트 완료
Step 2: ① 파인튜닝 → HuggingFace 모델 업로드
Step 3: ② RAG 인덱스 → 로컬 ChromaDB or Qdrant
Step 4: ③ 챗봇 → HuggingFace Spaces 배포 (라이브 URL 확보)
Step 5: ④ GitHub Action → 데모 레포에 설치
```

---

## 7. 포트폴리오 서사

### 인터뷰에서 설명하는 방법

> "저는 4개 프로젝트를 만든 게 아니라, 하나의 AI 플랫폼을 설계했습니다.
>
> 당근마켓처럼 `engine/`을 내부 플랫폼으로 만들고,
> 4개 프로젝트가 플랫폼 소비자로 동작하도록 설계했습니다.
>
> 덕분에 새 기능을 추가할 때 engine API 하나만 바꾸면
> 나머지 프로젝트는 수정 없이 개선 효과를 얻습니다.
> 이게 플랫폼 사고방식입니다."

### GitHub 레포 구조 (외부에서 보이는 것)

```
github.com/[username]/devmind
├── engine/              ← 플랫폼 코어 (별도 레포 or 공유 패키지)
├── projects/
│   ├── 01-finetune/     ← 파인튜닝 파이프라인
│   ├── 02-rag-code/     ← RAG over 코드베이스
│   ├── 03-rag-chatbot/  ← 프로덕션 RAG 챗봇
│   └── 04-issue-to-pr/  ← Issue-to-PR 에이전트
└── README.md            ← 플랫폼 전체 설명 + 아키텍처 다이어그램
```

### 증거 체인 (포트폴리오 링크 목록)

| 단계 | 증거 | URL |
|------|------|-----|
| 모델 학습 | HuggingFace 모델 허브 | huggingface.co/[username]/korean-gpt2-124m |
| 파인튜닝 | 학습 그래프 + 모델 카드 | HuggingFace 모델 카드 |
| RAG 코드베이스 | 데모 GIF + 성능 비교 | GitHub README |
| RAG 챗봇 | 실제 접속 가능한 URL | HuggingFace Spaces |
| Issue-to-PR | 실제 PR 링크 | GitHub PR (데모 레포) |

---

## 8. 구현 우선순위 (Quick Win 먼저)

```
Week 1: engine/ 정리
  - server.py inference API 확정
  - memory.py ChromaDB 연결 확인
  - agent.py ReAct 루프 테스트

Week 2-3: ① 파인튜닝 (학습된 모델 바로 활용)
  - alignment.py SFT 구현
  - A2A 에이전트 파이프라인 연결
  - HuggingFace 업로드

Week 4-5: ② RAG over 코드베이스 (가장 임팩트)
  - AST 파서 + GraphRAG 인덱서
  - HyDE + Hybrid Search 구현
  - 데모 GIF 제작

Week 6: ③ RAG 챗봇 (②재활용, 배포만)
  - CRAG 신뢰도 평가 추가
  - Prompt Caching 설정
  - HuggingFace Spaces 배포

Week 7-8: ④ Issue-to-PR (최종 차별화)
  - SWE-agent 도구 구현
  - Reflexion 루프 추가
  - GitHub Action 설치
```

---

## 9. 기술 차별화 요약

```
일반 포트폴리오:       "RAG 챗봇 만들었습니다"
DevMind 포트폴리오:    "AI 플랫폼을 설계하고,
                       그 위에 4개 프로덕션 서비스를 올렸습니다.
                       플랫폼 레이어 덕에 각 서비스는
                       AI 로직에만 집중할 수 있었습니다."

차별화 포인트:
├── 직접 사전학습 (GPT-2 124M, 2.1B 토큰)
├── 플랫폼 사고방식 (engine/ = 내부 플랫폼)
├── GraphRAG (일반 RAG 대비 관계 기반 검색)
├── A2A/ANP/x402 (에이전트 간 자율 협업)
└── 실제 배포 URL + PR 링크 (말이 아닌 증거)
```
