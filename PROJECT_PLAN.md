# 포트폴리오 프로젝트 플랜

> 한국어 GPT-2 124M 학습 완료 후 순서대로 진행
> 목표 포지션: ML 엔지니어 / AI 엔지니어

---

## 현재 진행중

**한국어 GPT-2 124M 사전학습**
- 스크립트: `train_a100.py`
- 플랫폼: RunPod A100 80GB
- 데이터: 231lee/korean-gpt2-dataset (10GB, ~2.1B 토큰)
- 체크포인트: ckpt_step960 → 200,000 steps 목표
- 예상 완료: 오늘 오후 3시 20분경
- 산출물: HuggingFace 모델 허브 업로드

---

## 프로젝트 ① 파인튜닝 파이프라인

**캡스톤 #7 — 엔드-투-엔드 파인튜닝 파이프라인**

### 파인튜닝 도메인
**AI/ML 개념 설명 (한국어)** — 124M 사이즈에서 현실적으로 가장 효과적인 도메인

- SFT 데이터: 커리큘럼 RAG(8420 청크)에서 자동 생성한 AI/ML Q&A 50~100 pairs
  ```
  {'user': 'DPO가 뭐야?', 'assistant': 'DPO는 보상 모델 없이 선호도 정렬을 하는 방법입니다. ...'}
  ```
- DPO 데이터: `chosen` = 정확하고 자연스러운 설명, `rejected` = 모호하거나 틀린 설명
- Constitutional AI 기준: 정확성 / 자연스러운 한국어 / 핵심 개념 포함 여부

### 핵심 기술
| 기술 | 역할 |
|------|------|
| SFT (Supervised Fine-Tuning) | instruction following 학습 |
| DPO (Direct Preference Optimization) | 선호도 정렬 |
| Constitutional AI | 품질 자동 검증 |
| LoRA | 메모리 효율적 파인튜닝 (engine/에 추가 예정) |
| ANP (신원) ↔ A2A (통신) | 에이전트 간 자율 운영 |

### 에이전트 구조
```
[데이터 생성 에이전트]
    커리큘럼 RAG 검색 → SFT/DPO 데이터 자동 생성
    ANP 신원 증명
    ↕ A2A
[학습 에이전트]
    load_checkpoint → LoRA 주입 → SFT → DPO 순차 학습
    ↕ A2A
[평가 에이전트]
    Constitutional AI 품질 검증
    perplexity / 생성 샘플 비교 (before SFT vs after DPO)
```

### 남은 구현 과제
1. `data/sft_data.json` — 커리큘럼 RAG로 자동 생성
2. `data/dpo_data.json` — preferred/rejected 쌍
3. `engine/lora.py` — LoRA 주입 (Phase 11-08 기반)
4. `engine/eval.py` — perplexity + 생성 샘플 비교
5. `finetune.py` — 전체 파이프라인 진입점

### 산출물
- HuggingFace 모델 카드 (학습 그래프 포함)
- AI/ML 개념을 한국어로 설명하는 instruction-following 모델
- A2A 에이전트 파이프라인 데모

### 활용 파일
- `engine/alignment.py` (SFT/DPO 구현 완료)
- `engine/load_checkpoint.py` (체크포인트 → engine 변환 완료)
- `engine/protocol.py` (ANP/A2A/x402 구현)

---

## 프로젝트 ② RAG over 코드베이스

**캡스톤 #2 — RAG over 코드베이스**

### 핵심 기술
| 기술 | 역할 | 효과 |
|------|------|------|
| GraphRAG | 코드 의존성 그래프 탐색 | 관계 기반 검색 |
| AST 파싱 | 함수/클래스 단위 청킹 | 코드 구조 보존 |
| HyDE | 가상 답변으로 벡터 검색 | 검색 정확도 +30% |
| Hybrid Search | BM25 + 벡터 결합 | 검색 커버리지 향상 |
| Cross-encoder Reranker | 검색 결과 재정렬 | 최종 품질 향상 |

### 동작 예시
```
질문: "train 함수가 의존하는 모든 것은?"

GraphRAG 탐색:
[train()] --호출--> [DataLoader()]
          --호출--> [optimizer.step()]
          --사용--> [CONFIG 딕셔너리]
          --저장--> [save_checkpoint()]

→ 관련 파일 + 라인 번호 + 의존성 그래프 반환
```

### 산출물
- GitHub 레포 + 데모 GIF
- 코드베이스 자연어 질의 시스템
- 일반 RAG vs GraphRAG 성능 비교

### 활용 파일
- `engine/memory.py` (벡터 저장)
- `phases/19-capstone-projects/02-rag-over-codebase/code/main.py`

---

## 프로젝트 ③ 프로덕션 RAG 챗봇

**캡스톤 #8 — 프로덕션 RAG 챗봇**

### 핵심 기술
| 기술 | 역할 | 효과 |
|------|------|------|
| GraphRAG | 관계 기반 개인화 검색 | 맥락 파악 |
| CRAG | 검색 결과 신뢰도 평가 | 할루시네이션 감소 |
| Hybrid Search | BM25 + 벡터 결합 | 검색 커버리지 |
| Prompt Caching | 공통 prefix 캐싱 | 비용 70% 절감 |

### CRAG 동작
```
검색 결과 → "이 결과로 답할 수 있나?" 평가
    → 신뢰도 낮음 → 웹 검색 보완
    → 신뢰도 높음 → 바로 답변 생성
```

### 산출물
- 라이브 URL (HuggingFace Spaces or Vercel)
- 실제 접속 가능한 챗봇 데모
- 비용 최적화 리포트

### 활용 파일
- `engine/server.py` (KV-cache 추론 서버)
- `engine/memory.py` (RAG 메모리)
- `phases/19-capstone-projects/08-production-rag-chatbot/code/main.py`

---

## 프로젝트 ④ Study Log → 블로그 자동 PR 에이전트

**커스텀 — 공부한 내용을 GitHub 블로그 포스트로 자동 PR**

> "오늘 DPO 공부했어" 한 마디로 `vanillaturtlechips.github.io`에 PR이 열린다

### 핵심 기술
| 기술 | 역할 | 효과 |
|------|------|------|
| Curriculum RAG | 공부 주제 → 핵심 개념 자동 검색 | 포스트 내용 충실화 |
| ReAct 에이전트 | Thought → Action → Observation 루프 | 포스트 구조화 |
| Reflexion | "이 포스트가 핵심 개념을 커버하는가?" 자기 검토 | 품질 자동 개선 |
| GitHub API | 브랜치 생성 → 커밋 → PR 오픈 | 완전 자동화 |

### 동작 흐름
```
입력: "오늘 DPO 공부했어"
    ↓
[1] Curriculum RAG 검색
    search_curriculum("DPO") → 핵심 개념/코드/수식 청크 수집
    ↓
[2] ReAct 루프로 포스트 초안 생성
    Thought: "도입부 → 수식 → 코드 예시 → 요약 순서로 써야 한다"
    Action: write_section("개요"), write_section("수식"), ...
    ↓
[3] Reflexion 자기 검토
    "Bradley-Terry loss를 언급했는가? 코드 예시가 있는가?"
    → 부족하면 RAG 재검색 후 보완
    ↓
[4] GitHub PR 자동 생성
    브랜치: post/2026-05-11-dpo
    파일: _posts/2026-05-11-dpo.md
    PR 제목: "[Auto] DPO: 보상 모델 없는 선호도 정렬"
    ↓
출력: PR 링크 반환
```

### 포스트 구조 (자동 생성 — Chirpy 테마 형식)
```markdown
---
title: "DPO: 보상 모델 없는 선호도 정렬"
date: 2026-05-11 00:00:00 +0900
categories: [ML, Alignment]
tags: [dpo, rlhf, alignment]
math: true
---

## 핵심 개념
## 왜 DPO인가 (vs RLHF)
## 수식 풀이
## 코드로 보기
## 정리
```

- 파일 경로: `_posts/ml/2026-05-11-dpo.md`
- 브랜치: `post/2026-05-11-dpo`

### 산출물
- 실제 동작하는 에이전트 스크립트 (`blog_agent.py`)
- `vanillaturtlechips.github.io`에 열린 실제 PR 링크
- 에이전트가 생성한 포스트 품질 before/after Reflexion 비교
- 주 1회 자동 실행 가능한 GitHub Action 설정

### 왜 포트폴리오에 강한가
- **직접 사용하는 도구** — 면접에서 "실제로 씁니다"라고 말할 수 있음
- **관찰 가능한 증거** — 실제 PR 링크를 포트폴리오에 첨부
- **RAG + ReAct + Reflexion** 세 기술을 하나의 실용 도구에 통합

### 활용 파일
- `engine/agent.py` (ReAct 루프)
- `engine/tools.py` (ToolRegistry — GitHub 도구 추가)
- `engine/memory.py` (RAG 메모리)
- `curriculum-rag/query.py` (커리큘럼 검색)

---

## 전체 기술 스택 요약

```
공통 기반
├── GraphRAG            ② ③
├── Hybrid Search       ② ③
├── ANP/A2A             ① (선택적으로 ④)
├── Curriculum RAG      ① (데이터 생성) + ④ (포스트 생성)
└── engine/ 스택        전체 공통

차별화 포인트
├── HyDE                ② 코드 검색 정확도
├── CRAG                ③ 할루시네이션 방지
├── Reflexion           ④ 자기 개선 루프
└── Constitutional AI   ① 품질 자동 검증
```

---

## 진행 순서

> **원칙: 각 프로젝트를 독립적으로 완성 → 마지막에 하나의 에코시스템으로 통합**

```
[완료 예정] 한국어 GPT-2 124M 사전학습
     ↓
[①] 파인튜닝 파이프라인 (독립 완성)    AI/ML Q&A 도메인, SFT/DPO, A2A
     ↓
[②] RAG over 코드베이스 (독립 완성)    GraphRAG + HyDE
     ↓
[③] 프로덕션 RAG 챗봇 (독립 완성)     CRAG + 배포
     ↓
[④] 블로그 PR 에이전트 (독립 완성)     ReAct + Reflexion + GitHub API
     ↓
[⑤] 에코시스템 통합                    ①②③④를 A2A로 연결
```

---

## Phase ⑤ 에코시스템 통합

> ①②③④가 각자 독립 동작한 뒤, A2A로 연결해 하나의 시스템으로 만든다

### 통합 구조
```
사용자: "오늘 DPO 공부했어"
    ↓
[④ 블로그 에이전트]
    RAG 검색 → 포스트 초안 → Reflexion → PR 오픈
    ↕ A2A
[③ RAG 챗봇]
    포스트 내용을 챗봇 지식베이스에 자동 인덱싱
    ↕ A2A
[② 코드베이스 RAG]
    포스트에 등장하는 코드(engine/)를 자동 연결
    ↕ A2A
[① 파인튜닝 모델]
    실제 한국어 GPT-2가 예시 답변 생성
    ("이 개념을 모델이 직접 설명해드립니다")
```

### 통합 시나리오
```
입력: "SFT에 대해 검색해줘"
→ ③ 챗봇이 ② 코드베이스 RAG로 관련 코드 탐색
→ ① 파인튜닝 모델이 한국어로 개념 설명 생성
→ ④ 에이전트가 대화 내용을 블로그 포스트로 정리 → PR

입력: "engine/alignment.py의 SFT 코드 설명해줘"
→ ② AST 파싱으로 SFTTrainer 의존성 그래프 탐색
→ ① 모델이 코드 설명 생성
→ ③ 챗봇 응답 + ④ 블로그 자동 저장 (선택)
```

### 에코시스템 산출물
- A2A로 연결된 4개 에이전트가 동시에 동작하는 데모 영상
- `ecosystem/main.py` — 통합 진입점
- "혼자 공부하는 AI 엔지니어의 학습 시스템" 포트폴리오 스토리

---

## 포트폴리오 최종 구성

| 프로젝트 | 증거 | 어필 포인트 |
|---------|------|-----------|
| GPT-2 학습 | HuggingFace 모델 링크 | "직접 학습까지 했네" |
| 파인튜닝 | 학습 그래프 + 모델 카드 | "파이프라인 전체 구현" |
| RAG 코드베이스 | GitHub + 데모 GIF | "GraphRAG까지 아네" |
| RAG 챗봇 | 라이브 URL | "배포도 할 줄 아네" |
| 블로그 PR 에이전트 | 실제 PR 링크 + 자동화 영상 | "직접 쓰는 도구까지 만들었네" |
| 에코시스템 통합 | 4개 에이전트 동작 데모 영상 | "시스템 설계까지 생각하네" |
