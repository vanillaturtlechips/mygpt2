"""
ReAct agent loop.

  AgentLoop   — Thought → Action → Observation cycle
  OllamaEngine — calls a local Ollama model (e.g. qwen2.5:7b)
  MockEngine  — scripted responses for testing

Prompt format the LLM must follow:
  Thought: <reasoning>
  Action: <tool_name>
  Args: <json>
"""

import re
import json
import time
import urllib.request
from typing import Callable, List, Optional, Tuple

from .tools import ToolRegistry
from .memory import MemorySystem


# =============================================================================
# LLM BACKENDS
# =============================================================================

class OllamaEngine:
    """
    Calls a locally running Ollama model.
    Default: qwen2.5:7b — this is the real LLM for the ticketing agent.

    Start Ollama first:
      ollama pull qwen2.5:7b
      ollama serve
    """

    def __init__(self, model: str = 'gemma3:4b',
                 host: str = 'localhost', port: int = 11434):
        self.url   = f'http://{host}:{port}/api/generate'
        self.model = model

    def __call__(self, prompt: str) -> str:
        body = json.dumps({
            'model':  self.model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.2, 'top_k': 20},
        }).encode()
        req  = urllib.request.Request(
            self.url, data=body,
            headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req, timeout=60)
        return json.loads(resp.read())['response']

    def available(self) -> bool:
        try:
            urllib.request.urlopen(
                f'http://{self.url.split("/api")[0].split("//")[1]}/api/tags',
                timeout=2)
            return True
        except Exception:
            return False


class InferenceEngineAdapter:
    """Wraps our InferenceEngine for use as an agent LLM backend."""

    def __init__(self, engine, tokenizer):
        self.engine    = engine
        self.tokenizer = tokenizer

    def __call__(self, prompt: str) -> str:
        ids     = self.tokenizer.encode(prompt)
        # Keep only the last N tokens to fit within max_seq_len
        ids     = ids[-(self.engine.max_seq_len - 80):]
        out_ids = self.engine.generate(ids, max_tokens=120,
                                        temperature=0.3, top_k=10)
        return self.tokenizer.decode(out_ids[len(ids):])


class MockEngine:
    """Returns scripted responses — used for unit tests and demos."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self._idx       = 0

    def __call__(self, prompt: str) -> str:
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return 'Action: answer\nArgs: {"text": "스크립트 종료"}'


# =============================================================================
# REACT AGENT LOOP
# =============================================================================

SYSTEM_PROMPT = """당신은 티켓팅 에이전트입니다. 사용자의 요청을 도구를 사용해 처리하세요.

[규칙]
1. 매 턴마다 반드시 아래 형식으로 출력하세요.
2. Action은 반드시 사용 가능한 도구 중 하나여야 합니다.
3. 최종 답변은 answer 도구를 사용하세요.

[출력 형식]
Thought: (현재 상황 분석 및 다음 행동 결정)
Action: (도구 이름)
Args: {"key": "value"}
"""


class AgentLoop:

    def __init__(self,
                 generate_fn: Callable[[str], str],
                 memory:      MemorySystem,
                 tools:       ToolRegistry,
                 max_steps:   int = 8,
                 verbose:     bool = True):
        self.generate  = generate_fn
        self.memory    = memory
        self.tools     = tools
        self.max_steps = max_steps
        self.verbose   = verbose

    # ── prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, user_input: str,
                      history: List[Tuple],
                      context: str) -> str:
        tool_block = self.tools.descriptions()
        parts      = [
            SYSTEM_PROMPT,
            f"[사용 가능한 도구]\n{tool_block}",
        ]
        if context.strip():
            parts.append(context)

        parts.append(f"[사용자 요청]\n{user_input}\n")

        for i, (thought, action, args, obs) in enumerate(history, 1):
            parts.append(
                f"Thought: {thought}\n"
                f"Action: {action}\n"
                f"Args: {json.dumps(args, ensure_ascii=False)}\n"
                f"Observation: {obs}"
            )

        parts.append("Thought:")
        return '\n\n'.join(parts)

    # ── response parser ───────────────────────────────────────────────────────

    @staticmethod
    def _parse(text: str) -> Tuple[str, str, dict]:
        """Extract (thought, action, args) from LLM output."""
        # Thought
        tm = re.search(r'(?:Thought:\s*)(.+?)(?=\nAction:|\Z)', text, re.DOTALL)
        thought = tm.group(1).strip() if tm else text[:100].strip()

        # Action
        am = re.search(r'Action:\s*(\w+)', text)
        action = am.group(1).strip() if am else 'answer'

        # Args — find first {...} after "Args:"
        arm = re.search(r'Args:\s*(\{.*?\})', text, re.DOTALL)
        if arm:
            try:
                args = json.loads(arm.group(1))
            except json.JSONDecodeError:
                args = {'text': thought}
        else:
            args = {'text': thought}

        return thought, action, args

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> str:
        context = self.memory.build_context(user_input, max_recall=3)
        history: List[Tuple] = []

        if self.verbose:
            print(f"\n{'='*55}")
            print(f"[사용자] {user_input}")
            print('='*55)

        for step in range(1, self.max_steps + 1):
            prompt   = self._build_prompt(user_input, history, context)
            raw      = self.generate(prompt)
            full     = 'Thought: ' + raw           # prepend since prompt ends at "Thought:"
            thought, action, args = self._parse(full)

            if self.verbose:
                print(f"\n[Step {step}]")
                print(f"  Thought : {thought[:80]}")
                print(f"  Action  : {action}")
                print(f"  Args    : {json.dumps(args, ensure_ascii=False)[:80]}")

            # ── final answer ──────────────────────────────────────────────
            if action == 'answer':
                answer_text = args.get('text', thought)
                if self.verbose:
                    print(f"\n[에이전트] {answer_text}")
                self.memory.remember(
                    f"사용자: {user_input} → 에이전트: {answer_text}",
                    {'type': 'conversation', 'steps': step})
                return answer_text

            # ── tool call ─────────────────────────────────────────────────
            if action not in self.tools:
                observation = f"알 수 없는 도구: {action}"
            else:
                observation = self.tools.run(action, args)

            if self.verbose:
                print(f"  Obs     : {observation[:100]}")

            history.append((thought, action, args, observation))

        # Max steps reached
        fallback = f"최대 스텝({self.max_steps})에 도달했습니다."
        if self.verbose:
            print(f"\n[에이전트] {fallback}")
        return fallback


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import tempfile, os
    import numpy as np
    from .tokenizer import BPETokenizer
    from .transformer import GPT
    from .trainer import TextDataset, Trainer
    from .server import InferenceEngine
    from .memory import MemorySystem, Embedder
    from .tools import default_registry

    np.random.seed(42)

    # ── 모델 & 메모리 준비 ────────────────────────────────────────────────────
    corpus = """
인공지능은 인간의 지능을 모방하는 기술입니다.
티켓팅 에이전트는 KTX와 콘서트 예매를 도와드립니다.
KTX 예약은 출발일 1개월 전부터 가능합니다.
콘서트 예매는 오픈 당일 오전 10시에 시작됩니다.
좌석은 창가와 통로 중 선택할 수 있습니다.
예매 취소는 출발 하루 전까지 가능합니다.
결제 수단은 카드와 카카오페이를 지원합니다.
""" * 50

    print("=== 모델 초기화 (학습 생략) ===")
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=400)
    model   = GPT.nano(tok.vocab_size)
    engine  = InferenceEngine(model)
    embedder = Embedder(engine, tok)

    tmpdir = tempfile.mkdtemp()
    memory = MemorySystem(embedder, recall_size=100,
                          archive_path=os.path.join(tmpdir, 'archive.jsonl'))

    memory.core.set('persona', '티켓팅 에이전트. KTX·콘서트·항공권 예매 전문.')
    memory.core.set('user',    '사용자: 김철수 | 선호 좌석: 창가 | 결제: 카카오페이')
    memory.remember("지난달 BTS 콘서트 A구역 예매 완료",      {'type': 'history'})
    memory.remember("KTX 서울→부산 창가 좌석 자주 이용",       {'type': 'preference'})

    tools = default_registry(memory)

    # ── 시나리오 1: MockEngine으로 루프 구조 검증 ─────────────────────────────
    print("\n" + "="*55)
    print("시나리오 1: KTX 예매 (MockEngine — 구조 검증용)")
    print("="*55)

    ktx_script = [
        # Step 1 — 잔여석 조회
        """KTX 잔여석부터 확인해야겠다.
Action: search_ktx
Args: {"date": "2026-05-15", "from": "서울", "to": "부산"}""",

        # Step 2 — 예매 진행
        """창가 좌석이 있다. 바로 예매한다.
Action: book_ktx
Args: {"date": "2026-05-15", "time": "10:00", "from": "서울", "to": "부산", "seat_type": "창가"}""",

        # Step 3 — 완료
        """예매가 완료됐다. 결과를 알린다.
Action: answer
Args: {"text": "KTX 서울→부산 5월 15일 오전 10시 창가 좌석 예매 완료했습니다!"}""",
    ]

    agent = AgentLoop(MockEngine(ktx_script), memory, tools)
    agent.run("서울에서 부산까지 5월 15일 KTX 창가 좌석 예매해줘")

    # ── 시나리오 2: 콘서트 예매 ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("시나리오 2: 콘서트 예매 (MockEngine)")
    print("="*55)

    concert_script = [
        """IU 콘서트 티켓 잔여석을 확인한다.
Action: search_concert
Args: {"artist": "IU", "date": "2026-06-01"}""",

        """A구역이 있다. 1매 예매한다.
Action: book_concert
Args: {"artist": "IU", "date": "2026-06-01", "zone": "A구역", "count": 1}""",

        """예매 완료.
Action: answer
Args: {"text": "IU 콘서트 6월 1일 A구역 1매 예매 완료했습니다!"}""",
    ]

    agent2 = AgentLoop(MockEngine(concert_script), memory, tools)
    agent2.run("IU 콘서트 6월 1일 티켓 1장 예매해줘")

    # ── 시나리오 3: 계산기 도구 ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("시나리오 3: 계산기 도구")
    print("="*55)

    calc_script = [
        """KTX 왕복 요금을 계산한다.
Action: calculator
Args: {"expr": "59800 * 2"}""",

        """왕복 요금은 119600원이다.
Action: answer
Args: {"text": "서울↔부산 KTX 왕복 요금은 119,600원입니다."}""",
    ]

    agent3 = AgentLoop(MockEngine(calc_script), memory, tools)
    agent3.run("서울 부산 KTX 왕복 요금이 얼마야? 편도 59800원이야")

    # ── Ollama 연결 가능 여부 확인 ────────────────────────────────────────────
    print("\n" + "="*55)
    print("Ollama 연결 테스트")
    print("="*55)

    ollama = OllamaEngine(model='qwen2.5:7b')
    if ollama.available():
        print("Ollama 사용 가능 — 실제 LLM으로 에이전트 실행")
        agent_real = AgentLoop(ollama, memory, tools)
        agent_real.run("내일 서울에서 부산 가는 KTX 예매해줘")
    else:
        print("Ollama 미실행 — MockEngine 결과로 구조 검증 완료")
        print("\n실제 LLM 사용 시:")
        print("  ollama pull qwen2.5:7b")
        print("  ollama serve")
        print("  → AgentLoop(OllamaEngine('gemma3:4b'), memory, tools).run('KTX 예매해줘')")

    # ── 메모리에 쌓인 대화 확인 ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("대화 후 메모리 상태")
    print("="*55)
    print(f"recall 항목: {memory.stats()['recall_entries']}개")
    recent = memory.search_recall("예매 완료", top_k=3)
    for r in recent:
        print(f"  - {r.text[:70]}")
