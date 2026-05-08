"""
Full ticketing agent — end-to-end stack integration.

Layer map:
  engine/transformer.py  → GPT model
  engine/server.py       → InferenceEngine (KV-cache forward + embeddings)
  engine/memory.py       → 3-layer memory (Core / Recall / Archive)
  engine/tools.py        → ToolRegistry + local mock tools
  engine/agent.py        → ReAct AgentLoop
  engine/protocol.py     → A2A/ANP/x402 (remote sub-agents over HTTP)

Architecture:
  ┌─────────────────────────────────────────────┐
  │            티켓팅 에이전트 (AgentLoop)         │
  │  memory: Core + Recall + Archive             │
  │  tools:  calculator, ask_user, answer,       │
  │          search_ktx_remote  ─→  KTX A2A      │
  │          book_ktx_remote    ─→  KTX A2A      │
  │          search_concert_remote → Concert A2A │
  │          book_concert_remote   → Concert A2A │
  └───────────────┬─────────────────┬────────────┘
                  │ x402 $0.01      │ x402 $0.005
    ┌─────────────▼──┐    ┌─────────▼──────────┐
    │  KTX A2A Server │    │ Concert A2A Server  │
    │  port 19102     │    │ port 19103          │
    └────────────────┘    └────────────────────┘
"""

import json
import random
import tempfile
import os
import time

import numpy as np

from .tokenizer   import BPETokenizer
from .transformer import GPT
from .server      import InferenceEngine
from .memory      import MemorySystem, Embedder
from .tools       import Tool, ToolRegistry, Calculator, AskUser, Answer, RecallSearch
from .agent       import AgentLoop, OllamaEngine, MockEngine
from .protocol    import (AgentCard, Skill, DIDRegistry, PaymentLedger,
                          A2AServer, A2AClient)


# =============================================================================
# REMOTE SUB-AGENT SERVERS  (실제 서비스라면 별도 프로세스/서버에서 실행)
# =============================================================================

def _make_ktx_server(port: int) -> A2AServer:
    random.seed(0)

    card = AgentCard(
        did         = 'did:local:ktx-agent',
        name        = 'KTX 예매 에이전트',
        description = 'KTX 잔여석 조회 및 예매 전문',
        endpoint    = f'http://127.0.0.1:{port}',
        skills      = [
            Skill('search_ktx', 'KTX 잔여석 조회', price_usd=0.0),
            Skill('book_ktx',   'KTX 예매',        price_usd=0.01),
        ],
    )
    srv = A2AServer(card)

    def search_ktx(date: str, from_city: str = '서울',
                   to_city: str = '부산') -> dict:
        random.seed(hash(date) % 9999)
        times = ['08:00', '10:00', '12:00', '14:00', '16:00', '18:00']
        seats = [
            {'time': t,
             'seat_type': random.choice(['창가', '통로']),
             'count': random.randint(1, 8)}
            for t in random.sample(times, 3)
        ]
        return {'available': True, 'from': from_city,
                'to': to_city, 'date': date, 'seats': seats}

    def book_ktx(date: str, time: str,
                 seat_type: str = '창가', **kw) -> dict:
        bid = f"KTX-{random.randint(100000, 999999)}"
        return {
            'booking_id': bid, 'date': date,
            'time': time, 'seat_type': seat_type,
            'message': f'예매 완료. 예약번호: {bid}',
        }

    srv.register_skill('search_ktx', search_ktx, price=0.0)
    srv.register_skill('book_ktx',   book_ktx,   price=0.01)
    srv.start(port=port)
    return srv


def _make_concert_server(port: int) -> A2AServer:
    random.seed(1)

    card = AgentCard(
        did         = 'did:local:concert-agent',
        name        = '콘서트 예매 에이전트',
        description = '콘서트·공연 티켓 예매 전문',
        endpoint    = f'http://127.0.0.1:{port}',
        skills      = [
            Skill('search_concert', '콘서트 잔여석 조회', price_usd=0.0),
            Skill('book_concert',   '콘서트 예매',        price_usd=0.005),
        ],
    )
    srv = A2AServer(card)

    def search_concert(artist: str, date: str = '') -> dict:
        random.seed(hash(artist) % 9999)
        zones = [
            {'zone': 'A구역', 'price': 165000,
             'count': random.randint(0, 20)},
            {'zone': 'B구역', 'price': 132000,
             'count': random.randint(0, 30)},
            {'zone': '스탠딩', 'price': 99000,
             'count': random.randint(0, 50)},
        ]
        available = [z for z in zones if z['count'] > 0]
        return {'artist': artist, 'available': bool(available),
                'zones': available}

    def book_concert(artist: str, zone: str = 'B구역',
                     count: int = 1, **kw) -> dict:
        bid = f"CNT-{random.randint(100000, 999999)}"
        prices = {'A구역': 165000, 'B구역': 132000, '스탠딩': 99000}
        total  = prices.get(zone, 132000) * count
        return {
            'booking_id': bid, 'artist': artist,
            'zone': zone, 'count': count, 'total_price': total,
            'message': f'예매 완료. 예약번호: {bid} | 금액: {total:,}원',
        }

    srv.register_skill('search_concert', search_concert, price=0.0)
    srv.register_skill('book_concert',   book_concert,   price=0.005)
    srv.start(port=port)
    return srv


# =============================================================================
# A2A TOOLS  (ToolRegistry에 등록돼 에이전트 루프가 호출)
# =============================================================================

class SearchKTXRemote(Tool):
    name        = 'search_ktx'
    description = ('KTX 잔여석 조회 (원격 A2A). '
                   'Args: {"date":"YYYY-MM-DD","from_city":"출발","to_city":"도착"}')

    def __init__(self, client: A2AClient):
        self._client = client

    def run(self, date: str,
            from_city: str = '서울', to_city: str = '부산') -> str:
        r = self._client.send_task(
            'did:local:ktx-agent', 'search_ktx',
            {'date': date, 'from_city': from_city, 'to_city': to_city})
        return json.dumps(r, ensure_ascii=False)


class BookKTXRemote(Tool):
    name        = 'book_ktx'
    description = ('KTX 예매 (원격 A2A, $0.01). '
                   'Args: {"date":"YYYY-MM-DD","time":"HH:MM","seat_type":"창가/통로"}')

    def __init__(self, client: A2AClient):
        self._client = client

    def run(self, date: str, time: str, seat_type: str = '창가') -> str:
        r = self._client.send_task(
            'did:local:ktx-agent', 'book_ktx',
            {'date': date, 'time': time, 'seat_type': seat_type})
        return json.dumps(r, ensure_ascii=False)


class SearchConcertRemote(Tool):
    name        = 'search_concert'
    description = ('콘서트 잔여석 조회 (원격 A2A). '
                   'Args: {"artist":"아티스트명","date":"YYYY-MM-DD"}')

    def __init__(self, client: A2AClient):
        self._client = client

    def run(self, artist: str, date: str = '') -> str:
        r = self._client.send_task(
            'did:local:concert-agent', 'search_concert',
            {'artist': artist, 'date': date})
        return json.dumps(r, ensure_ascii=False)


class BookConcertRemote(Tool):
    name        = 'book_concert'
    description = ('콘서트 예매 (원격 A2A, $0.005). '
                   'Args: {"artist":"아티스트명","zone":"구역","count":매수}')

    def __init__(self, client: A2AClient):
        self._client = client

    def run(self, artist: str, zone: str = 'B구역', count: int = 1) -> str:
        r = self._client.send_task(
            'did:local:concert-agent', 'book_concert',
            {'artist': artist, 'zone': zone, 'count': count})
        return json.dumps(r, ensure_ascii=False)


# =============================================================================
# TICKETING AGENT  (전체 스택 조립)
# =============================================================================

class TicketingAgent:
    """
    End-to-end ticketing agent.
    Combines: memory + ReAct loop + A2A tools + x402 payments.
    """

    KTX_PORT     = 19202
    CONCERT_PORT = 19203
    AGENT_DID    = 'did:local:ticketing-orchestrator'

    def __init__(self,
                 llm_engine    = None,
                 initial_funds: float = 5.0,
                 verbose:       bool  = True):
        # ── 서브 에이전트 서버 시작 ─────────────────────────────────────────
        self._ktx_srv     = _make_ktx_server(self.KTX_PORT)
        self._concert_srv = _make_concert_server(self.CONCERT_PORT)
        time.sleep(0.2)    # 서버 준비 대기

        # ── x402 지갑 ───────────────────────────────────────────────────────
        self._ledger = PaymentLedger.global_instance()
        self._ledger.deposit(self.AGENT_DID, initial_funds)

        # ── 오케스트레이터 카드 ─────────────────────────────────────────────
        self._my_card = AgentCard(
            did         = self.AGENT_DID,
            name        = '티켓팅 오케스트레이터',
            description = 'KTX·콘서트 예매 통합 에이전트',
            endpoint    = '',
            skills      = [],
        )
        self._client = A2AClient(self._my_card)

        # ── 메모리 ──────────────────────────────────────────────────────────
        self._memory = self._build_memory()

        # ── 도구 레지스트리 ─────────────────────────────────────────────────
        self._tools = self._build_tools()

        # ── LLM ─────────────────────────────────────────────────────────────
        self._llm = llm_engine or OllamaEngine(model='gemma3:4b')

        # ── 에이전트 루프 ────────────────────────────────────────────────────
        self._agent = AgentLoop(
            generate_fn = self._llm,
            memory      = self._memory,
            tools       = self._tools,
            max_steps   = 10,
            verbose     = verbose,
        )

    # ── 내부 빌더 ────────────────────────────────────────────────────────────

    def _build_memory(self) -> MemorySystem:
        """간단한 메모리 — 임베더 없이 키워드 기반으로 초기화."""
        tmpdir = tempfile.mkdtemp()

        class _DummyEmbedder:
            """텍스트를 단어 빈도 벡터(dim=64)로 근사."""
            dim = 64
            def embed(self, text: str):
                v = np.zeros(self.dim)
                for i, ch in enumerate(text[:self.dim]):
                    v[i % self.dim] += ord(ch) / 65536
                norm = np.linalg.norm(v)
                return v / (norm + 1e-9)

        memory = MemorySystem(
            embedder     = _DummyEmbedder(),
            recall_size  = 200,
            archive_path = os.path.join(tmpdir, 'archive.jsonl'),
        )
        memory.core.set('persona',
            '당신은 티켓팅 에이전트입니다. KTX·콘서트 예매를 도와드립니다.')
        memory.core.set('notes',
            'search_ktx/book_ktx: KTX 조회·예매. '
            'search_concert/book_concert: 콘서트 조회·예매.')
        return memory

    def _build_tools(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(Calculator())
        reg.register(AskUser())
        reg.register(Answer())
        reg.register(RecallSearch(self._memory))
        reg.register(SearchKTXRemote(self._client))
        reg.register(BookKTXRemote(self._client))
        reg.register(SearchConcertRemote(self._client))
        reg.register(BookConcertRemote(self._client))
        return reg

    # ── 퍼블릭 API ───────────────────────────────────────────────────────────

    def chat(self, message: str) -> str:
        return self._agent.run(message)

    def balance(self) -> float:
        return self._ledger.balance(self.AGENT_DID)

    def stop(self):
        self._ktx_srv.stop()
        self._concert_srv.stop()


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':

    # ── 시나리오 스크립트 (MockEngine) ────────────────────────────────────────

    KTX_SCRIPT = [
        # step 1: 잔여석 조회
        """서울→부산 5월 20일 KTX 잔여석부터 확인한다.
Action: search_ktx
Args: {"date": "2026-05-20", "from_city": "서울", "to_city": "부산"}""",

        # step 2: 예매
        """잔여석 있음. 창가 좌석 10:00편 예매.
Action: book_ktx
Args: {"date": "2026-05-20", "time": "10:00", "seat_type": "창가"}""",

        # step 3: 답변
        """예매 완료.
Action: answer
Args: {"text": "서울→부산 5월 20일 오전 10시 창가 좌석 KTX 예매 완료했습니다! A2A + x402 결제도 성공."}""",
    ]

    CONCERT_SCRIPT = [
        """IU 콘서트 잔여석 확인.
Action: search_concert
Args: {"artist": "IU", "date": "2026-06-01"}""",

        """A구역 있다. 1매 예매.
Action: book_concert
Args: {"artist": "IU", "zone": "A구역", "count": 1}""",

        """예매 완료.
Action: answer
Args: {"text": "IU 콘서트 6월 1일 A구역 1매 예매 완료. 예약번호와 금액은 메시지를 확인하세요."}""",
    ]

    CALC_SCRIPT = [
        """KTX 왕복 요금 계산.
Action: calculator
Args: {"expr": "59800 * 2"}""",

        """왕복 119,600원.
Action: answer
Args: {"text": "서울↔부산 KTX 왕복 요금은 119,600원입니다."}""",
    ]

    # ── 에이전트 생성 ─────────────────────────────────────────────────────────
    print("="*60)
    print("티켓팅 에이전트 초기화")
    print("="*60)

    agent = TicketingAgent(
        llm_engine    = None,  # Ollama 연결 시도 전 mock으로 대체
        initial_funds = 5.0,
        verbose       = True,
    )

    print(f"초기 지갑 잔액: ${agent.balance():.4f}\n")

    # ── 시나리오 1: KTX 예매 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("시나리오 1: KTX 예매 (MockEngine + A2A + x402)")
    print("="*60)
    agent._agent.generate = MockEngine(KTX_SCRIPT)
    result = agent.chat("서울에서 부산까지 5월 20일 KTX 창가 좌석 예매해줘")
    print(f"\n지갑 잔액: ${agent.balance():.4f}  (KTX $0.01 차감)")

    # ── 시나리오 2: 콘서트 예매 ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("시나리오 2: 콘서트 예매 (MockEngine + A2A + x402)")
    print("="*60)
    agent._agent.generate = MockEngine(CONCERT_SCRIPT)
    result = agent.chat("IU 콘서트 6월 1일 A구역 1매 예매해줘")
    print(f"\n지갑 잔액: ${agent.balance():.4f}  (콘서트 $0.005 추가 차감)")

    # ── 시나리오 3: 계산기 ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("시나리오 3: 계산기 (로컬 도구)")
    print("="*60)
    agent._agent.generate = MockEngine(CALC_SCRIPT)
    result = agent.chat("서울 부산 KTX 왕복 요금이 얼마야? 편도 59800원이야")
    print(f"\n지갑 잔액: ${agent.balance():.4f}  (계산기는 무료)")

    # ── 원장 최종 상태 ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("최종 원장 상태")
    print("="*60)
    ledger   = PaymentLedger.global_instance()
    registry = DIDRegistry.global_instance()
    for did, bal in ledger.state().items():
        card  = registry.resolve(did)
        label = card.name if card else did
        print(f"  {label:25s}: ${bal:.4f}")

    # ── 실제 Ollama LLM 테스트 ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("실제 LLM 테스트 (Ollama)")
    print("="*60)
    ollama = OllamaEngine(model='gemma3:4b')
    if ollama.available():
        print("Ollama 연결 성공 — 실제 LLM으로 에이전트 실행\n")
        agent._agent.generate = ollama
        agent.chat("내일 서울에서 부산 가는 KTX 창가 좌석 예매해줘")
        print(f"\n지갑 잔액: ${agent.balance():.4f}")
    else:
        print("Ollama 미실행 — MockEngine 검증만 완료")
        print("실제 LLM 사용 시:")
        print("  ollama serve  (터미널 1)")
        print("  python3 -m engine.ticketing_agent  (터미널 2)")

    agent.stop()
    print("\n=== 완료 ===")
