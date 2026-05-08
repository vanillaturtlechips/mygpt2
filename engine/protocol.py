"""
A2A / ANP / x402 protocol stack for agent-to-agent communication.

  ANP  — Agent identity via DID (Decentralized Identifier)
  A2A  — Task delegation between agents over HTTP
  x402 — Autonomous micropayments (HTTP 402 flow)

Flow:
  Client → POST /a2a (no payment)
  Server → 402  {"payment_required": 0.01, "payment_to": "did:..."}
  Client → pays via x402 ledger → gets tx_id
  Client → POST /a2a (with payment_tx_id)
  Server → 200  {"status": "success", "result": {...}}
"""

import json
import uuid
import time
import hashlib
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler


# =============================================================================
# ANP — AGENT IDENTITY
# =============================================================================

@dataclass
class Skill:
    id:          str
    description: str
    price_usd:   float = 0.0   # 0 = free


@dataclass
class AgentCard:
    """
    An agent's public identity card.
    In production this is served at /.well-known/did.json
    and referenced by a DID such as did:web:ticketing.myai.com
    """
    did:         str
    name:        str
    description: str
    endpoint:    str            # base URL of the agent's A2A server
    skills:      List[Skill]

    def to_dict(self) -> dict:
        return {
            'did':         self.did,
            'name':        self.name,
            'description': self.description,
            'endpoint':    self.endpoint,
            'skills': [
                {'id': s.id, 'description': s.description, 'price_usd': s.price_usd}
                for s in self.skills
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AgentCard':
        return cls(
            did         = d['did'],
            name        = d['name'],
            description = d['description'],
            endpoint    = d['endpoint'],
            skills      = [Skill(**s) for s in d.get('skills', [])],
        )


class DIDRegistry:
    """
    Local DID registry (shared singleton across agents in the same process).
    In production: each agent serves its own DID document over HTTPS.
    """

    _instance: Optional['DIDRegistry'] = None

    def __init__(self):
        self._store: Dict[str, AgentCard] = {}

    @classmethod
    def global_instance(cls) -> 'DIDRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, card: AgentCard):
        self._store[card.did] = card

    def resolve(self, did: str) -> Optional[AgentCard]:
        if did in self._store:
            return self._store[did]
        # HTTP fallback: did:web:host → https://host/.well-known/did.json
        if did.startswith('did:web:'):
            return self._resolve_web(did)
        return None

    def _resolve_web(self, did: str) -> Optional[AgentCard]:
        host = did[len('did:web:'):]
        try:
            resp = urllib.request.urlopen(
                f'https://{host}/.well-known/did.json', timeout=5)
            return AgentCard.from_dict(json.loads(resp.read()))
        except Exception:
            return None

    def list_agents(self) -> List[AgentCard]:
        return list(self._store.values())


# =============================================================================
# x402 — MICROPAYMENT LEDGER
# =============================================================================

@dataclass
class PaymentProof:
    tx_id:     str
    from_did:  str
    to_did:    str
    amount:    float
    timestamp: float


class PaymentLedger:
    """
    Simulated payment ledger (shared singleton).
    In production: on-chain transactions (Base / Ethereum / Solana).
    Replace transfer() and verify() with real blockchain calls.
    """

    _instance: Optional['PaymentLedger'] = None

    def __init__(self):
        self._balances: Dict[str, float] = {}
        self._txs:      Dict[str, PaymentProof] = {}
        self._lock = threading.Lock()

    @classmethod
    def global_instance(cls) -> 'PaymentLedger':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def deposit(self, did: str, amount: float):
        with self._lock:
            self._balances[did] = self._balances.get(did, 0.0) + amount

    def balance(self, did: str) -> float:
        return self._balances.get(did, 0.0)

    def transfer(self, from_did: str, to_did: str,
                 amount: float) -> Optional[PaymentProof]:
        with self._lock:
            if self._balances.get(from_did, 0.0) < amount - 1e-9:
                return None                 # insufficient funds
            self._balances[from_did] -= amount
            self._balances[to_did]    = self._balances.get(to_did, 0.0) + amount
            proof = PaymentProof(
                tx_id     = hashlib.sha256(
                    f"{from_did}{to_did}{amount}{time.time()}".encode()
                ).hexdigest()[:16],
                from_did  = from_did,
                to_did    = to_did,
                amount    = amount,
                timestamp = time.time(),
            )
            self._txs[proof.tx_id] = proof
            return proof

    def verify(self, tx_id: str, expected_to: str,
               min_amount: float, ttl: float = 300.0) -> bool:
        proof = self._txs.get(tx_id)
        if not proof:
            return False
        return (proof.to_did == expected_to
                and proof.amount >= min_amount - 1e-9
                and time.time() - proof.timestamp < ttl)

    def state(self) -> dict:
        return {did: round(bal, 6) for did, bal in self._balances.items()}


# =============================================================================
# A2A — TASK FORMAT
# =============================================================================

@dataclass
class Task:
    task_id:        str
    skill_id:       str
    params:         Dict[str, Any]
    caller_did:     str
    payment_tx_id:  Optional[str] = None


@dataclass
class TaskResult:
    task_id:          str
    status:           str          # success | error | payment_required
    result:           Optional[Dict]  = None
    error:            Optional[str]   = None
    payment_required: Optional[float] = None
    payment_to:       Optional[str]   = None


# =============================================================================
# A2A — SERVER
# =============================================================================

class A2AServer:
    """
    HTTP server that exposes agent skills to other agents.

    Endpoints:
      GET  /.well-known/did.json  — returns AgentCard
      POST /a2a                   — accepts Task, returns TaskResult
    """

    def __init__(self,
                 card:     AgentCard,
                 registry: DIDRegistry   = None,
                 ledger:   PaymentLedger = None):
        self.card     = card
        self.registry = registry or DIDRegistry.global_instance()
        self.ledger   = ledger   or PaymentLedger.global_instance()
        self._skills: Dict[str, tuple] = {}   # skill_id → (handler, price)
        self._http:   Optional[HTTPServer] = None
        self.registry.register(card)

    def register_skill(self, skill_id: str,
                       handler: Callable,
                       price:   float = 0.0):
        self._skills[skill_id] = (handler, price)

    # ── request handling ─────────────────────────────────────────────────────

    def _handle(self, body: dict) -> tuple:
        """Returns (http_status, response_dict)."""
        try:
            task = Task(
                task_id       = body.get('task_id', str(uuid.uuid4())[:8]),
                skill_id      = body['skill_id'],
                params        = body.get('params', {}),
                caller_did    = body.get('caller_did', ''),
                payment_tx_id = body.get('payment_tx_id'),
            )
        except KeyError as e:
            return 400, {'status': 'error', 'error': f'Missing field: {e}'}

        if task.skill_id not in self._skills:
            return 404, {'status': 'error',
                         'error': f"Unknown skill '{task.skill_id}'"}

        handler, price = self._skills[task.skill_id]

        # ── x402 gate ────────────────────────────────────────────────────────
        if price > 0:
            if not task.payment_tx_id:
                return 402, {
                    'task_id':          task.task_id,
                    'status':           'payment_required',
                    'payment_required': price,
                    'payment_to':       self.card.did,
                    'currency':         'USD (simulated)',
                }
            if not self.ledger.verify(task.payment_tx_id,
                                      self.card.did, price):
                return 402, {
                    'task_id':          task.task_id,
                    'status':           'payment_invalid',
                    'payment_required': price,
                    'payment_to':       self.card.did,
                }

        # ── execute ──────────────────────────────────────────────────────────
        try:
            result = handler(**task.params)
            return 200, {'task_id': task.task_id,
                         'status':  'success',
                         'result':  result}
        except Exception as e:
            return 500, {'task_id': task.task_id,
                         'status':  'error',
                         'error':   str(e)}

    # ── HTTP server ───────────────────────────────────────────────────────────

    def start(self, host: str = '127.0.0.1', port: int = 8090) -> 'A2AServer':
        _self = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ('/did.json', '/.well-known/did.json'):
                    body = json.dumps(
                        _self.card.to_dict(), ensure_ascii=False).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/a2a':
                    length = int(self.headers.get('Content-Length', 0))
                    data   = json.loads(self.rfile.read(length))
                    code, resp = _self._handle(data)
                    body   = json.dumps(resp, ensure_ascii=False).encode()
                    self.send_response(code)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *_):
                pass

        self._http = HTTPServer((host, port), _Handler)
        t = threading.Thread(target=self._http.serve_forever, daemon=True)
        t.start()
        return self

    def stop(self):
        if self._http:
            self._http.shutdown()


# =============================================================================
# A2A — CLIENT
# =============================================================================

class A2AClient:
    """
    Sends tasks to remote agents.
    Automatically handles the x402 payment round-trip.
    """

    def __init__(self,
                 caller_card: AgentCard,
                 registry:    DIDRegistry   = None,
                 ledger:      PaymentLedger = None):
        self.card     = caller_card
        self.registry = registry or DIDRegistry.global_instance()
        self.ledger   = ledger   or PaymentLedger.global_instance()

    def send_task(self, target_did: str,
                  skill_id: str,
                  params:   Dict[str, Any]) -> dict:
        target = self.registry.resolve(target_did)
        if not target:
            return {'status': 'error',
                    'error': f'Cannot resolve DID: {target_did}'}

        payload = {
            'task_id':    str(uuid.uuid4())[:8],
            'skill_id':   skill_id,
            'params':     params,
            'caller_did': self.card.did,
        }

        # ── first attempt (no payment) ────────────────────────────────────
        resp = self._post(target.endpoint + '/a2a', payload)

        # ── handle 402 ───────────────────────────────────────────────────
        if resp.get('status') in ('payment_required', 'payment_invalid'):
            amount = resp['payment_required']
            to_did = resp['payment_to']
            proof  = self.ledger.transfer(self.card.did, to_did, amount)
            if proof is None:
                bal = self.ledger.balance(self.card.did)
                return {'status': 'error',
                        'error': f'잔액 부족 (필요: ${amount:.4f}, '
                                 f'보유: ${bal:.4f})'}
            payload['payment_tx_id'] = proof.tx_id
            resp = self._post(target.endpoint + '/a2a', payload)

        return resp

    @staticmethod
    def _post(url: str, data: dict) -> dict:
        body = json.dumps(data, ensure_ascii=False).encode()
        req  = urllib.request.Request(
            url, data=body,
            headers={'Content-Type': 'application/json'})
        try:
            r = urllib.request.urlopen(req, timeout=10)
            return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import random

    random.seed(42)
    registry = DIDRegistry.global_instance()
    ledger   = PaymentLedger.global_instance()

    # ── 1. 에이전트 카드 정의 (ANP) ───────────────────────────────────────────
    print("=== ANP: 에이전트 신원 등록 ===")

    ticketing_card = AgentCard(
        did         = 'did:local:ticketing-agent',
        name        = '티켓팅 에이전트',
        description = 'KTX·콘서트·항공권 예매 통합 에이전트',
        endpoint    = 'http://127.0.0.1:19001',
        skills      = [
            Skill('orchestrate', '예매 오케스트레이션', price_usd=0.0),
        ],
    )

    ktx_card = AgentCard(
        did         = 'did:local:ktx-agent',
        name        = 'KTX 예매 에이전트',
        description = 'KTX 잔여석 조회 및 예매 전문',
        endpoint    = 'http://127.0.0.1:19002',
        skills      = [
            Skill('search_ktx', 'KTX 잔여석 조회', price_usd=0.0),
            Skill('book_ktx',   'KTX 예매',        price_usd=0.01),
        ],
    )

    concert_card = AgentCard(
        did         = 'did:local:concert-agent',
        name        = '콘서트 예매 에이전트',
        description = '콘서트·공연 티켓 예매 전문',
        endpoint    = 'http://127.0.0.1:19003',
        skills      = [
            Skill('search_concert', '콘서트 잔여석 조회', price_usd=0.0),
            Skill('book_concert',   '콘서트 예매',        price_usd=0.005),
        ],
    )

    for card in [ticketing_card, ktx_card, concert_card]:
        registry.register(card)
        print(f"  등록: {card.did}  ({len(card.skills)} skills)")

    # ── 2. 에이전트 서버 시작 (A2A) ──────────────────────────────────────────
    print("\n=== A2A: 에이전트 서버 시작 ===")

    # KTX 에이전트 서버
    ktx_server = A2AServer(ktx_card)

    def skill_search_ktx(date: str, from_city: str = '서울',
                         to_city: str = '부산') -> dict:
        random.seed(hash(date) % 9999)
        times = ['08:00', '10:00', '14:00', '18:00']
        seats = [{'time': t, 'seat_type': random.choice(['창가', '통로']),
                  'count': random.randint(1, 8)}
                 for t in random.sample(times, 3)]
        return {'available': True, 'from': from_city,
                'to': to_city, 'date': date, 'seats': seats}

    def skill_book_ktx(date: str, time: str,
                       seat_type: str = '창가', **kw) -> dict:
        bid = f"KTX-{random.randint(100000,999999)}"
        return {'booking_id': bid, 'date': date,
                'time': time, 'seat_type': seat_type,
                'message': f'예매 완료. 예약번호: {bid}'}

    ktx_server.register_skill('search_ktx', skill_search_ktx, price=0.0)
    ktx_server.register_skill('book_ktx',   skill_book_ktx,   price=0.01)
    ktx_server.start(port=19002)
    print("  KTX 에이전트    → http://127.0.0.1:19002")

    # 콘서트 에이전트 서버
    concert_server = A2AServer(concert_card)

    def skill_search_concert(artist: str, date: str = '') -> dict:
        random.seed(hash(artist) % 9999)
        zones = [
            {'zone': 'A구역', 'price': 165000, 'count': random.randint(0, 20)},
            {'zone': 'B구역', 'price': 132000, 'count': random.randint(0, 30)},
            {'zone': '스탠딩', 'price': 99000,  'count': random.randint(0, 50)},
        ]
        available = [z for z in zones if z['count'] > 0]
        return {'artist': artist, 'available': bool(available), 'zones': available}

    def skill_book_concert(artist: str, zone: str = 'B구역',
                           count: int = 1, **kw) -> dict:
        bid = f"CNT-{random.randint(100000,999999)}"
        prices = {'A구역': 165000, 'B구역': 132000, '스탠딩': 99000}
        return {'booking_id': bid, 'artist': artist,
                'zone': zone, 'count': count,
                'total_price': prices.get(zone, 132000) * count,
                'message': f'예매 완료. 예약번호: {bid}'}

    concert_server.register_skill('search_concert', skill_search_concert, price=0.0)
    concert_server.register_skill('book_concert',   skill_book_concert,   price=0.005)
    concert_server.start(port=19003)
    print("  콘서트 에이전트 → http://127.0.0.1:19003")
    time.sleep(0.3)   # 서버 준비 대기

    # ── 3. x402 초기 잔액 충전 ───────────────────────────────────────────────
    print("\n=== x402: 초기 잔액 충전 ===")
    ledger.deposit(ticketing_card.did, 1.0)   # $1.00 충전
    print(f"  티켓팅 에이전트 잔액: ${ledger.balance(ticketing_card.did):.4f}")

    # ── 4. DID 조회 (ANP) ────────────────────────────────────────────────────
    print("\n=== ANP: DID 조회 ===")
    resolved = registry.resolve('did:local:ktx-agent')
    print(f"  조회: did:local:ktx-agent")
    print(f"  이름: {resolved.name}")
    print(f"  스킬: {[s.id for s in resolved.skills]}")

    # GET /.well-known/did.json 도 동작하는지 확인
    r = urllib.request.urlopen('http://127.0.0.1:19002/did.json')
    card_json = json.loads(r.read())
    print(f"  HTTP 조회: {card_json['name']} ({card_json['did']})")

    # ── 5. A2A + x402 흐름 — KTX 예매 ──────────────────────────────────────
    print("\n=== A2A + x402: KTX 예매 흐름 ===")
    client = A2AClient(ticketing_card)

    print("\n[Step 1] KTX 잔여석 조회 (무료)")
    result = client.send_task(
        'did:local:ktx-agent', 'search_ktx',
        {'date': '2026-05-20', 'from_city': '서울', 'to_city': '부산'})
    print(f"  상태: {result['status']}")
    if result['status'] == 'success':
        seats = result['result']['seats']
        print(f"  잔여석: {seats}")

    print(f"\n[Step 2] KTX 예매 (유료 $0.01) — 잔액: ${ledger.balance(ticketing_card.did):.4f}")
    result = client.send_task(
        'did:local:ktx-agent', 'book_ktx',
        {'date': '2026-05-20', 'time': '10:00', 'seat_type': '창가'})
    print(f"  상태: {result['status']}")
    if result['status'] == 'success':
        print(f"  결과: {result['result']['message']}")
    print(f"  결제 후 잔액: ${ledger.balance(ticketing_card.did):.4f}")
    print(f"  KTX 에이전트 수익: ${ledger.balance(ktx_card.did):.4f}")

    # ── 6. A2A + x402 흐름 — 콘서트 예매 ───────────────────────────────────
    print("\n=== A2A + x402: 콘서트 예매 흐름 ===")

    print("\n[Step 1] IU 콘서트 잔여석 조회 (무료)")
    result = client.send_task(
        'did:local:concert-agent', 'search_concert',
        {'artist': 'IU', 'date': '2026-06-01'})
    print(f"  상태: {result['status']}")
    if result['status'] == 'success':
        print(f"  잔여: {result['result']['zones']}")

    print(f"\n[Step 2] IU 콘서트 예매 (유료 $0.005) — 잔액: ${ledger.balance(ticketing_card.did):.4f}")
    result = client.send_task(
        'did:local:concert-agent', 'book_concert',
        {'artist': 'IU', 'zone': 'A구역', 'count': 1})
    print(f"  상태: {result['status']}")
    if result['status'] == 'success':
        r = result['result']
        print(f"  결과: {r['message']} | 금액: {r['total_price']:,}원")
    print(f"  결제 후 잔액: ${ledger.balance(ticketing_card.did):.4f}")

    # ── 7. 잔액 부족 테스트 ──────────────────────────────────────────────────
    print("\n=== x402: 잔액 부족 테스트 ===")
    broke_card = AgentCard(
        did='did:local:broke-agent', name='파산 에이전트',
        description='', endpoint='', skills=[])
    ledger.deposit(broke_card.did, 0.001)   # 잔액 부족
    broke_client = A2AClient(broke_card)
    result = broke_client.send_task(
        'did:local:ktx-agent', 'book_ktx',
        {'date': '2026-05-20', 'time': '08:00'})
    print(f"  상태: {result['status']}")
    print(f"  메시지: {result['error']}")

    # ── 8. 최종 원장 상태 ────────────────────────────────────────────────────
    print("\n=== 최종 원장 상태 ===")
    for did, bal in ledger.state().items():
        name = registry.resolve(did)
        label = name.name if name else did
        print(f"  {label:20s}: ${bal:.4f}")

    ktx_server.stop()
    concert_server.stop()
    print("\n완료.")
