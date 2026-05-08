"""
3-layer agent memory system.

  CoreMemory    — fixed slots always injected into context (persona, user, task)
  RecallMemory  — recent memories with numpy cosine-similarity search
  ArchiveMemory — append-only long-term log, keyword/date search
  MemorySystem  — orchestrates all three layers
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any

from .server import InferenceEngine
from .tokenizer import BPETokenizer


# =============================================================================
# EMBEDDER
# =============================================================================

class Embedder:
    """
    Converts text → fixed-size float32 vector using our GPT hidden states.
    No external embedding API needed.
    """

    def __init__(self, engine: InferenceEngine, tokenizer: BPETokenizer):
        self.engine    = engine
        self.tokenizer = tokenizer
        self.dim       = engine.embed_dim

    def embed(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text)
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        ids = ids[:self.engine.max_seq_len]
        return self.engine.encode(ids)          # (C,)


# =============================================================================
# MEMORY ENTRY
# =============================================================================

class MemoryEntry:
    def __init__(self, text: str, embedding: np.ndarray,
                 metadata: Optional[Dict] = None,
                 created_at: Optional[str] = None):
        self.text       = text
        self.embedding  = embedding
        self.metadata   = metadata or {}
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            'text':       self.text,
            'embedding':  self.embedding.tolist(),
            'metadata':   self.metadata,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryEntry':
        return cls(
            text       = d['text'],
            embedding  = np.array(d['embedding'], dtype=np.float32),
            metadata   = d.get('metadata', {}),
            created_at = d.get('created_at'),
        )


# =============================================================================
# LAYER 1 — CORE MEMORY
# =============================================================================

class CoreMemory:
    """
    Fixed named slots that are ALWAYS included in the LLM context.
    Think of it as the agent's working memory / whiteboard.

    Slots:
      persona  — who the agent is
      user     — persistent facts about the user
      task     — what the agent is currently doing
      notes    — scratch space for temporary information
    """

    SLOTS = ('persona', 'user', 'task', 'notes')

    def __init__(self):
        self._store: Dict[str, str] = {s: '' for s in self.SLOTS}

    def set(self, slot: str, value: str):
        if slot not in self._store:
            raise KeyError(f"Unknown slot '{slot}'. Available: {self.SLOTS}")
        self._store[slot] = value

    def get(self, slot: str) -> str:
        return self._store.get(slot, '')

    def update(self, slot: str, value: str):
        """Append to an existing slot value."""
        current = self._store.get(slot, '')
        self._store[slot] = (current + '\n' + value).strip()

    def to_string(self) -> str:
        lines = [f"[{s}] {v}" for s, v in self._store.items() if v]
        return '\n'.join(lines)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self._store.update(json.load(f))


# =============================================================================
# LAYER 2 — RECALL MEMORY
# =============================================================================

class RecallMemory:
    """
    Recent memories stored with embeddings.
    Retrieval: cosine similarity (numpy dot product — no external vector DB).

    When capacity is exceeded, the oldest entries are drained out
    (MemorySystem moves them to ArchiveMemory).
    """

    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []

    def __len__(self):
        return len(self._entries)

    def add(self, entry: MemoryEntry):
        self._entries.append(entry)

    def search(self, query_emb: np.ndarray,
               top_k: int = 5,
               min_score: float = 0.2) -> List[MemoryEntry]:
        """Return top-k entries by cosine similarity to query_emb."""
        if not self._entries:
            return []

        mat    = np.stack([e.embedding for e in self._entries])   # (N, C)
        q_unit = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms  = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        scores = (mat / norms) @ q_unit                            # (N,)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self._entries[i] for i in top_idx if scores[i] >= min_score]

    def drain_oldest(self, n: int) -> List[MemoryEntry]:
        """Remove and return the n oldest entries for archiving."""
        n       = min(n, len(self._entries))
        drained = self._entries[:n]
        self._entries = self._entries[n:]
        return drained

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([e.to_dict() for e in self._entries],
                      f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self._entries = [MemoryEntry.from_dict(d) for d in json.load(f)]


# =============================================================================
# LAYER 3 — ARCHIVE MEMORY
# =============================================================================

class ArchiveMemory:
    """
    Append-only long-term storage (JSONL file).
    No embeddings stored here — search is by keyword or date range.
    Embeddings would be too expensive to keep indefinitely.
    """

    def __init__(self, path: str = '/tmp/memory_archive.jsonl'):
        self.path = path

    def add(self, entries: List[MemoryEntry]):
        with open(self.path, 'a', encoding='utf-8') as f:
            for e in entries:
                record = {
                    'text':       e.text,
                    'metadata':   e.metadata,
                    'created_at': e.created_at,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def search(self, keyword: Optional[str] = None,
               date_from: Optional[str] = None,
               date_to:   Optional[str] = None,
               limit: int = 20) -> List[dict]:
        results = []
        if not os.path.exists(self.path):
            return results

        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if keyword and keyword.lower() not in rec['text'].lower():
                    continue
                ts = rec.get('created_at', '')
                if date_from and ts < date_from:
                    continue
                if date_to   and ts > date_to:
                    continue

                results.append(rec)
                if len(results) >= limit:
                    break

        return results

    def count(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())


# =============================================================================
# MEMORY SYSTEM  (orchestrator)
# =============================================================================

class MemorySystem:
    """
    Orchestrates all three memory layers.

    Data flow:
      new event → remember() → RecallMemory
      recall gets full (80%) → drain oldest 25% → ArchiveMemory

    Context assembly:
      build_context(query) → CoreMemory + top-k relevant recall entries
    """

    def __init__(self,
                 embedder:      Embedder,
                 recall_size:   int = 500,
                 archive_path:  str = '/tmp/memory_archive.jsonl',
                 consolidate_at: float = 0.8):
        self.embedder  = embedder
        self.core      = CoreMemory()
        self.recall    = RecallMemory(max_entries=recall_size)
        self.archive   = ArchiveMemory(archive_path)
        self._cap      = recall_size
        self._thresh   = int(recall_size * consolidate_at)

    # ── write ─────────────────────────────────────────────────────────────────

    def remember(self, text: str, metadata: Optional[Dict] = None):
        """Embed text and add to recall. Auto-consolidates when near capacity."""
        emb   = self.embedder.embed(text)
        entry = MemoryEntry(text, emb, metadata)
        self.recall.add(entry)

        if len(self.recall) >= self._thresh:
            self._consolidate()

    def _consolidate(self):
        """Move oldest 25 % of recall entries to archive."""
        n       = max(1, len(self.recall) // 4)
        drained = self.recall.drain_oldest(n)
        self.archive.add(drained)

    # ── read ──────────────────────────────────────────────────────────────────

    def search_recall(self, query: str,
                      top_k: int = 5,
                      min_score: float = 0.2) -> List[MemoryEntry]:
        q_emb = self.embedder.embed(query)
        return self.recall.search(q_emb, top_k, min_score)

    def search_archive(self, keyword: Optional[str] = None,
                       date_from: Optional[str] = None,
                       date_to:   Optional[str] = None) -> List[dict]:
        return self.archive.search(keyword, date_from, date_to)

    # ── context assembly ──────────────────────────────────────────────────────

    def build_context(self, query: str, max_recall: int = 3) -> str:
        """
        Assemble a context string for the LLM:
          1. Core memory (always)
          2. Top-k semantically relevant recall memories
        """
        parts = []

        core_str = self.core.to_string()
        if core_str:
            parts.append("=== 기본 정보 ===\n" + core_str)

        relevant = self.search_recall(query, top_k=max_recall)
        if relevant:
            lines = '\n'.join(
                f"- [{e.created_at[:10]}] {e.text}" for e in relevant
            )
            parts.append("=== 관련 기억 ===\n" + lines)

        return '\n\n'.join(parts)

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'recall_entries':  len(self.recall),
            'archive_entries': self.archive.count(),
            'recall_capacity': self._cap,
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        self.core.save(  os.path.join(dir_path, 'core.json'))
        self.recall.save(os.path.join(dir_path, 'recall.json'))

    def load(self, dir_path: str):
        c_path = os.path.join(dir_path, 'core.json')
        r_path = os.path.join(dir_path, 'recall.json')
        if os.path.exists(c_path):
            self.core.load(c_path)
        if os.path.exists(r_path):
            self.recall.load(r_path)


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    import tempfile
    from .tokenizer import BPETokenizer
    from .transformer import GPT
    from .trainer import TextDataset, Trainer

    np.random.seed(42)

    # ── tiny trained model ────────────────────────────────────────────────────
    corpus = """
인공지능은 인간의 지능을 모방하는 기술입니다.
머신러닝은 데이터로부터 패턴을 학습합니다.
딥러닝은 신경망을 사용하는 머신러닝 방법입니다.
트랜스포머는 어텐션 메커니즘을 기반으로 합니다.
GPT는 트랜스포머 기반의 언어 모델입니다.
티켓팅 에이전트는 콘서트 예매를 도와드립니다.
KTX 예약은 출발일 1개월 전부터 가능합니다.
결제 수단은 카드, 카카오페이를 지원합니다.
좌석은 창가와 통로 중 선택할 수 있습니다.
예매 취소는 출발 하루 전까지 가능합니다.
""" * 40

    print("=== 모델 학습 ===")
    tok   = BPETokenizer()
    tok.train(corpus, vocab_size=400)
    model = GPT.nano(tok.vocab_size)
    trainer = Trainer(model, tok, lr=3e-4)
    trainer.train(TextDataset(tok.encode(corpus), seq_len=32),
                  batch_size=4, epochs=3, log_every=99999)

    engine   = InferenceEngine(model)
    embedder = Embedder(engine, tok)

    # ── verify embedder ───────────────────────────────────────────────────────
    print("\n=== 임베딩 유사도 테스트 ===")
    texts = ["KTX 예약", "기차 예매", "콘서트 티켓", "인공지능 기술"]
    embs  = [embedder.embed(t) for t in texts]

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    print(f"'KTX 예약' ↔ '기차 예매'    : {cos(embs[0], embs[1]):.3f}  (높을수록 좋음)")
    print(f"'KTX 예약' ↔ '콘서트 티켓'  : {cos(embs[0], embs[2]):.3f}")
    print(f"'KTX 예약' ↔ '인공지능 기술' : {cos(embs[0], embs[3]):.3f}  (낮을수록 좋음)")

    # ── build memory system ───────────────────────────────────────────────────
    print("\n=== 메모리 시스템 구성 ===")
    tmpdir = tempfile.mkdtemp()
    mem = MemorySystem(embedder, recall_size=20,
                       archive_path=os.path.join(tmpdir, 'archive.jsonl'))

    # Core memory 설정
    mem.core.set('persona', '티켓팅 에이전트. 콘서트, KTX, 항공권 예매를 도와드립니다.')
    mem.core.set('user',    '사용자: 김철수 | 선호 좌석: 창가 | 결제: 카카오페이')
    mem.core.set('task',    '대기 중')

    # Recall에 과거 기억 추가
    memories = [
        ("지난달 BTS 콘서트 티켓 2매 예매 완료. 구역 A, 좌석 12열.",
         {'type': 'ticketing', 'event': 'BTS 콘서트'}),
        ("KTX 서울→부산 12월 24일 오전 10시. 창가 좌석. 예매 성공.",
         {'type': 'ktx', 'route': '서울-부산'}),
        ("항공권 제주 왕복 1월 15일. 결제 카카오페이 완료.",
         {'type': 'flight', 'destination': '제주'}),
        ("KTX 서울→대구 예매 시도. 매진으로 실패. 대기열 등록함.",
         {'type': 'ktx', 'route': '서울-대구', 'status': 'failed'}),
        ("사용자가 창가 좌석을 강하게 선호함. 통로석 거절한 적 있음.",
         {'type': 'preference'}),
        ("콘서트 예매 시 오전 10시 오픈런 패턴. 자동화 요청함.",
         {'type': 'pattern'}),
    ]

    for text, meta in memories:
        mem.remember(text, meta)

    print(f"recall 저장 완료: {mem.stats()['recall_entries']}개")

    # ── recall 검색 테스트 ────────────────────────────────────────────────────
    print("\n=== Recall 검색 테스트 ===")
    queries = [
        "KTX 예매 내역 알려줘",
        "콘서트 티켓 예매한 적 있어?",
        "좌석 선호도가 어떻게 돼?",
    ]

    for q in queries:
        print(f"\n질문: {q}")
        results = mem.search_recall(q, top_k=2)
        for r in results:
            print(f"  → {r.text[:50]}...")

    # ── Archive 테스트 ────────────────────────────────────────────────────────
    print("\n=== Archive 강제 이동 테스트 ===")
    # recall_size=20이므로 많이 넣으면 자동으로 archive로 이동
    for i in range(20):
        mem.remember(f"테스트 기억 {i}: 임의의 과거 이벤트 데이터", {'type': 'test'})

    stats = mem.stats()
    print(f"recall: {stats['recall_entries']}개, "
          f"archive: {stats['archive_entries']}개")

    archive_results = mem.search_archive(keyword='KTX')
    print(f"archive에서 'KTX' 검색: {len(archive_results)}건")
    for r in archive_results[:2]:
        print(f"  [{r['created_at'][:10]}] {r['text'][:60]}")

    # ── context 조립 ──────────────────────────────────────────────────────────
    print("\n=== Context 조립 ===")
    ctx = mem.build_context("KTX 예매해줘")
    print(ctx)

    # ── 저장 / 로드 ───────────────────────────────────────────────────────────
    print("\n=== 저장 / 로드 ===")
    mem.save(tmpdir)
    mem2 = MemorySystem(embedder, recall_size=20,
                        archive_path=os.path.join(tmpdir, 'archive.jsonl'))
    mem2.load(tmpdir)
    print(f"로드 후 recall: {mem2.stats()['recall_entries']}개")
    r = mem2.search_recall("좌석 선호", top_k=1)
    print(f"로드 후 검색 결과: {r[0].text[:50] if r else '없음'}")
