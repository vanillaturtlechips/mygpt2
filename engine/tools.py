"""
Tool registry and built-in tools for the agent loop.

Tools:
  calculator     — evaluate math expressions
  ask_user       — prompt the user for input
  answer         — emit final answer and stop the loop
  recall_search  — semantic search over agent memory
  search_ktx     — mock KTX seat availability
  book_ktx       — mock KTX booking
  search_concert — mock concert ticket availability
  book_concert   — mock concert booking
"""

import json
import random
import numpy as np
from typing import Dict, Any


# =============================================================================
# BASE CLASSES
# =============================================================================

class Tool:
    name:        str = ''
    description: str = ''    # shown to the LLM so it knows when/how to call the tool

    def run(self, **kwargs) -> str:
        raise NotImplementedError

    def schema(self) -> str:
        return f"{self.name}: {self.description}"


class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def run(self, name: str, kwargs: Dict[str, Any]) -> str:
        if name not in self._tools:
            available = ', '.join(self._tools.keys())
            return f"오류: '{name}' 도구 없음. 사용 가능: {available}"
        try:
            return self._tools[name].run(**kwargs)
        except TypeError as e:
            return f"인자 오류: {e}"
        except Exception as e:
            return f"실행 오류: {e}"

    def descriptions(self) -> str:
        return '\n'.join(t.schema() for t in self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# =============================================================================
# GENERAL-PURPOSE TOOLS
# =============================================================================

class Calculator(Tool):
    name        = 'calculator'
    description = '수식 계산. Args: {"expr": "1 + 2 * 3"}'

    _SAFE = {k: v for k, v in vars(__builtins__
                                    if isinstance(__builtins__, dict)
                                    else __builtins__).items()
             if k in ('abs', 'round', 'min', 'max', 'sum', 'int', 'float')} \
           if not isinstance(__builtins__, dict) else {}

    def run(self, expr: str) -> str:
        try:
            result = eval(expr, {"__builtins__": {}}, {})   # no builtins — safe
            return str(result)
        except Exception as e:
            return f"계산 오류: {e}"


class AskUser(Tool):
    name        = 'ask_user'
    description = '사용자에게 추가 정보를 요청. Args: {"message": "질문 내용"}'

    def run(self, message: str) -> str:
        print(f"\n[에이전트 질문] {message}")
        return input("[사용자 입력] ").strip()


class Answer(Tool):
    """Special tool — calling this ends the ReAct loop."""
    name        = 'answer'
    description = '최종 답변을 제공하고 루프를 종료. Args: {"text": "답변 내용"}'

    def run(self, text: str) -> str:
        return text     # AgentLoop detects this tool and stops


class RecallSearch(Tool):
    name        = 'recall_search'
    description = '과거 기억/대화 이력 검색. Args: {"query": "검색어"}'

    def __init__(self, memory):
        self.memory = memory

    def run(self, query: str) -> str:
        results = self.memory.search_recall(query, top_k=3)
        if not results:
            return "관련 기억 없음"
        return '\n'.join(f"- {r.text}" for r in results)


# =============================================================================
# TICKETING TOOLS  (mock — replace with real API calls)
# =============================================================================

class SearchKTX(Tool):
    name        = 'search_ktx'
    description = ('KTX 잔여석 조회. '
                   'Args: {"date": "YYYY-MM-DD", "from": "출발역", "to": "도착역"}')

    def run(self, date: str, **kwargs) -> str:
        random.seed(hash(date) % 1000)
        if random.random() < 0.15:
            return json.dumps({"available": False, "message": "매진"},
                               ensure_ascii=False)
        times = ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00"]
        seats = [
            {"time": t,
             "seat_type": random.choice(["창가", "통로"]),
             "count": random.randint(1, 8)}
            for t in random.sample(times, 3)
        ]
        return json.dumps({"available": True, "seats": seats}, ensure_ascii=False)


class BookKTX(Tool):
    name        = 'book_ktx'
    description = ('KTX 예매. '
                   'Args: {"date":"YYYY-MM-DD","time":"HH:MM",'
                   '"from":"출발역","to":"도착역","seat_type":"창가/통로"}')

    def run(self, date: str, time: str, seat_type: str = "창가", **kwargs) -> str:
        booking_id = f"KTX-{random.randint(100000, 999999)}"
        return json.dumps({
            "status":     "success",
            "booking_id": booking_id,
            "date":       date,
            "time":       time,
            "seat_type":  seat_type,
            "message":    f"예매 완료. 예약번호: {booking_id}",
        }, ensure_ascii=False)


class SearchConcert(Tool):
    name        = 'search_concert'
    description = ('콘서트 티켓 잔여 조회. '
                   'Args: {"artist": "아티스트명", "date": "YYYY-MM-DD"}')

    def run(self, artist: str, date: str = "", **kwargs) -> str:
        random.seed(hash(artist + date) % 1000)
        if random.random() < 0.3:
            return json.dumps({"available": False, "message": "매진"},
                               ensure_ascii=False)
        zones = [
            {"zone": "A구역", "price": 165000, "count": random.randint(0, 20)},
            {"zone": "B구역", "price": 132000, "count": random.randint(0, 30)},
            {"zone": "스탠딩", "price": 99000,  "count": random.randint(0, 50)},
        ]
        available = [z for z in zones if z["count"] > 0]
        return json.dumps({"available": bool(available), "zones": available},
                           ensure_ascii=False)


class BookConcert(Tool):
    name        = 'book_concert'
    description = ('콘서트 티켓 예매. '
                   'Args: {"artist":"아티스트명","date":"YYYY-MM-DD",'
                   '"zone":"구역","count":매수}')

    def run(self, artist: str, zone: str = "B구역",
            count: int = 1, **kwargs) -> str:
        booking_id = f"CNT-{random.randint(100000, 999999)}"
        price_map  = {"A구역": 165000, "B구역": 132000, "스탠딩": 99000}
        unit_price = price_map.get(zone, 132000)
        return json.dumps({
            "status":      "success",
            "booking_id":  booking_id,
            "artist":      artist,
            "zone":        zone,
            "count":       count,
            "total_price": unit_price * count,
            "message":     f"예매 완료. 예약번호: {booking_id}",
        }, ensure_ascii=False)


# =============================================================================
# FACTORY
# =============================================================================

def default_registry(memory=None) -> ToolRegistry:
    """Create a ToolRegistry with all standard + ticketing tools."""
    reg = ToolRegistry()
    reg.register(Calculator())
    reg.register(AskUser())
    reg.register(Answer())
    reg.register(SearchKTX())
    reg.register(BookKTX())
    reg.register(SearchConcert())
    reg.register(BookConcert())
    if memory is not None:
        reg.register(RecallSearch(memory))
    return reg
