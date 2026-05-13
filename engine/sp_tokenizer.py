"""
SentencePiece 토크나이저를 ChatTokenizer 인터페이스로 래핑.

기존 vocab(32000) 끝에 특수 토큰 2개를 추가:
  user_id  = 32000  (<|user|>)
  asst_id  = 32001  (<|assistant|>)

사용 전 반드시 expand_vocab(model, tok.vocab_size) 호출 필요.
"""

import sentencepiece as spm


class SPChatTokenizer:

    def __init__(self, model_path: str):
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        self._model_path = model_path

        base         = self._sp.get_piece_size()   # 32000
        self.user_id = base                         # 32000
        self.asst_id = base + 1                     # 32001
        self.eos_id  = self._sp.eos_id()            # 3
        self.pad_id  = self._sp.pad_id()            # 0

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size() + 2        # 32002

    def encode(self, text: str) -> list:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list, skip_special: bool = True) -> str:
        base = self._sp.get_piece_size()
        # 확장 특수 토큰(32000, 32001)은 SP가 모르므로 먼저 제거
        filtered = [i for i in ids if i < base]
        if not skip_special:
            # eos/pad 등 SP 내부 특수 토큰은 SP가 알아서 처리
            return self._sp.decode(filtered)
        sp_special = {self._sp.bos_id(), self._sp.eos_id(), self._sp.pad_id()}
        filtered = [i for i in filtered if i not in sp_special]
        return self._sp.decode(filtered)
