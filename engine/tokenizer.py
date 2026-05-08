import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple


# =============================================================================
# BYTE-LEVEL BPE TOKENIZER
# =============================================================================

class BPETokenizer:

    SPECIAL_TOKENS = ['<|pad|>', '<|unk|>', '<|bos|>', '<|eos|>', '<|endoftext|>']

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._special_ids: Dict[str, int] = {}

    # ── byte-level base ───────────────────────────────────────────────────────

    def _build_base_vocab(self):
        self.vocab = {}
        idx = 0
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = idx
            self._special_ids[tok] = idx
            idx += 1
        for b in range(256):
            token = bytes([b]).decode('latin-1')
            self.vocab[token] = idx
            idx += 1

    def _text_to_byte_tokens(self, text: str) -> List[str]:
        return [bytes([b]).decode('latin-1') for b in text.encode('utf-8')]

    # ── training ──────────────────────────────────────────────────────────────

    def _get_word_freqs(self, text: str) -> Dict[Tuple[str, ...], int]:
        freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        for word in re.findall(r'\S+|\s', text):
            key = tuple(self._text_to_byte_tokens(word))
            freqs[key] += 1
        return freqs

    def _get_pair_freqs(self, word_freqs: Dict) -> Dict[Tuple[str, str], int]:
        pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq
        return pair_freqs

    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict) -> Dict:
        merged = pair[0] + pair[1]
        new_freqs = {}
        for word, freq in word_freqs.items():
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def train(self, text: str, vocab_size: int = 1000, verbose: bool = False):
        self._build_base_vocab()
        word_freqs = self._get_word_freqs(text)

        while len(self.vocab) < vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < 2:
                break
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            word_freqs = self._merge_pair(best_pair, word_freqs)
            if verbose:
                print(f"merge {len(self.merges):4d}: vocab={len(self.vocab)}")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    # ── encode / decode ───────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> List[str]:
        tokens = self._text_to_byte_tokens(word)
        for pair in self.merges:
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self._special_ids['<|bos|>'])
        for word in re.findall(r'\S+|\s', text):
            for token in self._tokenize_word(word):
                ids.append(self.vocab.get(token, self._special_ids['<|unk|>']))
        if add_eos:
            ids.append(self._special_ids['<|eos|>'])
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        special_ids = set(self._special_ids.values())
        byte_list = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            token = self.inverse_vocab.get(i, '')
            byte_list.extend(token.encode('latin-1'))
        return bytes(byte_list).decode('utf-8', errors='replace')

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab, 'merges': self.merges}, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = [tuple(m) for m in data['merges']]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._special_ids = {tok: self.vocab[tok] for tok in self.SPECIAL_TOKENS if tok in self.vocab}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.vocab_size


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    corpus = """
    인공지능은 인간의 지능을 모방하는 기술입니다.
    머신러닝은 인공지능의 한 분야로, 데이터로부터 학습합니다.
    딥러닝은 머신러닝의 한 방법으로, 신경망을 사용합니다.
    트랜스포머는 딥러닝 모델의 핵심 아키텍처입니다.
    자연어 처리는 인공지능이 언어를 이해하는 분야입니다.
    """ * 30

    print("=== Training BPE ===")
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=600, verbose=False)
    print(f"vocab size : {tok.vocab_size}")
    print(f"merges     : {len(tok.merges)}")

    print("\n=== Encode / Decode ===")
    text = "인공지능은 딥러닝을 사용합니다."
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"원문   : {text}")
    print(f"IDs    : {ids}")
    print(f"복원   : {decoded}")
    print(f"일치   : {text == decoded}")

    print("\n=== Special Tokens ===")
    ids_bos = tok.encode("안녕하세요", add_bos=True, add_eos=True)
    print(f"bos+eos IDs    : {ids_bos}")
    print(f"skip special   : {tok.decode(ids_bos, skip_special=True)}")
    print(f"keep special   : {tok.decode(ids_bos, skip_special=False)}")

    print("\n=== Compression ratio ===")
    sample = "트랜스포머는 딥러닝 모델의 핵심 아키텍처입니다."
    char_count = len(sample)
    token_count = len(tok.encode(sample))
    print(f"글자 수  : {char_count}")
    print(f"토큰 수  : {token_count}")
    print(f"압축률   : {char_count / token_count:.2f}x")

    print("\n=== Save / Load ===")
    tok.save('/tmp/tokenizer.json')
    tok2 = BPETokenizer()
    tok2.load('/tmp/tokenizer.json')
    print(f"저장 후 로드 일치: {tok.encode(text) == tok2.encode(text)}")

    print("\nAll checks passed.")
