import math
import copy
import numpy as np
from typing import Optional
from .autograd import Tensor
from .transformer import GPT
from .tokenizer import BPETokenizer
from .optim import AdamW
from .data import Dataset


# =============================================================================
# CHAT TOKENIZER
# =============================================================================

class ChatTokenizer(BPETokenizer):
    """BPETokenizer with <|user|> and <|assistant|> role tokens added."""

    SPECIAL_TOKENS = ['<|pad|>', '<|unk|>', '<|bos|>', '<|eos|>', '<|endoftext|>',
                      '<|user|>', '<|assistant|>']

    @property
    def user_id(self) -> int:
        return self._special_ids['<|user|>']

    @property
    def asst_id(self) -> int:
        return self._special_ids['<|assistant|>']

    @property
    def eos_id(self) -> int:
        return self._special_ids['<|eos|>']


# =============================================================================
# SFT DATASET
# =============================================================================

class SFTDataset(Dataset):
    """
    Formats conversations as: <|user|> prompt <|assistant|> response <|eos|>
    Loss mask = 0 on prompt tokens, 1 on response tokens.
    """

    def __init__(self, conversations: list, tokenizer: ChatTokenizer, seq_len: int):
        self.samples = []
        for conv in conversations:
            user_ids  = [tokenizer.user_id]  + tokenizer.encode(conv['user'])
            asst_ids  = [tokenizer.asst_id]  + tokenizer.encode(conv['assistant']) + [tokenizer.eos_id]
            ids  = user_ids + asst_ids
            mask = [0] * len(user_ids) + [1] * len(asst_ids)
            if len(ids) > seq_len + 1:
                ids, mask = ids[:seq_len + 1], mask[:seq_len + 1]
            self.samples.append((
                np.array(ids,  dtype=np.int32),
                np.array(mask, dtype=np.float32),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, mask = self.samples[idx]
        # x = input tokens, y = target tokens (shifted by 1), m = loss mask on targets
        return ids[:-1], ids[1:], mask[1:]


# =============================================================================
# SFT TRAINER
# =============================================================================

class SFTTrainer:

    def __init__(self, model: GPT, tokenizer: ChatTokenizer,
                 lr: float = 1e-4, weight_decay: float = 0.1,
                 grad_clip: float = 1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.grad_clip = grad_clip
        self._base_lr = lr
        self.optimizer = AdamW(model.parameters(), lr=lr,
                               betas=(0.9, 0.95), weight_decay=weight_decay)
        self.step = 0

    def _compute_loss(self, x: np.ndarray, y: np.ndarray,
                      mask: np.ndarray) -> Optional[Tensor]:
        n_asst = mask.sum()
        if n_asst == 0:
            return None

        logits = self.model.forward(x[np.newaxis, :])   # (1, T, V)
        T, V = logits.data.shape[1], logits.data.shape[2]
        logits_2d = logits.reshape(T, V)                # keep in graph

        # build one-hot targets zeroed on prompt positions
        one_hot = np.zeros((T, V), dtype=np.float32)
        one_hot[np.arange(T), y] = 1.0
        one_hot *= mask[:, np.newaxis]

        # fused masked cross-entropy with explicit backward
        m     = logits_2d.data.max(axis=-1, keepdims=True)
        shift = logits_2d.data - m
        exp_s = np.exp(shift)
        s_exp = exp_s.sum(axis=-1, keepdims=True)
        softmax = exp_s / (s_exp + 1e-10)
        log_probs = logits_2d.data - (m + np.log(s_exp + 1e-10))

        loss_val = np.float32(-(log_probs * one_hot).sum() / n_asst)
        out = Tensor(loss_val, requires_grad=logits_2d.requires_grad)
        out._prev = {logits_2d}
        out._op   = 'sft_loss'

        def _backward():
            if logits_2d.requires_grad:
                grad = (softmax - one_hot) / n_asst
                grad *= mask[:, np.newaxis]          # zero out prompt grads
                logits_2d._accumulate(out.grad * grad)

        out._backward = _backward
        return out

    def _clip_gradients(self) -> float:
        total_norm = sum(np.sum(p.grad ** 2)
                         for p in self.model.parameters() if p.grad is not None)
        total_norm = math.sqrt(total_norm)
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-8)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad *= scale
        return total_norm

    def train(self, dataset: SFTDataset, epochs: int = 3, log_every: int = 10):
        self.model.train()
        print(f"SFT 학습 시작 | 샘플 수: {len(dataset)} | epochs: {epochs}")
        print("-" * 50)

        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_steps = 0.0, 0
            for idx in np.random.permutation(len(dataset)):
                x, y, mask = dataset[int(idx)]
                self.optimizer.zero_grad()
                loss = self._compute_loss(x, y, mask)

                if loss is None or not np.isfinite(loss.data):
                    self.step += 1
                    continue

                loss.backward()
                norm = self._clip_gradients()
                self.optimizer.step()
                epoch_loss  += loss.data
                epoch_steps += 1
                self.step   += 1

                if self.step % log_every == 0:
                    ppl = math.exp(min(loss.data, 20))
                    print(f"epoch {epoch} | step {self.step:4d} | "
                          f"loss {loss.data:.4f} | ppl {ppl:.2f} | norm {norm:.2f}")

            avg = epoch_loss / max(epoch_steps, 1)
            print(f"\nepoch {epoch} 완료 | avg loss {avg:.4f} | "
                  f"ppl {math.exp(min(avg, 20)):.2f}\n")

    def save(self, path: str):
        import json
        weights = {str(i): p.data.tolist() for i, p in enumerate(self.model.parameters())}
        config = {
            'vocab_size': self.model.vocab_size,
            'embed_dim':  self.model.embed_dim,
            'num_heads':  self.model.num_heads,
            'num_layers': self.model.num_layers,
            'max_seq_len': self.model.max_seq_len,
            'dropout':    self.model.dropout_p,
        }
        with open(path, 'w') as f:
            json.dump({'config': config, 'weights': weights}, f)

    def load(self, path: str):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        for i, p in enumerate(self.model.parameters()):
            p.data = np.array(data['weights'][str(i)], dtype=np.float32)


# =============================================================================
# PREFERENCE DATASET  (for DPO)
# =============================================================================

class PreferenceDataset(Dataset):
    """
    Each sample: (prompt + chosen_response) and (prompt + rejected_response).
    prompt_len marks where the response starts so log-prob is computed only over the response.
    """

    def __init__(self, preferences: list, tokenizer: ChatTokenizer, seq_len: int):
        self.samples = []
        for pref in preferences:
            prompt_ids   = ([tokenizer.user_id] + tokenizer.encode(pref['prompt'])
                            + [tokenizer.asst_id])
            chosen_ids   = tokenizer.encode(pref['chosen'])   + [tokenizer.eos_id]
            rejected_ids = tokenizer.encode(pref['rejected']) + [tokenizer.eos_id]

            chosen_full   = np.array(prompt_ids + chosen_ids,   dtype=np.int32)[:seq_len]
            rejected_full = np.array(prompt_ids + rejected_ids, dtype=np.int32)[:seq_len]

            self.samples.append({
                'chosen':     chosen_full,
                'rejected':   rejected_full,
                'prompt_len': len(prompt_ids),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# DPO TRAINER
# =============================================================================

class DPOTrainer:
    """
    Direct Preference Optimization.

    loss = -log σ( β * [ logπ(y_w|x) - logπ_ref(y_w|x)
                        - logπ(y_l|x) + logπ_ref(y_l|x) ] )

    ref_model is kept frozen (numpy only, no gradient).
    """

    def __init__(self, model: GPT, ref_model: GPT, tokenizer: ChatTokenizer,
                 lr: float = 1e-5, beta: float = 0.1, grad_clip: float = 1.0):
        self.model     = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta      = beta
        self.grad_clip = grad_clip
        self.optimizer = AdamW(model.parameters(), lr=lr,
                               betas=(0.9, 0.95), weight_decay=0.0)
        self.step = 0

    # ── log-prob helpers ─────────────────────────────────────────────────────

    def _log_prob_np(self, model: GPT,
                     input_ids: np.ndarray, prompt_len: int) -> float:
        """Mean response log-prob as plain float (for frozen ref model)."""
        T = len(input_ids)
        response_len = T - prompt_len - 1
        if response_len <= 0:
            return 0.0

        ldata = model.forward(input_ids[np.newaxis, :]).data[0]  # (T, V)
        m     = ldata.max(axis=-1, keepdims=True)
        shift = ldata - m
        log_p = shift - np.log(np.exp(shift).sum(axis=-1, keepdims=True) + 1e-10)

        total = sum(log_p[t, input_ids[t + 1]] for t in range(prompt_len, T - 1))
        return total / response_len

    def _log_prob_tensor(self, input_ids: np.ndarray, prompt_len: int) -> Tensor:
        """Mean response log-prob as Tensor (gradient flows through current model)."""
        T = len(input_ids)
        response_len = T - prompt_len - 1
        if response_len <= 0:
            return Tensor(np.float32(0.0))

        logits = self.model.forward(input_ids[np.newaxis, :])   # (1, T, V)
        V = logits.data.shape[-1]
        logits_2d = logits.reshape(T, V)                         # (T, V) in graph

        # numerically stable log_softmax: subtract max (constant) then chain through Tensor ops
        m_t    = Tensor(logits.data[0].max(axis=-1, keepdims=True))   # constant
        shifted = logits_2d - m_t
        log_sum_exp = shifted.exp().sum(axis=-1, keepdims=True).log() + m_t
        log_probs_2d = logits_2d - log_sum_exp                  # (T, V)

        # gather target-token log-probs via one-hot (preserves graph)
        gather = np.zeros((T, V), dtype=np.float32)
        for t in range(prompt_len, T - 1):
            gather[t, input_ids[t + 1]] = 1.0

        token_lp = (log_probs_2d * Tensor(gather)).sum(axis=-1)  # (T,)

        # average over response positions only
        resp_weight = np.zeros(T, dtype=np.float32)
        resp_weight[prompt_len: T - 1] = 1.0 / response_len
        return (token_lp * Tensor(resp_weight)).sum()

    # ── DPO loss ─────────────────────────────────────────────────────────────

    def _dpo_loss(self, chosen: np.ndarray,
                  rejected: np.ndarray, prompt_len: int) -> Tensor:
        log_p_c = self._log_prob_tensor(chosen,   prompt_len)
        log_p_r = self._log_prob_tensor(rejected, prompt_len)

        self.ref_model.eval()
        log_r_c = float(self._log_prob_np(self.ref_model, chosen,   prompt_len))
        log_r_r = float(self._log_prob_np(self.ref_model, rejected, prompt_len))

        delta = (
            (log_p_c - Tensor(np.float32(log_r_c))) -
            (log_p_r - Tensor(np.float32(log_r_r)))
        ) * Tensor(np.float32(self.beta))

        return -(delta.sigmoid().log())

    # ── gradient clip ─────────────────────────────────────────────────────────

    def _clip_gradients(self) -> float:
        total_norm = sum(np.sum(p.grad ** 2)
                         for p in self.model.parameters() if p.grad is not None)
        total_norm = math.sqrt(total_norm)
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-8)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad *= scale
        return total_norm

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, dataset: PreferenceDataset,
              epochs: int = 2, log_every: int = 5):
        self.model.train()
        print(f"DPO 학습 시작 | 샘플 수: {len(dataset)} | epochs: {epochs}")
        print("-" * 50)

        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_steps = 0.0, 0

            for i, idx in enumerate(np.random.permutation(len(dataset))):
                sample = dataset[int(idx)]
                self.optimizer.zero_grad()
                loss = self._dpo_loss(
                    sample['chosen'], sample['rejected'], sample['prompt_len'])

                if not np.isfinite(loss.data):
                    self.step += 1
                    continue

                loss.backward()
                norm = self._clip_gradients()
                self.optimizer.step()
                epoch_loss  += loss.data
                epoch_steps += 1
                self.step   += 1

                if (i + 1) % log_every == 0:
                    print(f"epoch {epoch} | step {self.step:4d} | "
                          f"dpo_loss {loss.data:.4f} | norm {norm:.2f}")

            avg = epoch_loss / max(epoch_steps, 1)
            print(f"\nepoch {epoch} 완료 | avg DPO loss {avg:.4f}\n")


# =============================================================================
# UTILS
# =============================================================================

def clone_gpt(model: GPT) -> GPT:
    """Deep-copy a GPT model (for use as frozen reference in DPO)."""
    ref = GPT(
        vocab_size  = model.vocab_size,
        embed_dim   = model.embed_dim,
        num_heads   = model.num_heads,
        num_layers  = model.num_layers,
        max_seq_len = model.max_seq_len,
        dropout     = model.dropout_p,
    )
    for rp, mp in zip(ref.parameters(), model.parameters()):
        rp.data = mp.data.copy()
    ref.eval()
    return ref


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    corpus = """
인공지능은 인간의 지능을 모방하는 기술입니다.
머신러닝은 데이터로부터 패턴을 학습합니다.
딥러닝은 신경망을 사용하는 머신러닝 방법입니다.
트랜스포머는 어텐션 메커니즘을 기반으로 합니다.
자연어 처리는 컴퓨터가 언어를 이해하는 분야입니다.
GPT는 트랜스포머 기반의 언어 모델입니다.
""" * 80

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print("=== ChatTokenizer 학습 ===")
    tok = ChatTokenizer()
    tok.train(corpus, vocab_size=400)
    print(f"vocab size : {tok.vocab_size}")
    print(f"<|user|>   : {tok.user_id}")
    print(f"<|asst|>   : {tok.asst_id}")

    # ── pretrained model ──────────────────────────────────────────────────────
    print("\n=== 기반 모델 생성 ===")
    model = GPT.nano(tok.vocab_size)
    print(f"파라미터: {model.num_params():,}")

    # ── SFT ───────────────────────────────────────────────────────────────────
    print("\n=== SFT 학습 ===")
    conversations = [
        {'user': '인공지능이 뭐야?',
         'assistant': '인공지능은 인간의 지능을 모방하는 기술입니다.'},
        {'user': '머신러닝 설명해줘.',
         'assistant': '머신러닝은 데이터로부터 패턴을 학습하는 방법입니다.'},
        {'user': '딥러닝은 뭐야?',
         'assistant': '딥러닝은 신경망을 사용하는 머신러닝의 한 종류입니다.'},
        {'user': 'GPT가 뭐야?',
         'assistant': 'GPT는 트랜스포머 기반의 언어 모델입니다.'},
        {'user': '트랜스포머 알려줘.',
         'assistant': '트랜스포머는 어텐션 메커니즘을 사용하는 딥러닝 모델입니다.'},
    ] * 10   # 50 samples

    sft_dataset = SFTDataset(conversations, tok, seq_len=64)
    sft_trainer = SFTTrainer(model, tok, lr=1e-3)
    sft_trainer.train(sft_dataset, epochs=3, log_every=30)

    # ── DPO ───────────────────────────────────────────────────────────────────
    print("\n=== DPO 학습 ===")
    ref_model = clone_gpt(model)   # snapshot of SFT model, stays frozen

    preferences = [
        {'prompt':   '인공지능이 뭐야?',
         'chosen':   '인공지능은 인간의 지능을 모방하는 기술입니다.',
         'rejected': '모르겠어요.'},
        {'prompt':   '머신러닝 설명해줘.',
         'chosen':   '머신러닝은 데이터로부터 패턴을 학습하는 방법입니다.',
         'rejected': '어려운 질문이네요.'},
        {'prompt':   '딥러닝은 뭐야?',
         'chosen':   '딥러닝은 신경망을 사용하는 머신러닝의 한 종류입니다.',
         'rejected': '잘 모릅니다.'},
        {'prompt':   'GPT가 뭐야?',
         'chosen':   'GPT는 트랜스포머 기반의 언어 모델입니다.',
         'rejected': '모르겠습니다.'},
    ] * 5   # 20 samples

    dpo_dataset = PreferenceDataset(preferences, tok, seq_len=64)
    dpo_trainer = DPOTrainer(model, ref_model, tok, lr=1e-5, beta=0.1)
    dpo_trainer.train(dpo_dataset, epochs=2, log_every=5)

    # ── generation ────────────────────────────────────────────────────────────
    print("\n=== 생성 테스트 ===")
    model.eval()
    for prompt in ['인공지능이 뭐야?', 'GPT가 뭐야?']:
        prompt_ids = [tok.user_id] + tok.encode(prompt) + [tok.asst_id]
        out_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.7, top_k=20)
        print(f"Q: {prompt}")
        print(f"A: {tok.decode(out_ids[len(prompt_ids):])}\n")
