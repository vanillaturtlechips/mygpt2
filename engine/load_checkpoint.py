"""
train_a100.py (PyTorch GPT2) 체크포인트 → engine/transformer.py (numpy GPT) 변환

사용법:
    from engine.load_checkpoint import load_from_pytorch_ckpt
    model = load_from_pytorch_ckpt("checkpoints/ckpt_step200000.pt")
"""

import numpy as np


def load_from_pytorch_ckpt(ckpt_path: str):
    """
    PyTorch 체크포인트를 로드해서 engine.transformer.GPT 인스턴스를 반환.

    주요 변환:
      - qkv (2304, 768) → q_proj (768,768) / k_proj (768,768) / v_proj (768,768) 분리
      - q/k/v bias 제거 (engine은 bias=False)
      - LayerNorm weight/bias → gamma/beta
      - weight tying 유지 (head.weight = tok_emb.weight)
    """
    import torch
    from .transformer import GPT

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    # ── 컴파일된 모델은 키 앞에 "_orig_mod." 붙음 ─────────────────────────────
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    cfg = ckpt.get("config", {})
    vocab_size  = cfg.get("vocab_size",  sd["tok_emb.weight"].shape[0])
    embed_dim   = cfg.get("d_model",     sd["tok_emb.weight"].shape[1])
    num_heads   = cfg.get("n_heads",     12)
    num_layers  = cfg.get("n_layers",    12)
    max_seq_len = cfg.get("max_seq_len", sd["pos_emb.weight"].shape[0])

    model = GPT(
        vocab_size  = vocab_size,
        embed_dim   = embed_dim,
        num_heads   = num_heads,
        num_layers  = num_layers,
        max_seq_len = max_seq_len,
        dropout     = 0.0,   # 추론/파인튜닝 시 dropout 끔
    )

    def t(key: str) -> np.ndarray:
        return sd[key].float().numpy()

    # ── 임베딩 ────────────────────────────────────────────────────────────────
    model.token_emb.weight.data = t("tok_emb.weight")   # (V, E)
    model.pos_emb.weight.data   = t("pos_emb.weight")   # (T, E)

    # ── 최종 LayerNorm ────────────────────────────────────────────────────────
    model.ln_f.gamma.data = t("ln_f.weight")
    model.ln_f.beta.data  = t("ln_f.bias")

    # ── LM head (weight tying: tok_emb와 공유) ────────────────────────────────
    model.head.weight.data = t("tok_emb.weight")   # tied

    # ── 트랜스포머 블록 ────────────────────────────────────────────────────────
    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}"

        # LayerNorm 1
        block.ln1.gamma.data = t(f"{p}.ln1.weight")
        block.ln1.beta.data  = t(f"{p}.ln1.bias")

        # Attention: qkv (2304, 768) → Q / K / V 분리
        qkv_w = t(f"{p}.attn.qkv.weight")   # (3E, E)
        E = embed_dim
        block.attn.q_proj.weight.data = qkv_w[0*E : 1*E]   # (E, E)
        block.attn.k_proj.weight.data = qkv_w[1*E : 2*E]
        block.attn.v_proj.weight.data = qkv_w[2*E : 3*E]
        # q/k/v bias=False → 무시

        # out_proj
        block.attn.out_proj.weight.data = t(f"{p}.attn.proj.weight")
        block.attn.out_proj.bias.data   = t(f"{p}.attn.proj.bias")

        # LayerNorm 2
        block.ln2.gamma.data = t(f"{p}.ln2.weight")
        block.ln2.beta.data  = t(f"{p}.ln2.bias")

        # FFN: net = Sequential(Linear, GELU, Linear, Dropout)
        block.ff.net.layers[0].weight.data = t(f"{p}.ffn.fc1.weight")
        block.ff.net.layers[0].bias.data   = t(f"{p}.ffn.fc1.bias")
        block.ff.net.layers[2].weight.data = t(f"{p}.ffn.fc2.weight")
        block.ff.net.layers[2].bias.data   = t(f"{p}.ffn.fc2.bias")

    print(f"로드 완료: {ckpt_path}")
    print(f"  vocab={vocab_size}, embed={embed_dim}, heads={num_heads}, "
          f"layers={num_layers}, seq={max_seq_len}")
    print(f"  파라미터: {model.num_params():,}")
    return model


def verify_mapping(ckpt_path: str, n_tokens: int = 16) -> bool:
    """PyTorch 모델과 engine 모델의 출력이 일치하는지 검증."""
    import torch
    from .transformer import GPT

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # ── PyTorch forward ───────────────────────────────────────────────────────
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from train_a100 import GPT2
    cfg = ckpt.get("config", {})
    pt_model = GPT2(
        vocab_size  = cfg.get("vocab_size",  32000),
        d_model     = cfg.get("d_model",     768),
        n_heads     = cfg.get("n_heads",     12),
        n_layers    = cfg.get("n_layers",    12),
        max_seq_len = cfg.get("max_seq_len", 1024),
        dropout     = 0.0,
    )
    pt_model.load_state_dict(sd)
    pt_model.eval()

    ids = np.random.randint(0, cfg.get("vocab_size", 32000), (1, n_tokens))
    with torch.no_grad():
        pt_out, _ = pt_model(torch.tensor(ids))
    pt_np = pt_out.numpy()[0]   # (T, V)

    # ── engine forward ────────────────────────────────────────────────────────
    eng_model = load_from_pytorch_ckpt(ckpt_path)
    eng_model.eval()
    eng_out = eng_model.forward(ids).data[0]   # (T, V)

    max_diff = np.abs(pt_np - eng_out).max()
    print(f"\n검증 결과: max_diff = {max_diff:.6f}")
    ok = max_diff < 1e-3
    print("PASS ✓" if ok else f"FAIL ✗  (허용 오차 1e-3 초과)")
    return ok


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/checkpoints/ckpt_step200000.pt"
    model = load_from_pytorch_ckpt(path)
    verify_mapping(path)
