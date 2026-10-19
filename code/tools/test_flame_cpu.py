import sys
sys.path.insert(0, ".")
import torch

from train import Config, NanoGPT, apply_flame_preset
from moe import flame_param_counts, FLAME_SCALES


def make_cfg(twin=False, seed=0):
    cfg = Config(
        data_dir="data", out_dir="/tmp/flame_test_out",
        vocab_size=256, sequence_len=64, n_layer=2, n_embd=512, n_head=16,
        device_batch_size=2, total_batch_size=4,
        enable_nanogpt_ngram_ve=False, enable_bigram_ve=False,
        enable_trigram_ve=False,
        flame_profile="tiny", flame_dense_twin=twin,
    )
    apply_flame_preset(cfg)
    torch.manual_seed(seed)
    return cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_flame_moe_structure():
    cfg = make_cfg(twin=False)
    assert cfg.n_layer == 9 and cfg.n_embd == 256 and cfg.n_head == 16
    assert cfg.d_ff == 1368 and cfg.moe_d_e == 176
    assert cfg.norm_type == "rmsnorm" and cfg.pos_type == "rope"
    assert cfg.untied_head and cfg.ffn_type == "swiglu" and not cfg.bias
    model = NanoGPT(cfg)
    # layer 0 dense SwiGLU, layers 1..8 MoE
    assert hasattr(model.transformer.h[0].mlp, "w1"), "layer0 should be dense SwiGLU"
    assert not hasattr(model.transformer.h[0], "moe_arch")
    for i in range(1, 9):
        assert hasattr(model.transformer.h[i], "moe_arch"), f"layer {i} should be MoE"
        arch = model.transformer.h[i].moe_arch
        assert arch.d_e == 176 and arch.e_routed == 64 and arch.k_routed == 6
        assert arch.shared_units == 2
    # untied: wte and lm_head are different tensors
    assert model.lm_head.weight is not model.transformer.wte.weight
    # param count must match the exact ledger
    counts = flame_param_counts(cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.d_ff,
                                cfg.moe_d_e, cfg.vocab_size, dense_twin=False)
    actual = count_params(model)
    assert actual == counts["total"], f"total mismatch: {actual} vs {counts['total']}"
    print(f"[t1] flame tiny MoE structure + ledger exact: {actual:,} params")


def test_flame_dense_twin_structure():
    cfg = make_cfg(twin=True)
    model = NanoGPT(cfg)
    for i in range(1, 9):
        mlp = model.transformer.h[i].mlp
        assert hasattr(mlp, "w1"), f"twin layer {i} should be dense SwiGLU"
        assert mlp.w1.out_features == 8 * 176, f"twin width should be 8*d_e"
    counts = flame_param_counts(cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.d_ff,
                                cfg.moe_d_e, cfg.vocab_size, dense_twin=True)
    actual = count_params(model)
    assert actual == counts["total"] == counts["dense_twin_total"], \
        f"twin total mismatch: {actual} vs {counts['total']}"
    assert actual == counts["active"], "dense twin: total == active"
    print(f"[t2] flame tiny dense-twin structure + ledger exact: {actual:,} params")


def test_flame_forward_backward():
    cfg = make_cfg(twin=False)
    cfg.max_steps = 1
    model = NanoGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    tgt = torch.randint(0, cfg.vocab_size, (2, 64))
    loss = model(idx, targets=tgt)
    assert loss.requires_grad
    extra = model.moe_extra_loss()
    assert extra is not None and extra.requires_grad
    (loss + extra).backward()
    # router grads exist
    for i in range(1, 9):
        rg = model.transformer.h[i].mlp.router.weight.grad
        assert rg is not None and rg.abs().sum() > 0, f"router {i} no grad"
    # aux/z losses are logged separately (loss contract)
    assert model.moe_active
    print(f"[t3] flame forward/backward OK (loss {float(loss):.3f}, "
          f"extra {float(extra):.4f})")


def test_flame_twin_forward():
    cfg = make_cfg(twin=True)
    model = NanoGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    tgt = torch.randint(0, cfg.vocab_size, (2, 64))
    loss = model(idx, targets=tgt)
    loss.backward()
    assert model.moe_extra_loss() is None
    assert not model.moe_active
    print(f"[t4] flame dense-twin forward/backward OK (loss {float(loss):.3f})")


def test_dense_default_unchanged():
    # default Config (no flame) must build the legacy dense model untouched
    cfg = Config(data_dir="data", out_dir="/tmp/flame_test_out",
                 vocab_size=256, sequence_len=64, n_layer=2, n_embd=64, n_head=2,
                 device_batch_size=2, total_batch_size=4)
    assert cfg.flame_profile == "" and cfg.norm_type == "layernorm"
    assert cfg.pos_type == "learned" and cfg.ffn_type == "gelu" and cfg.d_ff == 0
    torch.manual_seed(0)
    model = NanoGPT(cfg)
    assert model.lm_head.weight is model.transformer.wte.weight  # tied
    assert hasattr(model.transformer, "wpe")
    for blk in model.transformer.h:
        assert type(blk.ln_1).__name__ == "LayerNorm"
        assert not hasattr(blk.attn, "rope_cos")
        assert hasattr(blk.mlp, "c_fc")
    idx = torch.randint(0, 256, (2, 64))
    tgt = torch.randint(0, 256, (2, 64))
    loss = model(idx, targets=tgt)
    loss.backward()
    print("[t5] legacy dense default path unchanged")


if __name__ == "__main__":
    test_flame_moe_structure()
    test_flame_dense_twin_structure()
    test_flame_forward_backward()
    test_flame_twin_forward()
    test_dense_default_unchanged()
    print("ALL FLAME TESTS PASSED")
