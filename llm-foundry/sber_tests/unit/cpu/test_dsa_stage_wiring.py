"""Wiring around DSA that has no kernel in it and used to be untested:

  * `freeze_main_for_warmup` (stage-1 trunk freeze) on model-level AND module-level parameter names;
  * the attention registry entry when a DSA kernel dependency is missing: a clear ImportError naming
    the dependency, not a bare KeyError at model build;
  * config-level and block-level `dense_attention_layers` validation are the SAME rules
    (one shared helper), so a config rejected by `GigaMixConfig` is rejected by the block and vice versa.
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

from llmfoundry.models.utils.dense_attention_layers import (
    normalize_dense_attention_layers,
    resolve_attention_type,
)


class _Toy(nn.Module):
    """Names mimic `model.layers.N.self_attn.indexer.*` plus a bare `indexer.*` sub-tree."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.layers:
            layer.self_attn = nn.Module()
            layer.self_attn.indexer = nn.Linear(4, 4)
            layer.self_attn.o_proj = nn.Linear(4, 4)
            layer.mlp = nn.Linear(4, 4)
        self.embed = nn.Embedding(8, 4)


@pytest.mark.unit
@pytest.mark.fast
def test_freeze_main_for_warmup_keeps_only_indexers_trainable():
    dsa_attention = pytest.importorskip("llmfoundry.models.layers.dsa_attention")
    model = _Toy()
    n_train, n_total = dsa_attention.freeze_main_for_warmup(model)
    trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert trainable == sorted(n for n, _ in model.named_parameters() if ".indexer." in n)
    assert n_train == 4 and n_total == len(list(model.parameters()))
    # a bare attention module (names start with "indexer.") is handled too
    attn = model.layers[0].self_attn
    for p in attn.parameters():
        p.requires_grad_(True)
    dsa_attention.freeze_main_for_warmup(attn)
    assert all(p.requires_grad for p in attn.indexer.parameters())
    assert not any(p.requires_grad for p in attn.o_proj.parameters())


@pytest.mark.unit
@pytest.mark.fast
def test_freeze_main_for_warmup_rejects_model_without_indexers():
    dsa_attention = pytest.importorskip("llmfoundry.models.layers.dsa_attention")
    with pytest.raises(ValueError, match="no DSA indexer parameters"):
        dsa_attention.freeze_main_for_warmup(nn.Linear(2, 2))


@pytest.mark.unit
def test_missing_dsa_dependency_gives_an_explicit_import_error():
    """Simulate `flash_mla` absent in a subprocess: the registry key must exist and explain itself."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["flash_mla"] = None          # makes `import flash_mla` raise ImportError
        from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY, DSA_IMPORT_ERROR
        assert "LlamaLatentSparseAttention" in ATTN_CLASS_REGISTRY
        assert DSA_IMPORT_ERROR is not None
        try:
            ATTN_CLASS_REGISTRY["LlamaLatentSparseAttention"](config=None, layer_idx=0)
        except ImportError as e:
            msg = str(e)
            assert "LlamaLatentSparseAttention" in msg and "flash_mla" in msg, msg
            print("OK")
        else:
            raise SystemExit("stub did not raise")
        """
    )
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr


class TestDenseAttentionLayersSharedRules:
    def _cfg(self, **kw):
        from types import SimpleNamespace
        base = dict(attention_type="LlamaLatentSparseAttention", num_hidden_layers=8,
                    linear_attention_type=None, full_attention_layers=None, dense_attention_layers=None)
        base.update(kw)
        return SimpleNamespace(**base)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_resolution_order(self):
        cfg = self._cfg(linear_attention_type="Qwen3NextGatedDeltaNet", full_attention_layers=[3, 7],
                        dense_attention_layers=[7])
        assert resolve_attention_type(cfg, 0) == "Qwen3NextGatedDeltaNet"
        assert resolve_attention_type(cfg, 3) == "LlamaLatentSparseAttention"
        assert resolve_attention_type(cfg, 7) == "LlamaLatentAttention"

    @pytest.mark.unit
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "bad, match",
        [
            ((1,), "must be a list"),
            ([True], "only integer"),
            ([1, 1], "unique"),
            ([9], "out-of-range"),
            ([2], "subset of full_attention_layers"),
        ],
    )
    def test_config_and_block_reject_the_same_inputs(self, bad, match):
        kw = dict(attention_type="LlamaLatentSparseAttention", num_hidden_layers=8,
                  linear_attention_type="Qwen3NextGatedDeltaNet", full_attention_layers=[3, 7])
        with pytest.raises(ValueError, match=match):
            normalize_dense_attention_layers(bad, **kw)
        with pytest.raises(ValueError, match=match):
            resolve_attention_type(self._cfg(dense_attention_layers=bad, **kw), 3)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gigamix_config_uses_the_shared_helper(self):
        from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
        with pytest.raises(ValueError, match="dense_attention_layers is supported only"):
            GigaMixConfig(num_hidden_layers=4, attention_type="LlamaLatentAttention",
                          dense_attention_layers=[1], kv_lora_rank=512, q_lora_rank=1536,
                          qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)
        cfg = GigaMixConfig(num_hidden_layers=4, attention_type="LlamaLatentSparseAttention",
                            dense_attention_layers=[1], kv_lora_rank=512, q_lora_rank=1536,
                            qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                            indexer_n_heads=64, indexer_head_dim=128, indexer_topk=128,
                            aux_kl_loss_koeff=0.01, dsa_stage=2)
        assert cfg.dense_attention_layers == [1]

@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.parametrize(
    "model_cfg, expected",
    [
        ({"aux_loss_free": False}, False),
        ({"aux_loss_free": True}, True),
        ({"aux_loss_free": True, "dsa_stage": 2}, True),
        ({"aux_loss_free": True, "dsa_stage": 1}, False),
    ],
)
def test_router_bias_update_is_off_while_the_trunk_is_frozen(model_cfg, expected):
    """UpdateBias asserts the layer is trainable, and DSA stage 1 freezes all of them."""
    train_py = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "scripts", "train"
    )
    src = textwrap.dedent(
        f"""
        import sys, warnings
        sys.path.insert(0, {train_py!r})
        from train import wants_router_bias_update
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = wants_router_bias_update({model_cfg!r})
        print(int(got), len(caught))
        """
    )
    env = dict(os.environ, WORLD_SIZE="1", NNODES="1", RANK="0", LOCAL_RANK="0",
               MASTER_ADDR="127.0.0.1", MASTER_PORT="29571")
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    got, warned = (int(x) for x in out.stdout.strip().splitlines()[-1].split())
    assert bool(got) is expected
    # the disabled case has to say so out loud, the others stay quiet
    assert warned == (1 if model_cfg.get("dsa_stage") == 1 and model_cfg["aux_loss_free"] else 0)
