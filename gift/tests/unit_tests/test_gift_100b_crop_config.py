"""The development crop preserves the production widths and training controls."""

from tests.unit_tests.torch_free import run_blocked


def test_crop_resolves_and_validates_without_training_runtime():
    result = run_blocked(r"""
from math import sqrt
import re
from torchtitan.config.build import prepare_document
from torchtitan.config.loader import load_raw_document
from torchtitan.config.schema.models.common.attention.linear.gdn import GatedDeltaNetConfig
from torchtitan.config.schema.models.common.attention.softmax.mla import MLAConfig
from torchtitan.config.schema.models.common.nn_modules import LinearConfig
from torchtitan.config.schema.models.common.norms.gated_norm import GatedRMSNormConfig
from torchtitan.config.schema.protocols.module import iter_module_configs
from torchtitan.config.schema.models.gift_moe.const import MLAKVAInit, MLAKVBInit, RenormKeepStd
from torchtitan.config.schema.protocols.module import XavierNormal, XavierUniform, Zeros
from torchtitan.config.validation import validate_typed_config, ValidationContext, RuntimeInfo
x = prepare_document(load_raw_document('configs/experiments/gift_100b_crop8_ep8_fp8_compile_cute.yaml'))
c, m = x.trainer_config, x.model_config
report = validate_typed_config(c, m, ValidationContext(stage='in_container', runtime=RuntimeInfo(world_size=8, local_world_size=8, device_count=8)))
assert not report.errors, report.errors
assert m.dim == 4096 and m.vocab_size == 128256
assert len(m.layers) == 8 and m.num_mtp == 1
assert [type(b.attention) for b in m.layers] == [GatedDeltaNetConfig] * 3 + [MLAConfig] + [GatedDeltaNetConfig] * 3 + [MLAConfig]
assert m.layers[0].feed_forward.hidden_dim == 10944
assert isinstance(m.lm_head.param_init['weight'], XavierNormal)
assert isinstance(m.tok_embeddings.param_init['weight'], XavierNormal)
assert isinstance(m.norm.param_init['weight'], Zeros)
assert m.norm.hidden_z_loss.coef == 1e-7 and m.norm.hidden_z_loss.on == 'input'
for i, b in enumerate(m.layers):
    for n in (b.attention_norm, b.ffn_norm, b.post_attn_norm, b.post_ffn_norm):
        assert n.norm_backend == 'cute' and n.zero_centered
        assert n.hidden_z_loss is None
        assert isinstance(n.param_init['gate_up_weight'], XavierNormal)
    a = b.attention
    if isinstance(a, GatedDeltaNetConfig):
        assert (a.num_k_heads, a.num_v_heads) == (16, 32)
        assert isinstance(a.qkv.conv.param_init['weight'], XavierUniform)
    else:
        assert (a.q_lora_rank, a.kv_lora_rank, a.n_heads) == (0, 512, 32)
        assert isinstance(a.wkv_a.param_init['weight'], MLAKVAInit)
        assert isinstance(a.wkv_b.param_init['weight'], MLAKVBInit)
    if i:
        e, r = b.moe.experts, b.moe.router
        assert (e.hidden_dim, e.num_experts, r.top_k) == (1536, 128, 8)
        assert e.param_init['w13_E2FD'].std == sqrt(2 / (4096 + 1536))
        assert isinstance(r.gate.param_init['weight'], RenormKeepStd)
        assert isinstance(r.gate.param_init['weight'].init, XavierNormal)
        assert b.moe.shared_experts.hidden_dim == 1536
        assert b.moe.load_balance_strategy == 'adaptive'
        d = e.token_dispatcher
        assert d.pad_multiple == 128 and d.non_blocking_capacity_factor is None
        assert d.token_drop.expert_capacity_tokens == 50000
        assert d.token_drop.drop_by_group_capacity
        assert d.token_drop.source_capacity(num_tokens=8192, top_k=8, num_experts=128, ep_size=8) == 6250
assert (c.training.local_batch_size, c.training.seq_len, c.training.global_batch_size) == (2, 4096, 512)
assert c.training.global_batch_size // (8 * c.training.local_batch_size) == 32
assert c.training.steps == c.lr_scheduler.total_steps == 65000
assert c.debug.seed == 2048
assert c.lr_scheduler.warmup_steps == 2000
assert c.lr_scheduler.decay_ratio is None
assert c.lr_scheduler.decay_type == 'cosine'
assert c.lr_scheduler.min_lr_factor == 0.0098
assert (c.loss.num_chunks, c.loss.num_chunks_on_monitoring_steps) == (2, None)
assert c.loss.loss_weights == [0.3]
assert c.parallelism.data_parallel_shard_degree == c.parallelism.expert_parallel_degree == 8
assert c.parallelism.fsdp_backend == 'gfsdp2' and c.parallelism.gfsdp_match_fsdp2_init
assert c.compile.enable and c.compile.components == ['model', 'loss']
assert c.fp8.sparse_use_grouped_gemm and c.fp8.sparse_deep_gemm_num_sms == 116
assert c.fp8.dense_include_linear_layers == r'.*\.(w1|w2|w3|wo|o_proj|qkv_proj|z_proj)$'
modules = dict(iter_module_configs(m))
gated_norms = {name: module for name, module in modules.items() if isinstance(module, GatedRMSNormConfig)}
assert len(gated_norms) == 43
assert 'layers.3.attention.kv_norm' in gated_norms
assert 'mtp.layers.0.token_norm' in gated_norms
assert 'mtp.layers.0.block.attention.kv_norm' in gated_norms
for name, norm in gated_norms.items():
    assert norm.norm_backend == 'cute' and norm.zero_centered, name
    assert norm.gate_rank == 16 and norm.normalized_shape % 128 == 0, name
pattern = re.compile(c.fp8.dense_include_linear_layers)
selected = {name for name, module in modules.items() if isinstance(module, LinearConfig) and pattern.search(name)}
expected = {
    f'layers.{i}.{projection}'
    for i in (0, 1, 2, 4, 5, 6)
    for projection in ('attention.qkv.qkv_proj', 'attention.o_proj', 'attention.z_proj')
}
expected |= {'layers.3.attention.wo', 'layers.7.attention.wo', 'mtp.layers.0.block.attention.wo'}
expected |= {f'layers.0.feed_forward.w{i}' for i in (1, 2, 3)}
expected |= {f'layers.{layer}.moe.shared_experts.w{i}' for layer in range(1, 8) for i in (1, 2, 3)}
expected |= {f'mtp.layers.0.block.feed_forward.w{i}' for i in (1, 2, 3)}
assert selected == expected, (selected - expected, expected - selected)
assert not any(pattern.search(name) for name in ('lm_head', 'layers.1.moe.router.gate', 'layers.0.attention.ba_proj', 'layers.3.attention.wkv_a'))
assert c.callbacks['ema'].start_step == 10000 and c.callbacks['ema'].decay == .99
assert set(c.callbacks) == {'ema'}
assert not c.checkpoint.enable and not c.profiler.enable_profiling
assert not c.profiler.enable_memory_snapshot and not c.training.enable_cpu_offload
assert c.dataloader.dataset.streams[0].local.endswith('/artifacts/raw/mds4096')
print('crop config verified')
""")
    assert result.stdout.strip() == "crop config verified"
