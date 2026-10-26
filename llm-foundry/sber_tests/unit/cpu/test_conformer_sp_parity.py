"""CPU gloo parity tests for conformer exact-halo sequence parallelism.

Runs a small conformer (4 layers, d64, attn_chunk 8 -> halo 32, k5, dropout 0,
SDPA attention) at sp=WORLD_SIZE against a single-process reference on
identical inputs and checks forward outputs (all rows, pads included), input
grads and every weight grad after summing the per-rank partials. Cases: the
encoder alone on one stack, the component through its length-bucketing loop
(bucket-then-slice: one group shards, a short group falls back to replicated
when its core < halo, a short row is all padding on the trailing ranks), the
frozen-encoder trap (component in eval mode, top-level model training), a
batch that differs across the SP group (gate must replicate), and the SDPA
path on an all-padding row.

The model runs in float64 so the check is decisive: a halo defect stays
O(1e-2..1) while summation-order noise between rank partials and the reference
is ~1e-14. ``ConformerLayer.forward`` casts its output to bfloat16
unconditionally (a production choice for activation checkpointing that adds no
math); the test lifts that one cast in its own process, see ``_lift_bf16_cast``.

Launch:  torchrun --nproc_per_node=N sber_tests/unit/cpu/test_conformer_sp_parity.py --sp-size S
The pytest entries spawn that torchrun; (4, 2) runs two SP groups on
different data and proves the gate's all-reduces and the gather stay on the
SP subgroup. composer is replaced by the shared shim when it is not importable.
"""

import argparse
import copy
import inspect
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from sber_tests.test_utils.composer_dist_shim import install_composer_dist_shim
from sber_tests.test_utils.conformer_cpu import import_conformer_modules

pytestmark = [pytest.mark.unit]

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
FEAT_IN = 16
ATTN_CHUNK = 8
N_LAYERS = 4
SUBSAMPLING = 4
HALO = ATTN_CHUNK * N_LAYERS
FLOOR = HALO * SUBSAMPLING  # per-rank geometric floor in mel frames
CFG = dict(
    feat_in=FEAT_IN,
    n_layers=N_LAYERS,
    d_model=64,
    n_heads=4,
    ff_expansion_factor=2,
    attn_chunk_size=ATTN_CHUNK,
    conv_kernel_size=5,
    conv_norm_type="layer_norm",
    dropout=0.0,
    dropout_emb=0.0,
    dropout_att=0.0,
    subsampling="striding_conv1d",
    subsampling_factor=SUBSAMPLING,
    subsampling_conv_channels=64,
    max_position_seq_len=4096,
    attention_backend="sdpa",
    length_bucketing=True,
)


class _TorchWithoutBf16Cast:
    """Stand-in for the ``torch`` name inside the modeling module: ``bfloat16``
    reads as float64 so the layer-boundary cast is a no-op here; every other
    attribute is torch's own."""

    bfloat16 = torch.float64

    def __getattr__(self, name):
        return getattr(torch, name)


def _lift_bf16_cast(modeling):
    source = inspect.getsource(modeling)
    assert source.count("torch.bfloat16") == 1, (
        "the cast lift assumes the layer-boundary cast is the module's only bf16 reference"
    )
    modeling.torch = _TorchWithoutBf16Cast()


def _build_component(modeling, configuration, seed=0):
    torch.manual_seed(seed)
    model = modeling.ConformerForAudioLanguageModeling(
        configuration.ConformerConfig(**CFG)
    )
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    return model.double()


def _make_sp_group(sp_size):
    """Consecutive blocks of ``sp_size`` ranks; every rank creates every group
    (required by new_group) and keeps the one it belongs to."""
    world = dist.get_world_size()
    assert world % sp_size == 0, (world, sp_size)
    my_group = None
    for start in range(0, world, sp_size):
        ranks = list(range(start, start + sp_size))
        group = dist.new_group(ranks=ranks)
        if dist.get_rank() in ranks:
            my_group = group
    return my_group


def _mel_batch(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(
        len(lengths), max(lengths), FEAT_IN, generator=g, dtype=torch.float64
    )
    for i, length in enumerate(lengths):
        x[i, length:] = 0
    return x, torch.tensor(lengths, dtype=torch.long)


def _flat_bundle(lengths, seed=1):
    x, lens = _mel_batch(lengths, seed)
    flat = torch.cat([x[i, :length] for i, length in enumerate(lengths)])
    return flat, lens


def _loss_weights(shape, seeds):
    return sum(
        torch.randn(
            shape, generator=torch.Generator().manual_seed(s), dtype=torch.float64
        )
        for s in seeds
    )


class _GatherCounter:
    """Wraps the SP gather so a case can assert how many collectives ran."""

    def __init__(self, modeling):
        self.modeling, self.count = modeling, 0
        self._orig = modeling.gather_from_sequence_parallel_region

    def __enter__(self):
        def counted(x):
            self.count += 1
            return self._orig(x)

        self.modeling.gather_from_sequence_parallel_region = counted
        return self

    def __exit__(self, *exc):
        self.modeling.gather_from_sequence_parallel_region = self._orig


def _assert_gathers(counter, expected, group, what):
    counts = [torch.zeros(1, dtype=torch.long) for _ in range(group.size())]
    dist.all_gather(counts, torch.tensor([counter.count]), group=group)
    counts = [int(c) for c in counts]
    assert counts == [expected] * group.size(), (
        f"{what}: gathers per rank {counts} != {expected}"
    )


def _assert_close(name, a, b, stats, atol=1e-9, rtol=1e-7):
    diff = (a - b).abs().max().item()
    stats[name] = max(stats.get(name, 0.0), diff)
    torch.testing.assert_close(
        a, b, atol=atol, rtol=rtol, msg=f"{name}: max|diff|={diff}"
    )


def _run_encoder(model, x, lengths, is_sp_active, weight_seeds, train, halo_shrink=0):
    enc = copy.deepcopy(model)
    enc.train(train)
    if halo_shrink:
        orig_layout = enc._sp_layout

        def short_layout(seq_len):
            layout = orig_layout(seq_len)
            if layout is None:
                return None
            rank, size, padded, overlap = layout
            return rank, size, padded, overlap - halo_shrink

        enc._sp_layout = short_layout
    x = x.clone().requires_grad_(True)
    out, out_len = enc._forward_stack(x, lengths, is_sp_active=is_sp_active)
    grads = None
    if train:
        loss = (out * _loss_weights(out.shape, weight_seeds)).sum()
        loss.backward()
        grads = {"input": x.grad.detach().clone()}
        grads.update({n: p.grad.detach().clone() for n, p in enc.named_parameters()})
    return out.detach(), out_len, grads


def _run_component(model, flat, lengths, top_level_training, weight_seeds, train):
    comp = copy.deepcopy(model)
    comp.train(train)
    comp.set_top_level_training(top_level_training)
    flat = flat.clone().requires_grad_(True)
    out, out_len = comp([{"spectrograms": flat, "spectrogram_lengths": lengths}])
    grads = None
    if train:
        loss = (out * _loss_weights(out.shape, weight_seeds)).sum()
        loss.backward()
        grads = {"input": flat.grad.detach().clone()}
        grads.update({n: p.grad.detach().clone() for n, p in comp.named_parameters()})
    return out.detach(), out_len, grads


def _compare(name, sp, ref, group, stats):
    sp_out, sp_len, sp_grads = sp
    ref_out, ref_len, ref_grads = ref
    assert sp_out.shape == ref_out.shape, (name, sp_out.shape, ref_out.shape)
    assert sp_len.tolist() == ref_len.tolist(), name
    _assert_close(f"{name}/forward", sp_out, ref_out, stats)
    for key in ref_grads:
        summed = sp_grads[key].clone()
        dist.all_reduce(summed, group=group)
        _assert_close(f"{name}/grad/{key}", summed, ref_grads[key], stats)


def _encoder_case(modeling, model, lengths, world, rank, group, log):
    """Encoder alone (``_forward_stack`` with is_sp_active) on one stack."""
    x, lens = _mel_batch(lengths, seed=1 + dist.get_rank() // world)
    all_seeds = [1000 + r for r in range(world)]
    stats = {}
    with _GatherCounter(modeling) as c:
        ref = _run_encoder(model, x, lens, False, all_seeds, train=True)
    _assert_gathers(c, 0, group, "encoder/reference")
    # sp=world: this rank's loss shard is its own random weight; the shards sum
    # to the reference loss, so summed partial grads must equal the reference.
    with _GatherCounter(modeling) as c:
        sp = _run_encoder(model, x, lens, True, [1000 + rank], train=True)
    _assert_gathers(c, 1, group, "encoder/train")
    _compare("encoder", sp, ref, group, stats)

    ev_ref, _, _ = _run_encoder(model, x, lens, False, all_seeds, train=False)
    with _GatherCounter(modeling) as c:
        ev_sp, _, _ = _run_encoder(model, x, lens, True, all_seeds, train=False)
    _assert_gathers(c, 1, group, "encoder/eval")
    _assert_close("encoder/eval_forward", ev_sp, ev_ref, stats)

    # Mutation check: a halo one chunk short must be caught (the last layer's
    # conv leak lands inside the core), otherwise this test proves nothing.
    # The correct halo is exact up to summation order (~1e-14); the leak of a
    # one-chunk-short halo at this init is ~1e-11, so demand two orders of
    # magnitude over the correct-halo diff and a floor above float64 noise.
    bad_out, _, _ = _run_encoder(
        model, x, lens, True, all_seeds, train=False, halo_shrink=ATTN_CHUNK
    )
    bad_diff = (bad_out - ev_ref).abs().max().item()
    good_diff = stats["encoder/eval_forward"]
    assert bad_diff > 1e-12 and bad_diff > 100 * good_diff, (
        f"halo shrunk by one chunk went undetected (diff={bad_diff}, correct halo diff={good_diff})"
    )
    grad_keys = [k for k in stats if "/grad/" in k]
    log(
        f"[encoder] sp={world} lengths={lengths} forward={stats['encoder/forward']:.3e} "
        f"eval={stats['encoder/eval_forward']:.3e} grad_input={stats['encoder/grad/input']:.3e} "
        f"grad_weights_max={max(stats[k] for k in grad_keys if not k.endswith('/input')):.3e} "
        f"short_halo_diff={bad_diff:.3e}"
    )


def _component_case(
    modeling, model, name, lengths, expected_gathers, world, rank, group, log
):
    """Component forward (gate + length-bucketing loop) vs the same component
    with the top-level model in eval (gate off -> sp1 path, byte for byte
    the bucketing branch). Each SP group gets its own data (seed by group)."""
    flat, lens = _flat_bundle(lengths, seed=1 + dist.get_rank() // world)
    all_seeds = [1000 + r for r in range(world)]
    stats = {}
    with _GatherCounter(modeling) as c:
        ref = _run_component(model, flat, lens, False, all_seeds, train=True)
    _assert_gathers(c, 0, group, f"{name}/reference")
    with _GatherCounter(modeling) as c:
        sp = _run_component(model, flat, lens, True, [1000 + rank], train=True)
    _assert_gathers(c, expected_gathers, group, f"{name}/train")
    _compare(name, sp, ref, group, stats)

    # Frozen-encoder trap: the component is in eval mode (dropout off), the
    # top-level model is training -> still sharded, same numbers.
    with _GatherCounter(modeling) as c:
        frozen = _run_component(model, flat, lens, True, all_seeds, train=False)
    _assert_gathers(c, expected_gathers, group, f"{name}/frozen")
    _assert_close(f"{name}/frozen_forward", frozen[0], ref[0], stats)

    grad_keys = [k for k in stats if "/grad/" in k]
    log(
        f"[{name}] sp={world} lengths={lengths} gathers={expected_gathers} "
        f"forward={stats[f'{name}/forward']:.3e} frozen={stats[f'{name}/frozen_forward']:.3e} "
        f"grad_input={stats[f'{name}/grad/input']:.3e} "
        f"grad_weights_max={max(stats[k] for k in grad_keys if not k.endswith('/input')):.3e}"
    )


def _layout_checks(modeling, model, world, log):
    """Geometry the parity cases rely on, stated explicitly."""
    padded = modeling._sp_padded_len(75 * (world // 2), world, ATTN_CHUNK)
    assert padded % (world * ATTN_CHUNK) == 0
    for r in range(world):
        s, f = modeling._sp_slice_bounds(padded, r, world, HALO)
        assert s % ATTN_CHUNK == 0 and f % ATTN_CHUNK == 0 and 0 <= s < f <= padded
        core = padded // world
        expected = core + HALO * ((r > 0) + (r < world - 1))
        assert f - s == expected, (r, s, f, expected)
    # too short -> None (replicated), never an assert
    assert model._sp_layout(ATTN_CHUNK) is None
    log(f"[layout] sp={world} padded={padded} halo={HALO} ok")


def _sdpa_all_padding_check(modeling, log):
    """SDPA on rows whose every key is masked (an SP slice past a short clip's
    end): finite forward (bias-only output), finite backward, zero q/k/v grads."""
    torch.manual_seed(0)
    attn = modeling.ChunkRotaryPositionMultiHeadedAttention(
        n_head=4,
        n_feat=64,
        dropout=0.0,
        attn_chunk_size=ATTN_CHUNK,
        attention_backend="sdpa",
    ).double()
    x = torch.randn(2, 16, 64, dtype=torch.float64, requires_grad=True)
    pos = modeling.ChunkRotaryPositionalEmbedding(16, 5000, ATTN_CHUNK, 64)
    cos, sin = pos.create_pe(16, "cpu")
    pos_emb = (cos.double(), sin.double())
    pad_mask = torch.ones(2, 16, dtype=torch.bool)
    out = attn(x, x, x, pos_emb=pos_emb, pad_mask=pad_mask)
    torch.testing.assert_close(out, attn.linear_out.bias.detach().expand_as(out))
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    for lin in (attn.linear_q, attn.linear_k, attn.linear_v):
        assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
        assert lin.weight.grad.abs().sum() == 0
    # mixed: one valid row next to an all-padding row leaves the valid row intact.
    x2 = x.detach().clone().requires_grad_(True)
    mixed_mask = torch.ones(2, 16, dtype=torch.bool)
    mixed_mask[0, :11] = False
    out_mixed = attn(x2, x2, x2, pos_emb=pos_emb, pad_mask=mixed_mask)
    ref_single = attn(x2[:1], x2[:1], x2[:1], pos_emb=pos_emb, pad_mask=mixed_mask[:1])
    torch.testing.assert_close(out_mixed[:1], ref_single)
    out_mixed.sum().backward()
    assert torch.isfinite(x2.grad).all()
    log(
        "[sdpa_all_padding] finite forward/backward, bias-only output, zero q/k/v grads"
    )


def _worker(sp_size):
    dist.init_process_group("gloo")
    group = _make_sp_group(sp_size)
    # From here on `world`/`rank` are the SP group's size and this rank's
    # position in it; with sp_size < WORLD_SIZE the groups run side by side
    # on different data, so a collective that leaked to the world group
    # (wrong shapes or wrong peers) fails loudly.
    world, rank = group.size(), group.rank()
    torch.use_deterministic_algorithms(True)
    composer_dist = install_composer_dist_shim()
    modeling, configuration = import_conformer_modules()
    assert modeling.composer_dist is composer_dist, (
        "conformer did not pick up composer.utils.dist"
    )
    assert modeling.gather_from_sequence_parallel_region is not None
    composer_dist._SEQUENCE_PARALLEL_GROUP = group
    _lift_bf16_cast(modeling)
    model = _build_component(modeling, configuration)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    _layout_checks(modeling, model, world, log)
    _sdpa_all_padding_check(modeling, log)
    # Post-stem length = ceil(ceil(L/2)/2); halo 32 => core >= 32 => padded
    # stack >= 32*sp. Scale the mel lengths with sp so each case keeps at least
    # one SP-feasible group and clears the gate floor (FLOOR*sp mel);
    # sp=2: [300,296] -> T=75 -> padded 80 -> core 40.
    k = world // 2
    _encoder_case(modeling, model, [300 * k, 296 * k], world, rank, group, log)
    _component_case(
        modeling, model, "single_group", [300 * k, 296 * k], 1, world, rank, group, log
    )
    # groups (pad_waste_cap 0.5): [1000k,1000k,300] + [300,300]. The 300-row is
    # all padding on every rank whose slice starts past its 75 frames (sp2:
    # rank 1 starts at 96); the second group's core (40 @sp2, 24 @sp4, 16 @sp8)
    # is below the halo from sp4 on -> replicated there (1 gather), sharded at
    # sp2 (2 gathers).
    _component_case(
        modeling,
        model,
        "two_groups",
        [1000 * k, 1000 * k, 300, 300, 300],
        2 if world == 2 else 1,
        world,
        rank,
        group,
        log,
    )
    dist.barrier()
    log(f"PASS sp={world} world={dist.get_world_size()}")
    dist.destroy_process_group()


def _launch(nproc, sp_size):
    env = {
        k: v
        for k, v in os.environ.items()
        if not (
            k.startswith("TORCHELASTIC_")
            or k.startswith("PET_")
            or k
            in (
                "RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
            )
        )
    }
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        os.path.abspath(__file__),
        "--sp-size",
        str(sp_size),
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)


@pytest.mark.parametrize("nproc,sp_size", [(2, 2), (4, 2), (4, 4), (8, 8)])
def test_conformer_sp_parity(nproc, sp_size):
    res = _launch(nproc, sp_size)
    assert res.returncode == 0 and f"PASS sp={sp_size} world={nproc}" in res.stdout, (
        res.stdout + res.stderr
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-size", type=int, default=None)
    args = parser.parse_args()
    _worker(args.sp_size or int(os.environ["WORLD_SIZE"]))
