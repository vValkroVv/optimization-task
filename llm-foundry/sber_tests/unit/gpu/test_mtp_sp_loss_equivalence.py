"""Sequence parallelism must not change the MTP objective.

Each rank's losses normalize as `sp_size * local / global`, so the objective is
the mean over the SP group; this puts one fixed batch through both topologies
and compares. Why a fixed batch rather than two trainer runs: the trainer
would add optimizer state, data order and dropout to the comparison. Why the
documents sit on the slice boundary: that is where the document gate
(`same_doc_next_mask`, derived before the split and carried in the split
tuple beside `labels`) has to hand each rank the correct value -- a mask
derived after the split, or split in a different frame than the labels, would
disagree here and nowhere else. Why the groups are real rather than patched:
the attention reads the SP size in `__init__` to decide whether it
all-to-alls, and the loss normalization reads it to reduce.

The straddling batch is the control. It must NOT agree -- the per-slice tail
of `_build_shifted_labels` costs it supervision (labels are split before the
per-depth roll, so the last `h` positions of every slice lose head `h`'s
target) -- because a bound loose enough to accept it could not see an SP
error either.

Adapted from samoylovlesha/angelspec_eagle's test of the same name for the
independent-heads layout (no `mtp_recurrent`, no on-policy TTT): the objective
compared is the main CE, the MTP loss and its per-head vector.
"""

import os
from unittest.mock import patch

import composer.utils.dist as composer_dist
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from sber_tests.unit.gpu.sp_tests_base import cleanup_distributed

# 2 is what a CI device pair can hold; set MTP_SP_WORLD=8 on an eight-device
# job to run the production topology.
SP_WORLD = int(os.environ.get("MTP_SP_WORLD", "2"))
SEQ_LEN = 128
DEPTHS = 3
# Tolerances from the recurrent-layout original (aligned batch measured at
# 8.6e-08 against 6.0e-03 on the straddling one); re-measure if this layout's
# aligned agreement lands elsewhere and tighten accordingly.
RTOL, ATOL = 1e-4, 1e-6
ALIGNED = (SEQ_LEN // SP_WORLD,) * SP_WORLD
STRADDLING = (SEQ_LEN,)


def _init_sp(rank, world_size):
    """Build the real SP group. `composer.initialize_dist` sets up the SP group
    itself; initializing torch.distributed first would make it skip that."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="12371", NODE_RANK="0",
        RANK=str(rank), LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size), LOCAL_WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    composer_dist.initialize_dist(device="gpu", sp_size=world_size)
    assert composer_dist.get_sp_group_size() == world_size


def _sp_config(num_hidden_layers: int = 2):
    """Tiny stand-in for the mtp123 configs: dense MTP block on MLA with gated
    attention, `layernorm_type: pre_post`, `varlen_input: true`. Only the sizes
    shrink; `norm_type` is `LlamaRMSNorm` to avoid the `torch.compile`
    dependency of the zero-centered gated norm (every norm is position-local,
    so none can move information between positions)."""
    config = GigaMixConfig(
        # --- sizes: tiny ---
        vocab_size=256,
        hidden_size=256,
        intermediate_size=128,
        dense_intermediate_size=512,
        num_hidden_layers=num_hidden_layers,
        # The a2a scatters heads across the SP group, so the head count must
        # cover the world; production carries 32 against SP8.
        num_attention_heads=max(2, SP_WORLD),
        num_key_value_heads=max(2, SP_WORLD),
        max_position_embeddings=max(SEQ_LEN, 128),
        # --- MTP head: the independent-heads layout ---
        use_mtp=True,
        mtp_predictor_num=DEPTHS,
        mtp_block_type="dense",
        # --- attention: MLA, as production ---
        attention_type="LlamaLatentAttention",
        linear_attention_type=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=64,
        q_lora_rank=128,
        num_krope_heads=1,
        use_mla_scaling_factor=True,
        use_custom_rotary_kernel=False,
        gated_attention=True,
        attention_bias=False,
        attention_dropout=0.0,
        deterministic_attention=True,
        varlen_input=True,
        # --- norms / MLP ---
        norm_type="LlamaRMSNorm",
        layernorm_type="pre_post",
        fused_mlp=False,
        init_type="deepseek",
        # --- parallelism ---
        sp_split_type="equal",
        tp_size=1,
        use_cache=False,
        init_device="cpu",
    )
    config.first_k_dense_replace = config.num_hidden_layers
    return config


def _batch(device, doc_lengths):
    """A packed batch with unshifted labels, the position after each seam masked
    as `data/packing.py` masks it. Identical on every rank -- the model splits."""
    torch.manual_seed(7)
    input_ids = torch.randint(1, 256, (1, SEQ_LEN))
    labels = input_ids.clone()
    seams = [sum(doc_lengths[:i]) for i in range(1, len(doc_lengths))]
    labels[:, seams] = -100
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "position_ids": torch.cat(
            [torch.arange(length) for length in doc_lengths]
        ).unsqueeze(0).to(device),
    }


def _build(config):
    """The attention reads the SP size in `__init__` to decide whether it
    all-to-alls; the seed is what makes two builds the same weights."""
    torch.manual_seed(1234)
    return GigaMixForCausalLM(config).to("cuda").to(torch.bfloat16).train()


def _losses(model, batch):
    """`[total, ce, mtp, *per_head]`, read off the forward tuple's fixed tail.

    With no z-losses and no draft-loss kinds configured the tuple ends in
    `(loss_ce, loss_mtp, loss_mtp_per_head)`.
    """
    outputs = model(**batch, return_dict=False)
    total, loss_ce, loss_mtp, per_head = (
        outputs[0], outputs[-3], outputs[-2], outputs[-1]
    )
    assert per_head.shape == (DEPTHS,), (
        f"expected the per-head vector at [-1], got {tuple(per_head.shape)}"
    )
    return torch.stack(
        [total.float(), loss_ce.float(), loss_mtp.float(), *per_head.float()]
    )


def _sp_mean(local):
    """What the trainer logs: one rank's share, averaged over the SP group."""
    total = local.clone()
    dist.all_reduce(total, group=composer_dist.get_sp_group())
    return total / composer_dist.get_sp_group_size()


def _run(rank, world_size):
    _init_sp(rank, world_size)
    try:
        config = _sp_config()
        device = torch.device("cuda")
        aligned, straddling = _batch(device, ALIGNED), _batch(device, STRADDLING)

        model_sp = _build(config)
        aligned_sp = _sp_mean(_losses(model_sp, aligned))
        straddling_sp = _sp_mean(_losses(model_sp, straddling))
        if rank != 0:
            return

        with patch("composer.utils.dist.get_sp_group_size", return_value=1):
            model_sp1 = _build(config)
            aligned_sp1 = _losses(model_sp1, aligned)
            straddling_sp1 = _losses(model_sp1, straddling)

        torch.testing.assert_close(
            aligned_sp, aligned_sp1, rtol=RTOL, atol=ATOL,
            msg=lambda hint: (
                f"{hint}\n[total, ce, mtp, *per_head] at sp1 {aligned_sp1.tolist()}"
            ),
        )
        assert not torch.allclose(
            straddling_sp, straddling_sp1, rtol=RTOL, atol=ATOL
        ), (
            "the control agreed: "
            f"{straddling_sp1.tolist()} against {straddling_sp.tolist()}"
        )
    finally:
        cleanup_distributed()


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.requires_gpu
def test_sp_reports_the_same_objective_as_sp1():
    if torch.cuda.device_count() < SP_WORLD:
        pytest.skip(f"needs {SP_WORLD} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_run, args=(SP_WORLD,), nprocs=SP_WORLD)
