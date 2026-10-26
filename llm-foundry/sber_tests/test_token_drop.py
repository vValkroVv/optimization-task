import os
import re
import torch

import warnings

import numpy as np
import random
from typing import Optional, Literal

import yaml

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig

from llmfoundry.models.layers.moe.dispatchers import TokenDispatcher

from composer.utils import reproducibility


os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["NUMEXPR_MAX_THREADS"] = "16"
ITER_NUM = 1

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)

def configure_deterministic_mode():
    """Configure PyTorch deterministic mode.

    .. note::

        When using the :class:`.Trainer`, you can use the ``deterministic_mode`` flag
        instead of invoking this function directly.
        For example:

        .. testsetup::

            import warnings

            warnings.filterwarnings(action="ignore", message="Deterministic mode is activated.")

        .. doctest::

            >>> trainer = Trainer(deterministic_mode=True)

        .. testcleanup::

            warnings.resetwarnings()

        However, to configure deterministic mode for operations before the trainer is initialized, manually invoke this
        function at the beginning of your training script.

    .. note::

        When training on a GPU, this function must be invoked before any CUDA operations.

    .. note::

        Deterministic mode degrades performance. Do not use outside of testing and debugging.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    # and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # warnings.warn('Deterministic mode is activated. This will negatively impact performance.', category=UserWarning)
    from torch._inductor import config
    config.fallback_random = True


def generate_first_topk_input(
    num_tokens: int = 4096,
    num_experts: int = 8,
    experts_per_token: int = 2,
    torch_dtype = torch.float32,
):

    routing_map = torch.zeros(size=(num_tokens, num_experts), dtype=torch_dtype, device="cuda:0")
    routing_map[:, :experts_per_token] = 1

    probs = routing_map.abs() / routing_map.abs().sum(1, keepdim=True)

    return routing_map, probs

def generate_uniform_input(
    num_tokens: int = 4096,
    num_experts: int = 8,
    experts_per_token: int = 2,
    torch_dtype = torch.float32,
):

    pattern = torch.cat([torch.ones(experts_per_token, dtype=torch_dtype, device="cuda:0"), torch.zeros(num_experts - experts_per_token, dtype=torch_dtype, device="cuda:0")])
    shifts = (torch.arange(num_tokens) * experts_per_token) % num_experts
    routing_map = torch.stack([pattern.roll((s.item(),)) for s in shifts])

    probs = routing_map.abs() / routing_map.abs().sum(1, keepdim=True)

    return routing_map, probs

def init_token_dispatcher(
        dispatcher_balancing_strategy: Literal['capacity_factor', 'fixed_capacity']="fixed_capacity",
        expert_capacity_tokens:Optional[int]=None, # очень грубая оценка приблизительного значения при ep=1
        expert_capacity_factor:Optional[float]=None,
        drop_by_group_capacity:bool=True,
        num_experts_per_tok:int=2,
        num_routed_experts:int=8,
        peak_capacity_microbatchsize:int=1,
        max_position_embeddings:int=4096

    ):

    config = GigaMixConfig(
        dispatcher_balancing_strategy=dispatcher_balancing_strategy,
        expert_capacity_tokens=expert_capacity_tokens,
        expert_capacity_factor=expert_capacity_factor,
        drop_by_group_capacity=drop_by_group_capacity,
        num_experts_per_tok=num_experts_per_tok,
        num_routed_experts=num_routed_experts,
        peak_capacity_microbatchsize=peak_capacity_microbatchsize,
        max_position_embeddings=max_position_embeddings
    )

    token_dispatcher = TokenDispatcher(config)

    return token_dispatcher

def test_max_token_drop_fixed_capacity():

    configure_deterministic_mode()

    dispatcher_balancing_strategy="fixed_capacity"
    expert_capacity_tokens=200
    drop_by_group_capacity=False

    num_experts_per_tok=2
    num_routed_experts=8


    mbs = 1
    seq_len = 4096
    num_tokens = seq_len * mbs

    torch_dtype = torch.float32

    routing_map, probs = generate_first_topk_input(
                                num_tokens=num_tokens,
                                num_experts=num_routed_experts,
                                experts_per_token=num_experts_per_tok,
                                torch_dtype=torch_dtype,
                            )

    token_dispatcher = init_token_dispatcher(
        dispatcher_balancing_strategy=dispatcher_balancing_strategy,
        expert_capacity_tokens=expert_capacity_tokens,
        drop_by_group_capacity=drop_by_group_capacity,
        num_experts_per_tok=num_experts_per_tok,
        num_routed_experts=num_routed_experts,
        max_position_embeddings=seq_len,
        peak_capacity_microbatchsize=mbs
    )

    routing_map_new, probs_new = token_dispatcher._drop_token_by_capacity_factor(
        routing_map=routing_map,
        probs=probs
    )

    assert int(routing_map_new.sum(0).max()) == expert_capacity_tokens, "tokens should be dropped to capacity threshold level"
    assert int(routing_map_new.sum(0).min()) == 0, "there shold be empty experts"
    assert int(probs_new.sum(0).max()) == int(expert_capacity_tokens*(1/num_experts_per_tok)), "probs of dropped tokens should be dropped"


def test_max_token_drop_relative_capacity():

    configure_deterministic_mode()

    dispatcher_balancing_strategy="capacity_factor"
    expert_capacity_factor=2
    drop_by_group_capacity=False

    num_experts_per_tok=2
    num_routed_experts=8


    mbs = 1
    seq_len = 4096
    num_tokens = seq_len * mbs

    torch_dtype = torch.float32

    routing_map, probs = generate_first_topk_input(
                                num_tokens=num_tokens,
                                num_experts=num_routed_experts,
                                experts_per_token=num_experts_per_tok,
                                torch_dtype=torch_dtype,
                            )

    token_dispatcher = init_token_dispatcher(
        dispatcher_balancing_strategy=dispatcher_balancing_strategy,
        expert_capacity_factor=expert_capacity_factor,
        drop_by_group_capacity=drop_by_group_capacity,
        num_experts_per_tok=num_experts_per_tok,
        num_routed_experts=num_routed_experts,
        max_position_embeddings=seq_len,
        peak_capacity_microbatchsize=mbs
    )

    routing_map_new, probs_new = token_dispatcher._drop_token_by_capacity_factor(
        routing_map=routing_map,
        probs=probs
    )

    target_max_tokens = int(expert_capacity_factor*(num_tokens*num_experts_per_tok//num_routed_experts))

    assert int(routing_map_new.sum(0).max()) == target_max_tokens, "tokens should be dropped to capacity threshold level"
    assert int(routing_map_new.sum(0).min()) == 0, "there shold be empty experts"
    assert int(probs_new.sum(0).max()) == int(target_max_tokens*(1/num_experts_per_tok)), "probs of dropped tokens should be dropped"


def test_max_token_drop_peak_capacity():

    configure_deterministic_mode()

    dispatcher_balancing_strategy="peak_capacity_factor"
    expert_capacity_factor=2
    drop_by_group_capacity=False


    num_experts_per_tok=2
    num_routed_experts=8


    mbs = 1
    seq_len = 4096
    num_tokens = seq_len * mbs

    torch_dtype = torch.float32

    routing_map, probs = generate_first_topk_input(
                                num_tokens=num_tokens,
                                num_experts=num_routed_experts,
                                experts_per_token=num_experts_per_tok,
                                torch_dtype=torch_dtype,
                            )

    token_dispatcher = init_token_dispatcher(
        dispatcher_balancing_strategy=dispatcher_balancing_strategy,
        expert_capacity_factor=expert_capacity_factor,
        drop_by_group_capacity=drop_by_group_capacity,
        num_experts_per_tok=num_experts_per_tok,
        num_routed_experts=num_routed_experts,
        max_position_embeddings=seq_len,
        peak_capacity_microbatchsize=mbs
    )

    routing_map_new, probs_new = token_dispatcher._drop_token_by_capacity_factor(
        routing_map=routing_map,
        probs=probs
    )

    target_max_tokens = int(expert_capacity_factor*(num_tokens*num_experts_per_tok//num_routed_experts))

    assert int(routing_map_new.sum(0).max()) == target_max_tokens, "tokens should be dropped to capacity threshold level"
    assert int(routing_map_new.sum(0).min()) == 0, "there shold be empty experts"
    assert int(probs_new.sum(0).max()) == int(target_max_tokens*(1/num_experts_per_tok)), "probs of dropped tokens should be dropped"
