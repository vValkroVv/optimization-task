import os
import re
import torch

import warnings

import numpy as np
import random
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

import yaml

from llmfoundry.models.layers.attention import LlamaPackedAttention, LlamaLatentAttention
from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.layers.moe.experts import GroupedLlamaMLP
from llmfoundry.models.layers.blocks import LlamaMixDecoderLayer

from modeling_deepseek_corrected import DeepseekV2Attention, DeepseekV2FlashAttention2, DeepseekV2MoE

from modeling_deepseek_corrected import flash_attn_func as flash_attn_func_deepseek
from llmfoundry.models.layers.attention import flash_attn_func as flash_attn_func_gigamix

from configuration_deepseek import DeepseekV2Config

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


class CustomDeepSeekMoe(DeepseekV2MoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def custom_forward(
        self,
        sorted_tokens,
        tokens_per_expert,
        topk_weight
    ):

        # cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        # cnts.scatter_(1, topk_ids, 1)
        # tokens_per_expert = cnts.sum(dim=0)
        # idxs = topk_ids.view(-1).argsort()
        # sorted_tokens = x[idxs // topk_ids.shape[1]]
        # sorted_tokens_shape = sorted_tokens.shape
        # tokens_per_expert = tokens_per_expert.cpu().numpy()
        # print(f"шейпы внутри {tokens_per_expert.shape}, ")
        # idxs = topk_ids.view(-1).argsort()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            print(f"tokens idxs {start_idx}, {end_idx}")
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        print(f"шейпы внутри outs {len(outputs)}, {outputs[0].shape}, {topk_weight.shape}")
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        return outs * topk_weight.unsqueeze(dim=-1)

        # new_x = torch.empty_like(outs)
        # new_x[idxs] = outs
        # final_out = (
        #     new_x.view(*topk_ids.shape, -1)
        #     .type(topk_weight.dtype)
        #     .mul_(topk_weight.unsqueeze(dim=-1))
        #     # .mul_(topk_weight)
        #     # .sum(dim=1)
        #     # .type(new_x.dtype)
        # )
        # return final_out


# def convert_weights_deepseek_to_gigamix(
#     deepseek_model,
#     gigamix_model,
#     deepseek_config
# ):

#     config = deepseek_model.config
#     INTERMIDEATE_SIZE = config.moe_intermediate_size

#     target_moe_upgate = torch.zeros(config.hidden_size, config.n_routed_experts * config.moe_intermediate_size*2)
#     target_moe_down = torch.zeros(config.n_routed_experts * config.moe_intermediate_size, config.hidden_size)

#     with torch.no_grad():
#         for layer_name, layer_weight in deepseek_model.named_parameters():
#             # print(layer_name)
#             if "experts" in layer_name and ".down_proj" in layer_name and not ("shared" in layer_name):
#                 match_numbers = re.search(r'experts\.(\d+)\.down_proj', layer_name)
#                 if match_numbers:
#                     expert_number = int(match_numbers.group(1))

#                 target_moe_down[expert_number*INTERMIDEATE_SIZE : (expert_number+1)*INTERMIDEATE_SIZE, :] = layer_weight.clone().t()
#                 continue

#             if "experts" in layer_name and ".up_proj" in layer_name and not ("shared" in layer_name):
#                 match_numbers = re.search(r'experts\.(\d+)\.up_proj', layer_name)
#                 if match_numbers:
#                     expert_number = int(match_numbers.group(1))

#                 target_moe_upgate[:, (2*expert_number+1)*INTERMIDEATE_SIZE : 2*(expert_number+1)*INTERMIDEATE_SIZE] = layer_weight.clone().t()
#                 continue

#             if "experts" in layer_name and ".gate_proj" in layer_name and not ("shared" in layer_name):
#                 match_numbers = re.search(r'experts\.(\d+)\.gate_proj', layer_name)
#                 if match_numbers:
#                     expert_number = int(match_numbers.group(1))

#                 target_moe_upgate[:, 2*expert_number*INTERMIDEATE_SIZE : (2*expert_number+1)*INTERMIDEATE_SIZE] = layer_weight.clone().t()
#                 continue

#         gigamix_model.weight1 = torch.nn.Parameter(target_moe_upgate.t().contiguous())
#         gigamix_model.weight2 = torch.nn.Parameter(target_moe_down.t().contiguous())



def convert_weights_deepseek_to_gigamix(
    deepseek_model,
    gigamix_model,
    deepseek_config
):

    config = deepseek_model.config
    INTERMIDEATE_SIZE = config.moe_intermediate_size

    target_moe_upgate = torch.zeros(config.n_routed_experts, config.hidden_size, 2, config.moe_intermediate_size).transpose(1,2)
    target_moe_down = torch.zeros(config.n_routed_experts * config.moe_intermediate_size, config.hidden_size)

    with torch.no_grad():
        for layer_name, layer_weight in deepseek_model.named_parameters():
            if "experts" in layer_name and ".down_proj" in layer_name and not ("shared" in layer_name):
                match_numbers = re.search(r'experts\.(\d+)\.down_proj', layer_name)
                if match_numbers:
                    expert_number = int(match_numbers.group(1))

                target_moe_down[expert_number*INTERMIDEATE_SIZE : (expert_number+1)*INTERMIDEATE_SIZE, :] = layer_weight.clone().t()
                # target_moe_down[i * ffn_dim:(i + 1) * ffn_dim, :]
                continue

            if "experts" in layer_name and ".up_proj" in layer_name and not ("shared" in layer_name):
                match_numbers = re.search(r'experts\.(\d+)\.up_proj', layer_name)
                if match_numbers:
                    expert_number = int(match_numbers.group(1))

                up_w = layer_weight.clone().t()
                target_moe_upgate[expert_number, 1, :, :] = up_w
                continue

            if "experts" in layer_name and ".gate_proj" in layer_name and not ("shared" in layer_name):
                match_numbers = re.search(r'experts\.(\d+)\.gate_proj', layer_name)
                if match_numbers:
                    expert_number = int(match_numbers.group(1))

                gate_w = layer_weight.clone().t()
                target_moe_upgate[expert_number, 0, :, :] = gate_w
                continue

        gigamix_model.weight1 = torch.nn.Parameter(target_moe_upgate.transpose(1,2).view(config.hidden_size, config.n_routed_experts*2*config.moe_intermediate_size).contiguous())
        gigamix_model.weight2 = torch.nn.Parameter(target_moe_down.contiguous())



def convert_weights_gigamix_to_deepseek(
    deepseek_model,
    gigamix_model,
    deepseek_config
):

    config = deepseek_model.config
    N_EXPERTS = config.n_routed_experts


    with torch.no_grad():

        target_moe_upgate = gigamix_model.weight1.clone()
        target_moe_down = gigamix_model.weight2.clone()

        experts_ffn_dim, hidden_dim = target_moe_down.shape
        ffn_dim = experts_ffn_dim // N_EXPERTS

        assert ffn_dim != hidden_dim
        for i in range(N_EXPERTS):
            weights_ = target_moe_down[i * ffn_dim:(i + 1) * ffn_dim, :].t().clone().contiguous()
            deepseek_model.experts[i].down_proj.weight = torch.nn.Parameter(weights_)


        hidden_dim, total_ffn = target_moe_upgate.shape
        D = total_ffn // (2 * N_EXPERTS)
        w1 = target_moe_upgate.view(N_EXPERTS, hidden_dim, 2 * D)
        for i in range(N_EXPERTS):
            gate_part, up_part = w1[i].chunk(2, dim=-1)
            gate_w = gate_part.t().contiguous()
            up_w = up_part.t().contiguous()

            deepseek_model.experts[i].gate_proj.weight = torch.nn.Parameter(gate_w)
            deepseek_model.experts[i].up_proj.weight = torch.nn.Parameter(up_w)



def init_weights(
    module,
    init_type: str = "const",
    factor: float = None
):
    if init_type == "random":
        print("after_random init")
        for name, weight in module.named_parameters():
            with torch.no_grad():
                weight.normal_(mean=0.0, std=3e-2)
            # print(f"{name} -- {weight.shape} -- {weight.max()}")

    elif init_type == "const":
        print("after_const init")
        assert factor is not None
        for name, weight in module.named_parameters():
            with torch.no_grad():
                weight.fill_(factor)
            # print(f"{name} -- {weight.shape} -- {weight.max()}")
    else:
        raise NameError("'init_type' should be in ['random', 'const']")


def init_deepseek_moe():

    config_path = "/home/jovyan/sozhbannikov/convertation_exps/checkpoints/hf/DeepSeek-V2-Lite/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config_dict["max_position_embeddings"] = 4096
    config_dict["torch_dtype"] = "bfloat16"
    # config_dict["torch_dtype"] = "float32"
    config_dict["rope_scaling"] = None
    config = DeepseekV2Config(**config_dict)
    # moe = DeepseekV2Attention(config)
    moe = CustomDeepSeekMoe(config)
    moe.eval()

    # print("deepseek moe")
    # for name, weight in moe.named_parameters():
    #     print(f"{name} -- {weight.shape} -- {weight.max()}")

    return moe, config


def init_gigamix_moe():
    config_path = "/home/jovyan/sozhbannikov/llm-foundry/sber_configs/deepseek_lite_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # print(config_dict["model"]["config_overrides"])
    config = GigaMixConfig(**config_dict["model"]["config_overrides"])
    moe = GroupedLlamaMLP(config, ep_size=1)
    moe.eval()


    # TOKENS_PER_EXPERT = config['num_experts_per_tok']

    # print("gigamix moe")
    # for name, weight in moe.named_parameters():
    #     print(f"{name} -- {weight.shape} -- {weight.max()}")

    return moe, config

def generate_input(
    size: Optional[tuple] = None,
    torch_dtype=torch.float32
):
    size = size or (1, 4096, 2048)
    input_x = torch.empty(size=size, dtype=torch_dtype).normal_(mean=0.0, std=3e-2)
    # input_x = torch.ones(size=size, dtype=torch_dtype) * 6e-3
    return input_x

def check_result(
    logits_1,
    logits_2,
    atol=1e-7
):

    print(f"""
    Output shape: {logits_1.shape}, {logits_2.shape},
    diff {(logits_1-logits_2).abs().max()}
    diff mean {(logits_1-logits_2).abs().mean()}
    mean el magnitude {logits_1.abs().mean()}, {logits_2.abs().mean()},
    max el {logits_1.max()} {logits_2.max()},
    min el {logits_1.min()} {logits_2.min()},
    """)

    print(logits_1[55, :30])
    print(logits_2[55, :30])

    print("-----------")

    print(logits_1[4, :30])
    print(logits_2[4, :30])

    print("-----------")

    print(f"Средние {logits_1.mean()} {logits_2.mean()}")
    print(f"дисперсии {logits_1.std()} {logits_2.std()}")

    print(f"типы {logits_1.type()}, {logits_2.type()}")
    assert torch.allclose(logits_1, logits_2, atol=atol)


def prepare_input(
    input_x,
    deepseek_moe,
    torch_dtype
):

    # deepseek_config = deepseek_moe.config

    # bsz, seq_len, hid_dim = input_x.shape
    # idx_topk, probs_topk, _ = deepseek_moe.gate(input_x)

    # input_tokens = bsz * seq_len
    # input_x = input_x.view(input_tokens, hid_dim).to(torch_dtype)
    # idx_topk_dense = torch.zeros((input_tokens, deepseek_config.n_routed_experts), device="cuda:0", dtype=torch.int64)
    # idx_topk_dense[idx_topk] = 1

    # print("idx_topk_dense[:4, :]")

    # probs_topk_dense = torch.zeros((input_tokens, deepseek_config.n_routed_experts), device="cuda:0", dtype=torch_dtype)
    # probs_topk_dense[idx_topk] = probs_topk.to(torch_dtype)

    # print(probs_topk_dense[:4, :])

    # probs_topk_repeated = probs_topk.flatten().to(torch_dtype)
    # input_x_repeated = input_x.repeat_interleave(deepseek_config.num_experts_per_tok, dim=0)

    # tokens_per_expert = idx_topk_dense.sum(axis = 0).to(torch.int64).cpu()

    deepseek_config = deepseek_moe.config
    bsz, seq_len, hid_dim = input_x.shape

    input_tokens = bsz * seq_len
    input_x = input_x.view(input_tokens, hid_dim).to(torch_dtype)

    random_indices = torch.argsort(torch.rand((input_tokens, deepseek_config.n_routed_experts)), dim=1)
    idx_topk = torch.arange(deepseek_config.n_routed_experts).expand((input_tokens, deepseek_config.n_routed_experts))
    idx_topk = torch.gather(idx_topk, 1, random_indices)
    idx_topk = idx_topk[:, :deepseek_config.num_experts_per_tok]

    probs = torch.randn((input_tokens, deepseek_config.n_routed_experts)).abs()
    probs = probs[:, :deepseek_config.num_experts_per_tok] / probs.sum(axis=1, keepdim=True)

    input_x_repeated = input_x.repeat_interleave(deepseek_config.num_experts_per_tok, dim=0)
    idx_topk_repeated = idx_topk.flatten()
    probs_topk_repeated = probs.flatten()

    routing_sort = idx_topk_repeated.argsort()
    print(f"routing sort0 {routing_sort.shape}, {routing_sort}")
    print(f"idx_topk {idx_topk.shape}, {idx_topk_repeated.shape}, {input_x_repeated.shape}")

    input_x_repeated = input_x_repeated[routing_sort]
    probs_topk_repeated = probs_topk_repeated[routing_sort]
    tokens_per_expert = torch.bincount(idx_topk_repeated, minlength=deepseek_config.n_routed_experts)

    print(f"routing sort {routing_sort}")
    print(f"tok_per_exp {tokens_per_expert}")
    print(f"probs_topk_repeated {probs_topk_repeated}")
    print(f"shapes {input_x_repeated.shape}, {probs_topk_repeated.shape}, {tokens_per_expert.shape}")


    return input_x.to(device="cuda:0").to(dtype=torch_dtype), input_x_repeated.to(device="cuda:0").to(dtype=torch_dtype), probs_topk_repeated.to(device="cuda:0").to(dtype=torch_dtype), tokens_per_expert.to(dtype=torch.int64)

# def test_constant_forward_moe():

#     # torch_dtype = torch.float32
#     torch_dtype = torch.bfloat16
#     deepseek_moe, deepseek_config = init_deepseek_moe()
#     gigamix_moe, gigamix_config = init_gigamix_moe()
#     factor = 0.01

#     print(deepseek_config)
#     print(gigamix_config)

#     init_weights(
#         module = deepseek_moe,
#         init_type = "const",
#         factor = factor
#     )

#     init_weights(
#         module = gigamix_moe,
#         init_type = "const",
#         factor = factor
#     )


#     reproducibility.seed_all(42)
#     deepseek_moe.to("cuda:0")
#     deepseek_moe = deepseek_moe.to(torch_dtype)
#     reproducibility.seed_all(42)

#     reproducibility.seed_all(42)
#     gigamix_moe.to("cuda:0")
#     gigamix_moe = gigamix_moe.to(torch_dtype)
#     reproducibility.seed_all(42)

#     for _ in range(ITER_NUM):
#         input_x = generate_input(
#             (1, 4096, 2048),
#             torch_dtype=torch_dtype
#             )

#         input_x, input_x_repeated, probs_topk_repeated, tokens_per_expert = prepare_input(
#                                                                                     input_x,
#                                                                                     deepseek_moe,
#                                                                                     torch_dtype,
#                                                                                 )


#         # print(f"вероятности {probs_topk_dense[4, :20]}")

#         # print(f"шейпы {idx_topk.shape}, {probs_topk.shape}")
#         # print(f"значения {idx_topk}, {probs_topk}")
#         # print(f"шейпы2 {tokens_per_expert.shape}, {probs_topk_repeated.shape}, {input_x_repeated.shape}")

#         logits_1 = gigamix_moe(
#             permuted_local_hidden_states=input_x_repeated,
#             tokens_per_expert=tokens_per_expert,
#             permuted_probs=probs_topk_repeated
#         )
#         logits_2 = deepseek_moe.custom_forward(
#             input_x_repeated,
#             tokens_per_expert.numpy(),
#             probs_topk_repeated,
#             input_x
#         )

#         check_result(
#             logits_1,
#             logits_2,
#             atol=1e-7
#         )


def test_convertation_forward_moe():

    configure_deterministic_mode()

    # min_bfloat16_value = torch.finfo(torch.bfloat16).tiny
    # print(f'Минимальное нормальное число для bfloat16: {min_bfloat16_value}')
    # return

    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    deepseek_moe, deepseek_config = init_deepseek_moe()
    gigamix_moe, gigamix_config = init_gigamix_moe()
    factor = 6e-3

    # print(deepseek_config)
    # print(gigamix_config)

    init_weights(
        module = deepseek_moe,
        init_type = "random"
        # init_type = "const",
        # factor = factor
    )

    # init_weights(
    #     module = gigamix_moe,
    #     # init_type = "random"
    #     init_type = "const",
    #     factor = factor

    # )
    # g_diff = (gigamix_moe.weight1[:, :deepseek_config.moe_intermediate_size] - deepseek_moe.experts[0].gate_proj.weight.T).abs().max()
    # u_diff = (gigamix_moe.weight1[:, deepseek_config.moe_intermediate_size:2;deepseek_config.moe_intermediate_size] - deepseek_moe.experts[0].gate_proj.weight.T).abs().max()
    # print(f"дифф в весах {g_diff}")
    # print(f"дифф в весах {u_diff}")

    convert_weights_deepseek_to_gigamix(
        deepseek_moe,
        gigamix_moe,
        deepseek_config
    )



    reproducibility.seed_all(42)
    deepseek_moe.to("cuda:0")
    deepseek_moe = deepseek_moe.to(torch_dtype)
    reproducibility.seed_all(42)

    reproducibility.seed_all(42)
    gigamix_moe.to("cuda:0")
    gigamix_moe = gigamix_moe.to(torch_dtype)
    reproducibility.seed_all(42)

    for _ in range(ITER_NUM):
        input_x = generate_input(
            (1, 4096, 2048),
            torch_dtype=torch_dtype
            )

        input_x, input_x_repeated, probs_topk_repeated, tokens_per_expert = prepare_input(
                                                                                    input_x,
                                                                                    deepseek_moe,
                                                                                    torch_dtype,
                                                                                )

        # print(f"вероятности {probs_topk_repeated[:30]}")

        # print(f"шейпы {idx_topk.shape}, {probs_topk.shape}")
        # print(f"значения , {tokens_per_expert}")
        # print(f"шейпы2 {tokens_per_expert.shape}, {probs_topk_repeated.shape}, {input_x_repeated.shape}")

        logits_1 = gigamix_moe(
            permuted_local_hidden_states=input_x_repeated,
            tokens_per_expert=tokens_per_expert,
            permuted_probs=probs_topk_repeated
        )
        logits_2 = deepseek_moe.custom_forward(
            input_x_repeated,
            tokens_per_expert.numpy(),
            probs_topk_repeated,
        )

        check_result(
            logits_1,
            logits_2,
            atol=1e-7
        )


def test_convertation_gigamix2deepseek_forward_moe():

    configure_deterministic_mode()

    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    deepseek_moe, deepseek_config = init_deepseek_moe()
    gigamix_moe, gigamix_config = init_gigamix_moe()
    factor = 6e-3

    # print(deepseek_config)
    # print(gigamix_config)

    # init_weights(
    #     module = deepseek_moe,
    #     init_type = "random"
    #     # init_type = "const",
    #     # factor = factor
    # )

    init_weights(
        module = gigamix_moe,
        init_type = "random"
        # init_type = "const",
        # factor = factor

    )

    convert_weights_gigamix_to_deepseek(
        deepseek_moe,
        gigamix_moe,
        deepseek_config
    )



    reproducibility.seed_all(42)
    deepseek_moe.to("cuda:0")
    deepseek_moe = deepseek_moe.to(torch_dtype)
    reproducibility.seed_all(42)

    reproducibility.seed_all(42)
    gigamix_moe.to("cuda:0")
    gigamix_moe = gigamix_moe.to(torch_dtype)
    reproducibility.seed_all(42)

    for _ in range(ITER_NUM):
        input_x = generate_input(
            (1, 4096, 2048),
            torch_dtype=torch_dtype
            )

        input_x, input_x_repeated, probs_topk_repeated, tokens_per_expert = prepare_input(
                                                                                    input_x,
                                                                                    deepseek_moe,
                                                                                    torch_dtype,
                                                                                )

        # print(f"вероятности {probs_topk_repeated[:30]}")

        # print(f"шейпы {idx_topk.shape}, {probs_topk.shape}")
        # print(f"значения , {tokens_per_expert}")
        # print(f"шейпы2 {tokens_per_expert.shape}, {probs_topk_repeated.shape}, {input_x_repeated.shape}")

        logits_1 = gigamix_moe(
            permuted_local_hidden_states=input_x_repeated,
            tokens_per_expert=tokens_per_expert,
            permuted_probs=probs_topk_repeated
        )
        logits_2 = deepseek_moe.custom_forward(
            input_x_repeated,
            tokens_per_expert.numpy(),
            probs_topk_repeated,
        )

        check_result(
            logits_1,
            logits_2,
            atol=1e-7
        )


def test_convertation_cycle_forward_moe():

    configure_deterministic_mode()

    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    deepseek_moe, deepseek_config = init_deepseek_moe()
    gigamix_moe, gigamix_config = init_gigamix_moe()
    gigamix_moe_preconv, _ = init_gigamix_moe()
    factor = 6e-3

    # print(deepseek_config)
    # print(gigamix_config)


    # init_weights(
    #     module = deepseek_moe,
    #     init_type = "random"
    #     # init_type = "const",
    #     # factor = factor
    # )


    init_weights(
        module = gigamix_moe,
        init_type = "random"
        # init_type = "const",
        # factor = factor

    )

    gigamix_moe_preconv.load_state_dict(gigamix_moe.state_dict())


    convert_weights_gigamix_to_deepseek(
        deepseek_moe,
        gigamix_moe,
        deepseek_config
    )

    init_weights(
        module = gigamix_moe,
        # init_type = "random"
        init_type = "const",
        factor = 0
    )

    convert_weights_deepseek_to_gigamix(
        deepseek_moe,
        gigamix_moe,
        deepseek_config
    )


    reproducibility.seed_all(42)
    gigamix_moe_preconv.to("cuda:0")
    gigamix_moe_preconv = gigamix_moe_preconv.to(torch_dtype)
    reproducibility.seed_all(42)

    reproducibility.seed_all(42)
    gigamix_moe.to("cuda:0")
    gigamix_moe = gigamix_moe.to(torch_dtype)
    reproducibility.seed_all(42)

    for _ in range(ITER_NUM):
        input_x = generate_input(
            (1, 4096, 2048),
            torch_dtype=torch_dtype
            )

        input_x, input_x_repeated, probs_topk_repeated, tokens_per_expert = prepare_input(
                                                                                    input_x,
                                                                                    deepseek_moe,
                                                                                    torch_dtype,
                                                                                )

        # print(f"вероятности {probs_topk_repeated[:30]}")

        # print(f"шейпы {idx_topk.shape}, {probs_topk.shape}")
        # print(f"значения , {tokens_per_expert}")
        # print(f"шейпы2 {tokens_per_expert.shape}, {probs_topk_repeated.shape}, {input_x_repeated.shape}")

        logits_1 = gigamix_moe(
            permuted_local_hidden_states=input_x_repeated,
            tokens_per_expert=tokens_per_expert,
            permuted_probs=probs_topk_repeated
        )
        logits_2 = gigamix_moe_preconv(
            permuted_local_hidden_states=input_x_repeated,
            tokens_per_expert=tokens_per_expert,
            permuted_probs=probs_topk_repeated
        )

        check_result(
            logits_1,
            logits_2,
            atol=1e-7
        )