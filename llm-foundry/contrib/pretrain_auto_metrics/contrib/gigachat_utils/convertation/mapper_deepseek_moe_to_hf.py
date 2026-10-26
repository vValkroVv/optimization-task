#!/usr/bin/env python
# coding: utf-8
# script for convertion megablocks moe checkpoint to huggingface MixtralModel model
import argparse
import json
from pprint import pprint
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
import numpy as np
import warnings
from utils.file_utils import vendor_deepseek_runtime_code

# requires >4.36.0
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM
)

def convert_deepseek_to_hf(ckpt: Dict,
                           args: Optional[argparse.Namespace] = None) -> None:
    HF_LLAMA_CHECKPOINT_FOLDER = args.output_checkpoint_path

    assert args.torch_dtype in ('float16', 'bfloat16'), \
        f"{args.torch_dtype=} not supported"
    assert not Path(HF_LLAMA_CHECKPOINT_FOLDER).exists() or len(list(Path(HF_LLAMA_CHECKPOINT_FOLDER).iterdir())) == 0, \
        f"{HF_LLAMA_CHECKPOINT_FOLDER=} already exists"

    config = ckpt['state']['integrations']['huggingface']['model']['config']['content']
    print("config:", config)
    if args.tokenizer_path:
        TOKENIZER_PATH = args.tokenizer_path
    else:
        TOKENIZER_PATH = config['_name_or_path']
    print("Tokenizer path:", TOKENIZER_PATH)

    del config['architectures']

    model_name = "deepseek-ai/DeepSeek-V3"

    config['moe_intermediate_size'] = config['intermediate_size']
    config['intermediate_size'] = config['dense_intermediate_size']
    config['n_routed_experts'] = config['num_routed_experts']
    config['aux_loss_alpha'] = config['router_aux_loss_coef']
    config['torch_dtype'] = args.torch_dtype

    assert config['attention_type'] in ("LlamaLatentAttention", "LlamaLatentSparseAttention"), (
        f"This mapper supports only attention_type in ('LlamaLatentAttention', 'LlamaLatentSparseAttention'), got: {config['attention_type']}"
    )
    assert config['num_krope_heads'] == 1, "Only num_krope_heads = 1 is supported"

    AUX_FREE = config['aux_loss_free']
    if not AUX_FREE:
        warnings.warn(
            "DeepseekMapper was tested with aux_loss_free=True. "
            "It should work with aux_loss_free=False, but correctness is not guaranteed.",
            RuntimeWarning,
        )

    config['q_lora_rank'] = None if config.get('q_lora_rank') == 0 else config.get('q_lora_rank')

    assert config['scoring_func'] == "sigmoid", (
        "This mapper expects scoring_func='sigmoid'. "
        "DeepSeek uses 'sigmoid' by default. If you want to use softmax gating, "
        "patch the modeling file to handle 'softmax', e.g. in modeling_deepseek.py (~line ~429):\n"
        "    if self.scoring_func == 'sigmoid':\n"
        "        scores = logits.sigmoid()\n"
        "    elif self.scoring_func == 'softmax':\n"
        "        scores = logits.softmax(dim=-1)\n"
    )

    g_topk = config["moe_router_group_topk"]
    n_groups = config["moe_router_num_groups"]

    if g_topk is None and n_groups is None:
        config["n_group"] = 1
        config["topk_group"] = 1
    elif (g_topk, n_groups) == (4, 8):
        config["n_group"] = n_groups
        config["topk_group"] = g_topk
    else:
        warnings.warn("This configuration is not tested yet", RuntimeWarning)
        config["n_group"] = n_groups if n_groups is not None else 1
        config["topk_group"] = g_topk if g_topk is not None else 1

    config['routed_scaling_factor'] = config['moe_router_routed_scaling_factor']

    if config.get("use_mtp"):
        raise ValueError(
            "MTP is not supported in this pipeline. Please use the 'map_gigafsdp_to_hf_safetensor' mapper instead. "
            "It is still possible to use this mapper, but MTP module will be ignored. "
            "If this is what you want (visiion moe conversion), comment out this raise."
        )

    config['num_nextn_predict_layers'] = 0

    scaling = config.get("rope_scaling")
    allowed = {"linear", "dynamic", "yarn"}

    if isinstance(scaling, dict):
        scaling["type"] = scaling.get("type") or scaling.pop("rope_type", None)
        if scaling["type"] not in allowed:
            raise ValueError(f"Unknown RoPE scaling type {scaling['type']!r}")
    elif isinstance(scaling, str):
        if scaling not in allowed:
            raise ValueError(f"Unknown RoPE scaling type {scaling!r}")
        scaling = {"type": scaling}
    elif scaling is not None:
        raise TypeError("rope_scaling must be dict, str or None")

    config["rope_scaling"] = scaling

    torch.set_default_dtype(getattr(torch, args.torch_dtype))
    cfg = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        **config
    )
    cfg.__dict__.pop("quantization_config", None)

    cfg.bos_token_id = 1
    cfg.eos_token_id = 2
    cfg.pad_token_id = 2

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=getattr(torch, args.torch_dtype))
    pprint(model.config)
    pprint(model)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"{count_trainable_parameters(model)=}")

    model_state_d = {k[len('model.'):]: v for k, v in ckpt['state']['model'].items()}
    merge_model_dict = {}

    kv_a_buf = {}  # {layer_base: {"dkv": Tensor, "kr": Tensor}}
    kv_b_buf = {}  # {layer_base: {"uv": Tensor,  "uk": Tensor}}

    N_EXPERTS = cfg.n_routed_experts
    for (layer_name, weights) in model_state_d.items():
        #skip MTP layers
        if "mtp" in layer_name:
            continue

        # MLA
        # q_lora_rank == 0
        if layer_name.endswith(".self_attn._attn.q_proj.weight"):
            final = layer_name.replace(".self_attn._attn.q_proj.", ".self_attn.q_proj.")
            merge_model_dict[final] = weights
            continue

        # q_lora_rank != 0:
        # 1. q_a <- _attn.dq
        if layer_name.endswith(".self_attn._attn.dq_proj.weight"):
            final = layer_name.replace(".self_attn._attn.dq_proj.", ".self_attn.q_a_proj.")
            merge_model_dict[final] = weights
            continue

        # 2. q_a_layernorm <- _attn.q_norm
        if layer_name.endswith(".self_attn._attn.q_norm.weight"):
            final = layer_name.replace(".self_attn._attn.q_norm.", ".self_attn.q_a_layernorm.")
            merge_model_dict[final] = weights
            continue

        # 3. q_b <- _attn.uq
        if layer_name.endswith(".self_attn._attn.uq_proj.weight"):
            final = layer_name.replace(".self_attn._attn.uq_proj.", ".self_attn.q_b_proj.")
            merge_model_dict[final] = weights
            continue

        # DSA indexer: keep module names, remove the training-time _attn wrapper.
        # prod_dsa names the indexer query proj `indexer.wq`; older DeepSeek-V3.2 uses `indexer.wq_b`.
        if ".self_attn._attn.indexer." in layer_name:
            final = layer_name.replace(".self_attn._attn.indexer.", ".self_attn.indexer.")
            final = final.replace(".indexer.wq.", ".indexer.wq_b.")
            merge_model_dict[final] = weights
            continue

        # o_proj <- _attn.o_proj
        if layer_name.endswith(".self_attn._attn.o_proj.weight"):
            final = layer_name.replace(".self_attn._attn.o_proj.", ".self_attn.o_proj.")
            merge_model_dict[final] = weights
            continue

        # kv_a_layernorm <- _attn.kv_norm
        if layer_name.endswith(".self_attn._attn.kv_norm.weight"):
            final = layer_name.replace(".self_attn._attn.kv_norm.", ".self_attn.kv_a_layernorm.")
            merge_model_dict[final] = weights
            continue

        # kv_a_proj_with_mqa: dkv_proj + kr_proj
        if layer_name.endswith(".self_attn._attn.dkv_proj.weight"):
            base = layer_name.split(".self_attn._attn.dkv_proj.weight")[0]
            kv_a_buf.setdefault(base, {})["dkv"] = weights
            continue

        if layer_name.endswith(".self_attn._attn.kr_proj.weight"):
            base = layer_name.split(".self_attn._attn.kr_proj.weight")[0]
            kv_a_buf.setdefault(base, {})["kr"] = weights
            continue

        # kv_b_proj: uk_proj + uv_proj
        if layer_name.endswith(".self_attn._attn.uv_proj.weight"):
            base = layer_name.split(".self_attn._attn.uv_proj.weight")[0]
            kv_b_buf.setdefault(base, {})["uv"] = weights
            continue

        if layer_name.endswith(".self_attn._attn.uk_proj.weight"):
            base = layer_name.split(".self_attn._attn.uk_proj.weight")[0]
            kv_b_buf.setdefault(base, {})["uk"] = weights
            continue

        if layer_name.endswith(".self_attn.input_layernorm.weight"):
            final = layer_name.replace(".self_attn.input_layernorm.", ".input_layernorm.")
            merge_model_dict[final] = weights
            continue

        if layer_name.endswith(".self_attn.post_attention_layernorm.weight"):
            final = layer_name.replace(".self_attn.post_attention_layernorm.", ".post_attention_layernorm.")
            merge_model_dict[final] = weights
            continue

        # self_attn.mlp.* -> .mlp.*
        if ".self_attn.mlp." in layer_name and layer_name.endswith(".weight"):
            final = layer_name.replace(".self_attn.mlp.", ".mlp.")
            merge_model_dict[final] = weights
            continue

        # shared_experts под self_attn -> mlp.shared_experts
        if ".self_attn.shared_experts." in layer_name and layer_name.endswith(".weight"):
            final = layer_name.replace(".self_attn.shared_experts.", ".mlp.shared_experts.")
            merge_model_dict[final] = weights
            continue

        # self_attn._gate.gate.weight -> mlp.gate.weight
        if layer_name.endswith(".self_attn._gate.gate.weight"):
            final = layer_name.replace(".self_attn._gate.gate.", ".mlp.gate.")
            merge_model_dict[final] = weights
            continue

        # AUX_FREE = True -- недоступно с deepseek-ai/DeepSeek-V2-Lite
        if layer_name.endswith(".self_attn._gate.expert_bias"):
            target_name = layer_name if layer_name.startswith("model.") else f"model.{layer_name}"
            final = target_name.replace(
                ".self_attn._gate.expert_bias",
                ".mlp.gate.e_score_correction_bias",
            )
            merge_model_dict[final] = weights
            continue

        if 'block_sparse_moe' not in layer_name:
            merge_model_dict[layer_name] = weights
            continue

        if layer_name.endswith("block_sparse_moe.experts.weight2"):
            # If sonic format, convert to g_gemm convention before splitting
            if config.get("moe_impl") == "SonicGMMMoeBlock" or config.get("moe_kernel") == "sonic": # TODO: del "moe_kernel" == "sonic" afer refactor
                E = N_EXPERTS
                I = weights.shape[0] // E
                H = weights.shape[1]
                weights = weights.view(E, H, I).permute(0, 2, 1).contiguous().view(I * E, H)

            experts_ffn_dim, hidden_dim = weights.shape
            ffn_dim = experts_ffn_dim // N_EXPERTS
            assert ffn_dim != hidden_dim
            for i in range(N_EXPERTS):
                weights_ = weights[i * ffn_dim:(i + 1) * ffn_dim, :].t().clone().contiguous()
                final_layer_name = layer_name.replace(
                    "block_sparse_moe.experts.weight2",
                    f"mlp.experts.{i}.down_proj.weight"
                )
                merge_model_dict[final_layer_name] = weights_
            continue

        if layer_name.endswith("block_sparse_moe.experts.weight1"):
            # If sonic format, convert to g_gemm convention before splitting
            if config.get("moe_impl") == "SonicGMMMoeBlock" or config.get("moe_kernel") == "sonic": # TODO: del "moe_kernel" == "sonic" afer refactor
                E = N_EXPERTS
                H = weights.shape[0]
                I = weights.shape[1] // (2 * E)
                w = weights.view(E, 2 * I, H).permute(0, 2, 1).contiguous()  # (E, H, 2*I) interleaved
                gate = w[:, :, 0::2]
                up   = w[:, :, 1::2]
                weights = torch.cat([gate, up], dim=-1).contiguous().view(H, 2 * I * E)

            hidden_dim, total_ffn = weights.shape
            D = total_ffn // (2 * N_EXPERTS)
            w1 = weights.view(N_EXPERTS, hidden_dim, 2 * D)
            for i in range(N_EXPERTS):
                gate_part, up_part = w1[i].chunk(2, dim=-1)
                gate_w = gate_part.t().contiguous()
                up_w = up_part.t().contiguous()
                base = layer_name.replace("block_sparse_moe.experts.weight1", f"mlp.experts.{i}")
                merge_model_dict[f"{base}.gate_proj.weight"] = gate_w
                merge_model_dict[f"{base}.up_proj.weight"] = up_w
            continue

        raise RuntimeError(f"Unexpected MoE key: {layer_name}")

    for base, parts in kv_a_buf.items():
        dkv = parts.get("dkv", None)  # (kv_lora_rank, hidden_size)
        kr = parts.get("kr", None)    # ((num_krope_heads * qk_rope_head_dim), hidden_size) (qk_rope_head_dim=1)
        if dkv is None or kr is None:
            raise RuntimeError(f"Missing kv_a parts for {base}: have {list(parts.keys())}")

        assert dkv.shape[1] == kr.shape[1], f"d_model mismatch in kv_a for {base}"
        dk_rope = cfg.qk_rope_head_dim
        assert kr.shape[0] == dk_rope, f"unexpected kr shape: {kr.shape}"

        fused_w = torch.cat([dkv, kr], dim=0).contiguous()
        merge_model_dict[base + ".self_attn.kv_a_proj_with_mqa.weight"] = fused_w

    for base, parts in kv_b_buf.items():
        uk = parts.get("uk", None)  # (n_heads * qk_nope_head_dim, kv_lora_rank)
        uv = parts.get("uv", None)  # (n_heads * v_head_dim, kv_lora_rank)
        if uv is None or uk is None:
            raise RuntimeError(f"Missing kv_b parts for {base}: have {list(parts.keys())}")

        # [uk; uv] = [K_nope; V]
        n_head = cfg.num_attention_heads
        dk_nope = cfg.qk_nope_head_dim
        dv = cfg.v_head_dim
        r_kv = uk.shape[1]
        assert uk.shape == (n_head * dk_nope, r_kv)
        assert uv.shape == (n_head * dv, r_kv)

        uk_view = uk.view(n_head, dk_nope, r_kv)
        uv_view = uv.view(n_head, dv, r_kv)
        fused_w = torch.cat([uk_view, uv_view], dim=1).reshape(n_head * (dk_nope + dv), r_kv).contiguous()

        merge_model_dict[base + ".self_attn.kv_b_proj.weight"] = fused_w

    Path(HF_LLAMA_CHECKPOINT_FOLDER).mkdir(parents=True, exist_ok=True)
    missing_keys, unexpected_keys = model.load_state_dict(
        merge_model_dict,
        strict=False,
        assign=True
    )
    assert len(missing_keys) == 0 and len(unexpected_keys) == 0, \
        f"{missing_keys=}, {unexpected_keys=}"
    model = model.to(getattr(torch, args.torch_dtype))
    model.save_pretrained(str(HF_LLAMA_CHECKPOINT_FOLDER), safe_serialization=False)
    vendor_deepseek_runtime_code(HF_LLAMA_CHECKPOINT_FOLDER, cfg, model)
    tokenizer.save_pretrained(str(HF_LLAMA_CHECKPOINT_FOLDER))
    print(f"{HF_LLAMA_CHECKPOINT_FOLDER=}")

    with open(f'{HF_LLAMA_CHECKPOINT_FOLDER}/config.json', 'r') as file:
        data = json.load(file)

    data['auto_map'] = {
        "AutoConfig": "configuration_deepseek.DeepseekConfig",
        "AutoModel": "modeling_deepseek.DeepseekModel",
        "AutoModelForCausalLM": "modeling_deepseek.DeepseekForCausalLM"
    }

    data['architectures'] = ["DeepseekForCausalLM"]
    with open(f'{HF_LLAMA_CHECKPOINT_FOLDER}/config_vanilla.json', 'w') as file:
        json.dump(data, file, indent=2)

    print("config vanilla is created")


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """Meta initialization context manager.

    A context manager under which models are initialized with all parameters
    on the meta device, therefore creating an empty model. Useful when just
    initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    with init_on_device(torch.device('meta'),
                        include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """Device initialization context manager.

    A context manager under which models are initialized with all parameters
    on the specified device.

    Args:
        device (`torch.device`): Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(self: torch.nn.Module, name: str,
                                 param: Optional[torch.nn.Parameter]):
        old_register_parameter(self, name, param)
        if param is not None:
            parameter = self._parameters[name]
            assert parameter is not None

            param_cls = type(parameter)
            kwargs = parameter.__dict__

            self._parameters[name] = param_cls(parameter.to(device), **kwargs)

    def register_empty_buffer(self: torch.nn.Module,
                              name: str,
                              tensor: Optional[torch.Tensor],
                              persistent: bool = True):
        old_register_buffer(self, name, tensor, persistent=persistent)
        if tensor is not None:
            named_buffer = self._buffers[name]
            assert named_buffer is not None
            self._buffers[name] = named_buffer.to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ['empty', 'zeros', 'ones', 'full']
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn: Callable):

        def wrapper(*args: Any, **kwargs: Any):
            kwargs['device'] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter  # type: ignore
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer  # type: ignore
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch, torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items(
        ):
            setattr(torch, torch_function_name, old_torch_function)
