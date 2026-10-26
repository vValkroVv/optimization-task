"""PyTorch GigaMix model."""

import functools
import inspect
import warnings
from collections import UserDict
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import transformers
from composer.metrics.nlp import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMCExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningQAAccuracy,
    LanguageCrossEntropy,
    LanguagePerplexity,
)
from composer.models import HuggingFaceModel
from composer.utils import dist, global_actlog_state
from composer.utils.dist import get_ep_group_size, get_tp_group_size
from composer.utils.profiler_annotation import profiler_annotation

# from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss # overrides below
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import ModelOutput

from llmfoundry.models.giga_mix.configuration_giga_mix import GigaMixConfig
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device, prepare_hf_model_for_fsdp
from llmfoundry.models.layers import (
    BLOCK_CLASS_REGISTRY,
    FC_CLASS_REGISTRY,
    MTP_CLASS_REGISTRY,
    PARALLEL_EMBEDDING_REGISTRY,
    RING_ATTN_CLASSES,
    ColumnParallelLinear,
    ScaledEmbedding,
)
from llmfoundry.models.layers.mtp import (
    get_mtp_activation_checkpointing_modules,
    get_mtp_fsdp_modules_with_names,
    resolve_mtp_block_type,
)
from llmfoundry.models.layers.norm import resolve_norm_class
from llmfoundry.models.layers.moe import AbstractGMMMoeBlock, ScMoEBlock
from llmfoundry.models.ops.fused_cross_entropy import (
    CrossEntropyLoss as FusedCrossEntropyLoss,
)
from llmfoundry.models.ops.included_proj_z_loss import (
    QKV_Z_LOSS_SUPPORTED_ATTENTION,
    collect_included_proj_z_losses,
    included_proj_active,
    resolve_included_proj_coef,
)
from llmfoundry.models.ops.zloss import hidden_z_loss_fn, z_loss_fn
from llmfoundry.models.parallel.sequence.utils import get_tensors_sp_part
from llmfoundry.models.parallel.tensor.mappings import distribute_async_tp_embeddings
from llmfoundry.models.utils.configuration_utils import get_sp_split_type
from llmfoundry.models.utils.flash_seqlens import (
    get_cu_seqlens_from_pos_ids,
    get_llama3_cu_seqlens_from_pos_ids,
)
from llmfoundry.models.utils.param_init_fns import (
    MODEL_INIT_REGISTRY,
    set_module_custom_std,
)
from llmfoundry.utils.config_utils import write_lora_config, to_container


@dataclass
class MoeModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GigaMixPreTrainedModel(PreTrainedModel):
    config_class = GigaMixConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module: nn.Module):
        # `param_init_fn` has the conditioning for all modules classes
        self.param_init_fn(module)

    # Params initialization function needed for meta devie parameters initialization.
    # Has inner conditions for all possible modules that model can use.
    def param_init_fn(self, module: nn.Module):
        init_type = self.config.init_type
        if init_type == "giga":
            # Update gain if module has appropriate attribute. Used to set proper gain
            # for MoE layers.
            if hasattr(module, "init_gain"):
                gain = module.init_gain
            else:
                gain = 1.0

            init_config = {
                "name": "xavier_normal_",
                "init_gain": gain,
                "init_div_is_residual": False,
                "emb_init_std": None,
                "emb_init_uniform_lim": None,
            }

            if hasattr(module, "custom_init_std"):
                fan_in, fan_out = module.custom_init_std
                init_config = set_module_custom_std(init_config, fan_in, fan_out, gain)

        elif init_type == "deepseek":
            std = self.config.deepseek_init_std
            init_config = {
                "name": "baseline_",
                "init_std": std,
                "init_div_is_residual": False,
                "emb_init_std": None,
                "emb_init_uniform_lim": None,
            }

        else:
            raise ValueError(
                f'`init_type` "{init_type}" is not supported for GigaMix architecture. '
                'Use "giga" or "deepseek".'
            )

        init_fn_name = init_config["name"]
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self.config.num_hidden_layers,
            **init_config,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GigaMixModel):
            module.gradient_checkpointing = value


class GigaMixModel(GigaMixPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: GigarConfig
    """

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_z_loss_eps = config.hidden_z_loss_eps
        self.hidden_z_loss_attn_coef = getattr(config, "hidden_z_loss_attn_coef", 0.0)
        self.hidden_z_loss_moe_coef = getattr(config, "hidden_z_loss_moe_coef", 0.0)
        self.hidden_z_loss_dense_mlp_coef = getattr(
            config, "hidden_z_loss_dense_mlp_coef", 0.0
        )

        # Check that async TP is only used with EmbeddingParallelEmbedding
        if config.enable_async_tp:
            assert config.parallel_embedding_type == "EmbeddingParallelEmbedding", (
                "Async tensor parallelism only works with EmbeddingParallelEmbedding"
            )

        if config.tp_size == 1:
            self.embed_tokens = ScaledEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                embed_scale=config.embed_scale,
                device=config.init_device,
            )
        else:
            assert config.enable_async_tp, (
                "Tensor parallelism is supported only with async TP"
            )
            if config.parallel_embedding_type == "EmbeddingParallelEmbedding":
                # TODO: should probably move this logic to GigaMixConfig?
                parallel_output_style = (
                    "shard_hidden" if config.enable_async_tp else "replicate"
                )
                emb_kwargs = dict(parallel_output_style=parallel_output_style)
            else:
                emb_kwargs = dict()

            self.embed_tokens = PARALLEL_EMBEDDING_REGISTRY[
                config.parallel_embedding_type
            ](
                config.vocab_size,
                config.hidden_size,
                config=config,
                padding_idx=config.pad_token_id,
                **emb_kwargs,
            )

        self.use_embed_ln = config.use_embed_ln
        if self.use_embed_ln:
            norm_class = resolve_norm_class(config.norm_type)
            self.embed_ln = norm_class(
                config.hidden_size,
                eps=config.rms_norm_eps,
                device=config.init_device,
            )

        self.layers = nn.ModuleList(
            [
                BLOCK_CLASS_REGISTRY["LlamaMixDecoderLayer"](
                    config,
                    layer_idx=i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # Warn (do not raise) about layers whose attention class does not
        # implement the QKV hidden z-loss: the `qkv` coef is configured for them
        # but will have no effect. Supported layers still get it. (up/gate
        # projections exist on every dense MLP / routed-MoE expert, so only qkv
        # needs this check.)
        if included_proj_active(config, "qkv"):
            unsupported = []
            for i, layer in enumerate(self.layers):
                coef_i = resolve_included_proj_coef(config, "qkv", i)
                attn = getattr(getattr(layer, "self_attn", None), "_attn", None)
                if (
                    coef_i > 0
                    and type(attn).__name__ not in QKV_Z_LOSS_SUPPORTED_ATTENTION
                ):
                    unsupported.append((i, type(attn).__name__))
            if unsupported:
                warnings.warn(
                    "QKV hidden z-loss is configured but not supported for the "
                    f"attention class of these layers: {unsupported}. No qkv "
                    "z-loss will be added on them (supported classes: "
                    f"{sorted(QKV_Z_LOSS_SUPPORTED_ATTENTION)}).",
                    stacklevel=2,
                )

        if config.activation_checkpoint_layers_num:
            for i in range(config.num_hidden_layers):
                if i < config.activation_checkpoint_layers_num:
                    self.layers[i]._activation_checkpointing = True
                else:
                    self.layers[i]._activation_checkpointing = False

        norm_class = resolve_norm_class(self.config.norm_type)
        self.norm = norm_class.from_config(config)

        self.enable_async_tp = config.enable_async_tp

        self.use_mtp = config.use_mtp
        self.mtp_block = MTP_CLASS_REGISTRY["LlamaDecoderMTPBlock"](config)

        self.gradient_checkpointing = False

        if hasattr(self.config, "pad_token_id") and config.pad_token_id is not None:
            self._pad_token_id = int(config.pad_token_id)
        else:
            self._pad_token_id = 0

        print(f"using pad token id {self._pad_token_id}")

        # Initialize weights using Transformers' `post_init` method that uses
        # `_init_weights` method defined in GigaMixPreTrainedModel class
        # ---
        # Do only in case of a non meta init (cpu, cuda (possibly)) as the meta
        # device initialization is handled by the FSDP module and the `param_init_fn`.
        # ---
        # NOTE (Sbr): comment out as we use GigaMixForCausalLM model class that applies
        # its `post_init` function recursivly so this call would lead to be the second time we do
        # init here (third if we cound layers default init). Uncomment if you're going to
        # use this class for some other custom purposes.
        #
        # if config.init_device != "meta":
        #     self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
    ):  # pylint: disable=unused-argument
        # [bsz, seq_len]
        return attention_mask

    def _embed_draft_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        draft_embeds = self.embed_tokens(draft_ids)
        if self.enable_async_tp:
            draft_embeds = distribute_async_tp_embeddings(draft_embeds)
        if self.config.mtp_loss_detach:
            draft_embeds = draft_embeds.detach()
        return draft_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
        prefix_mask: Optional[torch.Tensor] = None,
        lm_head_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Force set `use_cache=False` as it might leak through config.json. However, `use_cache=True`
        # is supported for audio modality when using packed_attn + GQA, hence we provide a separate
        # parameter `use_cache_force` to explicitly enable caching in that case.
        use_cache = (
            False if not self.config.use_cache_force else self.config.use_cache_force
        )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if prefix_mask is None:
            prefix_mask = torch.ones_like(
                input_ids, device=input_ids.device, dtype=torch.int32
            )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            if padding_mask is None:
                padding_mask = (input_ids != self._pad_token_id).to(torch.int32)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            if padding_mask is not None:
                padding_mask = padding_mask.to(torch.int32)
            elif attention_mask is not None:
                padding_mask = attention_mask
                if (
                    dist.get_sp_group_size() is not None
                    and dist.get_sp_group_size() > 1
                    and self.training
                ):
                    (padding_mask,) = get_tensors_sp_part(
                        padding_mask,
                        split_type=self.config.sp_split_type,
                        dim=1,
                    )
            else:
                padding_mask = torch.ones_like(
                    inputs_embeds[:, :, 0],
                    device=inputs_embeds.device,
                    dtype=torch.int32,
                )

        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if self.enable_async_tp and logical_batch_size is None:
            logical_batch_size = batch_size

        # Adjst `seq_length` in SP case to construct proper `input_ids` and `attention_mask`
        # for the whole sequence.
        if dist.get_sp_group_size() is not None and dist.get_sp_group_size() > 1:
            # NOTE (Sbr): Do not adjust `seq_length` in eval mode. See GigaMixForCausalLM
            # forward note for explanation.
            # ---
            # TODO (Sbr): Remove if when the issue is fixed.
            if self.training:
                seq_length = seq_length * dist.get_sp_group_size()

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[1]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        global_actlog_state.use_monitor_variable(
            inputs_embeds, "model.model.embed_tokens", "_output.0"
        )
        # Note: make unified all2all for all types of input (input_ids or inputs_embeds)
        if self.enable_async_tp:
            # inputs_embeds shape: [bs, seq_len, hidden_size // tp_size]
            inputs_embeds = distribute_async_tp_embeddings(inputs_embeds)
            # inputs_embeds shape: [1, bs * seq_len // tp_size, hidden_size

        raw_inputs_embeds = inputs_embeds
        decoder_inputs_embeds = inputs_embeds
        if self.use_embed_ln:
            decoder_inputs_embeds = self.embed_ln(decoder_inputs_embeds)

        attention_mask = self._prepare_decoder_attention_mask(  # pylint: disable=protected-access
            attention_mask,
            (batch_size, seq_length),
            decoder_inputs_embeds,
            past_key_values_length,
        )

        hidden_states = decoder_inputs_embeds

        if self.training:
            if self.gradient_checkpointing and use_cache:
                transformers.logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        hidden_z_loss_attn_sum = None
        hidden_z_loss_moe_sum = None
        hidden_z_loss_dense_mlp_sum = None

        global_actlog_state.set_layer_idx_variable(-1)
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = global_actlog_state.apply_layer_counter(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    None,
                    cu_seqlens,
                    max_seqlen,
                    logical_batch_size,
                    None,
                    padding_mask,
                    prefix_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    logical_batch_size=logical_batch_size,
                    is_left_padded_eval=is_left_padded_eval,
                    padding_mask=padding_mask,
                    prefix_mask=prefix_mask,
                )

            hidden_states = layer_outputs[0]

            global_actlog_state.use_monitor_variable(
                hidden_states, "model.model.layers.{}", "_output.0"
            )

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # `LlamaMixDecoderLayer` appends
            # (hidden_z_loss_attn, hidden_z_loss_moe, hidden_z_loss_dense_mlp)
            # at the tail of `layer_outputs` — see blocks.py. Accumulate the
            # per-layer detached scalars for logging; backward is already wired
            # in-graph via `AddAuxiliaryLoss` inside the layer.
            layer_hidden_z_loss_attn = layer_outputs[-3]
            layer_hidden_z_loss_moe = layer_outputs[-2]
            layer_hidden_z_loss_dense_mlp = layer_outputs[-1]
            if layer_hidden_z_loss_attn is not None:
                hidden_z_loss_attn_sum = (
                    layer_hidden_z_loss_attn
                    if hidden_z_loss_attn_sum is None
                    else hidden_z_loss_attn_sum + layer_hidden_z_loss_attn
                )
            if layer_hidden_z_loss_moe is not None:
                hidden_z_loss_moe_sum = (
                    layer_hidden_z_loss_moe
                    if hidden_z_loss_moe_sum is None
                    else hidden_z_loss_moe_sum + layer_hidden_z_loss_moe
                )
            if layer_hidden_z_loss_dense_mlp is not None:
                hidden_z_loss_dense_mlp_sum = (
                    layer_hidden_z_loss_dense_mlp
                    if hidden_z_loss_dense_mlp_sum is None
                    else hidden_z_loss_dense_mlp_sum + layer_hidden_z_loss_dense_mlp
                )

        normed_hidden_states = (
            self.norm(hidden_states)
            if self.use_mtp and self.config.draft_input_postnorm
            else None
        )
        draft_input = (
            hidden_states if normed_hidden_states is None else normed_hidden_states
        )
        mtp_input_embeds = raw_inputs_embeds
        if self.config.mtp_loss_detach:
            draft_input = draft_input.detach()
            mtp_input_embeds = mtp_input_embeds.detach()

        # MTP logits/loss are produced only in training mode.
        if self.use_mtp:
            mtp_hidden_states = self.mtp_block(
                input_embeds=mtp_input_embeds,
                hidden_states=draft_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                is_left_padded_eval=is_left_padded_eval,
                logical_batch_size=logical_batch_size,
                padding_mask=padding_mask,
                prefix_mask=prefix_mask,
                embed_fn=self._embed_draft_ids,
                logits_fn=lm_head_fn,
            )
            for i, mtp_hidden_state in enumerate(mtp_hidden_states):
                global_actlog_state.use_monitor_variable(
                    mtp_hidden_state,
                    f"model.model.mtp_block.{i}",
                    "_output.0",
                    layer_depends=False,
                )
        else:
            mtp_hidden_states = None

        hidden_z_loss = None
        if self.hidden_z_loss_eps != 0.0:
            # Adjust `seq_length` for view here as we calculated it as for a full
            # sequence in the beginning of the method.
            if self.training:
                sp_seqlen_adjust_factor = dist.get_sp_group_size() or 1
            else:
                sp_seqlen_adjust_factor = 1

            flat_inputs = hidden_states.view(
                batch_size,
                seq_length // sp_seqlen_adjust_factor,
                hidden_states.size(dim=-1),
            )
            hidden_z_loss = hidden_z_loss_fn(flat_inputs, self.hidden_z_loss_eps)

        global_actlog_state.use_monitor_variable(
            hidden_states, "model.norm", "_input.0"
        )
        hidden_states = (
            self.norm(hidden_states)
            if normed_hidden_states is None
            else normed_hidden_states
        )
        global_actlog_state.use_monitor_variable(
            hidden_states, "model.norm", "_output.0"
        )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            elts = [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if self.use_mtp:
                elts.append(mtp_hidden_states)
            if hidden_z_loss is not None:
                elts.append(hidden_z_loss)
            if hidden_z_loss_attn_sum is not None:
                elts.append(hidden_z_loss_attn_sum)
            if hidden_z_loss_moe_sum is not None:
                elts.append(hidden_z_loss_moe_sum)
            if hidden_z_loss_dense_mlp_sum is not None:
                elts.append(hidden_z_loss_dense_mlp_sum)
            return tuple(v for v in elts if v is not None)

        if self.use_mtp:
            raise NotImplementedError("Returning 'dict' is not supported with mtp")

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GigaMixForCausalLM(GigaMixPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    # GigaOmni passes `padding_mask` only to language models that set this.
    accepts_padding_mask = True

    def __init__(self, config):
        super().__init__(config)

        assert not (
            config.pretraining_tp > 1
            and (config.tp_size > 1 or config.enable_async_tp or config.use_liger)
        ), "pretraining_tp cannot be used with tp, asynctp and use_liger options."

        assert not (config.use_liger and config.enable_async_tp), (
            "Fused Liger CrossEntropy Loss doesn't work with AsyncTP yet."
        )

        assert not (
            config.z_loss_eps != 0.0 and config.lm_head_logit_softcapping is not None
        ), (
            "Z-loss doesn't work with LM head logit softcapping (need to fix z-loss compute)"
        )

        self.config = config
        self.model = GigaMixModel(config)
        self.vocab_size = config.vocab_size
        self.z_loss_eps = config.z_loss_eps
        self.hidden_z_loss_eps = config.hidden_z_loss_eps
        self.hidden_z_loss_attn_coef = getattr(config, "hidden_z_loss_attn_coef", 0.0)
        self.hidden_z_loss_moe_coef = getattr(config, "hidden_z_loss_moe_coef", 0.0)
        self.hidden_z_loss_dense_mlp_coef = getattr(
            config, "hidden_z_loss_dense_mlp_coef", 0.0
        )
        self.lm_head: ColumnParallelLinear = FC_CLASS_REGISTRY["ColumnParallelLinear"](
            config.hidden_size,
            config.vocab_size,
            config=config,
            bias=False,
            lm_head=True,
            gather_output=False,
        )

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize layers and apply other related processing
        # ---
        # NOTE (Sbr): only done here, the GigaMixModel call is commented to avoid
        # unnecessary re-initialization calls.
        if config.init_device != "meta":
            self.post_init()

        self.use_mtp = config.use_mtp
        self.ignore_index = config.ignore_index
        self.loss_observer = None
        self.topk_observer = None

        self.mtp_ce_loss_weight = config.mtp_ce_loss_weight
        self.lk_loss_weight = getattr(config, "lk_loss_weight", 0.0)
        self.tv_loss_weight = getattr(config, "tv_loss_weight", 0.0)
        self.kl_loss_weight = getattr(config, "kl_loss_weight", 0.0)
        self.e2e_tv_loss_weight = getattr(config, "e2e_tv_loss_weight", 0.0)
        self.use_lk_loss = self.use_mtp and getattr(config, "use_lk_loss", False)
        self.use_tv_loss = self.use_mtp and getattr(config, "use_tv_loss", False)
        self.use_kl_loss = self.use_mtp and getattr(config, "use_kl_loss", False)
        self.use_e2e_tv_loss = self.use_mtp and getattr(
            config, "use_e2e_tv_loss", False
        )
        self.enable_adaptive_weight_loss = self.use_mtp and getattr(
            config, "enable_adaptive_weight_loss", False
        )
        self.adaptive_weight_loss_weight = (
            1.0 if self.enable_adaptive_weight_loss else 0.0
        )
        self.adaptive_weight_lambda_loss_weight = 0.0
        self.draft_loss_kinds = [
            kind
            for kind, enabled in (
                ("lk", self.use_lk_loss),
                ("tv", self.use_tv_loss),
                ("kl", self.use_kl_loss),
                ("adaptive_weight", self.enable_adaptive_weight_loss),
                ("adaptive_weight_lambda", self.enable_adaptive_weight_loss),
                ("e2e_tv", self.use_e2e_tv_loss),
            )
            if enabled
        ]
        if self.draft_loss_kinds and config.tp_size > 1:
            raise ValueError(
                "Draft losses (lk/tv/kl/adaptive_weight/e2e_tv) need a full-vocab "
                "softmax, but tp_size > 1 shards the lm_head output"
            )
        if self.use_mtp:
            active_mtp_weights = {
                "mtp_ce_loss_weight": self.mtp_ce_loss_weight,
                **{
                    f"{kind}_loss_weight": getattr(self, f"{kind}_loss_weight")
                    for kind in self.draft_loss_kinds
                },
            }
            if not any(weight != 0.0 for weight in active_mtp_weights.values()):
                warnings.warn(
                    "use_mtp=True but every MTP loss weight is 0 "
                    f"({active_mtp_weights}), so the MTP heads will receive zero "
                    "gradient. Set mtp_ce_loss_weight, or one of "
                    "lk/tv/kl/e2e_tv_loss_weight, or enable_adaptive_weight_loss.",
                    stacklevel=2,
                )

        if config.lm_head_logit_softcapping is not None:
            self.logit_softcap_fn = torch.compile(
                self.compute_logit_softcapping, fullgraph=True
            )

    @staticmethod
    def compute_logit_softcapping(logits: torch.Tensor, softcap_value: float):
        logits_in_fp32 = logits.float()
        softcapped_logits = softcap_value * torch.tanh(logits_in_fp32 / softcap_value)

        return softcapped_logits.to(dtype=logits.dtype)

    @staticmethod
    def _build_packed_loss_weights(
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        bs, seq_len = labels.shape
        device = labels.device

        # Flatten position_ids to 1-D; packed-sample boundaries are where pos == 0
        pos_flat = position_ids.reshape(-1).long()  # [bs * seq_len]
        # sample_id for each token: increment at every position reset to 0
        sample_id = (pos_flat == 0).cumsum(0) - 1  # [bs * seq_len], 0-indexed

        # Non-ignored mask, flattened
        valid_mask = (labels.reshape(-1) != ignore_index).float()  # [bs * seq_len]

        num_samples = int(sample_id[-1].item()) + 1

        # n_s: number of non-ignored tokens per packed sample
        n_s = torch.zeros(num_samples, dtype=torch.float32, device=device)
        n_s.scatter_add_(0, sample_id, valid_mask)  # [num_samples]

        S = (n_s > 0).float().sum()
        S = S.clamp_min_(1.0)
        sample_n_s = n_s[sample_id]  # [bs * seq_len]
        sample_n_s = sample_n_s.clamp_min_(1.0)  # avoid div by zero for empty samples

        weights_flat = valid_mask / (S * sample_n_s)

        return weights_flat.view(bs, seq_len)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _compute_topk_accuracy(self, logits, shift_labels):
        flat_logits = logits.view(-1, logits.size(-1))[: shift_labels.numel()]
        flat_labels = shift_labels.view(-1)
        mask = flat_labels != self.ignore_index
        topk_results = {}
        if self.topk_observer is not None and flat_labels.numel() >= 2**31:
            raise ValueError("rank-local top-k count exceeds Int32")
        local_counts = [mask.sum().to(torch.int32)] if self.topk_observer is not None else []

        max_k = max(self._topk_values)
        topk_ids = flat_logits.topk(max_k, dim=-1).indices
        for k in self._topk_values:
            if k == 1:
                correct = (topk_ids[:, 0] == flat_labels) & mask
            else:
                correct = (topk_ids[:, :k] == flat_labels.unsqueeze(-1)).any(
                    dim=-1
                ) & mask
            correct_count = correct.sum()
            if self.topk_observer is not None:
                local_counts.append(correct_count.to(torch.int32))
            topk_results[k] = correct_count.float()
        topk_results["num_tokens"] = mask.sum().float()
        if self.topk_observer is not None:
            self.topk_observer(torch.stack(local_counts))
        if self._compute_topk_this_step:
            self._topk_results = topk_results

    def _finalize_mtp_acc_len(self, all_argmaxes, shift_labels, same_doc_next=None):
        mtp_num = self.config.mtp_predictor_num
        shifted = self._build_shifted_labels(shift_labels, same_doc_next)
        base_pred = all_argmaxes[0]

        full_valid = shift_labels != self.ignore_index
        cumulative_gts = []
        cumulative_bases = []
        cumulative_gt = None
        cumulative_base = None
        for h in range(1, mtp_num + 1):
            pred_h = all_argmaxes[h]
            label_h = shifted[h]
            valid_h = label_h != self.ignore_index
            base_pred_h = base_pred.roll(shifts=(-h,), dims=(1,))

            match_gt = (pred_h == label_h) & valid_h
            match_base = (pred_h == base_pred_h) & valid_h
            cumulative_gt = (
                match_gt if cumulative_gt is None else (cumulative_gt & match_gt)
            )
            cumulative_base = (
                match_base
                if cumulative_base is None
                else (cumulative_base & match_base)
            )
            cumulative_gts.append(cumulative_gt)
            cumulative_bases.append(cumulative_base)
            full_valid = full_valid & valid_h

        self._mtp_acc_len_results = {
            "full_valid": full_valid.sum().float(),
            "chain_full_gt": torch.stack(
                [(cum & full_valid).sum().float() for cum in cumulative_gts]
            ),
            "chain_full_base": torch.stack(
                [(cum & full_valid).sum().float() for cum in cumulative_bases]
            ),
        }

    def _compute_mtp_acc_len(self, raw_logits, shift_labels, same_doc_next=None):
        if not self.use_mtp:
            return
        mtp_num = self.config.mtp_predictor_num
        if mtp_num <= 0:
            return

        bs, seq = shift_labels.shape
        n = bs * seq
        flat_logits = raw_logits.view(-1, raw_logits.size(-1))
        if flat_logits.size(0) < (mtp_num + 1) * n:
            return

        all_argmaxes = [
            flat_logits[h * n : (h + 1) * n].argmax(dim=-1).view(bs, seq)
            for h in range(mtp_num + 1)
        ]
        self._finalize_mtp_acc_len(all_argmaxes, shift_labels, same_doc_next)

    def _compute_mtp_acc_len_chunked(
        self, combined_hidden, shift_labels, chunk_size=2048, same_doc_next=None
    ):
        # Liger path: raw_logits is never materialized. Stream argmax per-head
        # in chunks so we never hold ((mtp+1)*bs*seq, vocab) in memory at once.
        if not self.use_mtp:
            return
        mtp_num = self.config.mtp_predictor_num
        if mtp_num <= 0:
            return
        if len(combined_hidden) != mtp_num + 1:
            return

        bs, seq = shift_labels.shape
        n = bs * seq
        all_argmaxes = []
        for h_idx in range(mtp_num + 1):
            hidden = combined_hidden[h_idx]
            hidden_flat = hidden.reshape(-1, hidden.size(-1))
            if hidden_flat.size(0) != n:
                return
            chunks = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                logits_chunk = self.lm_head(hidden_flat[start:end])
                chunks.append(logits_chunk.argmax(dim=-1))
                del logits_chunk
            all_argmaxes.append(torch.cat(chunks, dim=0).view(bs, seq))
        self._finalize_mtp_acc_len(all_argmaxes, shift_labels, same_doc_next)

    @staticmethod
    def same_doc_next_mask(
        position_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """`True` at `j` when token `j + 1` is in the same document as token `j`.

        Boundaries come from `position_ids` restarting at 0; the last column is `False`.
        """
        if position_ids is None:
            return None
        same = torch.zeros_like(position_ids, dtype=torch.bool)
        same[:, :-1] = position_ids[:, 1:] > position_ids[:, :-1]
        return same

    def _build_shifted_labels(
        self,
        shift_labels: torch.Tensor,
        same_doc_next: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Per-head labels: entry `h` holds `labels[j + h + 1]` at position `j`.

        With `same_doc_next` (from `same_doc_next_mask`), head `h` is supervised
        at `j` only when `j` and `j + h + 1` are the same document -- `packing.py`
        masks one position per seam, which covers head 0 alone. `None` keeps the
        ungated behaviour.
        """
        labels = [shift_labels]
        if self.use_mtp:
            cur = shift_labels
            gate = same_doc_next
            for _ in range(self.config.mtp_predictor_num):
                with torch.no_grad():
                    cur = cur.roll(shifts=(-1,), dims=(1,)).contiguous()
                    cur[:, -1] = self.ignore_index
                    if gate is not None:
                        gate = same_doc_next & gate.roll(shifts=(-1,), dims=(1,))
                        cur.masked_fill_(~gate, self.ignore_index)
                labels.append(cur)
        return labels

    def _compute_valid_counts(self, labels_stack):
        valid_counts = (labels_stack != self.ignore_index).sum(dim=-1)
        sp_size = dist.get_sp_group_size() or 1
        if sp_size > 1:
            dist.all_reduce(
                valid_counts,
                reduce_operation="SUM",
                group=dist.get_sp_group(),
            )
        return valid_counts

    def _mtp_position_weights(
        self, device, dtype=torch.float32, beta=None
    ) -> torch.Tensor:
        beta = beta or self.config.mtp_position_weight_beta
        k = torch.arange(self.config.mtp_predictor_num, device=device, dtype=dtype)
        weights = beta**k
        return weights / weights.sum()

    def _aggregate_per_head(self, per_head: torch.Tensor, beta=None) -> torch.Tensor:
        """Collapse a per-head loss vector with the exponential position decay `beta`."""
        beta = beta or self.config.mtp_position_weight_beta
        if beta == 1.0:
            return per_head.mean()
        weights = self._mtp_position_weights(per_head.device, per_head.dtype, beta)
        return (weights * per_head).sum()

    @staticmethod
    def _mtp_num_tokens(valid_counts: torch.Tensor) -> torch.Tensor:
        """Valid targets per MTP head, summed over the batch. `[mtp_predictor_num]`

        Normalizer for the draft losses only: the MTP cross-entropy normalizes per
        sample instead, so the two are not on the same scale.
        """
        return valid_counts[1:].sum(dim=-1).clamp_min(1).to(torch.float32)

    def _compute_mtp_loss(self, losses, valid_counts):
        if not self.use_mtp:
            return torch.zeros((), device=losses.device), None

        mtp_losses = losses[1:]  ## 0 is for main block
        num_tokens = valid_counts[1:].clamp_min(1).to(torch.float32)
        sp_size = dist.get_sp_group_size() or 1
        per_head = sp_size * (mtp_losses.sum(dim=-1) / num_tokens).mean(
            dim=-1
        )  # [mtp_predictor_num]
        return self._aggregate_per_head(per_head), per_head

    DRAFT_LOSS_BETA_ATTR = {
        "lk": "lk_position_weight_beta",
        "tv": "tv_position_weight_beta",
        "kl": "kl_position_weight_beta",
        "adaptive_weight": "adaptive_weight_position_weight_beta",
        "adaptive_weight_lambda": "adaptive_weight_position_weight_beta",
    }

    DRAFT_DIAGNOSTIC_KINDS = ("adaptive_weight_lambda",)

    DRAFT_LOSS_TERMS = {
        "lk": ("lk",),
        "tv": ("tv",),
        "kl": ("kl",),
        "adaptive_weight": ("kl", "tv"),
        "adaptive_weight_lambda": ("tv",),
        "e2e_tv": (),
    }

    def _draft_log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        """fp32 log-softmax of the vocab projection of `hidden`. `[*, vocab]`"""
        logits = self.lm_head(hidden)
        softcap = self.config.lm_head_logit_softcapping
        if softcap:
            logits = softcap * torch.tanh(logits.float() / softcap)
        return torch.log_softmax(logits, dim=-1, dtype=torch.float32)

    def _compute_draft_losses(
        self,
        combined_hidden,
        shift_labels,
        kinds,
        mtp_labels,
        valid_counts,
        chunk_size=None,
    ):
        """Distribution-matching losses between the main head (`p`, no grad) and the
        MTP heads (`q`).

        Kinds: `lk`, `tv`, `kl`, `adaptive_weight`, `adaptive_weight_lambda`
        (diagnostic) and `e2e_tv`. Returns `{kind: (loss, per_head)}`, each collapsed
        with its own `<kind>_position_weight_beta` decay; a kind whose weight is 0 is
        reduced under `no_grad` and only logged.

        Chunked by `draft_loss_chunk_size` -- lower it if a run OOMs here.
        """
        mtp_num = self.config.mtp_predictor_num
        if not kinds:
            return {}
        if len(combined_hidden) != mtp_num + 1:
            raise ValueError(
                f"draft losses need main + {mtp_num} MTP hidden states, "
                f"got {len(combined_hidden)}"
            )
        bs, seq_len = shift_labels.shape
        for h, hidden in enumerate(combined_hidden):
            if tuple(hidden.shape[:2]) != (bs, seq_len):
                raise ValueError(
                    f"combined_hidden[{h}] has shape {tuple(hidden.shape)}, expected "
                    f"({bs}, {seq_len}, hidden_size) to match shift_labels"
                )

        if chunk_size is None:
            chunk_size = getattr(self.config, "draft_loss_chunk_size", 2048)
        lk_eps = self.config.lk_loss_eps
        hidden_size = combined_hidden[0].size(-1)
        sp_size = dist.get_sp_group_size() or 1

        terms = []
        for kind in kinds:
            for term in self.DRAFT_LOSS_TERMS[kind]:
                if term not in terms:
                    terms.append(term)
        term_idx = {term: i for i, term in enumerate(terms)}
        want_chain = "e2e_tv" in kinds

        want_adaptive = any(k.startswith("adaptive_weight") for k in kinds)
        need_m = want_chain or "lk" in term_idx or "tv" in term_idx
        need_kl = "kl" in term_idx

        def trained(kind):
            return kind in kinds and getattr(self, f"{kind}_loss_weight") != 0.0

        grad_m = (
            trained("lk")
            or trained("tv")
            or trained("adaptive_weight")
            or trained("e2e_tv")
        )
        grad_kl = trained("kl") or trained("adaptive_weight")
        needs_grad = grad_m or grad_kl

        def draft_chunk(target_hidden, *drafts_and_valids):
            drafts = drafts_and_valids[:mtp_num]
            valids = drafts_and_valids[mtp_num:]
            rows = target_hidden.size(0)

            with torch.no_grad():
                log_p = self._draft_log_probs(target_hidden)

            log_q_all = self._draft_log_probs(torch.cat(drafts, dim=0))

            head_losses = []
            head_alphas = []
            for h_idx in range(mtp_num):
                log_q = log_q_all[h_idx * rows : (h_idx + 1) * rows]
                valid = valids[h_idx]

                m = kl = None
                with torch.no_grad():
                    p = log_p.exp()
                if need_m:
                    with torch.set_grad_enabled(grad_m):
                        m = torch.minimum(p, log_q.exp()).sum(-1)
                if need_kl:
                    with torch.set_grad_enabled(grad_kl):
                        kl = (p * (log_p - log_q)).sum(-1)
                del p

                per_term = []
                for term in terms:
                    if term == "lk":
                        per_token = -m.clamp_min(lk_eps).log()
                    elif term == "tv":
                        per_token = 1.0 - m
                    else:
                        per_token = kl
                    per_term.append((per_token * valid).sum())
                head_losses.append(
                    torch.stack(per_term)
                    if per_term
                    else valid.new_zeros(0, dtype=torch.float32)
                )
                if want_chain:
                    head_alphas.append(torch.where(valid > 0, m, torch.ones_like(m)))
            return (
                torch.stack(head_losses),
                torch.stack(head_alphas) if want_chain else None,
            )

        target_flat = combined_hidden[0].reshape(-1, hidden_size).detach()
        drafts_flat = []
        valids_flat = []
        for h in range(1, mtp_num + 1):
            drafts_flat.append(
                combined_hidden[h].roll(shifts=(h,), dims=(1,)).reshape(-1, hidden_size)
            )
            valid_h = (mtp_labels[h] != self.ignore_index).to(torch.float32)
            valids_flat.append(valid_h.roll(shifts=(h,), dims=(1,)).reshape(-1))

        n = target_flat.size(0)
        total = torch.zeros(
            mtp_num, len(terms), device=target_flat.device, dtype=torch.float32
        )
        alpha_chunks = []
        for start in range(0, n, chunk_size):
            chunk = slice(start, min(start + chunk_size, n))
            args = (
                target_flat[chunk],
                *(draft[chunk] for draft in drafts_flat),
                *(valid[chunk] for valid in valids_flat),
            )
            if needs_grad:
                reduced, alpha_chunk = torch.utils.checkpoint.checkpoint(
                    draft_chunk, *args, use_reentrant=False
                )
            else:
                with torch.no_grad():
                    reduced, alpha_chunk = draft_chunk(*args)
            if want_chain:
                alpha_chunks.append(alpha_chunk)
            total = total + reduced

        num_tokens = self._mtp_num_tokens(valid_counts)
        per_head_term = sp_size * total / num_tokens.unsqueeze(-1)

        lam = None
        if want_adaptive:
            tv_sum = total[:, term_idx["tv"]].detach().clone()
            alpha_tokens = num_tokens.detach().clone()
            if sp_size > 1:
                dist.all_reduce(
                    tv_sum,
                    reduce_operation="SUM",
                    group=dist.get_sp_group(),
                )
            fsdp_size = dist.get_fsdp_group_size() or 1
            if fsdp_size > 1:
                packed = torch.cat([tv_sum, alpha_tokens])
                dist.all_reduce(
                    packed,
                    reduce_operation="SUM",
                    group=dist.get_fsdp_group(),
                )
                tv_sum, alpha_tokens = packed.split(tv_sum.numel())
            alpha = (1.0 - tv_sum / alpha_tokens).clamp_(0.0, 1.0)
            lam = torch.exp(-self.config.nu_adaptive_weight * alpha)

        out = {}
        for kind in kinds:
            if kind == "e2e_tv":
                out[kind] = self._e2e_chain_loss(
                    torch.cat(alpha_chunks, dim=1).view(mtp_num, bs, seq_len),
                    mtp_labels,
                )
                continue
            if kind == "adaptive_weight":
                per_head = (
                    lam * per_head_term[:, term_idx["kl"]]
                    + (1.0 - lam) * per_head_term[:, term_idx["tv"]]
                )
            elif kind == "adaptive_weight_lambda":
                per_head = lam
            else:
                per_head = per_head_term[:, term_idx[kind]]
            out[kind] = (
                self._aggregate_per_head(
                    per_head,
                    getattr(self.config, self.DRAFT_LOSS_BETA_ATTR[kind]),
                ),
                per_head,
            )
        return out

    def _e2e_chain_loss(self, alpha, mtp_labels):
        """Expected accepted length over the draft chain:

            prefix[m] = prod_{i <= m} alpha_i
            loss      = 1 - sum_m prefix[m] / mtp_predictor_num

        Returns `(loss, per_depth)`. `per_depth` is not
        `mtp_acc_len/head{m}_cumulative` and the two cannot be differenced.
        """
        mtp_num = self.config.mtp_predictor_num
        alpha = torch.stack(
            [alpha[h - 1].roll(shifts=(-h,), dims=(1,)) for h in range(1, mtp_num + 1)]
        )

        valid_all = (torch.stack(mtp_labels[1:]) != self.ignore_index).all(dim=0)

        rho = self.config.e2e_alpha_floor
        alpha = alpha.clamp(0.0, 1.0)
        smoothed = rho + (1.0 - rho) * alpha if rho else alpha

        if self.config.e2e_direct:
            contrib = 1.0 - torch.cumprod(smoothed, dim=0)
        else:
            contrib = self._e2e_chain_weights(smoothed) * (1.0 - alpha)

        sp_size = dist.get_sp_group_size() or 1
        denom = valid_all.sum().to(torch.float32)
        if sp_size > 1:
            dist.all_reduce(denom, reduce_operation="SUM", group=dist.get_sp_group())
        denom = denom.clamp_min(1.0)
        mask = valid_all.unsqueeze(0).to(contrib.dtype)
        per_depth = sp_size * (contrib * mask).sum(dim=(1, 2)) / denom
        return per_depth.mean(), per_depth

    @staticmethod
    def _e2e_chain_weights(alpha: torch.Tensor) -> torch.Tensor:
        """Detached `pre[m] * S[m]` -- AngelSpec's `W_m` without its `1 / D`.

        The `.detach()` is load-bearing.
        """
        det = alpha.detach()
        pre = torch.cat([torch.ones_like(det[:1]), torch.cumprod(det[:-1], dim=0)])
        s = torch.ones_like(det)
        for m in range(det.size(0) - 2, -1, -1):
            s[m] = 1.0 + det[m + 1] * s[m + 1]
        return pre * s

    def _compute_loss(
        self,
        inputs,
        shift_labels,
        loss_fn,
        num_tokens,
        valid_counts=None,
        mtp_labels=None,
        **kwargs,
    ):
        return_z_loss = kwargs.pop("return_z_loss", False)
        # hidden_z_loss is added to the per-token loss below; it is never
        # forwarded to `loss_fn` (lm_head in the Liger path does not accept it).
        hidden_z_loss = kwargs.pop("hidden_z_loss", None)
        loss_weights = kwargs.pop("loss_weights", None)
        if kwargs:
            kwargs["return_z_loss"] = return_z_loss

        labels = (
            self._build_shifted_labels(shift_labels)
            if mtp_labels is None
            else mtp_labels
        )
        labels_stack = torch.stack(labels, dim=0)  ## [num_mtp + 1, bs, seq_len]

        if self.use_mtp and valid_counts is None:
            valid_counts = self._compute_valid_counts(labels_stack)

        flat_inputs = inputs.view(
            -1, inputs.size(dim=-1)
        )  ## [(num_mtp + 1) * bs * seq_len, hidden_dim]
        flat_labels = labels_stack.view(-1).to(
            flat_inputs.device
        )  ## [(num_mtp + 1) * bs * seq_len]
        loss_flat = loss_fn(flat_inputs, flat_labels, **kwargs)

        z_loss = None
        if (
            self.z_loss_eps != 0.0 and self.config.use_liger
        ):  ## same as if return_z_loss
            loss_flat, z_loss = loss_flat

        bs, seq_len = shift_labels.size()
        if self.loss_observer is not None and self.training and torch.is_grad_enabled():
            self.loss_observer(
                loss_flat.detach().view(len(labels), bs, seq_len),
                flat_labels.view(len(labels), bs, seq_len) != self.ignore_index,
            )

        # # NOTE(m1kol): Might be necessary, need to remove in the future if FA
        # # and Liger versions are better (see `lse_square_scale` in `loss_fct` above).
        if self.z_loss_eps != 0.0 and not self.config.use_liger:
            z_loss = z_loss_fn(flat_inputs, self.z_loss_eps)
            z_loss.masked_fill_(flat_labels == self.ignore_index, 0.0)
            loss_flat += z_loss

        losses = loss_flat.view(
            len(labels), bs, seq_len
        )  ## len(labels) = num_mtp + 1 , NOTE: (*2 for dpo)

        # calculate cross entropy loss for main block
        # ---
        # Norm CE loss by SP size (`num_tokens = num_tokens / sp_size`) as `num_tokens`
        # are provided from forward and calculated before splitting `labels` to SP chunks.
        sp_size = dist.get_sp_group_size() or 1
        loss = losses[0]
        if hidden_z_loss is not None:
            loss = loss + hidden_z_loss.view(bs, -1)
        loss_ce = loss.view(bs, -1)
        if loss_weights is not None:
            # Packed-sequence aggregation: each dialog's configured mass is spread
            # uniformly over its n_s valid tokens and normalized globally.
            # sp_size * sum(loss * w_local) recovers the full packed loss because
            # each SP rank owns a contiguous slice and the framework averages
            # gradients across the SP group (same invariant as the batch path).
            if getattr(self.config, "_data_profiler_enabled", False):
                # Profiler: fall back to batch-element normalization (partial under SP)
                loss_ce_profiler = loss_ce.sum(-1)
                loss_ce_profiler = sp_size * loss_ce_profiler / num_tokens
                self.config._data_profiler_per_sample_losses = loss_ce_profiler.detach()
            loss_ce = sp_size * (loss_ce * loss_weights).sum()
        else:
            loss_ce = loss_ce.sum(-1)
            loss_ce = sp_size * loss_ce / num_tokens
            # Upweight samples that pack multiple sequences: scale by sqrt(#packed).
            # if num_packed_sequences is not None:
            #     loss_ce = loss_ce * num_packed_sequences.to(loss_ce.dtype).sqrt()

            if getattr(self.config, "_data_profiler_enabled", False):
                self.config._data_profiler_per_sample_losses = loss_ce.detach()
            loss_ce = loss_ce.mean()

        # calculate avg cross entropy loss for mtp blocks
        loss_mtp, loss_mtp_per_head = self._compute_mtp_loss(losses, valid_counts)

        # calculate avg z-loss for main and mtp blocks
        if z_loss is not None:
            z_loss = z_loss.view(len(labels), bs, seq_len)
            steps = torch.arange(
                0, self.config.mtp_predictor_num + 1, device=num_tokens.device
            ).view(-1, 1)
            zloss_num_tokens = (
                (num_tokens.view(1, -1) - steps).clamp_min_(1).to(torch.float32)
            )
            z_loss = (z_loss.sum(dim=-1) / zloss_num_tokens).sum(dim=0)
            z_loss = sp_size * z_loss.mean() / len(labels)

        if hidden_z_loss is not None:
            hidden_z_loss = hidden_z_loss.view(bs, seq_len)
            hidden_z_loss = sp_size * (hidden_z_loss.sum(dim=-1) / num_tokens).mean()

        return loss_ce, loss_mtp, z_loss, hidden_z_loss, loss_mtp_per_head

    def calculate_liger_loss(
        self,
        hidden_states,
        outputs,
        shift_labels,
        num_tokens,
        return_z_loss,
        hidden_z_loss,
        mtp_labels=None,
        valid_counts=None,
    ):
        grad_output = None
        if not self.use_mtp:
            valid_counts = None
        if self.use_mtp:
            if isinstance(outputs, tuple):
                mtp_hidden_states = outputs[-1]
                # note (sbr):
                #  it is important to remove the last element from the tuple
                #  because the number of elements in the final outputs
                outputs = outputs[:-1]
            else:
                raise NotImplementedError("Returning 'dict' is not supported with mtp")

            if valid_counts is None:
                valid_counts = self._compute_valid_counts(
                    torch.stack(
                        mtp_labels
                        if mtp_labels is not None
                        else self._build_shifted_labels(shift_labels),
                        dim=0,
                    )
                )

            def compute_static_grad_output_for_mtp() -> torch.Tensor:
                """
                Statically сomputes gradient for loss_flat tensor of this operations:
                ```python
                losses = loss_flat.view(num_mtp + 1, bs, seq_len)

                loss_ce = losses[0].view(bs, -1).sum(-1)
                loss_ce = (sp_group_size * loss_ce / num_tokens).mean()

                ## `_compute_mtp_loss`:
                per_head = sp_group_size * (losses[1:].sum(-1) / tokens_stack).mean(-1)
                loss_mtp = (self._mtp_position_weights(...) * per_head).sum()

                loss = loss_ce + self.config.mtp_loss_weight * (
                    self.mtp_ce_loss_weight * loss_mtp
                )
                ```

                The draft losses (lk/tv/kl) also sit inside `mtp_loss_weight` in `forward`,
                but they are ordinary autograd graphs and so are not covered here.

                Liger does not support computational graphs where gradient losses contain unequal values in the backward pass.
                For this reason we compute grad_output and pass it trough Liger forward.
                """
                bs, seq_len = shift_labels.size()
                sp_size = dist.get_sp_group_size() or 1

                # fp32 computing for numerical stability inside Liger kernel
                with torch.amp.autocast("cuda", enabled=False):
                    tokens_stack = valid_counts[1:].clamp_min(1).to(torch.float32)

                    num_mtp = self.config.mtp_predictor_num
                    mtp_loss_weight = (
                        self.config.mtp_loss_weight * self.mtp_ce_loss_weight
                    )
                    total_elements = (num_mtp + 1) * bs * seq_len
                    loss_flat_grad = torch.zeros(
                        total_elements, device=num_tokens.device, dtype=torch.float32
                    )  # ((num_mtp+1)*bs*seq_len,)

                    # Main loss gradients
                    grad_main = sp_size / (num_tokens * bs)  # (bs,)
                    grad_main = grad_main.repeat_interleave(seq_len)  # (bs * seq_len,)
                    main_elements = bs * seq_len
                    loss_flat_grad[:main_elements] = grad_main

                    # MTP loss gradients
                    weights = self._mtp_position_weights(tokens_stack.device).view(
                        -1, 1
                    )
                    grad_mtp = (
                        sp_size * mtp_loss_weight * weights / (bs * tokens_stack)
                    )  # (num_mtp, bs)
                    grad_mtp = grad_mtp.unsqueeze(-1)  # (num_mtp, bs, 1)
                    grad_mtp = grad_mtp.expand(
                        -1, -1, seq_len
                    )  # (num_mtp, bs, seq_len)
                    grad_mtp = grad_mtp.reshape(-1)  # (num_mtp * bs * seq_len,)

                    loss_flat_grad[main_elements:] = grad_mtp

                    if (
                        torch.isnan(loss_flat_grad).any()
                        or torch.isinf(loss_flat_grad).any()
                    ):
                        raise ValueError("Invalid values in grad_output computation")

                return loss_flat_grad

            grad_output = compute_static_grad_output_for_mtp()
            assert isinstance(mtp_hidden_states, tuple)
            combined_hidden = (hidden_states,) + mtp_hidden_states
        else:

            def compute_static_grad_output_for_sft() -> torch.Tensor:
                """
                Statically сomputes gradient for loss_flat tensor of this operations:
                ```python
                loss_ce = loss.view(labels.shape[0], -1) # (bs, seq_len)
                loss_ce = loss_ce.sum(-1)
                loss_ce = sp_group_size * loss_ce / num_tokens
                loss_ce = loss_ce.mean()
                ```

                Liger does not support computational graphs where gradient losses contain unequal values in the backward pass.
                For this reason we compute grad_output and pass it trough Liger forward.
                """
                bs, seq_len = shift_labels.size()
                sp_size = dist.get_sp_group_size() or 1

                # fp32 computing for numerical stability inside Liger kernel
                with torch.amp.autocast("cuda", enabled=False):
                    # CE loss gradients
                    grad_loss = sp_size / (num_tokens * bs)  # (bs,)
                    grad_loss = grad_loss.repeat_interleave(seq_len)  # (bs * seq_len,)

                    if torch.isnan(grad_loss).any() or torch.isinf(grad_loss).any():
                        raise ValueError("Invalid values in grad_output computation")

                return grad_loss

            # If training is performed on the SFT dataset
            if num_tokens.numel() > 1 and not torch.all(
                num_tokens == num_tokens.reshape(-1)[0]
            ):
                grad_output = compute_static_grad_output_for_sft()

            combined_hidden = (hidden_states,)

        loss, loss_mtp, z_loss, hidden_z_loss, loss_mtp_per_head = self._compute_loss(
            torch.cat(combined_hidden, dim=0)
            if len(combined_hidden) > 1
            else combined_hidden[0],
            shift_labels,
            self.lm_head,
            num_tokens=num_tokens,
            valid_counts=valid_counts,
            mtp_labels=mtp_labels,
            use_liger=self.config.use_liger,
            process_group=dist.get_tp_group(),
            lse_square_scale=self.z_loss_eps,
            softcap_value=self.config.lm_head_logit_softcapping or 0.0,
            return_z_loss=return_z_loss,
            precomputed_grad_output=grad_output,
            hidden_z_loss=hidden_z_loss,
        )
        return loss, loss_mtp, z_loss, outputs, hidden_z_loss, loss_mtp_per_head

    def calculate_standard_loss(
        self,
        logits,
        shift_labels,
        num_tokens,
        return_z_loss,
        hidden_z_loss,
        mtp_labels=None,
        valid_counts=None,
        packed_loss_weights=None,
    ):
        loss_fn = FusedCrossEntropyLoss(
            inplace_backward=self.config.loss_inplace_backward,
            reduction="none",
            process_group=dist.get_tp_group(),
            softcap_value=self.config.lm_head_logit_softcapping or 0.0,
            # NOTE(m1kol): Disable Z-loss calculation via FA for now as
            # DCLM tests show it's broken somehow. Z-loss is calculated
            # later using Python (torch).
            # lse_square_scale=self.z_loss_eps,
            # return_z_loss=return_z_loss,
            ignore_index=self.ignore_index,
        )

        loss, loss_mtp, z_loss, hidden_z_loss, loss_mtp_per_head = self._compute_loss(
            logits.contiguous(),
            shift_labels,
            loss_fn,
            num_tokens=num_tokens,
            valid_counts=valid_counts,
            mtp_labels=mtp_labels,
            return_z_loss=return_z_loss,
            hidden_z_loss=hidden_z_loss,
            loss_weights=packed_loss_weights,
        )
        return loss, loss_mtp, z_loss, hidden_z_loss, loss_mtp_per_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logical_batch_size: Optional[int] = None,
        is_left_padded_eval: bool = False,
        prefix_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                0 at pad positions, for the MoE router / dispatcher and MTP only. Defaults to
                `input_ids != pad_token_id`, or to `attention_mask` with `inputs_embeds`.
                Full (pre-SP) sequence; split here with `input_ids`.
        """
        if input_ids is not None:
            bs, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            bs, seq_len, _ = inputs_embeds.shape
        else:
            raise RuntimeError(
                "Either `input_ids` or `inputs_embeds` must be provided, but both are None."
            )

        assert (input_ids is None) != (inputs_embeds is None), (
            "`input_ids` and `inputs_embeds` are mutually exclusive arguments"
        )

        if self.model.config.enable_async_tp and logical_batch_size is None:
            logical_batch_size = bs

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            num_tokens = (labels != self.ignore_index).sum(-1)

        # Build per-token loss weights for packed-sequence aggregation.
        # Must be computed BEFORE the SP split so it covers the full global sequence.
        packed_loss_weights = None
        if (
            labels is not None
            and getattr(self.config, "loss_reduction_mode", "batch") == "packed"
        ):
            if self.config.use_liger and self.training:
                raise NotImplementedError(
                    "loss_reduction_mode='packed' is not supported with use_liger=True."
                )
            if position_ids is not None:
                packed_loss_weights = self._build_packed_loss_weights(
                    position_ids,
                    labels,
                    self.ignore_index,
                )
            # If position_ids is None, fall back to default "batch" aggregation silently.

        # Split sequences for sequence parallel case
        is_sequence_parallel_active: bool = (
            dist.get_sp_group_size() is not None and dist.get_sp_group_size() > 1
        )
        is_ring_attn_type: bool = self.config.attention_type in RING_ATTN_CLASSES

        # NOTE (Sbr): Disable sequence parallel processing during eval mode as there's
        # an error on generative eval (gsm8k).
        # ---
        # TODO (Sbr): Make SP available during eval mode as well. Now there's a problem
        # that the gsm8k gets stuck in the end and fails due to NCCL timeout error.
        if not self.training:
            is_sequence_parallel_active = False

        sp_group_size = (
            dist.get_sp_group_size() if dist.get_sp_group_size() is not None else 1
        )
        # multimodal models might omit this argument
        if input_ids is not None:
            assert input_ids.shape[-1] % sp_group_size == 0, (
                "input_ids length should be divisible by sp_group_size"
            )

        if inputs_embeds is not None:
            assert inputs_embeds.shape[1] % sp_group_size == 0, (
                "inputs_embeds length should be divisible by tp_sp_size"
            )

        full_seq_len = (
            input_ids.shape[-1]
            if input_ids is not None
            else inputs_embeds.shape[1]
            if inputs_embeds is not None
            else None
        )
        if position_ids is not None and full_seq_len is not None:
            assert position_ids.shape[-1] == full_seq_len, (
                f"position_ids covers {position_ids.shape[-1]} of {full_seq_len} "
                "positions; varlen attention and the document gate would "
                "silently mis-place every boundary after the covered prefix"
            )

        cu_seqlens = max_seqlen = None
        # DSA + SFT: number of real (non-trailing-pad) tokens, from the GLOBAL
        # input_ids before any SP split. Sync-free: stays a 0-d device tensor.
        dsa_q_real_max = None
        if (
            input_ids is not None
            and getattr(self.config, "attention_type", None)
            == "LlamaLatentSparseAttention"
        ):
            _pad_id = getattr(self.config, "pad_token_id", None)
            _not_pad = input_ids[0] != (0 if _pad_id is None else int(_pad_id))
            dsa_q_real_max = input_ids.shape[-1] - _not_pad.flip(0).int().argmax()
        # In the llama3-ring-attention case we need to compute cu_seqlens before SP,
        # because this ring-attention only works with global cu_seqlens
        if (
            self.config.sp_split_type == "llama3"
            and position_ids is not None
            and (getattr(self.config, "varlen_input", False) or not self.training)
        ):
            cu_seqlens, max_seqlen = get_llama3_cu_seqlens_from_pos_ids(position_ids)

        same_doc_next = self.same_doc_next_mask(position_ids)

        # Split sequences for sequence parallel case
        if is_sequence_parallel_active:
            if position_ids is None and is_ring_attn_type:
                position_ids = torch.arange(
                    0,
                    input_ids.shape[-1],
                    dtype=torch.long,
                    device=input_ids.device,
                ).view(1, -1)

            # NOTE (Sbr): Would probably need to add `past_key_values` split if they will be used. They are not
            # used as we use `use_cache=Fasle`. Would probably also need to change code in other places too.

            # We caculate attention for the whole sequence so we don't split `attention_mask` and `position_ids`.

            (
                input_ids,
                inputs_embeds,
                labels,
                same_doc_next,
                packed_loss_weights,
                padding_mask,
            ) = get_tensors_sp_part(
                (
                    input_ids,
                    inputs_embeds,
                    labels,
                    same_doc_next,
                    packed_loss_weights,
                    padding_mask,
                ),
                split_type=self.config.sp_split_type if self.training else "equal",
                dim=1,
            )

            if is_ring_attn_type:
                (position_ids,) = get_tensors_sp_part(
                    position_ids, split_type=self.config.sp_split_type, dim=-1
                )

        # In the case of ring-attention types, except for llama3 we need to compute cu_seqlens after SP,
        # because these ring-attention types only work with locally distributed cu_seqlens
        if (
            self.config.sp_split_type != "llama3"
            and position_ids is not None
            and (getattr(self.config, "varlen_input", False) or not self.training)
        ):
            cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)
            if dsa_q_real_max is not None and cu_seqlens is not None:
                cu_seqlens._dsa_q_real_max = dsa_q_real_max

        if (
            cu_seqlens is None
            and self.training
            and getattr(self.config, "varlen_input", False)
        ):
            # Without boundary position_ids there is nothing to derive cu_seqlens from, and every
            # sample would silently be attended as ONE document - varlen_input would be a no-op
            raise ValueError(
                "varlen_input=true but the batch carries no position_ids, so document boundaries "
                "(cu_seqlens) cannot be derived. Use batch_type='flash_attn' in the dataloader so "
                "the collator emits boundary position_ids."
            )

        if prefix_mask is None:
            if labels is not None:
                prefix_mask = (labels != self.ignore_index).to(torch.int32)
            else:
                if input_ids is not None:
                    prefix_mask = torch.ones_like(
                        input_ids, device=input_ids.device, dtype=torch.int32
                    )
                else:
                    prefix_mask = torch.ones_like(
                        inputs_embeds[:, :, 0],
                        device=inputs_embeds.device,
                        dtype=torch.int32,
                    )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logical_batch_size=logical_batch_size,
            is_left_padded_eval=is_left_padded_eval,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            prefix_mask=prefix_mask,
            lm_head_fn=self.lm_head if self.config.on_policy_ttt else None,
            padding_mask=padding_mask,
        )

        hidden_states = outputs[0]

        per_gpu_vocab_size = self.config.vocab_size
        if self.training and dist.get_tp_group_size() is not None:
            per_gpu_vocab_size //= dist.get_tp_group_size()

        def check_shapes_match(labels_shape: tuple, inputs: torch.Tensor):
            assert labels_shape == inputs.shape[:2], (
                f"Inputs and labels (bs, seq_len) shape mismatch, got labels {labels_shape} and {inputs.shape[:2]} inputs."
            )

        # In evaluation mode we need logits, so we use only lm_head without cross-entropy.
        logits = None
        raw_logits = None
        hidden_z_loss = None
        hidden_z_loss_attn = None
        hidden_z_loss_moe = None
        hidden_z_loss_dense_mlp = None

        # Hidden z-loss is produced by `GigaMixModel.forward` (appended to its
        # output tuple) for both Liger and non-Liger training paths; strip it
        # here before any downstream unpacking of `outputs`.
        # Order in the tuple (from the model): ..., hidden_z_loss,
        # hidden_z_loss_attn, hidden_z_loss_moe, hidden_z_loss_dense_mlp.
        # Strip in reverse. (Per-projection qkv/up/gate losses are NOT in this
        # tuple — they are logged via the module-walk side-channel.)
        if self.hidden_z_loss_dense_mlp_coef != 0.0 and self.training:
            if not isinstance(outputs, tuple):
                raise NotImplementedError(
                    "Returning 'dict' is not supported with z_loss"
                )
            hidden_z_loss_dense_mlp = outputs[-1]
            outputs = outputs[:-1]
        if self.hidden_z_loss_moe_coef != 0.0 and self.training:
            if not isinstance(outputs, tuple):
                raise NotImplementedError(
                    "Returning 'dict' is not supported with z_loss"
                )
            hidden_z_loss_moe = outputs[-1]
            outputs = outputs[:-1]
        if self.hidden_z_loss_attn_coef != 0.0 and self.training:
            if not isinstance(outputs, tuple):
                raise NotImplementedError(
                    "Returning 'dict' is not supported with z_loss"
                )
            hidden_z_loss_attn = outputs[-1]
            outputs = outputs[:-1]
        if self.hidden_z_loss_eps != 0.0 and self.training:
            if not isinstance(outputs, tuple):
                raise NotImplementedError(
                    "Returning 'dict' is not supported with z_loss"
                )
            hidden_z_loss = outputs[-1]
            outputs = outputs[:-1]

        # Liger training fuses lm_head+CE in `calculate_liger_loss`, which also
        # strips MTP hidden states from `outputs` itself. Skip the outer MTP
        # stripping and raw-logit materialization to preserve Liger's memory
        # savings (otherwise the full vocab projection runs every step).
        if not (self.config.use_liger and self.training):
            if self.use_mtp and self.training:
                if isinstance(outputs, tuple):
                    mtp_hidden_states = outputs[-1]
                    outputs = outputs[:-1]  # remove MTP hidden states from outputs
                else:
                    raise NotImplementedError(
                        "Returning 'dict' is not supported with mtp"
                    )
                assert isinstance(mtp_hidden_states, tuple)
                combined_hidden = (hidden_states,) + mtp_hidden_states
            else:
                combined_hidden = (hidden_states,)

            if self.config.tp_size > 1 and self.config.enable_async_tp:
                logit_parts = [
                    self.lm_head(
                        hidden_state,
                        gather_output=not self.training,
                        logical_batch_size=logical_batch_size,
                    )
                    for hidden_state in combined_hidden
                ]
                if len(logit_parts) > 1:
                    raw_logits = torch.cat(logit_parts, dim=0).flatten(0, 1)
                else:
                    raw_logits = logit_parts[0]
            else:
                if len(combined_hidden) > 1:
                    # NOTE: for dpo ((mtp+1)*bs*2, seq_len, hidden_size)
                    cat_hidden = torch.cat(combined_hidden, dim=0)
                else:
                    cat_hidden = combined_hidden[0]
                # NOTE: for dpo ((mtp+1)*bs*2, seq_len, vocab_size)
                logits_cat = self.lm_head(
                    cat_hidden,
                    gather_output=not self.training,
                    logical_batch_size=logical_batch_size * len(combined_hidden)
                    if logical_batch_size
                    else logical_batch_size,
                )

                if len(combined_hidden) > 1:
                    # NOTE: for dpo (seq_len*(mtp+1)*bs*2, vocab_size)
                    raw_logits = logits_cat.flatten(0, 1)
                else:
                    raw_logits = logits_cat

        loss = None
        z_loss = None
        loss_ce = None
        loss_mtp = None
        loss_mtp_per_head = None
        draft_losses = {}
        if labels is not None:
            # Shift so that tokens < n predict n
            # ---
            # NOTE (Sbr): Add labels length slice to `shift_logits` to account for sequence parallel case
            # when the input is padded (strip last padded tokens).

            # cyclic shift labels and replace last element with mask (-100)
            with torch.no_grad():
                shift_labels = labels.roll(shifts=(-1,), dims=(1,)).contiguous()
                shift_labels[:, -1] = self.ignore_index
                if packed_loss_weights is not None:
                    packed_loss_weights = packed_loss_weights.roll(
                        shifts=(-1,), dims=(1,)
                    ).contiguous()
                    packed_loss_weights[:, -1] = 0

            shift_labels_shape = shift_labels.shape

            return_z_loss = (self.z_loss_eps != 0.0) and self.config.use_liger

            mtp_labels = self._build_shifted_labels(shift_labels, same_doc_next)
            mtp_valid_counts = (
                self._compute_valid_counts(torch.stack(mtp_labels, dim=0))
                if self.use_mtp
                else None
            )

            # The liger is only used during training because it fuses the lm_head and cross-entropy operations.
            # For this reason, it's not possible to get logits from lm_head, so it's incompatible with evaluation mode.
            if self.config.use_liger and self.training:
                if (
                    getattr(self, "_compute_mtp_acc_len_this_step", False)
                    and self.use_mtp
                    and isinstance(outputs, tuple)
                    and isinstance(outputs[-1], tuple)
                ):
                    with torch.no_grad():
                        self._compute_mtp_acc_len_chunked(
                            (hidden_states,) + outputs[-1],
                            shift_labels,
                            same_doc_next=same_doc_next,
                        )
                        self._compute_mtp_acc_len_this_step = False
                check_shapes_match(shift_labels_shape, hidden_states)
                if self.draft_loss_kinds:
                    if not (
                        isinstance(outputs, tuple) and isinstance(outputs[-1], tuple)
                    ):
                        raise NotImplementedError(
                            "Draft losses need the MTP hidden states in the model "
                            "outputs, but the last output is not a tuple"
                        )
                    draft_losses = self._compute_draft_losses(
                        (hidden_states,) + outputs[-1],
                        shift_labels,
                        self.draft_loss_kinds,
                        mtp_labels,
                        mtp_valid_counts,
                    )
                loss_ce, loss_mtp, z_loss, outputs, hidden_z_loss, loss_mtp_per_head = (
                    self.calculate_liger_loss(
                        hidden_states,
                        outputs,
                        shift_labels,
                        num_tokens,
                        return_z_loss,
                        hidden_z_loss,
                        mtp_labels=mtp_labels,
                        valid_counts=mtp_valid_counts,
                    )
                )
            else:
                if self.draft_loss_kinds and self.training:
                    draft_losses = self._compute_draft_losses(
                        combined_hidden,
                        shift_labels,
                        self.draft_loss_kinds,
                        mtp_labels,
                        mtp_valid_counts,
                    )
                loss_ce, loss_mtp, z_loss, hidden_z_loss, loss_mtp_per_head = (
                    self.calculate_standard_loss(
                        raw_logits,
                        shift_labels,
                        num_tokens,
                        return_z_loss,
                        hidden_z_loss,
                        mtp_labels=mtp_labels,
                        valid_counts=mtp_valid_counts,
                        packed_loss_weights=packed_loss_weights,
                    )
                )

            if self.use_mtp:
                loss_mtp_total = self.mtp_ce_loss_weight * loss_mtp
                for kind, (draft_loss, _) in draft_losses.items():
                    weight = getattr(self, f"{kind}_loss_weight")
                    if weight != 0.0:
                        loss_mtp_total = loss_mtp_total + weight * draft_loss
                loss = loss_ce + self.config.mtp_loss_weight * loss_mtp_total
            else:
                loss = loss_ce

        if raw_logits is not None and not (self.config.delete_logits and self.training):
            if self.config.lm_head_logit_softcapping is not None:
                logits = self.logit_softcap_fn(
                    logits=raw_logits,
                    softcap_value=self.config.lm_head_logit_softcapping,
                )
            else:
                logits = raw_logits

        # Gather tensors for the whole sequence in evaluation mode
        if not self.training and is_sequence_parallel_active:
            sp_group = dist.get_sp_group()
            logits = torch.concatenate(dist.all_gather(logits, group=sp_group), dim=1)[
                :, :seq_len
            ]
            if outputs.hidden_states is not None:
                outputs.hidden_states = [
                    torch.concatenate(dist.all_gather(hidden, group=sp_group), dim=1)[
                        :, :seq_len
                    ]
                    for hidden in outputs.hidden_states
                ]

        if (
            self.training
            and (getattr(self, "_compute_topk_this_step", False) or self.topk_observer is not None)
            and raw_logits is not None
            and labels is not None
        ):
            with torch.no_grad():
                self._compute_topk_accuracy(raw_logits, shift_labels)
                self._compute_topk_this_step = False

        if (
            self.training
            and getattr(self, "_compute_mtp_acc_len_this_step", False)
            and raw_logits is not None
            and labels is not None
        ):
            with torch.no_grad():
                self._compute_mtp_acc_len(raw_logits, shift_labels, same_doc_next)
                self._compute_mtp_acc_len_this_step = False

        if self.config.delete_logits and self.training:
            del logits
            logits = None

        if not return_dict:
            main_output = tuple() if loss is None else (loss,)
            # kv cache, hidden states and etc
            decoder_outputs = outputs[1:]
            addition_outputs = (loss_ce,)

            main_output += (logits,)
            if z_loss is not None:
                addition_outputs += (z_loss,)

            if self.use_mtp:
                addition_outputs += (loss_mtp,)
                if self.config.mtp_predictor_num > 1:
                    addition_outputs += (loss_mtp_per_head,)

            if hidden_z_loss is not None:
                addition_outputs += (hidden_z_loss,)

            if hidden_z_loss_attn is not None:
                addition_outputs += (hidden_z_loss_attn,)

            if hidden_z_loss_moe is not None:
                addition_outputs += (hidden_z_loss_moe,)

            if hidden_z_loss_dense_mlp is not None:
                addition_outputs += (hidden_z_loss_dense_mlp,)

            for kind in self.draft_loss_kinds:
                addition_outputs += draft_losses.get(kind, (None, None))

            return main_output + decoder_outputs + addition_outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _check_disbalance(self):
        skip_step = False
        for layer in self.model.layers:
            if hasattr(layer, "block_sparse_moe"):
                if layer.block_sparse_moe.token_dispatcher._disbalance_flag:
                    layer.block_sparse_moe.token_dispatcher._disbalance_flag = False
                    skip_step = True

        return skip_step

    def _skip_step(self, outputs_tuple):
        outputs_tuple = list(outputs_tuple)
        # zero gradients
        if torch.is_tensor(outputs_tuple[1]):
            outputs_tuple[1] = outputs_tuple[1].detach()
        # zero loss
        outputs_tuple[0] *= 0
        outputs_tuple[2] *= 0

        return tuple(outputs_tuple)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


class ComposerGigaMixCausalLM(HuggingFaceModel):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        trust_remote_code = om_model_config.get("trust_remote_code", True)
        use_auth_token = om_model_config.get("use_auth_token", False)
        # Get and use resolved device in case of "mixed" value init_device value
        resolved_init_device = hf_get_init_device(
            om_model_config.get("init_device", "cpu")
        )

        # TP config
        tp_size = om_model_config.get("tp_size", 1)
        assert tp_size >= 1, f"tp_size is expected to be >=1, got {tp_size}"
        if tp_size > 1:
            tp_group_size = get_tp_group_size()
            assert tp_size == tp_group_size, (
                f"Wrong tensor parallel group size. {tp_group_size} instead {tp_size}"
            )

        config = GigaMixConfig.from_pretrained(
            om_model_config.pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            tp_size=tp_size,
            init_device=resolved_init_device,
            enable_async_tp=om_model_config.get("enable_async_tp", False),
        )

        # write lora params if exist
        config = write_lora_config(config, om_model_config)

        config.varlen_input = om_model_config.get("varlen_input", False)
        config.loss_reduction_mode = om_model_config.get("loss_reduction_mode", "batch")
        config.delete_logits = True

        # Extract config_overrides and handle sp_split_type and tied_modules_groups separately
        config_overrides = om_model_config.get("config_overrides", {})
        sp_split_override = config_overrides.pop("sp_split_type", None)
        tied_modules_groups = config_overrides.pop("tied_modules_groups", [[]])

        # set config overrides
        for k, v in config_overrides.items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).'
                )

            attr = getattr(config, k)
            if isinstance(v, (ListConfig, DictConfig)):
                v = om.to_container(v, resolve=True)
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        "Config dict override got unknown keys. "
                        + f"Extra keys: {extra_keys}. "
                        + f"Expected (a subset of) keys: {list(attr.keys())}."
                    )
                getattr(config, k).update(v)
            # necessary case to allow for rope_scaling to be overriden in llama config
            elif attr is None and isinstance(v, Mapping):
                setattr(config, k, {})
                getattr(config, k).update(v)
            else:
                setattr(config, k, v)

        setattr(config, "sp_split_type", sp_split_override)
        config.sp_split_type = get_sp_split_type(
            split_type=config.sp_split_type,
            attention_type=config.attention_type,
            varlen_input=config.varlen_input,
        )

        setattr(config, "tied_modules_groups", tied_modules_groups)
        config.tied_modules_groups = [
            to_container(modules_group) for modules_group in tied_modules_groups
        ]

        with dist.run_local_rank_zero_first():
            model = GigaMixForCausalLM(config)

        prepare_hf_model_for_fsdp(model, resolved_init_device)

        # NOTE (Sbr): train cross entropy and perplexity stay disabled to save memory
        # and compute -- the same signal is already logged as `loss/train/loss_ce`.
        # The eval-side ones are cheap (they only run on `eval_loader` batches) and are
        # the only source of a validation NLL: Composer does not call `loss()` during
        # eval, so `metrics/eval/LanguageCrossEntropy` is the val NLL.
        # train_metrics = [LanguageCrossEntropy(), LanguagePerplexity()]
        train_metrics = []
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
        )
        self.model_forward_args = inspect.getfullargspec(self.model.forward).args
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.z_loss_eps = config.z_loss_eps
        self.hidden_z_loss_eps = config.hidden_z_loss_eps
        self.hidden_z_loss_attn_coef = getattr(config, "hidden_z_loss_attn_coef", 0.0)
        self.hidden_z_loss_moe_coef = getattr(config, "hidden_z_loss_moe_coef", 0.0)
        self.hidden_z_loss_dense_mlp_coef = getattr(
            config, "hidden_z_loss_dense_mlp_coef", 0.0
        )
        self.use_mtp = config.use_mtp
        self.draft_loss_kinds = model.draft_loss_kinds
        self.draft_diagnostic_kinds = model.DRAFT_DIAGNOSTIC_KINDS

    def loss(self, outputs: ModelOutput, batch: Mapping):
        z_loss = None
        hidden_z_loss = None
        hidden_z_loss_attn = None
        hidden_z_loss_moe = None
        hidden_z_loss_dense_mlp = None
        loss_ce = None
        loss_mtp = None
        loss_mtp_per_head = None
        draft_losses = {}
        if self.config.use_return_dict:
            loss = outputs["loss"]
            assert self.z_loss_eps == 0, (
                "CausalLMOutputWithPast does not support z_loss"
            )
            assert not self.use_mtp, "CausalLMOutputWithPast does not support use_mtp"
        else:
            loss = outputs[0]

            values = outputs[2:]

            # Strip order matches append order in `GigaMixForCausalLM.forward`:
            # ..., loss_ce, [z_loss], [loss_mtp], [loss_mtp_per_head],
            #      [hidden_z_loss], [hidden_z_loss_attn], [hidden_z_loss_moe],
            for kind in reversed(self.draft_loss_kinds):
                draft_losses[kind] = (values[-2], values[-1])
                values = values[:-2]

            if self.hidden_z_loss_dense_mlp_coef > 0:
                hidden_z_loss_dense_mlp = values[-1]
                values = values[:-1]

            if self.hidden_z_loss_moe_coef > 0:
                hidden_z_loss_moe = values[-1]
                values = values[:-1]

            if self.hidden_z_loss_attn_coef > 0:
                hidden_z_loss_attn = values[-1]
                values = values[:-1]

            if self.hidden_z_loss_eps > 0:
                hidden_z_loss = values[-1]
                values = values[:-1]

            if self.use_mtp:
                if self.config.mtp_predictor_num > 1:
                    loss_mtp_per_head = values[-1]
                    values = values[:-1]
                else:
                    loss_mtp_per_head = None
                loss_mtp = values[-1]
                values = values[:-1]

            if self.z_loss_eps > 0:
                z_loss = values[-1]
                values = values[:-1]

            loss_ce = values[-1]
            values = values[:-1]

        return_d = {"total": loss}

        if z_loss is not None:
            return_d["z_loss"] = z_loss

        if hidden_z_loss is not None:
            return_d["hidden_z_loss"] = hidden_z_loss

        if hidden_z_loss_attn is not None:
            return_d["hidden_z_loss_attn"] = hidden_z_loss_attn

        if hidden_z_loss_moe is not None:
            return_d["hidden_z_loss_moe"] = hidden_z_loss_moe

        if hidden_z_loss_dense_mlp is not None:
            return_d["hidden_z_loss_dense_mlp"] = hidden_z_loss_dense_mlp

        # Per-projection hidden z-loss scalars (qkv/up_mlp/up_moe/gate_*) stashed
        # on modules during forward — logging-only (gradient is in-graph).
        for proj, val in collect_included_proj_z_losses(self.model).items():
            return_d[f"hidden_z_loss_{proj}"] = val

        if loss_mtp is not None:
            return_d["loss_mtp"] = loss_mtp
            if loss_mtp_per_head is not None:
                for i, head_loss in enumerate(loss_mtp_per_head):
                    return_d[f"loss_mtp{i + 1}"] = head_loss

        for kind, (draft_loss, draft_loss_per_head) in draft_losses.items():
            if draft_loss is None:
                continue
            if kind in self.draft_diagnostic_kinds:
                name = kind
            else:
                name = f"{kind}_loss"
                return_d[name] = draft_loss
            for i, head_loss in enumerate(draft_loss_per_head):
                return_d[f"{name}{i + 1}"] = head_loss

        if loss_ce is not None:
            return_d["loss_ce"] = loss_ce

        return return_d

    def forward(self, batch: Mapping):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            with profiler_annotation.annotate("forward phase"):
                output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                "Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model"
            )
        return output

    @staticmethod
    def _activation_checkpointing_auto_wrap_policy(
        model: GigaMixForCausalLM, fsdp_config: dict
    ):
        """In non-EP settings `num_layers_to_checkpoint` are wrapped for checkpointing,
        in EP settings self_attn and expert submodules are wrapped separately
        """

        def _root_module_or_self(module: nn.Module) -> nn.Module:
            root_module = getattr(module, "root_module", None)
            return root_module if isinstance(root_module, nn.Module) else module

        def _checkpoint_target(module: nn.Module) -> nn.Module:
            raw_module = _root_module_or_self(module)  # noqa: F841
            return next(module.children())

        def _moe_checkpoint_target(
            moe_module: nn.Module, cur_layer, num_layers_to_checkpoint
        ) -> Optional[nn.Module]:
            raw_module = _root_module_or_self(moe_module)
            if cur_layer >= num_layers_to_checkpoint:
                if isinstance(raw_module, ScMoEBlock):
                    raise RuntimeError(
                        "ScMoEBlock only supports activation checkpointing"
                    )
                return None
            if isinstance(raw_module, ScMoEBlock):
                return None
            return _checkpoint_target(moe_module)

        def _checkpoint_policy_fn(m: nn.Module) -> bool:
            num_layers_to_checkpoint = fsdp_config["num_layers_to_checkpoint"]
            if num_layers_to_checkpoint is None:
                num_layers_to_checkpoint = len(model.model.layers)

            layers = model.model.layers[:num_layers_to_checkpoint]
            checkpointed_modules = {
                _checkpoint_target(module.self_attn) for module in layers
            }
            checkpointed_modules |= {
                target
                for cur_layer, module in enumerate(model.model.layers)
                if hasattr(module, "block_sparse_moe")
                if (
                    target := _moe_checkpoint_target(
                        module.block_sparse_moe, cur_layer, num_layers_to_checkpoint
                    )
                )
                is not None
            }

            # `post_feedforward_layernorm` (sparse layers only) is checkpointed as
            # its own region — it lives on the decoder layer (outside `block_sparse_moe`
            # so the MoE combine op is not recomputed) and the gated-norm forward is
            # expensive enough in activations that leaving it without AC is costly.
            post_norms = {
                module.post_feedforward_layernorm
                for module in layers
                if hasattr(module, "post_feedforward_layernorm")
            }

            if hasattr(model.model, "mtp_block"):
                checkpointed_modules |= set(
                    get_mtp_activation_checkpointing_modules(
                        model.model.mtp_block.mtp_layers,
                        moe_target_fn=lambda moe: _moe_checkpoint_target(
                            moe, 0, num_layers_to_checkpoint
                        ),
                    )
                )

            return m in post_norms or m in checkpointed_modules

        return functools.partial(
            torch.distributed.fsdp.wrap.lambda_auto_wrap_policy,
            lambda_fn=_checkpoint_policy_fn,
        )

    @staticmethod
    def _giga_fsdp_modules_to_wrap_with_names(
        model: GigaMixForCausalLM, fsdp_config: dict
    ):
        """In non-EP settings this function wraps every *MixDecoderLayer separately.
        In EP settings this function wraps attention and expert layers separately.
        """
        modules_with_names = [(model.model.embed_tokens, "model.embed_tokens")]
        for layer_idx, module in enumerate(model.model.layers):
            modules_with_names.append(
                (module.self_attn, f"model.layers.{layer_idx}.self_attn")
            )
            if hasattr(module, "block_sparse_moe"):
                module_name = f"model.layers.{layer_idx}.block_sparse_moe"
                mlp_submodule = module.block_sparse_moe
                modules_with_names.append((mlp_submodule, module_name))

        modules_with_names.append((model.lm_head, "lm_head"))
        if hasattr(model.model, "mtp_block"):
            modules_with_names.extend(
                get_mtp_fsdp_modules_with_names(
                    model.model.mtp_block.mtp_layers,
                    "model.mtp_block",
                )
            )

        return modules_with_names

    @staticmethod
    def _giga_fsdp_fp8_allgather_modules(
        model: GigaMixForCausalLM, fsdp_config: dict
    ) -> List[str]:
        modules_with_names = (
            ComposerGigaMixCausalLM._giga_fsdp_modules_to_wrap_with_names(
                model, fsdp_config
            )
        )
        excluded_patterns = {"model.embed_tokens", "mtp_block", "lm_head", "self_attn"}
        filter_modules = [
            module_name
            for _, module_name in modules_with_names
            if all(pattern not in module_name for pattern in excluded_patterns)
        ]
        print("FP8 all-gather filter modules: ", filter_modules)

        return filter_modules

    @staticmethod
    def _giga_fsdp_rogue_layer_norm_modules_with_names(model: GigaMixForCausalLM):
        """LayerNorm is rogue only in case it is not a part of STM (SuperTensorModule)
        STM wrapping is declared in giga_fsdp_modules_to_wrap_with_names.

        Note: for sparse layers, `post_feedforward_layernorm` lives on the
        decoder layer itself (see `LlamaMixDecoderLayer.__init__`) so it IS
        rogue — it is not part of any STM. It is additionally wrapped as a
        standalone activation checkpoint target in
        `_activation_checkpointing_auto_wrap_policy`. For dense layers the
        post-norm lives inside `AttnGate` (which is the `self_attn` STM), so
        it is not rogue for those layers.
        """
        rogue_layer_norms = {model.model.norm: "model.norm"}

        if hasattr(model.model, "embed_ln"):
            rogue_layer_norms[model.model.embed_ln] = "model.embed_ln"

        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "post_feedforward_layernorm"):
                rogue_layer_norms[layer.post_feedforward_layernorm] = (
                    f"model.layers.{i}.post_feedforward_layernorm"
                )

        if hasattr(model.model, "mtp_block"):
            rogue_layer_norms |= {
                m: f"model.mtp_block.mtp_norms.{i}"
                for i, m in enumerate(model.model.mtp_block.mtp_norms)
            }
            for i, mtp_layer in enumerate(model.model.mtp_block.mtp_layers):
                dec = mtp_layer.decoder_layer
                # для dense mtp весь модуль оборачивается целиком --> post_feedforward_layernorm там не является rogue
                if resolve_mtp_block_type(mtp_layer.config) == "moe" and hasattr(
                    dec, "post_feedforward_layernorm"
                ):
                    rogue_layer_norms[dec.post_feedforward_layernorm] = (
                        f"model.mtp_block.mtp_layers.{i}.decoder_layer.post_feedforward_layernorm"
                    )
        return rogue_layer_norms

    @staticmethod
    def _special_process_group_fn(module: nn.Module):
        ep_size = get_ep_group_size()
        if isinstance(module, (AbstractGMMMoeBlock, ScMoEBlock)):
            if ep_size is None or ep_size == 1:
                return (torch.distributed.distributed_c10d._get_default_group(), None)
            else:
                return (dist.get_ep_fsdp_group(), dist.get_ep_group())
        else:
            return None

    def gigafsdp_model_setup(self, fsdp_config: dict):
        fsdp_config["num_layers_to_checkpoint"] = (
            self.config.activation_checkpoint_layers_num
        )
        if fsdp_config.get("fp8_allgather_modules") is None:
            fsdp_config["fp8_allgather_modules"] = (
                self._giga_fsdp_fp8_allgather_modules(self.model, fsdp_config)
            )

        fsdp_config["gigafsdp_wrappers"]["model"] = dict(
            activation_checkpointing_auto_wrap_policy=self._activation_checkpointing_auto_wrap_policy(
                self.model, fsdp_config
            ),
            giga_fsdp_modules_to_wrap_with_names=self._giga_fsdp_modules_to_wrap_with_names(
                self.model, fsdp_config
            ),
            giga_fsdp_rogue_layer_norm_modules_with_names=self._giga_fsdp_rogue_layer_norm_modules_with_names(
                self.model
            ),
            # giga_fsdp_layer_norm_module_cls - allows gigafsdp to sync gradients of cls weights
            # works only in case if all_reduce_grads_across_model_parallel_group = true
            giga_fsdp_layer_norm_module_cls=resolve_norm_class(self.config.norm_type),
            special_process_group_fn=self._special_process_group_fn,
        )
