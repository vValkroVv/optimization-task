"""Sequence-classification head and Composer wrapper for the Gigar backbone.

This module keeps all classifier-specific logic out of the dense-model
``modelling_gigar.py`` and reuses the Gigar backbone classes
(``LlamaModel``, ``LlamaPreTrainedModel``, ``ComposerGigarCausalLM``) via import.
"""

import inspect
from typing import List, Mapping, Optional

import torch
from torch import nn
from torchmetrics import Metric

from composer.metrics.classification import (
    BinaryAccuracy,
    BinaryAUCROC,
    BinaryPrecision,
    BinaryRecall,
    ClassificationMSE,
    MultiClassCrossEntropy,
    MultiClassOverallAccuracy,
    MultiClassWeightedF1,
    SoftBinaryF1Score,
)
from composer.utils import dist
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmfoundry.models.gigar.configuration_gigar import GigarConfig
from llmfoundry.models.gigar.modelling_gigar import (
    ComposerGigarCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from llmfoundry.models.hf.hf_fsdp import hf_get_init_device
from llmfoundry.models.layers.norm import resolve_norm_class


class LlamaForClassification(LlamaPreTrainedModel):
    """
    LLaMA model adapted for sequence classification tasks.
    This model uses the standard LLaMA backbone (LlamaModel) and adds a
    classification head on top. The final hidden state of the last token
    (typically corresponding to an EOS token) is passed through
    a linear layer to produce logits for `num_labels` classes.
    """

    def __init__(self, config: GigarConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.num_labels = config.num_labels

        # Classification head: hidden_size -> num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply post_init processing
        if config.init_device != "meta":
            self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        return_dict = False

        # Determine batch size for logical batching (AsyncTP support)
        if input_ids is not None:
            bs = input_ids.shape[0]
        elif inputs_embeds is not None:
            bs = inputs_embeds.shape[0]

        if self.model.config.enable_async_tp and logical_batch_size is None:
            logical_batch_size = bs

        # Backbone forward — LlamaModel handles internal TP automatically
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
        )

        # hidden_states shape: [batch_size, seq_len, hidden_size]
        hidden_states = outputs[0]

        # The dataset guarantees a trailing EOS on every sequence and the
        # left-padding collator keeps it at position -1, so the last hidden
        # state is the EOS representation used for pooling.
        pooled_logits = hidden_states[:, -1, :]

        # Classification head: logits shape: [batch_size, num_labels]
        logits = self.score(pooled_logits)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            labels = labels.float()

            if labels.dim() == 1:
                # binary compat: scalar soft label -> [p_neg, p_pos]
                assert self.num_labels == 2, (
                    f"1D labels only supported for binary (num_labels=2), got num_labels={self.num_labels}"
                )
                labels = torch.stack([1 - labels, labels], dim=1)
            # for [B, num_labels] vectors (one-hot or soft) — used as-is

            loss = loss_fn(logits, labels)

        # Format is chosen for passing to GigaFSDP
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ComposerGigarClassifier(ComposerGigarCausalLM):
    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        # Get config similar to parent
        resolved_init_device = hf_get_init_device(
            om_model_config.get("init_device", "cpu")
        )
        tp_size = om_model_config.get("tp_size", 1)

        config = GigarConfig.from_pretrained(
            om_model_config.get("pretrained_model_name_or_path"),
            init_device=resolved_init_device,
            tp_size=tp_size,
            num_labels=om_model_config.get("num_labels", 2),  # Classification specific
            enable_async_tp=om_model_config.get("enable_async_tp", False),
        )

        config.delete_logits = False

        # Extract config_overrides and handle sp_split_type separately
        config_overrides = om_model_config.get("config_overrides", {})
        sp_split_override = config_overrides.pop("sp_split_type", None)

        # set config overrides
        for k, v in config_overrides.items():
            if not hasattr(config, k):
                raise ValueError(
                    f'config does not have attribute "{k}" to override ({k}: {v}).'
                )

            attr = getattr(config, k)
            if isinstance(attr, Mapping):
                extra_keys = [_k for _k in v.keys() if _k not in attr.keys()]
                if extra_keys:
                    raise ValueError(
                        f"Config dict override got unknown keys. "
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

        if config.tp_size > 1:
            assert config.pretraining_tp == 1, (
                "pretraining_tp is not supported with Parallel Loss"
            )

        # Create classification model
        with dist.run_local_rank_zero_first():
            model = LlamaForClassification(config)

            if config.freeze_non_embed:
                for param in model.parameters():
                    param.requires_grad = False

                for param in model.model.embed_tokens.parameters():
                    param.requires_grad = True

                for param in model.score.parameters():
                    param.requires_grad = True

            if config.non_freeze_layers_idxs is not None:
                for param in model.parameters():
                    param.requires_grad = False

                for i, layer in enumerate(model.model.layers):
                    if i in config.non_freeze_layers_idxs:
                        for param in layer.parameters():
                            param.requires_grad = True

        # Save config as instance attribute
        self.config = config

        # Minimal FSDP setup (the main setup is in gigafsdp_model_setup)
        model_block = model.model.layers
        if len(model_block) > 0:
            block_type = type(model_block[0])
            model.fsdp_wrap_fn = lambda module: isinstance(module, block_type)
            model.activation_checkpointing_fn = lambda module: isinstance(
                module, block_type
            )

        train_metrics = []

        num_labels = om_model_config.get("num_labels", 2)

        if num_labels == 2:
            eval_metrics = [
                MultiClassCrossEntropy(),
                BinaryAccuracy(),
                BinaryPrecision(),
                BinaryRecall(),
                SoftBinaryF1Score(),
                BinaryAUCROC(),
                ClassificationMSE(),
            ]
        else:
            eval_metrics = [
                MultiClassCrossEntropy(),
                MultiClassOverallAccuracy(),
                MultiClassWeightedF1(num_classes=num_labels),
                ClassificationMSE(),
            ]

        # Initialize HuggingFaceModel parent
        super(ComposerGigarCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=False,
        )

        self.model_forward_args = inspect.getfullargspec(self.model.forward).args
        self.z_loss_eps = 0.0
        self.use_mtp = False

    @staticmethod
    def _giga_fsdp_modules_to_wrap_with_names(model, fsdp_config: dict):
        """Override parent's method to handle classification models."""
        modules_with_names = [(model.model.embed_tokens, "model.embed_tokens")]

        for i, m in enumerate(model.model.layers):
            modules_with_names.append((m, f"model.layers.{i}"))

        if hasattr(model.model, "mtp_block"):
            for i, m in enumerate(model.model.mtp_block.mtp_layers):
                modules_with_names.append(
                    (m.decoder_layer, f"model.mtp_block.mtp_layers.{i}.decoder_layer")
                )

        # 'score' for classification, 'lm_head' for causal LM
        if hasattr(model, "score"):
            modules_with_names.append((model.score, "score"))
        if hasattr(model, "lm_head"):
            modules_with_names.append((model.lm_head, "lm_head"))
        if not (hasattr(model, "score") or hasattr(model, "lm_head")):
            raise AttributeError("Model has neither 'score' nor 'lm_head' attribute")

        if fsdp_config["separate_layer_norm_disabled"]:
            modules_with_names.append((model.model.norm, "model.norm"))

        return modules_with_names

    def gigafsdp_model_setup(self, fsdp_config: dict):
        """Override to ensure our static methods are used."""
        fsdp_config["num_layers_to_checkpoint"] = (
            self.config.activation_checkpoint_layers_num
        )
        fsdp_config.setdefault("separate_layer_norm_disabled", False)
        if self.model.config.enable_async_tp:
            fsdp_config["separate_layer_norm_disabled"] = False

        fsdp_config["gigafsdp_wrappers"]["model"] = dict(
            activation_checkpointing_auto_wrap_policy=self._activation_checkpointing_auto_wrap_policy(
                self.model, fsdp_config
            ),
            giga_fsdp_modules_to_wrap_with_names=self._giga_fsdp_modules_to_wrap_with_names(
                self.model, fsdp_config
            ),
            giga_fsdp_rogue_layer_norm_modules_with_names=None
            if fsdp_config["separate_layer_norm_disabled"]
            else self._giga_fsdp_rogue_layer_norm_modules_with_names(self.model),
            giga_fsdp_layer_norm_module_cls=None
            if fsdp_config["separate_layer_norm_disabled"]
            else resolve_norm_class(self.config.norm_type),
            special_process_group_fn=None,
        )

    def loss(self, outputs, batch):
        """Simpler loss for classification."""
        if isinstance(outputs, Mapping):
            return {"total": outputs["loss"]}
        return {"total": outputs[0]}

    def eval_forward(self, batch: Mapping, outputs=None, **kwargs):
        if outputs is not None:
            return outputs
        labels = batch.get("labels", None)  # grab before filtering
        model_batch = {
            k: v
            for k, v in batch.items()
            if k in self.model_forward_args and k != "labels"
        }
        with torch.no_grad():
            raw_outputs = self.model(**model_batch)
        logits = raw_outputs[0]  # (logits,) since no loss computed
        return logits, labels

    def update_metric(self, batch: Mapping, outputs, metric: Metric):
        logits, labels = outputs
        if labels is not None:
            metric.update(logits, labels.float())
