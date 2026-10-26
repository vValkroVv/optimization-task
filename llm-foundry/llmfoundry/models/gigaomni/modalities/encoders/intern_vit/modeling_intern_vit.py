import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Tuple, Union, List
from timm.models.layers import DropPath
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...base_component import BaseOmniComponent
from .configuration_intern_vit import InternVisionConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


try:
    from composer.utils.dist import get_sp_group_size
    from llmfoundry.models.parallel.sequence.utils import get_tensors_sp_part
    from llmfoundry.models.parallel.sequence.mappings import (
        gather_from_sequence_parallel_region_on_batch_size_dimension,
    )
except ImportError:
    logger.info(
        "Encountered exception while importing functions for training with sequence parallelism. "
        "It's normal behaviour if you are loading hf model."
    )

    def get_sp_group_size():
        return 1


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        logger.warning("Your model is using InternRMSNorm, which is not tested!")

        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


NORM2FN = {
    "rms_norm": InternRMSNorm,
    "layer_norm": nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config

        self.class_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, config.hidden_size)
        )

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int) -> torch.Tensor:
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(
                1,
                self.config.image_size // self.config.patch_size,
                self.config.image_size // self.config.patch_size,
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        if (
            H != self.config.image_size // self.config.patch_size
            or W != self.config.image_size // self.config.patch_size
        ):
            pos_embed = F.interpolate(
                pos_embed, size=(H, W), mode="bicubic", align_corners=False
            )

        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(target_dtype)
        )  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, C = hidden_states.shape

        qkv = (
            self.qkv(hidden_states)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = (
                self.q_norm(q.transpose(1, 2).flatten(-2, -1))
                .view(B_, N_, H_, D_)
                .transpose(1, 2)
            )
            k = (
                self.k_norm(k.transpose(1, 2).flatten(-2, -1))
                .view(B_, N_, H_, D_)
                .transpose(1, 2)
            )

        dropout_p = self.attn_drop.p if self.training else 0.0

        backend = (
            SDPBackend.FLASH_ATTENTION
            if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
            else SDPBackend.MATH
        )
        with sdpa_kernel(backend):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, scale=self.scale
            )

        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[torch.FloatTensor],
        Optional[Tuple[torch.FloatTensor]],
    ]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states)) * self.ls1
        )
        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states)) * self.ls2
        )
        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a [`InternVisionEncoderLayer`].

    Args:
        config (`InternConfig`) The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(config, dpr[idx])
                for idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        output_hidden_states: bool = True,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if return_dict is None:
            return_dict = self.config.use_return_dict

        encoder_states = []
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(hidden_states)

            hidden_states = encoder_layer(hidden_states)

        if output_hidden_states:
            encoder_states.append(hidden_states)

        encoder_states_tuple = tuple(encoder_states) if output_hidden_states else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states_tuple] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states_tuple
        )


class InternVisionModel(PreTrainedModel):
    _no_split_modules = ["InternVisionEncoderLayer"]
    config_class = InternVisionConfig

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None and pixel_embeds is None:
            raise ValueError("You have to specify pixel_values or pixel_embeds")

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            assert pixel_values is not None, "pixel_values must be provided."
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f"Wrong pixel_values size: {pixel_values.shape}")

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InternViTForVisualLanguageModeling(InternVisionModel, BaseOmniComponent):
    def forward(
        self,
        inputs: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, ...]:  # type: ignore
        pixel_values, sizes = self._concat_inputs(inputs)

        sp_group_size = get_sp_group_size() or 1
        initial_shape = pixel_values.shape[0]
        if sp_group_size > 1 and self.training:
            pixel_values = self._pad_pixel_values_to_sp_group_size(pixel_values)
            (pixel_values,) = get_tensors_sp_part(
                pixel_values, split_type="equal", dim=0
            )

        vit_outputs = super().forward(
            pixel_values=pixel_values,
            output_hidden_states=self.config.get_tokens_from_layer != -1,
            return_dict=True,
            pixel_embeds=None,
        )

        assert isinstance(vit_outputs, BaseModelOutputWithPooling), (
            "Expected BaseModelOutputWithPooling from InternVisionModel"
        )

        vit_outputs = self._feature_select(vit_outputs)

        if sp_group_size > 1 and self.training:
            # We need to clone image_features to prevent strange gigafsdp error:
            # `RuntimeError: Output 0 of TyingGateBackward is a view and its base
            # or another view of its base has been modified inplace.`
            vit_outputs = gather_from_sequence_parallel_region_on_batch_size_dimension(
                vit_outputs.clone(), tensor_parallel_output_grad=True
            )[:initial_shape]

        return self._split_outputs(vit_outputs, sizes)

    def _feature_select(self, vit_outputs: BaseModelOutputWithPooling) -> torch.Tensor:
        """
        Selects and extracts features from the model outputs.

        Args:
            vit_outputs (BaseModelOutputWithPooling): Output from the vision tower forward pass.

        Returns:
            torch.Tensor: Selected image features.
        """
        if self.config.get_tokens_from_layer == -1:
            assert vit_outputs.last_hidden_state is not None, (
                "`last_hidden_state` is required but was not returned by vision tower"
            )
            return vit_outputs.last_hidden_state[:, 1:]

        assert vit_outputs.hidden_states is not None, (
            "`hidden_states` is required but was not returned by vision tower"
        )

        image_features = vit_outputs.hidden_states[self.config.get_tokens_from_layer]
        image_features = image_features[:, 1:]
        return image_features

    def _concat_inputs(
        self, inputs: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[int]]:
        for input_item in inputs:
            assert "pixel_values" in input_item, (
                f"`pixel_values` is required, but not found in input_item: {input_item.keys()}"
            )

        pixel_values_list = [input_item["pixel_values"] for input_item in inputs]
        sizes_list = [len(pixel_values) for pixel_values in pixel_values_list]

        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, sizes_list

    def _split_outputs(
        self, vit_outputs: torch.Tensor, sizes_list: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        vit_outputs_list = torch.split(vit_outputs, sizes_list, dim=0)
        return tuple(vit_outputs_list)

    def _pad_pixel_values_to_sp_group_size(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        sp_group_size = get_sp_group_size() or 1
        if pixel_values.shape[0] % sp_group_size != 0:
            pad_size = sp_group_size - (pixel_values.shape[0] % sp_group_size)
            padding = torch.zeros(
                (
                    pad_size,
                    pixel_values.shape[1],
                    pixel_values.shape[2],
                    pixel_values.shape[3],
                ),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            pixel_values = torch.cat((pixel_values, padding), dim=0)
        return pixel_values

    def unwrap_tuple_output(
        self, inputs: List[Dict[str, torch.Tensor]], outputs: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:
        return [{"pixel_embeds": output} for output in outputs]

    def get_giga_fsdp_modules_to_wrap_with_names(
        self, prefix: str = ""
    ) -> List[Tuple[torch.nn.Module, str]]:
        if not self.config.use_per_layer_wrapping:
            return [(self, prefix)]

        modules_to_wrap_with_names = [(self.embeddings, "embeddings")] + [
            (module, f"encoder.layers.{idx}")
            for idx, module in enumerate(self.encoder.layers)
        ]

        if prefix and not prefix.endswith("."):
            prefix += "."

        modules_to_wrap_with_names = [
            (item[0], prefix + item[1]) for item in modules_to_wrap_with_names
        ]
        return modules_to_wrap_with_names

    def get_giga_fsdp_modules_for_activation_checkpointing(
        self,
    ) -> List[torch.nn.Module]:
        return [self.embeddings] + list(self.encoder.layers)
