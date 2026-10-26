# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator, Mapping
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torchtitan.config import Configurable
from torchtitan.models.common.decoder_blocks import Decoder
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from torchtitan.tools.logging import logger


_DecoderConfig = TypeVar("_DecoderConfig", bound=Decoder.Config)


class MoEStateDictAdapter(StateDictAdapter, Generic[_DecoderConfig]):
    """
    StateDictAdapter for MoE models.
    HF MoE models store experts as a module list each with 2D weights. In torchtitan, we
    store experts as a 3D param with the first dimension being num_experts. The functions
    in this class help convert 3D param into list of 2D params so that the checkpoint
    can be loaded without incurring local memory overhead, and then concatenate
    the results back to 3D param. ``w13_E2FD`` stacks gate and up on dim 1 of a 4D
    param; it is unstacked into the two 3D projections HF stores separately before
    the per-expert split, and re-stacked after the per-expert concat.
    """

    # HF stores gate and up per expert; titan stacks them on dim 1 of
    # ``w13_E2FD``. Titan-side names of the two slices in slot order, and the
    # keys the ``from_hf_map``s use for them.
    _W13_PROJECTIONS = ("w1_EFD", "w3_EFD")

    def __init__(
        self,
        model_config: _DecoderConfig,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        # Store metadata for GroupedExperts <-> individual experts conversion
        self.grouped_expert_weight_placements = {}  # {titan_abstract_key: placements}
        self.grouped_expert_weight_shape = {}  # {titan_abstract_key: shape}
        self.grouped_expert_weight_mesh = {}  # {titan_abstract_key: device_mesh}
        self.local_experts_indices = {}  # {titan_abstract_key: (start_idx, end_idx)}

    def _calculate_strided_shard_shard_indices(
        self,
        strided_shard_dim_degree: int,
        strided_shard_dim_rank: int,
        shard_dim_degree: int,
        shard_dim_rank: int,
        dim_size_to_split: int,
    ) -> tuple[int, int]:
        """
        Given a [StridedShard(dim=i), Shard(dim=i)] placement, calculate the start index
        and end index on dim-i for GPU rank (strided_shard_dim_degree, shard_dim_rank)

        GPU Layout (strided_shard_rank, shard_rank):

        StridedShard Rank                  Shard rank
                        ┌─────────────────┐
                    0   │    GPU(0, 0)    │  0
                    ────┼─────────────────┤
                    1   │    GPU(1, 0)    │
                    ────┼─────────────────┤
                    2   │    GPU(2, 0)    │
                  ──────┼─────────────────┼────
                    0   │    GPU(0, 1)    │  1
                    ────┼─────────────────┤
                    1   │    GPU(1, 1)    │
                    ────┼─────────────────┤
                    2   │    GPU(2, 1)    │
                        └─────────────────┘

        Calculate the start_index from inner dimension (Shard(dim=i)) to outer dimension (StridedShard(dim=i)).
        """

        block_size = dim_size_to_split // (strided_shard_dim_degree * shard_dim_degree)

        # Error out if can not evenly divided
        if (
            block_size * (strided_shard_dim_degree * shard_dim_degree)
            != dim_size_to_split
        ):
            raise ValueError(
                f"Not supported split for strided_shard_dim_degree {strided_shard_dim_degree}, "
                f"shard_dim_degree {shard_dim_degree}, dim_size_to_split {dim_size_to_split}"
            )

        start_index = block_size * (
            strided_shard_dim_degree * shard_dim_rank + strided_shard_dim_rank
        )
        end_index = start_index + block_size

        return start_index, end_index

    def _caculate_indices_from_placements(
        self,
        dim: int,
        dim_size: int,
        dtensor_placements: tuple,
        device_mesh: DeviceMesh,
    ) -> tuple[int, int]:
        mesh_names = []
        dim_i_placements = []

        # Find all the device mesh dimensios that shard on dim-i.
        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if isinstance(placement, (Shard, _StridedShard)) and placement.dim == dim:
                mesh_names.append(name)
                dim_i_placements.append(placement)

        # Calculate local expert indices based on sharding strategy
        start_index, end_index = 0, dim_size
        if len(dim_i_placements) == 2:
            # Handle StridedShard(i) + Shard(i) case
            assert isinstance(dim_i_placements[0], _StridedShard), (
                "Expected StridedShard as first placement"
            )

            strided_shard_mesh = device_mesh[mesh_names[0]]
            shard_mesh = device_mesh[mesh_names[1]]

            strided_degree = strided_shard_mesh.size()
            strided_rank = strided_shard_mesh.get_local_rank()
            shard_degree = shard_mesh.size()
            shard_rank = shard_mesh.get_local_rank()

            start_index, end_index = self._calculate_strided_shard_shard_indices(
                strided_degree, strided_rank, shard_degree, shard_rank, dim_size
            )

        elif len(dim_i_placements) == 1:
            # Handle single Shard(i) case
            assert not isinstance(dim_i_placements[0], _StridedShard), (
                "Expected regular Shard, not StridedShard"
            )

            shard_mesh = device_mesh[mesh_names[0]]
            shard_degree = shard_mesh.size()
            shard_rank = shard_mesh.get_local_rank()

            block_size = dim_size // shard_degree
            if block_size * shard_degree != dim_size:
                raise ValueError(
                    f"Dim {dim} size ({dim_size}) cannot be evenly divided by shard degree ({shard_degree})"
                )

            start_index = block_size * shard_rank
            end_index = start_index + block_size

        elif len(dim_i_placements) == 0:
            # No sharding on this dimension means all elements are local
            pass

        else:
            raise NotImplementedError(
                f"Unsupported DTensor placements for GroupedExperts: {dtensor_placements} {dim_i_placements} {mesh_names}"
            )

        return start_index, end_index

    def _get_local_experts_weights(
        self,
        abstract_key: str,
        titan_abstract_key: str,
        layer_id: str,
        grouped_expert_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Split GroupedExperts weight into individual expert weights for local processing.

        This method handles various sharding strategies for expert weights:
        - FSDP + EP: StridedShard(0)Shard(0) or Shard(0)
        - FSDP + ETP + EP: StridedShard(0)Shard(0)Shard(1/2) or StridedShard(1)Shard(0)Shard(1/2)

        Args:
            abstract_key: HuggingFace templage key with {} placeholders for layer and expert IDs
            titan_abstract_key: TorchTitan templage key with {} placeholders for layer and expert IDs
            layer_id: Layer identifier
            grouped_expert_weight: DTensor containing all experts' weights

        Returns:
            Dictionary mapping individual expert keys to their DTensor or plain tensor weights
        """
        device_mesh = grouped_expert_weight.device_mesh
        dtensor_placements = grouped_expert_weight.placements

        # Step 1: Extract dimension-0 placement information
        num_experts = grouped_expert_weight.shape[0]
        start_index, end_index = self._caculate_indices_from_placements(
            dim=0,
            dim_size=num_experts,
            dtensor_placements=dtensor_placements,
            device_mesh=device_mesh,
        )

        # Step 2: Store indices for potential future use in from_hf()
        self.local_experts_indices[titan_abstract_key] = (start_index, end_index)

        # Step 3: Identify mesh dimensions that shard on dim-0 (expert dimension)
        # exclude expert dimension
        # and build new sub-mesh/placements for individual expert weights
        sub_mesh_names = []
        sub_placements = []

        for i, name in enumerate(device_mesh.mesh_dim_names):
            placement = dtensor_placements[i]
            if isinstance(placement, Replicate):
                # Replicate (hybrid) doesn't shard any dim, keep in sub-mesh
                sub_mesh_names.append(name)
                sub_placements.append(Replicate())
            elif isinstance(placement, (Shard, _StridedShard)) and placement.dim == 0:
                # Shards on expert dim, exclude from sub-mesh
                pass
            elif isinstance(placement, Shard):
                # Shards on non-expert dim, keep in sub-mesh
                sub_mesh_names.append(name)
                sub_placements.append(Shard(placement.dim))
            elif isinstance(placement, _StridedShard):
                # Strided shard on non-expert dim, keep in sub-mesh
                sub_mesh_names.append(name)
                sub_placements.append(
                    _StridedShard(placement.dim, placement.split_factor)
                )
            else:
                raise ValueError(f"Unsupported placement type: {type(placement)}")

        # Step 4: Create sub-mesh excluding dim-0 sharding dimensions
        # If all mesh dimensions were sharding on dim-0, sub_mesh will be None (use plain tensors)
        sub_mesh = device_mesh[tuple(sub_mesh_names)] if sub_mesh_names else None

        # Step 5: Create individual expert tensors
        assert isinstance(grouped_expert_weight, DTensor), (
            "Expected DTensor for grouped expert weight"
        )

        local_grouped_weights = grouped_expert_weight._local_tensor
        expected_local_experts = end_index - start_index

        if local_grouped_weights.shape[0] != expected_local_experts:
            raise ValueError(
                f"Local tensor shape mismatch: expected {expected_local_experts} experts, "
                f"got {local_grouped_weights.shape[0]}"
            )

        local_expert_tensors = {}
        for expert_id in range(start_index, end_index):
            expert_key = abstract_key.format(layer_id, expert_id)
            local_expert_index = expert_id - start_index

            if sub_mesh is None:
                # Extract individual expert weight (2D) as plain tensor
                expert_weight = local_grouped_weights[local_expert_index, :, :]
            else:
                # Use slicing and unsqueeze get a 3D tensor, then create DTensor and squeeze
                expert_weight_3d = local_grouped_weights[
                    local_expert_index, :, :
                ].unsqueeze(0)
                expert_weight = DTensor.from_local(
                    expert_weight_3d,
                    sub_mesh,
                    sub_placements,
                    run_check=False,
                ).squeeze(0)
            local_expert_tensors[expert_key] = expert_weight

        return local_expert_tensors

    def _concatenate_expert_weights_dtensor(
        self,
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]],
        abstract_key: str,
        layer_num: str,
    ) -> torch.Tensor | None:
        """
        Args:
            expert_weights_by_layer: Dictionary tracking expert weights by layer, abstract key, and expert ID.
                Structure: {
                    layer_id: {
                        abstract_key: {
                            expert_id: tensor_weight
                        }
                    }
                }
                Used to collect individual expert weights before concatenating them into GroupedExperts.
            abstract_key: TorchTitan templage key with {} placeholders for layer and expert IDs
            layer_num: Layer identifier

        Returns:
            Concatenated GroupedExperts weight DTensor if all experts are available, otherwise None
        """
        # If we have all the experts for this abstract_key, concatenate them
        experts = expert_weights_by_layer[layer_num][abstract_key]
        start_index, end_index = self.local_experts_indices[abstract_key]
        expected_expert_ids = set(range(start_index, end_index))
        if len(experts) < len(expected_expert_ids):
            return None

        if set(experts) != expected_expert_ids:
            raise ValueError(
                f"Expected local expert IDs {sorted(expected_expert_ids)}, "
                f"got {sorted(experts)} for {abstract_key}."
            )

        sorted_expert_ids = sorted(experts.keys())
        sorted_experts = [experts[i] for i in sorted_expert_ids]

        # Stack experts - result may be DTensor or plain tensor depending on sub_mesh
        local_tensor = torch.stack(sorted_experts, dim=0)
        if isinstance(local_tensor, DTensor):
            local_tensor = local_tensor._local_tensor

        assert (
            abstract_key in self.grouped_expert_weight_placements
            and abstract_key in self.grouped_expert_weight_shape
            and abstract_key in self.grouped_expert_weight_mesh
        ), "GroupedExperts weight metadata (placements, shape, mesh) can not be None!"

        stacked_dtensor = DTensor.from_local(
            local_tensor,
            self.grouped_expert_weight_mesh[abstract_key],
            self.grouped_expert_weight_placements[abstract_key],
            run_check=False,
        )

        del expert_weights_by_layer[layer_num][abstract_key]
        if not expert_weights_by_layer[layer_num]:
            del expert_weights_by_layer[layer_num]

        return stacked_dtensor

    def _split_experts_weights(
        self, weight: torch.Tensor, num_experts: int
    ) -> tuple[torch.Tensor, ...]:
        """Split a grouped weight into one tensor per expert for offline conversion.

        This helper is intentionally limited to local tensors. Splitting a DTensor
        here could gather the weight and cause an OOM.
        """
        if weight.ndim != 3 or weight.shape[0] != num_experts:
            raise ValueError(
                "GroupedExperts weight must have shape "
                f"[{num_experts}, out_features, in_features], "
                f"got {tuple(weight.shape)}."
            )
        return torch.unbind(weight, dim=0)

    def _concatenate_expert_weights(
        self,
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]],
        abstract_key: str,
        layer_num: str,
        num_experts: int,
    ) -> torch.Tensor | None:
        """
        Concatenated GroupedExperts weight using torch.stack(). Used for offline conversion.

        Args:
            expert_weights_by_layer: Dictionary tracking expert weights by layer, abstract key, and expert ID.
                Structure: {
                    layer_id: {
                        abstract_key: {
                            expert_id: tensor_weight
                        }
                    }
                }
                Used to collect individual expert weights before concatenating them into GroupedExperts.
            abstract_key: TorchTitan templage key with {} placeholders for layer and expert IDs
            layer_num: Layer identifier
            num_experts: Number of experts in the GroupedExperts module

        Returns:
            Concatenated GroupedExperts weight if all experts are available, otherwise None
        """
        # If we have all the experts for this abstract_key, concatenate them
        experts = expert_weights_by_layer[layer_num][abstract_key]
        if len(experts) < num_experts:
            return None

        sorted_expert_ids = sorted(experts.keys())
        expected_expert_ids = set(range(num_experts))
        if set(experts) != expected_expert_ids:
            raise ValueError(
                f"Expected expert IDs {sorted(expected_expert_ids)}, "
                f"got {sorted(experts)} for {abstract_key}."
            )
        sorted_experts = [experts[i] for i in sorted_expert_ids]
        stacked_tensor = torch.stack(sorted_experts, dim=0)

        del expert_weights_by_layer[layer_num][abstract_key]
        if not expert_weights_by_layer[layer_num]:
            del expert_weights_by_layer[layer_num]

        return stacked_tensor

    def _num_routed_experts(self) -> int:
        """Routed-expert count of the main decoder's MoE layers."""
        moe_layer = next(
            layer
            for layer in self.model_config.placed_layers()
            if layer.moe is not None
        )
        assert moe_layer.moe is not None
        return moe_layer.moe.num_experts

    def _hf_projections(
        self, titan_abstract_key: str, value: Any
    ) -> Iterator[tuple[str, Any]]:
        """Yield ``(titan_abstract_key, weight)`` per projection HF stores on its own.
        ``w13_E2FD`` yields its gate and up slices under the ``w1_EFD`` /
        ``w3_EFD`` keys the HF maps use; any other weight is yielded as is.
        """
        if not titan_abstract_key.endswith(".w13_E2FD"):
            yield titan_abstract_key, value
            return
        prefix = titan_abstract_key.rsplit(".", 1)[0]
        for name, weight in zip(
            self._W13_PROJECTIONS, self.fused_to_separate_w13(value)
        ):
            yield f"{prefix}.{name}", weight

    def _grouped_experts_to_hf(
        self,
        value: Any,
        titan_abstract_key: str,
        to_hf_map: Mapping[str, str],
        hf_layer: str,
        num_experts: int,
    ) -> dict[str, Any]:
        """Split one GroupedExperts weight into per-expert HF entries.

        Shared by the main and MTP ``to_hf`` paths (they differ only in the key
        maps and the HF layer index). ``w13_E2FD`` is unstacked into the gate/up
        projections first. The online (DTensor) path records the grouped-weight
        metadata for ``from_hf`` and emits the locally-owned expert shards; the
        offline path splits all ``num_experts`` experts. A weight without an HF
        key (e.g. MTP experts in a format with no MTP experts) is skipped, as
        for any other entry; ``check_hf_export`` guards exports.
        """
        hf_state_dict: dict[str, Any] = {}
        for titan_key, weight in self._hf_projections(titan_abstract_key, value):
            hf_abstract_key = to_hf_map.get(titan_key)
            if hf_abstract_key is None:
                continue
            if isinstance(weight, DTensor):
                self.grouped_expert_weight_placements[titan_key] = weight.placements
                self.grouped_expert_weight_shape[titan_key] = weight.shape
                self.grouped_expert_weight_mesh[titan_key] = weight.device_mesh
                hf_state_dict.update(
                    self._get_local_experts_weights(
                        hf_abstract_key, titan_key, hf_layer, weight
                    )
                )
            else:
                split_values = self._split_experts_weights(weight, num_experts)
                hf_state_dict.update(
                    {
                        hf_abstract_key.format(hf_layer, expert_num): split_values[
                            expert_num
                        ]
                        for expert_num in range(num_experts)
                    }
                )
        return hf_state_dict

    def _grouped_experts_from_hf(
        self,
        expert_weights_by_layer: dict[str, Any],
        titan_abstract_key: str,
        layer_key: str,
        expert_num: str | int,
        value: Any,
        num_experts: int,
    ) -> tuple[str, Any] | None:
        """Accumulate one HF expert weight and return a completed titan entry.

        Shared by the main and MTP ``from_hf`` paths (they differ only in the key
        maps and the titan layer index). The online (DTensor) path is taken when
        ``to_hf`` recorded local expert indices for this key. A completed gate
        or up projection is parked under the layer's fused ``w13_E2FD`` key and
        returned, stacked, only once its partner has completed too.
        """
        expert_weights_by_layer.setdefault(layer_key, {}).setdefault(
            titan_abstract_key, {}
        )[int(expert_num)] = value
        if titan_abstract_key in self.local_experts_indices:
            stacked_value = self._concatenate_expert_weights_dtensor(
                expert_weights_by_layer, titan_abstract_key, layer_key
            )
        else:
            stacked_value = self._concatenate_expert_weights(
                expert_weights_by_layer,
                titan_abstract_key,
                layer_key,
                num_experts,
            )
        if stacked_value is None:
            return None

        prefix, name = titan_abstract_key.rsplit(".", 1)
        if name not in self._W13_PROJECTIONS:
            return titan_abstract_key.format(layer_key), stacked_value

        fused_abstract_key = f"{prefix}.w13_E2FD"
        slots = expert_weights_by_layer.setdefault(layer_key, {}).setdefault(
            fused_abstract_key, {}
        )
        slots[self._W13_PROJECTIONS.index(name)] = stacked_value
        if len(slots) < len(self._W13_PROJECTIONS):
            return None
        del expert_weights_by_layer[layer_key][fused_abstract_key]
        if not expert_weights_by_layer[layer_key]:
            del expert_weights_by_layer[layer_key]
        fused_value = self.separate_to_fused_w13(slots[0], slots[1]).contiguous()
        return fused_abstract_key.format(layer_key), fused_value

    @staticmethod
    def _validate_grouped_experts_from_hf(
        expert_weights_by_layer: dict[str, Any],
        *,
        context: str | None = None,
    ) -> None:
        """Reject an HF checkpoint with incomplete expert projections."""
        if not expert_weights_by_layer:
            return
        pending = {
            layer: sorted(keys) for layer, keys in expert_weights_by_layer.items()
        }
        # Only fused keys left means every expert arrived, but gate or up did not.
        only_fused = all(
            key.endswith(".w13_E2FD") for keys in pending.values() for key in keys
        )
        what = "gate/up expert weights" if only_fused else "expert weights"
        context_suffix = f" for {context}" if context else ""
        raise ValueError(f"Incomplete {what}{context_suffix}: {pending}")


def get_dense_model_nparams_and_flops(
    model: nn.Module,
    n_layers: int,
    n_heads: int,
    head_dims: int,
    seq_len: int,
    enable_weight_tying: bool = False,
) -> tuple[int, int]:
    """Param-count-based nparams and FLOPs/token (coarse ``6N`` + attention term).

    Fallback for models that do not expose structured per-block ``Config``
    objects to walk (e.g. the HuggingFace ``transformers`` modeling backend).
    Native torchtitan models use ``get_blockwise_nparams_and_flops`` instead,
    which sums the real matmul shapes block by block.

    Args:
        model: nn.Module representing the model.
        n_layers: The number of transformer layers.
        n_heads: The number of attention heads.
        head_dims: The sum of qk and v head dimensions.
        seq_len: The sequence length in training configs.
        enable_weight_tying: Whether weight tying is enabled.

    Returns:
        Tuple of (nparams, num_flops_per_token):
            nparams: Total number of model parameters.
            num_flops_per_token: Estimated number of floating point operations per token.
    """
    # model.parameters() de-duplicates shared parameters, so tied input/output
    # embeddings are counted once.
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, nn.Embedding)
    )

    # Reasoning behind the factor of 6 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul (attention scores and value aggregation,
    #    combined in head_dims, counted as 1) in the forward and 4 (counted as 2)
    #    in the backward                                                      (3)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. sparsity in causal attention is not accounted for
    # NOTE: (4) is the legacy convention, kept so MFU stays comparable with past
    # runs of these models. get_blockwise_nparams_and_flops halves the attention
    # core instead (see causal_softmax_core_flops), so the two are not comparable.
    # With tied embeddings, PyTorch's parameter iterator already counts the
    # shared input/output parameter once. That parameter still participates in
    # the lm_head matmul, so do not subtract it for either size or FLOPs.
    nparams_for_matmul = nparams if enable_weight_tying else nparams - nparams_embedding
    num_flops_per_token = (
        6 * nparams_for_matmul + 6 * n_layers * n_heads * head_dims * seq_len
    )

    return nparams, num_flops_per_token


# Training FLOPs = forward + 2x backward (dL/dX + dL/dW).
NUM_FLOPS_PER_FWD_FLOP = 3


def prediction_head_costing(model_config: Configurable.Config) -> tuple[str, int]:
    """Shared head FQN and number of passes (main prediction plus MTP heads)."""
    from torchtitan.models.common.decoder_blocks.mtp import MTPBlock

    num_heads = 1 + sum(1 for _ in model_config.traverse(MTPBlock.Config))
    return "lm_head", num_heads


def get_blockwise_nparams_and_flops(
    model_config: Decoder.Config,
    model: nn.Module,
    seq_len: int,
) -> tuple[int, int]:
    """Per-building-block analytical nparams and training FLOPs per token.

    Each building-block ``Config`` reports its forward matmul FLOPs per token
    (summed from real ``Linear`` shapes); this walks the per-layer configs, adds
    the lm_head matmul, and multiplies by 3 (forward + 2x backward). The count is
    the full-model per-token cost and is parallelism-invariant (the metrics
    processor already normalizes throughput by ``non_data_parallel_size``).
    Recompute (activation checkpointing, FlashAttention backward) is excluded,
    matching the MFU convention.

    Works for any model whose layers are common ``TransformerBlock`` configs
    (``attention`` + ``moe``/``feed_forward``) and whose ``Config`` exposes a
    ``lm_head`` ``Linear.Config``. Models without structured block configs (e.g.
    the HuggingFace backend) use ``get_dense_model_nparams_and_flops`` instead.

    Args:
        model_config: the model's ``Decoder.Config`` (per-layer block configs).
        model: the built ``nn.Module``, used only for the parameter count so that
            tied parameters are de-duplicated by ``model.parameters()``.
        seq_len: training sequence length (enters the attention core term).

    Returns:
        Tuple of ``(nparams, num_flops_per_token)``.
    """
    from torchtitan.models.common.moe import MoE
    from torchtitan.protocols.blocks import Block
    from torchtitan.protocols.module import Module

    # model.parameters() de-duplicates shared parameters (e.g. tied embeddings).
    nparams = sum(p.numel() for p in model.parameters())

    fwd_flops_per_token = 0
    unaccounted: list[str] = []
    # One walk over every block, wherever it sits. Each sums its own sublayers
    # (TransformerBlock.Config.fwd_flops_per_token), so neither a new
    # FLOP-bearing sublayer nor a new container can be silently dropped.
    for fqn, block_cfg, _, _ in model_config.traverse(Block.Config):
        if type(block_cfg).fwd_flops_per_token is Module.Config.fwd_flops_per_token:
            # A family that does not account for itself yet: the vision tower's
            # sequence is patches, not tokens, so "per token" needs a decision of
            # its own rather than a mechanical sum.
            unaccounted.append(fqn)
            continue
        fwd_flops_per_token += block_cfg.fwd_flops_per_token(seq_len)

    inactive_expert_params = 0
    # Experts not selected by the router don't run per token. Only the common MoE
    # config exposes the expert shape needed.
    for _, moe_cfg, _, _ in model_config.traverse(MoE.Config):
        e = moe_cfg.experts
        # A materialized tree has every width; a partial (compact) MoE does not.
        assert e.dim is not None, "FLOPs need the materialized model config"
        expert_params = moe_cfg.num_experts * 3 * e.dim * e.hidden_dim
        inactive_expert_params += (
            expert_params
            * (moe_cfg.num_experts - moe_cfg.router.top_k)
            // moe_cfg.num_experts
        )

    # The lm_head matmul runs once per prediction head (MTPLoss applies the
    # shared head per head), regardless of weight tying — tying only affects
    # nparams, taken from model.parameters() above.
    lm_head = model_config.lm_head
    if lm_head is None:
        raise ValueError("FLOP counting needs a completed LM head")
    head_in, head_out = lm_head.require_features()
    _, num_prediction_heads = prediction_head_costing(model_config)
    fwd_flops_per_token += num_prediction_heads * 2 * head_in * head_out

    if unaccounted:
        logger.warning(
            "FLOP counting skipped blocks that do not report their own FLOPs: "
            f"{unaccounted}. Reported tflops/mfu understate the real cost."
        )

    if inactive_expert_params:
        logger.info(
            f"Total parameter count: {nparams:,}, "
            f"active {nparams - inactive_expert_params:,}"
        )
    else:
        logger.info(f"Total parameter count: {nparams:,}")

    num_flops_per_token = NUM_FLOPS_PER_FWD_FLOP * fwd_flops_per_token
    return nparams, num_flops_per_token
