# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import functools
import json
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import Future
from typing import Any, cast, Protocol, runtime_checkable, TypeAlias

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.config.schema.components.checkpoint import CheckpointManagerConfig
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.protocols.checkpoint_adapter import CheckpointAdapter
from torchtitan.protocols.sharding_backend import (
    EMA_MODEL,
    ShardingBackend,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection, TORCH_DTYPE_MAP

_HF_TOKENIZER_EXTRAS = frozenset(
    {"special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt"}
)


def hf_remote_code_files(hf_config: dict[str, Any]) -> tuple[str, ...]:
    """The ``*.py`` files ``auto_map`` refers to, e.g. ``mod.Class -> mod.py``.

    Empty for an architecture ``transformers`` knows natively; a config that
    names ``auto_map`` is only loadable next to the modules it points at.
    """
    auto_map = hf_config.get("auto_map") or {}
    return tuple(sorted({f"{ref.rsplit('.', 1)[0]}.py" for ref in auto_map.values()}))


def _reject_hf_with_checkpoint_adapter(
    *,
    fsdp_backend: str,
    has_checkpoint_adapter: bool,
    save_in_hf: bool,
    last_save_in_hf: bool,
    initial_load_in_hf: bool,
) -> None:
    """HF conversion reads the plain model state dict, which a backend with its
    own checkpoint adapter does not provide; refuse it at construction rather
    than inside ``save()``, hours into a run.
    """
    flags = [
        name
        for name, enabled in (
            ("checkpoint.save_in_hf", save_in_hf),
            ("checkpoint.last_save_in_hf", last_save_in_hf),
            ("checkpoint.initial_load_in_hf", initial_load_in_hf),
        )
        if enabled
    ]
    if not flags or not has_checkpoint_adapter:
        return
    raise ValueError(
        f"{' and '.join(flags)} is not supported on fsdp_backend="
        f"{fsdp_backend!r}, which checkpoints through its own adapter: the HF "
        f"export reads a plain model state dict and {fsdp_backend} does not "
        f"produce one. Set {' and '.join(flags)} to False and keep the ordinary "
        "DCP checkpoint -- the autometrics callback submits that checkpoint "
        "itself on this backend, and the PAM job converts it to HF."
    )


def warn_unmapped_hf_entries(
    state_dict: dict[str, Any],
    loaded: dict[str, Any],
    *,
    outcome: str = "keep their initial values",
) -> None:
    """Warn about entries an HF load skipped: those the HF format cannot hold
    (LoRA adapters, expert biases, extra norms). ``outcome`` says what happens
    to them; in a training load they keep their initial values."""
    if skipped := sorted(set(state_dict) - set(loaded)):
        logger.warning(
            f"{len(skipped)} state entries have no HF mapping and {outcome}: "
            f"{skipped[:8]}"
        )


def resolve_hf_export(
    sd_adapter: Any,
    model_config: Any,
    hf_remote_code_dir: str,
    *,
    require_config: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """The ``config.json`` to write and where its remote code comes from.

    Raises ``ValueError`` on the two ways an export ends up unloadable: an
    adapter that cannot describe this model, and remote-code files that are
    not where ``auto_map`` needs them. Called at construction so a run dies
    before the first step, and again at save time to do the copying. Takes the
    adapter class or an instance -- Autometrics validates from the model spec,
    before any adapter is built.

    ``require_config=False`` is the checkpointer's mode: an adapter without an
    ``hf_config`` writer degrades to the upstream
    weights-plus-tokenizer export -- ``(None, None)`` with a warning -- instead
    of refusing the run. Autometrics keeps the default: PAM loads the directory,
    so an export without ``config.json`` is useless there.
    """
    if sd_adapter is None:
        raise ValueError("HF export needs a state dict adapter.")
    try:
        config = sd_adapter.hf_config(model_config)
    except NotImplementedError as exc:
        if not require_config:
            logger.warning(
                f"{getattr(sd_adapter, '__name__', type(sd_adapter).__name__)} has no "
                "hf_config() writer: the HF export will hold weights and tokenizer "
                "files but no config.json; point transformers at the original "
                "model's config to load it."
            )
            return None, None
        raise ValueError(
            f"{getattr(sd_adapter, '__name__', type(sd_adapter).__name__)}"
            ".hf_config() cannot describe this model, "
            f"so its HF export would have no config.json: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"{getattr(sd_adapter, '__name__', type(sd_adapter).__name__)}"
            ".hf_config() cannot describe this model, "
            f"so its HF export would have no config.json: {exc}"
        ) from exc
    files = hf_remote_code_files(config)
    if not files:
        return config, None
    code_dir = hf_remote_code_dir or sd_adapter.default_hf_remote_code_dir()
    if not code_dir:
        raise ValueError(
            f"HF export needs the remote-code files {list(files)} next to the "
            "weights and no directory is known; set checkpoint.hf_remote_code_dir."
        )
    missing = [f for f in files if not os.path.isfile(os.path.join(code_dir, f))]
    if missing:
        raise ValueError(
            f"HF export needs the remote-code files {list(files)} next to the "
            f"weights, but {code_dir} lacks {missing}. Check out the directory "
            "that ships them or set checkpoint.hf_remote_code_dir."
        )
    return config, code_dir


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"

RESERVED_CHECKPOINT_KEYS = frozenset(
    {MODEL, OPTIMIZER, LR_SCHEDULER, DATALOADER, TRAIN_STATE}
)


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


@runtime_checkable
class ModelWeightsSource(Protocol):
    """A checkpoint state exposing EMA weights and whether to export them."""

    holds_average: bool
    use_ema: bool

    def model_state_dict(self) -> dict[str, torch.Tensor]: ...


def _ema_weights_source(states: dict[str, Any]) -> ModelWeightsSource:
    sources = [
        state for state in states.values() if isinstance(state, ModelWeightsSource)
    ]
    if len(sources) != 1:
        raise ValueError(
            "checkpoint.last_save_ema_model_also requires exactly one checkpoint "
            f"state providing model_state_dict; found {len(sources)}. "
            "Enable one EMA callback."
        )
    return sources[0]


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        return state_dict

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))
        # `set_model_state_dict()` does change the keys of the input state_dict,
        # we will need to reinitialize the cache_state_dict.
        self.cache_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


def _cast_tensor_leaves(value: Any, dtype: torch.dtype) -> Any:
    """Cast every tensor in a possibly nested state dict, leaving the rest."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    if isinstance(value, dict):
        return {k: _cast_tensor_leaves(v, dtype) for k, v in value.items()}
    return value


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager(Configurable):
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is guaranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        config (Checkpoint): The config used to configure the checkpointing.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with config.folder
        sd_adapter (Optional[type[BaseStateDictAdapter]]): The adapter used to convert model state
            dicts between native format and other formats.

    """

    Config: TypeAlias = CheckpointManagerConfig

    def __init__(
        self,
        config: Config,
        *,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        sharding_backend: ShardingBackend,
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
    ) -> None:
        config.validate_save_options()
        if config.enable and not config.load_only and config.last_save_ema_model_also:
            _ema_weights_source(states)
        self.enable = config.enable
        self.load_only = config.load_only

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )
        self.sharding_backend = sharding_backend
        self.checkpoint_adapter: CheckpointAdapter | None = (
            sharding_backend.checkpoint_adapter(model_parts)
        )

        async_mode = config.async_mode.lower()
        self.enable_staging = (
            self.enable and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        )

        self.last_saved_path: str | None = None

        if not self.enable:
            return

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None
        self.pg: dist.ProcessGroup | None = None

        self.folder = os.path.join(base_folder, config.folder)

        # Checkpoint policy related fields.
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_path = config.initial_load_path
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized
        self.last_save_model_only = config.last_save_model_only
        self.save_in_hf = config.hf_save_enabled and config.save_in_hf
        self.last_save_in_hf = config.hf_save_enabled and config.last_save_in_hf
        self.last_save_ema_model_also = config.last_save_ema_model_also
        self.ema_folder = os.path.join(base_folder, config.ema_folder)
        if (self.save_in_hf or self.last_save_in_hf) and sd_adapter is None:
            flags = " and ".join(
                name
                for name, enabled in (
                    ("checkpoint.save_in_hf", self.save_in_hf),
                    ("checkpoint.last_save_in_hf", self.last_save_in_hf),
                )
                if enabled
            )
            raise ValueError(f"{flags} is set, but the model has no sd_adapter.")
        _reject_hf_with_checkpoint_adapter(
            fsdp_backend=sharding_backend.name,
            has_checkpoint_adapter=self.checkpoint_adapter is not None,
            save_in_hf=self.save_in_hf,
            last_save_in_hf=self.last_save_in_hf,
            initial_load_in_hf=config.initial_load_in_hf,
        )
        if self.save_in_hf:
            self._check_hf_export_weights()
        if self.last_save_ema_model_also and self._ema_store() is None:
            raise ValueError(
                "checkpoint.last_save_ema_model_also is set, but no checkpoint "
                "state carries the averaged weights under 'ema_model'. On "
                "fsdp2 add the EMA callback to 'callbacks'; on gfsdp2 the flag "
                "is not supported -- every checkpoint already carries "
                "ema_model.* tensors, which the converters read with use_ema."
            )
        self.sd_adapter = sd_adapter
        if self.save_in_hf or self.last_save_in_hf:
            # Fail here, before the first step: an export whose config.json
            # cannot be written (or whose remote code is missing) is discovered
            # at construction, not after the first checkpoint was saved.
            resolve_hf_export(
                sd_adapter,
                sd_adapter.model_config,
                config.hf_remote_code_dir,
                require_config=False,
            )
            sd_adapter.check_hf_export(self.states[MODEL].state_dict())
        self.export_dtype = TORCH_DTYPE_MAP[config.export_dtype]
        self.hf_remote_code_dir = config.hf_remote_code_dir
        self.exclude_from_loading = config.exclude_from_loading
        self.interval = config.interval
        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint

        # Async checkpoint related fields.
        async_mode = config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

        self.keep_latest_k = config.keep_latest_k
        self.purge_thread: threading.Thread | None = None
        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(f"Unknown checkpoint async_mode {config.async_mode}")

        if (
            self.async_mode != AsyncMode.DISABLED
            and self.checkpoint_adapter is not None
        ):
            # The adapter's entries alias live parameter buffers, which an
            # asynchronous save reads after the step has moved on.
            raise ValueError(
                f"checkpoint.async_mode={config.async_mode!r} is not supported "
                f"on fsdp_backend={sharding_backend.name!r}: its checkpoint "
                "state aliases live parameter buffers. Use async_mode='disabled'."
            )

        if self.save_in_hf and self.async_mode != AsyncMode.DISABLED:
            raise ValueError(
                "checkpoint.save_in_hf runs a synchronous DCP save and "
                "safetensors consolidation on every interval save, which "
                "cancels asynchronous checkpointing (checkpoint.async_mode="
                f"'{config.async_mode}'). Use async_mode='disabled' with "
                "save_in_hf, or last_save_in_hf for one export at the end."
            )

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                # pyrefly: ignore [missing-attribute]
                self.mp_queue_send.put(Terminate())
                self.mp.join()
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        checkpoint_save_id: str | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None
        if to_hf:
            assert self.sd_adapter is not None, (
                "trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
            state_dict = self.sd_adapter.to_hf(state_dict)

            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            if fqn_to_index_mapping:
                storage_writer = HuggingFaceStorageWriter(
                    path=os.path.join(checkpoint_id, "sharded"),
                    save_distributed=True,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    enable_consolidation=False,
                )
            else:
                # the reason for only enabling consolidation if there is
                # no mapping is because no mapping implies that we save all fqns
                # to one file. This means we only need one rank to consolidate.
                # Otherwise we should use consolidate_safetensors_files_on_every_rank
                storage_writer = HuggingFaceStorageWriter(
                    path=checkpoint_id,
                    save_distributed=True,
                    enable_consolidation=True,
                )

        else:
            checkpoint_save_id = checkpoint_id

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        if to_hf and fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
        is_model_part_of_load: bool = True,
    ) -> None:
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.
            is_model_part_of_load (bool): Whether the model is part of this load.
        """

        if from_hf:
            assert self.sd_adapter is not None, (
                "trying to load checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(
                hf_state_dict,
                storage_reader=hf_storage_reader,
            )

            loaded = self.sd_adapter.from_hf(hf_state_dict)
            warn_unmapped_hf_entries(state_dict, loaded)
            self.states[MODEL].load_state_dict(loaded)
        elif self.checkpoint_adapter is not None:
            self.checkpoint_adapter.load(
                state_dict,
                checkpoint_id=checkpoint_id,
                process_group=self.pg,
                load_model=is_model_part_of_load,
            )
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if is_model_part_of_load and MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @sl.log_trace_span("checkpoint_save")
    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``last_step`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == trainer_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.

        Returns:
            bool: True if a checkpoint was written (or staged, for async modes) on
            this step.
        """
        if not self._should_save(curr_step, last_step):
            return False

        sl.add_step_tag("checkpoint_save")
        begin = time.monotonic()
        logger.info("Saving the checkpoint (or staging if async is enabled).")
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self.last_saved_path = checkpoint_id
        self._async_wait()
        # This GC is called for async checkpoint as it is useless to do
        # GC right after async_save -- the CPU memory is not able to be
        # freed until _async_wait()
        if last_step:
            self._save_last_step(curr_step)
        else:
            states = self._flattened_model_states_sd()
            if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                if self.stager is None:
                    self.stager = DefaultStager(StagingOptions(True, True, True, True))
                result = self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=self.async_mode,
                )
                assert isinstance(result, AsyncSaveResponse)
                self.save_future = result.upload_completion
                self.staging_future = result.staging_completion
                self.staging = True
            elif self.async_mode == AsyncMode.ASYNC:
                GarbageCollection.collect("GC collection invoked by checkpointer.")
                self.save_future = self.dcp_save(
                    states, checkpoint_id=checkpoint_id, async_mode=self.async_mode
                )
                GarbageCollection.collect("GC collection invoked by checkpointer.")
            else:
                self.dcp_save(
                    states,
                    checkpoint_id=checkpoint_id,
                    async_mode=AsyncMode.DISABLED,
                    enable_garbage_collection=True,
                )
        if self.save_in_hf and not (last_step and self.last_save_in_hf):
            # The last-step path exports through last_save_in_hf itself;
            # exporting here too wrote the same model twice.
            self._save_hf_export(checkpoint_id)
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled) "
            f"in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    @sl.log_trace_span("checkpoint_load")
    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """
        if not self.enable:
            return False

        model_only = False
        from_hf = False
        from_quantized = False
        if not os.path.exists(self.folder):
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized
            if from_hf:
                assert model_only, (
                    "Only model can be loaded when loading from HF's safetensors checkpoint."
                )

            if from_quantized:
                assert from_hf, (
                    "Quantized checkpoint can only be loaded from HuggingFace format."
                )

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "checkpoint.initial_load_path is specified but the path is not valid."
                    )
                if from_hf:
                    logger.info(
                        f"loading from HF safetensors from --checkpoint.initial_load_path: {self.initial_load_path}"
                    )
            elif from_hf:
                assert (
                    self.sd_adapter is not None
                    and self.sd_adapter.hf_assets_path is not None
                ), "from_hf is True but sd_adapter or hf_assets_path is not provided."
                hf_assets_path = self.sd_adapter.hf_assets_path
                checkpoint_id = hf_assets_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "hf_assets_path is being used to load HF weights but the path is not valid. \
                        Either make sure hf_assets_path is correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    f"loading HF safetensors from --hf_assets_path: {hf_assets_path}"
                )
            else:
                return False
        else:
            if self.initial_load_path:
                logger.warning(
                    "checkpoint.initial_load_path is provided but the checkpoint.folder exists. "
                    f"Checkpointer will use the checkpoints from the checkpoint.folder {self.folder}."
                )
            if self.initial_load_in_hf:
                logger.warning(
                    "checkpoint.initial_load_in_hf is True but the checkpoint.folder exists. "
                    "Checkpointer will not load from HF safetensors"
                )
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} but checkpoint {checkpoint_id} is not found."
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(model_only)
        is_model_part_of_load = model_only or MODEL not in self.exclude_from_loading
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
            is_model_part_of_load=is_model_part_of_load,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            assert self.staging_future is not None
            self.staging_future.result()
            self.staging = False

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
            dcp_metadata_probe = os.path.join(folder, filename, ".metadata")
            safetensors_metadata_probe = os.path.join(
                folder, filename, "model.safetensors.index.json"
            )
            if match and os.path.isfile(dcp_metadata_probe):
                step_counts.append(int(match.group(1)))
            elif match and os.path.isfile(safetensors_metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None, *, for_load: bool = False
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.

        Note that other states, such as optimizer states, are not flattened.
        With a checkpoint adapter the model is written from the adapter's state
        and loaded by the adapter itself, so a load target carries no model.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        adapter = self.checkpoint_adapter
        if adapter is not None and OPTIMIZER in sd:
            sd[OPTIMIZER] = adapter.wrap_optimizer_state(sd[OPTIMIZER])
        if MODEL in states:
            if adapter is None:
                sd.update(states[MODEL].state_dict())
            elif not for_load:
                sd.update(adapter.model_save_state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.

        This API is used to determine which states to load based on the
        configurations.

        Args:
            model_only (bool): Whether to load the model only.

        Returns:
            Dict[str, Any]: The states to load for the given step.
        """
        # For the first step, we will only load the model.
        if model_only:
            # A checkpoint adapter loads the model itself.
            if self.checkpoint_adapter is not None:
                return {}
            return self.states[MODEL].state_dict()

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        states_to_load = self._flattened_model_states_sd(states_to_load, for_load=True)

        return states_to_load

    def _ema_store(self) -> ModelWeightsSource | None:
        """The EMA store, or None when the run does not average."""
        return self.states.get(EMA_MODEL)

    def _check_hf_export_weights(self) -> None:
        ema = self._ema_store()
        if ema is None:
            logger.info("HF export takes the model weights: no EMA in this run.")
            return
        if not ema.use_ema:
            logger.info("HF export takes the model weights: the EMA has use_ema=False.")
            return
        unknown = ema.model_state_dict().keys() - self.states[MODEL].state_dict().keys()
        if unknown:
            raise ValueError(
                f"The EMA shadows {sorted(unknown)[:5]} are not model weights, so "
                "an HF export cannot lay them over the model's state dict."
            )
        logger.info(
            "HF export takes the EMA weights from the step the average starts; "
            "until then the model weights, with a warning."
        )

    def _hf_export_states(self) -> tuple[dict[str, Any], str]:
        """The weights an HF export publishes, and which kind they are."""
        states = dict(self.states[MODEL].state_dict())
        ema = self._ema_store()
        if ema is None or not ema.use_ema:
            return states, "live"
        if not ema.holds_average:
            logger.warning(
                "HF export takes the model weights: the EMA has not reached "
                "its start_step, so its shadows are still the weights the run "
                "started from."
            )
            return states, "live"

        shadows = ema.model_state_dict()
        states.update(shadows)
        logger.info(
            "HF export takes the EMA weights (%d of %d tensors averaged).",
            len(shadows),
            len(states),
        )
        return states, "ema"

    def _save_hf_export(self, checkpoint_id: str) -> None:
        """Write a model-only HF copy beside the DCP checkpoint.

        Lands in `{checkpoint}/artifacts/hf_checkpoint`, which is where the
        evaluation service looks. Model only and always synchronous: TODO: make asynchronous
        """
        model_states, weights_kind = self._hf_export_states()
        if self.export_dtype != torch.float32:
            model_states = {k: v.to(self.export_dtype) for k, v in model_states.items()}
        hf_path = os.path.join(checkpoint_id, "artifacts", "hf_checkpoint")
        logger.info(f"Exporting an HF safetensors copy to {hf_path}.")
        self.dcp_save(
            model_states,
            checkpoint_id=hf_path,
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=False,
            to_hf=True,
        )
        self._write_hf_metadata(hf_path, weights_kind=weights_kind)

    def _write_hf_metadata(self, hf_dir: str, *, weights_kind: str) -> None:
        """Make an exported HF directory loadable on its own, and say what is in it."""
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            try:
                self._write_hf_metadata_rank0(hf_dir, weights_kind)
            except Exception:
                logger.exception(
                    f"HF export {hf_dir}: config.json/tokenizer/remote code were not "
                    "written; the directory is not loadable as an HF model. Training "
                    "continues."
                )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _write_hf_metadata_rank0(self, hf_dir: str, weights_kind: str) -> None:
        with open(os.path.join(hf_dir, "export_info.json"), "w") as f:
            json.dump({"weights": weights_kind}, f)
            f.write("\n")

        config, code_dir = resolve_hf_export(
            self.sd_adapter,
            self.sd_adapter.model_config,
            self.hf_remote_code_dir,
            require_config=False,
        )
        if config is not None:
            # Older transformers read torch_dtype, newer ones dtype; write both.
            dtype = str(self.export_dtype).removeprefix("torch.")
            config["torch_dtype"] = dtype
            config["dtype"] = dtype
            with open(os.path.join(hf_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
                f.write("\n")
        assets = self.sd_adapter.hf_assets_path
        if assets and os.path.isdir(assets):
            for name in os.listdir(assets):
                src = os.path.join(assets, name)
                if os.path.isfile(src) and (
                    name.startswith("tokenizer") or name in _HF_TOKENIZER_EXTRAS
                ):
                    shutil.copyfile(src, os.path.join(hf_dir, name))
        else:
            logger.warning(
                f"HF export {hf_dir}: hf_assets_path {assets!r} is not a "
                "directory, no tokenizer files were copied."
            )
        for name in hf_remote_code_files(config or {}):
            shutil.copyfile(os.path.join(code_dir, name), os.path.join(hf_dir, name))
        logger.info(
            f"Wrote export_info.json, config.json, tokenizer and remote code to {hf_dir}."
        )

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So this
        # won't affect preemption and training resume. We also only allow dtype
        # conversion when we are checkpointing model only and the current dtype
        # is not the same as the export dtype at the end of the training.

        if self.last_save_model_only:
            states = (
                self.states[MODEL].state_dict()
                if self.checkpoint_adapter is None
                else self.checkpoint_adapter.model_save_state_dict()
            )

            if self.export_dtype != torch.float32:
                states = _cast_tensor_leaves(states, self.export_dtype)
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        if self.last_save_in_hf and not self.last_save_model_only:
            raise ValueError(
                "checkpoint.last_save_in_hf requires checkpoint.last_save_model_only=true"
            )

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )
        if self.last_save_in_hf:
            self._write_hf_metadata(
                self._create_checkpoint_id(curr_step), weights_kind="live"
            )
        if self.last_save_ema_model_also:
            self._save_last_step_ema_model(curr_step)

    def _save_last_step_ema_model(self, curr_step: int) -> None:
        source = self._ema_store()
        if source is None:
            logger.warning(
                "checkpoint.last_save_ema_model_also=True, but the run has no "
                "EMA callback. No EMA checkpoint is written."
            )
            return
        if not source.holds_average:
            logger.warning(
                "checkpoint.last_save_ema_model_also=True, but %s has not "
                "averaged anything: the run never reached the EMA's start_step, "
                "so the shadows are still the weights it started from. No EMA "
                "checkpoint is written.",
                type(source).__name__,
            )
            return

        states = source.model_state_dict()
        if self.export_dtype != torch.float32:
            states = {k: v.to(self.export_dtype) for k, v in states.items()}
        logger.info(
            f"Saving an EMA model-only checkpoint in {self.export_dtype} "
            f"at last step, step {curr_step}."
        )
        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step, folder=self.ema_folder),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
        )

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled."
            )

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints.

        Extracted so subclasses (e.g. FTCheckpointManager) can add
        additional guards (like participating_rank) without duplicating
        the purge loop in _purge_stale_checkpoints.
        """
        return (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
        )

    def _purge_stale_checkpoints(self):
        if self._should_purge():
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
