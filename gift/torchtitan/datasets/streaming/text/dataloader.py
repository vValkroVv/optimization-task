import os
import uuid
from dataclasses import replace
from typing import Any, TypeAlias

from torchtitan.config.schema.datasets.streaming.text.dataloader import (
    _has_enabled_fim,
    _stream_fim_rates,
    StreamingTextDataLoaderConfig,
)
from torchtitan.distributed.data_world import (
    DataWorld,
    get_dataloader_shared_memory_namespace,
)
from torchtitan.datasets.streaming.core.dataloader import (
    BaseStreamingDataLoader,
    use_streaming_data_world,
)
from torchtitan.datasets.streaming.core.streams import build_streams
from torchtitan.datasets.streaming.text.collators import TextPretrainCollator
from torchtitan.datasets.streaming.text.datasets import StreamingTextDataset
from torchtitan.datasets.streaming.text.transforms import (
    FIMTransform,
)


class StreamingTextDataLoader(BaseStreamingDataLoader):
    """Titan dataloader for already-tokenized streaming text MDS data"""

    # pyrefly: ignore [bad-override]
    Config: TypeAlias = StreamingTextDataLoaderConfig

    def __init__(
        self,
        config: Config,
        *,
        data_world: DataWorld,
        tokenizer: Any,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = None,
    ) -> None:
        dataset_config = config.dataset
        stream_rates, default_fim_rates = _stream_fim_rates(dataset_config)
        transforms = []
        if _has_enabled_fim(stream_rates, default_fim_rates):
            transforms.append(
                FIMTransform(
                    tokenizer=tokenizer,
                    max_seq_len=seq_len,
                    seed=dataset_config.shuffle_seed,
                    stream_fim_rates=stream_rates,
                    default_fim_rates=default_fim_rates,
                )
            )

        streams = build_streams(dataset_config.streams)
        if data_world.shared_memory_namespace is None:
            if data_world.local_world_size > 1:
                master_addr = os.environ.get("MASTER_ADDR", "standalone")
                master_port = os.environ.get("MASTER_PORT", "0")
                shared_memory_namespace = (
                    f"torchtitan_streaming_text_{master_addr}:{master_port}"
                )
            else:
                shared_memory_namespace = get_dataloader_shared_memory_namespace(
                    "streaming_text",
                    f"{os.environ.get('RANK', '0')}_loader_{uuid.uuid4().hex[:8]}",
                )
            data_world = replace(
                data_world, shared_memory_namespace=shared_memory_namespace
            )
        with use_streaming_data_world(data_world):
            dataset = StreamingTextDataset(
                streams=streams,
                seq_len=seq_len,
                token_field=config.token_field,
                transforms=transforms,
                transforms_epoch=config.transforms_epoch,
                epoch_size=dataset_config.epoch_size,
                predownload=dataset_config.predownload,
                cache_limit=dataset_config.cache_limit,
                partition_algo=dataset_config.partition_algo,
                num_canonical_nodes=dataset_config.num_canonical_nodes,
                ncn_multiplier=dataset_config.ncn_multiplier,
                batch_size=local_batch_size,
                replication=(
                    data_world.replication if data_world.replication > 1 else None
                ),
                shuffle=dataset_config.shuffle,
                shuffle_algo=dataset_config.shuffle_algo,
                shuffle_seed=dataset_config.shuffle_seed,
                shuffle_block_size=dataset_config.shuffle_block_size,
                sampling_method=dataset_config.sampling_method,
                batching_method=dataset_config.batching_method,
                legacy=dataset_config.legacy,
            )

        # eos_id need for block_causal mask type support
        if getattr(tokenizer, "eos_id", None) is None:
            raise ValueError(
                "StreamingTextDataLoader requires tokenizer.eos_id to separate packed documents"
            )
        collate_fn = TextPretrainCollator(
            seq_len=seq_len, eos_token_id=tokenizer.eos_id
        )

        self._init_streaming_loader(
            dataset=dataset,
            data_world=data_world,
            batch_size=local_batch_size,
            collate_fn=collate_fn,
            drop_last=config.drop_last,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            infinite=config.infinite,
        )
