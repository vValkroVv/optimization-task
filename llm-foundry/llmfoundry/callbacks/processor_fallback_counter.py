from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import json
import time
import torch
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s |  %(message)s"
)
log = logging.getLogger(__name__)


@dataclass
class ProcessorFallbackReport:
    ts: float
    reason: str
    step: int
    total_fallbacks: int
    max_processor_fallback_count: Optional[int]


class ProcessorFallbackCounter(Callback):
    """
    Callback that counts processor fallbacks and stops training when limit is exceeded.
    It monitors batches via BEFORE_BATCH_SPLIT_DATALOADER hook to track fallback events.

    Args:
        max_processor_fallback_count: Maximum allowed processor fallbacks
            If None, no limit is set
        dump_path: Path to save JSON report when fallback limit is exceeded
            Can be absolute or relative to the working directory

    Raises:
        RuntimeError: When max_processor_fallback_count is exceeded
    """

    def __init__(
        self,
        max_processor_fallback_count: Optional[int] = 10000,
        dump_path: str = "FAIL_REASON.json",
    ):
        assert (
            max_processor_fallback_count is None or max_processor_fallback_count >= 0
        ), (
            f"max_processor_fallback_count must be >= 0 or None, got {max_processor_fallback_count}"
        )

        assert dump_path is not None, "dump_path must be provided"

        self.max_processor_fallback_count = max_processor_fallback_count
        self.dump_path = dump_path

        self.total_processor_fallbacks: int = 0
        self.modality_fallbacks: Dict[str, int] = {}
        self._ordered_modalities: Optional[List[str]] = None
        self.print_marker = "[ProcessorFallbackCounter]"

    @staticmethod
    def _extract_non_text_modalities_from_collate_fn(collate_fn: Any) -> List[str]:
        """Resolve the fixed modality order from an omni collator.

        Args:
            collate_fn: Active dataloader collate function.

        Returns:
            List[str]: Ordered non-text modalities expected from the dataloader.
        """
        collators = getattr(collate_fn, "collators", None)
        assert collators is not None, (
            "ProcessorFallbackCounter requires an omni collate_fn with a 'collators' attribute"
        )

        ordered_modalities: List[str] = []
        for collator in collators:
            data_modality = collator.data_modality.value
            if data_modality == "text":
                continue

            if data_modality not in ordered_modalities:
                ordered_modalities.append(data_modality)

        assert ordered_modalities, (
            "ProcessorFallbackCounter requires at least one non-text modality in the active omni collator"
        )
        return ordered_modalities

    def _resolve_ordered_modalities(self, state: State) -> List[str]:
        """Cache the fixed modality order for the active train dataloader.

        Args:
            state: Current trainer state.

        Returns:
            List[str]: Ordered non-text modalities expected from each batch.
        """
        if self._ordered_modalities is not None:
            return self._ordered_modalities

        dataloader = state.dataloader or state.train_dataloader
        assert dataloader is not None, (
            "ProcessorFallbackCounter requires state.dataloader or state.train_dataloader to be set"
        )

        collate_fn = getattr(dataloader, "collate_fn", None)
        assert collate_fn is not None, (
            "ProcessorFallbackCounter requires the active dataloader to expose collate_fn"
        )

        self._ordered_modalities = self._extract_non_text_modalities_from_collate_fn(
            collate_fn
        )
        return self._ordered_modalities

    @staticmethod
    def _build_local_fallback_counts(
        modality_inputs: Mapping[str, Mapping[str, Any]],
        ordered_modalities: Sequence[str],
    ) -> List[int]:
        """Build a dense local fallback vector with zero-fill for missing modalities.

        Args:
            modality_inputs: Sparse modality payloads from the current raw batch.
            ordered_modalities: Fixed modality order derived from the dataloader.

        Returns:
            List[int]: Per-modality fallback counts aligned with ``ordered_modalities``.
        """
        unexpected_modalities = sorted(
            set(str(modality) for modality in modality_inputs.keys())
            - set(ordered_modalities)
        )
        assert not unexpected_modalities, (
            "ProcessorFallbackCounter encountered modalities outside the active dataloader schema: "
            f"{unexpected_modalities}"
        )

        local_fallback_counts: List[int] = []
        for modality in ordered_modalities:
            if modality not in modality_inputs:
                local_fallback_counts.append(0)
                continue

            modality_payload = modality_inputs[modality]
            assert "processor_fallback_count" in modality_payload, (
                f"ProcessorFallbackCounter expected 'processor_fallback_count' in modality_inputs[{modality!r}]"
            )
            local_fallback_counts.append(
                int(modality_payload["processor_fallback_count"])
            )

        return local_fallback_counts

    @staticmethod
    def _get_reduce_device() -> torch.device:
        """Return the device for distributed fallback-count reductions.

        Returns:
            torch.device: CUDA device for NCCL collectives when available, otherwise CPU.
        """
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    def before_batch_split_dataloader(self, state: State, logger: Logger):
        assert "modality_inputs" in state.raw_batch, (
            "ProcessorFallbackCounter works only with StreamingOmniDataset, so modality_inputs must be in raw_batch"
        )

        modality_inputs = state.raw_batch["modality_inputs"]
        ordered_modalities = self._resolve_ordered_modalities(state)
        local_processor_fallbacks = self._build_local_fallback_counts(
            modality_inputs=modality_inputs,
            ordered_modalities=ordered_modalities,
        )

        if dist.get_world_size() > 1:
            processor_fallbacks_tensor = torch.tensor(
                local_processor_fallbacks,
                dtype=torch.int64,
                device=self._get_reduce_device(),
            )
            dist.all_reduce(processor_fallbacks_tensor, reduce_operation="SUM")
            global_processor_fallbacks = processor_fallbacks_tensor.cpu().tolist()
        else:
            global_processor_fallbacks = local_processor_fallbacks

        modality_metrics = {}
        for modality, global_fallback_count in zip(
            ordered_modalities, global_processor_fallbacks
        ):
            self.modality_fallbacks[modality] = (
                self.modality_fallbacks.get(modality, 0) + global_fallback_count
            )
            self.total_processor_fallbacks += global_fallback_count

            metric_name = f"trainer/{modality}_processor_fallbacks"
            modality_metrics[metric_name] = self.modality_fallbacks[modality]

        if dist.get_global_rank() == 0:
            logger.log_metrics(
                {
                    **modality_metrics,
                    "trainer/total_processor_fallbacks": self.total_processor_fallbacks,
                }
            )

        processor_fallback_count_exceeded = (
            self.max_processor_fallback_count is not None
            and self.total_processor_fallbacks > self.max_processor_fallback_count
        )

        if processor_fallback_count_exceeded:
            self._trigger_fail(
                state,
                logger,
                reason="processor_fallback_count_exceeded_threshold",
                step=int(state.timestamp.batch),
                total_fallbacks=self.total_processor_fallbacks,
                max_processor_fallback_count=self.max_processor_fallback_count,
            )

    def _trigger_fail(
        self,
        state: State,
        logger: Logger,
        *,
        reason: str,
        step: int,
        total_fallbacks: int,
        max_processor_fallback_count: Optional[int],
    ) -> None:
        msg = (
            f"{self.print_marker}: {reason} | "
            f"step={step} | "
            f"total_fallbacks_count={total_fallbacks} | "
            f"max_processor_fallback_count={max_processor_fallback_count}"
        )

        if dist.get_global_rank() == 0:
            report = ProcessorFallbackReport(
                ts=time.time(),
                reason=reason,
                step=step,
                total_fallbacks=total_fallbacks,
                max_processor_fallback_count=max_processor_fallback_count,
            )
            try:
                dump_dir = os.path.dirname(self.dump_path)
                if dump_dir:
                    os.makedirs(dump_dir, exist_ok=True)
                with open(self.dump_path, "w") as f:
                    json.dump(asdict(report), f, indent=2)

                log.info(f"{self.print_marker} Report saved to {self.dump_path}")

            except Exception as e:
                log.error(
                    f"{self.print_marker} Could not write processor fallback report: {e}",
                    exc_info=True,
                )

        log.error(msg)
        raise RuntimeError(msg)

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_processor_fallbacks": self.total_processor_fallbacks,
            "modality_fallbacks": self.modality_fallbacks,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.total_processor_fallbacks = state_dict.get("total_processor_fallbacks", 0)
        self.modality_fallbacks = state_dict.get("modality_fallbacks", {})

        if dist.get_global_rank() == 0:
            log.info(
                f"{self.print_marker} Restored state: "
                f"total_processor_fallbacks={self.total_processor_fallbacks}, "
                f"modality_fallbacks={self.modality_fallbacks}"
            )
