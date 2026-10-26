import json
import torch
import logging
import inspect
import tempfile

from pathlib import Path
from torchmetrics import Metric
from omegaconf import DictConfig
from typing import Any, Dict, List, Mapping, Optional
from transformers import PreTrainedTokenizerBase
from transformers.utils.generic import ModelOutput

from composer.models import ComposerModel
from composer.metrics.nlp import InContextLearningMultipleChoiceAccuracy
from llmfoundry.models.ops.included_proj_z_loss import (
    collect_included_proj_z_losses,
)
from llmfoundry.utils.config_utils import write_lora_config

from .configuration_gigaomni import GigaOmniConfig
from .modelling_gigaomni import GigaOmniForCausalLM
from .gigafsdp_wrappers_gigaomni import _build_gigafsdp_wrappers_config


logging.basicConfig(
    format="%(asctime)s | %(levelname)s |  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ComposerGigaOmniForCausalLM(ComposerModel):
    def __init__(
        self, om_model_config: DictConfig, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        super().__init__()

        self.config = GigaOmniConfig(
            om_model_config.language_model,
            om_model_config.get("encoders", []),
            om_model_config.get("adapters", []),
            om_model_config.get("heads", []),
            om_model_config.get("modalities", []),
        )

        self.config = write_lora_config(self.config, om_model_config)

        self.model = GigaOmniForCausalLM(self.config)
        self.tokenizer = tokenizer

        self.train_metrics = []
        self.val_metrics = [InContextLearningMultipleChoiceAccuracy()]

        self.model_forward_args = inspect.getfullargspec(self.model.forward).args

        self.labels = None

    def forward(self, batch: Mapping) -> ModelOutput:
        if isinstance(batch, dict):
            batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            output = self.model(**batch)
        else:
            raise ValueError(
                "Unexpected batch type. Expected a dictionary with keys corresponding "
                "the forward function arguments of the GigaOmniForCausalLM model."
            )
        return output

    def loss(self, outputs: ModelOutput, batch: Mapping) -> Dict[str, torch.Tensor]:
        assert not self.model.language_model.config.return_dict, (
            "ComposerGigaOmniForCausalLM does not support return_dict!"
        )

        # NOTE: Order of possible losses must be exactly the same as their order in the GigaOmniForCausalLM outputs but in reverse
        possible_losses: List[Dict[str, Any]] = [
            {
                "name": "hidden_z_loss_dense_mlp",
                "used": self.model.config.language_model.get("hidden_z_loss_dense_mlp_coef", 0) > 0,
            },
            {
                "name": "hidden_z_loss_moe",
                "used": self.model.config.language_model.get("hidden_z_loss_moe_coef", 0) > 0,
            },
            {
                "name": "hidden_z_loss_attn",
                "used": self.model.config.language_model.get("hidden_z_loss_attn_coef", 0) > 0,
            },
            {
                "name": "hidden_z_loss",
                "used": self.model.config.language_model.get("hidden_z_loss_eps", 0) > 0,
            },
            {
                "name": "loss_mtp_per_head",
                "used": self.model.config.language_model.get("mtp_predictor_num") > 1,
            },
            {
                "name": "mtp_loss",
                "used": self.model.config.language_model.get("use_mtp", False),
            },
            {
                "name": "z_loss",
                "used": self.model.config.language_model.get("z_loss_eps", 0) > 0,
            },
            {"name": "loss_ce", "used": True},

        ]

        losses_dict: Dict[str, Any] = {"total": outputs[0]}

        # Per-projection hidden z-loss scalars (qkv/up_mlp/up_moe/gate_*) stashed
        # on modules during forward. They are NOT part of the output tuple above —
        # the regularizer gradient is spliced in-graph at each hook, so these are
        # logging-only and must be read before the tuple is stripped.
        for proj, val in collect_included_proj_z_losses(self.model).items():
            losses_dict[f"hidden_z_loss_{proj}"] = val

        # The draft-loss pairs are appended after everything in `possible_losses`
        # (GigaMixForCausalLM.forward), so they come off first, exactly as
        # ComposerGigaMixCausalLM.loss strips them.
        language_model = self.model.language_model
        draft_loss_kinds = getattr(language_model, "draft_loss_kinds", [])
        diagnostic_kinds = getattr(language_model, "DRAFT_DIAGNOSTIC_KINDS", ())
        draft_losses = {}
        for kind in reversed(draft_loss_kinds):
            draft_losses[kind] = (outputs[-2], outputs[-1])
            outputs = outputs[:-2]
        for kind, (draft_loss, draft_loss_per_head) in draft_losses.items():
            if draft_loss is None:
                continue
            if kind in diagnostic_kinds:
                name = kind
            else:
                name = f"{kind}_loss"
                losses_dict[name] = draft_loss
            for idx, head_loss in enumerate(draft_loss_per_head):
                losses_dict[f"{name}{idx + 1}"] = head_loss

        for possible_loss in possible_losses:
            if possible_loss["used"] and possible_loss["name"] == "loss_mtp_per_head":
                loss_mtp_per_head = outputs[-1]
                outputs = outputs[:-1]
                for idx, head_loss in enumerate(loss_mtp_per_head):
                    losses_dict[f"loss_mtp_{idx + 1}"] = head_loss
            elif possible_loss["used"] and outputs[-1] is not None:
                losses_dict[possible_loss["name"]] = outputs[-1]
                outputs = outputs[:-1]

        return losses_dict

    def eval_forward(
        self,
        batch: Mapping,
        outputs: Optional[ModelOutput] = None,
        inner_microbatch_size: Optional[int] = None,
    ):
        # TODO: implement support for inner_microbatch_size, it is just ignored for now

        if batch.get("mode", None) == "icl_task":
            # pop labels first to avoid computing loss
            self.labels = batch.pop("labels")  # type: ignore
            self.labels[:, :-1] = self.labels[:, 1:].clone()
            self.labels[:, -1] = -100

            outputs = outputs if outputs else self.forward(batch)
            output = outputs[1] if len(outputs[0].shape) == 0 else outputs[0]
        else:
            raise ValueError(
                f"Mode `mode`={batch.get('mode', None)} is not supported for ComposerGigaOmniForCausalLM!"
            )

        return output

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            return {metric.__class__.__name__: metric for metric in self.train_metrics}
        return {metric.__class__.__name__: metric for metric in self.val_metrics}

    def update_metric(self, batch: Mapping, outputs: ModelOutput, metric: Metric):
        if isinstance(metric, InContextLearningMultipleChoiceAccuracy):
            assert self.labels is not None
            metric.update(batch=batch, outputs=outputs, labels=self.labels)
        else:
            raise ValueError(
                f"Metric {metric.__class__.__name__} is not supported for ComposerGigaOmniForCausalLM!"
            )

    def get_metadata(self) -> Dict[str, Any]:
        config = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.config.save_pretrained(Path(tmp_dir))
            with open(Path(tmp_dir) / "config.json") as f:
                config = json.load(f)

        return {"model": {"config": {"content": config}}}

    def gigafsdp_model_setup(self, fsdp_config: Dict[str, Any]):
        _build_gigafsdp_wrappers_config(
            self.model, self.config.language_model["name"], fsdp_config
        )
