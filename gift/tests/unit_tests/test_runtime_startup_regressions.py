"""CPU coverage of startup bindings and model preparation after schema splitting."""

from types import SimpleNamespace

import torch

from tests.unit_tests.config_helpers import load_experiment_config


def test_trainer_import_and_optimizer_abs_max_pipeline():
    from torchtitan.callbacks.sources.optimizer_source import _ABS_MAX_PIPELINE
    from torchtitan.trainer import Trainer

    assert Trainer.Config.owner() is Trainer
    assert [stage.op for stage in _ABS_MAX_PIPELINE] == ["abs_max"]


def test_flux_model_preparation_and_meta_construction():
    config = load_experiment_config("flux_debug.yaml")
    model_config = config.model_spec.model
    model_config.owner().prepare_config(model_config, config=config)
    with torch.device("meta"):
        model = model_config.build()
    assert all(parameter.is_meta for parameter in model.parameters())
    nparams, flops = model_config.owner().get_nparams_and_flops(
        model_config, model, 512
    )
    assert nparams > 0
    assert flops > 0


def test_flux_trainer_initializes_encoders_from_runtime_schema(monkeypatch):
    import torchtitan.models.flux.trainer as flux
    from torchtitan.config.schema.models.flux.model.hf_embedder import (
        FluxEmbedderConfig,
    )

    config = load_experiment_config("flux_debug.yaml")
    seen = []

    def base_init(self, incoming):
        self.device = torch.device("cpu")
        self.parallel_dims = SimpleNamespace(dp_shard_enabled=False)

    monkeypatch.setattr(flux.Trainer, "__init__", base_init)
    monkeypatch.setattr(flux.dist_utils, "set_determinism", lambda *a, **kw: None)
    monkeypatch.setattr(flux, "load_ae", lambda path, cfg, **kw: seen.append(cfg))
    monkeypatch.setattr(FluxEmbedderConfig, "build", lambda self: torch.nn.Identity())
    monkeypatch.setattr(
        flux, "parallelize_encoders", lambda **kw: (kw["t5_model"], kw["clip_model"])
    )
    config.validator.enable = False
    trainer = flux.FluxTrainer(config)
    assert seen == [config.model_spec.model.autoencoder]
    assert isinstance(trainer.t5_encoder, torch.nn.Identity)
    assert isinstance(trainer.clip_encoder, torch.nn.Identity)
    assert config.training.seq_len == 512
