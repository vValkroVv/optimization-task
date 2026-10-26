import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
from types import ModuleType

import pytest
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).parents[3]


class _Dist:
    @staticmethod
    def get_world_size():
        return 1

    @staticmethod
    def all_reduce(tensor, reduce_operation):
        return None


def _load_callback_module():
    monitor_spec = importlib.util.spec_from_file_location(
        "isolated_dsa_monitor",
        ROOT / "llmfoundry" / "models" / "layers" / "dsa_monitor.py",
    )
    assert monitor_spec is not None and monitor_spec.loader is not None
    monitor_module = importlib.util.module_from_spec(monitor_spec)
    sys.modules[monitor_spec.name] = monitor_module
    monitor_spec.loader.exec_module(monitor_module)

    composer = ModuleType("composer")
    composer_core = ModuleType("composer.core")
    composer_core.Callback = object
    composer_core.State = object
    composer_loggers = ModuleType("composer.loggers")
    composer_loggers.Logger = object
    composer_utils = ModuleType("composer.utils")
    composer_utils.dist = _Dist
    layers = ModuleType("llmfoundry.models.layers")
    layers.dsa_monitor = monitor_module
    models = ModuleType("llmfoundry.models")
    models.layers = layers
    llmfoundry = ModuleType("llmfoundry")
    llmfoundry.models = models
    stubs = {
        "composer": composer,
        "composer.core": composer_core,
        "composer.loggers": composer_loggers,
        "composer.utils": composer_utils,
        "llmfoundry": llmfoundry,
        "llmfoundry.models": models,
        "llmfoundry.models.layers": layers,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        callback_spec = importlib.util.spec_from_file_location(
            "isolated_dsa_callback",
            ROOT / "llmfoundry" / "callbacks" / "dsa_kl_monitor.py",
        )
        assert callback_spec is not None and callback_spec.loader is not None
        callback_module = importlib.util.module_from_spec(callback_spec)
        sys.modules[callback_spec.name] = callback_module
        callback_spec.loader.exec_module(callback_module)
        return callback_module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


dsa_kl_monitor = _load_callback_module()


class _Logger:
    def __init__(self):
        self.metrics = None

    def log_metrics(self, metrics):
        self.metrics = metrics


def _state_with_dsa_layer(indexer_topk=2048, dsa_stage=2):
    layer = SimpleNamespace(
        dsa_stage=dsa_stage,
        indexer=object(),
        indexer_topk=indexer_topk,
        loss=torch.tensor(0.25),
    )
    model = SimpleNamespace(
        named_modules=lambda: [("model.layers.3.self_attn", layer)],
        parameters=lambda: iter([torch.zeros(1)]),
    )
    state = SimpleNamespace(
        model=model,
        timestamp=SimpleNamespace(batch=SimpleNamespace(value=0)),
    )
    return state, layer


def test_all_monitors_are_disabled_by_default(monkeypatch):
    callback = dsa_kl_monitor.DSAMonitor()
    layer = SimpleNamespace(
        _compute_kl_value=False,
        _compute_attention_mass_value=False,
        _compute_indexer_topk_value=False,
    )
    callback.layers = [layer]
    callback.tags = ["layer_3"]
    state = SimpleNamespace(
        timestamp=SimpleNamespace(batch=SimpleNamespace(value=100)),
    )
    all_reduce_calls = []
    monkeypatch.setattr(
        dsa_kl_monitor.dist,
        "all_reduce",
        lambda *args, **kwargs: all_reduce_calls.append((args, kwargs)),
    )

    callback.batch_start(state, _Logger())
    callback.batch_end(state, _Logger())

    assert not callback._any_enabled
    assert not layer._compute_kl_value
    assert not layer._compute_attention_mass_value
    assert not layer._compute_indexer_topk_value
    assert all_reduce_calls == []


def test_disabled_callback_does_not_inspect_model():
    callback = dsa_kl_monitor.DSAMonitor()
    model = SimpleNamespace(
        named_modules=lambda: pytest.fail("disabled monitor inspected the model"),
    )
    state = SimpleNamespace(model=model)

    callback.fit_start(state, _Logger())

    assert callback.layers == []
    assert callback.tags == []
    assert callback._device is None


def test_attention_mass_reduces_raw_sp_sums(monkeypatch):
    callback = dsa_kl_monitor.DSAMonitor(
        attention_mass={"enabled": True, "topk_values": [2, 4]},
    )
    local_stats = torch.tensor(
        [
            [0.8, 1.0, 0.0, 0.0],
            [1.6, 1.0, 0.0, 0.0],
        ]
    )
    peer_stats = torch.tensor(
        [
            [
                [1.0, 2.0, 4.0, 10.0],
                [2.0, 2.0, 4.0, 10.0],
            ]
        ]
    )
    layer = SimpleNamespace(attention_mass_monitor_stats=local_stats)
    callback.layers = [layer]
    callback.tags = ["layer_3"]
    callback._device = torch.device("cpu")
    callback._attention_mass_now = True

    monkeypatch.setattr(dsa_kl_monitor.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dsa_kl_monitor.dist,
        "all_reduce",
        lambda tensor, reduce_operation: tensor.add_(peer_stats),
    )
    logger = _Logger()

    callback.batch_end(None, logger)

    assert logger.metrics is not None
    assert logger.metrics[
        "dsa_attention_mass/topk_2/attention_mass/layer_3"
    ] == pytest.approx(0.6)
    assert logger.metrics[
        "dsa_attention_mass/topk_4/attention_mass/layer_3"
    ] == pytest.approx(1.2)
    assert logger.metrics["dsa_attention_mass/monitored_q_count"] == 3
    assert logger.metrics["dsa_attention_mass/eligible_q_count"] == 4
    assert logger.metrics["dsa_attention_mass/total_q_count"] == 10
    assert logger.metrics["dsa_attention_mass/monitored_q_fraction"] == 0.3
    assert logger.metrics["dsa_attention_mass/eligible_q_fraction"] == 0.4
    assert not any(key.startswith("dsa_indexer_topk/") for key in logger.metrics)
    assert layer.attention_mass_monitor_stats is None


def test_indexer_topk_reduces_raw_sp_sums(monkeypatch):
    callback = dsa_kl_monitor.DSAMonitor(
        indexer_topk={"enabled": True, "topk_values": [2, 4]},
    )
    local_stats = torch.tensor(
        [
            [2.0, 3.0, 1.0, 1.0, 0.0, 0.0],
            [4.0, 5.0, 2.0, 1.0, 0.0, 0.0],
        ]
    )
    peer_stats = torch.tensor(
        [
            [
                [6.0, 9.0, 3.0, 2.0, 4.0, 10.0],
                [8.0, 10.0, 4.0, 2.0, 4.0, 10.0],
            ]
        ]
    )
    layer = SimpleNamespace(indexer_topk_monitor_stats=local_stats)
    callback.layers = [layer]
    callback.tags = ["layer_3"]
    callback._device = torch.device("cpu")
    callback._indexer_topk_now = True

    monkeypatch.setattr(dsa_kl_monitor.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        dsa_kl_monitor.dist,
        "all_reduce",
        lambda tensor, reduce_operation: tensor.add_(peer_stats),
    )
    logger = _Logger()

    callback.batch_end(None, logger)

    assert logger.metrics is not None
    assert logger.metrics[
        "dsa_indexer_topk/topk_2/position_range/layer_3"
    ] == pytest.approx(8.0 / 3.0)
    assert logger.metrics[
        "dsa_indexer_topk/topk_4/position_range/layer_3"
    ] == pytest.approx(4.0)
    assert logger.metrics["dsa_indexer_topk/monitored_q_count"] == 3
    assert logger.metrics["dsa_indexer_topk/eligible_q_count"] == 4
    assert logger.metrics["dsa_indexer_topk/total_q_count"] == 10
    assert not any(key.startswith("dsa_attention_mass/") for key in logger.metrics)
    assert layer.indexer_topk_monitor_stats is None


def test_monitor_configuration_validation():
    with pytest.raises(ValueError, match="attention_mass.num_queries"):
        dsa_kl_monitor.DSAMonitor(
            attention_mass={"num_queries": 0},
        )
    with pytest.raises(ValueError, match="Unknown indexer_topk options"):
        dsa_kl_monitor.DSAMonitor(
            indexer_topk={"tail_buckets": 16},
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        dsa_kl_monitor.DSAMonitor(
            indexer_topk={"topk_values": [128, 128]},
        )
    callback = dsa_kl_monitor.DSAMonitor(
        attention_mass={"enabled": True, "num_queries": None},
        indexer_topk={
            "enabled": False,
            "q_start_pos": 4096,
            "topk_values": OmegaConf.create([512, 128]),
        },
    )
    assert callback.attention_mass.num_queries is None
    assert callback.attention_mass.enabled
    assert not callback.indexer_topk.enabled
    assert callback.indexer_topk.q_start_pos == 4096
    assert callback.indexer_topk.topk_values == (128, 512)


def test_nested_config_only_registers_sparse_layers_in_hybrid_model():
    config = OmegaConf.create(
        {
            "kl_loss": {"enabled": False, "log_interval": 20},
            "attention_mass": {
                "enabled": True,
                "log_interval": 100,
                "num_queries": 128,
                "q_start_pos": 65536,
                "topk_values": [128, 512, 1024, 2048],
            },
            "indexer_topk": {
                "enabled": True,
                "log_interval": 100,
                "num_queries": 128,
                "q_start_pos": 65536,
                "topk_values": [128, 512, 1024, 2048],
            },
        }
    )
    sparse_layer = SimpleNamespace(
        dsa_stage=1,
        indexer=object(),
        indexer_topk=2048,
        loss=torch.tensor(0.25),
    )
    dense_layer = SimpleNamespace()
    model = SimpleNamespace(
        named_modules=lambda: [
            ("model.layers.3.self_attn", dense_layer),
            ("model.layers.7.self_attn", sparse_layer),
        ],
        parameters=lambda: iter([torch.zeros(1)]),
    )
    state = SimpleNamespace(model=model)
    callback = dsa_kl_monitor.DSAMonitor(**config)

    callback.fit_start(state, _Logger())

    assert callback.layers == [sparse_layer]
    assert callback.tags == ["layer_7"]
    assert not hasattr(dense_layer, "_compute_attention_mass_value")
    assert not hasattr(dense_layer, "_compute_indexer_topk_value")
    assert not sparse_layer._compute_attention_mass_value
    assert not sparse_layer._compute_indexer_topk_value


def test_indexer_topk_values_are_validated_against_stage_2_model_topk():
    stage_1_state, stage_1_layer = _state_with_dsa_layer(
        indexer_topk=2048,
        dsa_stage=1,
    )
    stage_1 = dsa_kl_monitor.DSAMonitor(
        indexer_topk={"enabled": True, "topk_values": [4096]},
    )

    stage_1.fit_start(stage_1_state, _Logger())

    assert stage_1.indexer_topk_values == (4096,)
    assert stage_1_layer.indexer_topk_monitor_topk_values == (4096,)

    stage_2_state, stage_2_layer = _state_with_dsa_layer(
        indexer_topk=2048,
        dsa_stage=2,
    )
    callback = dsa_kl_monitor.DSAMonitor(
        indexer_topk={"enabled": True, "topk_values": [128, 512]},
    )

    callback.fit_start(stage_2_state, _Logger())

    assert callback.indexer_topk_values == (128, 512)
    assert stage_2_layer.indexer_topk_monitor_topk_values == (128, 512)

    invalid = dsa_kl_monitor.DSAMonitor(
        indexer_topk={"enabled": True, "topk_values": [4096]},
    )
    with pytest.raises(ValueError, match="must not exceed model.indexer_topk"):
        invalid.fit_start(stage_2_state, _Logger())


def test_attention_mass_is_stage_1_only():
    stage_1_state, stage_1_layer = _state_with_dsa_layer(
        indexer_topk=2048,
        dsa_stage=1,
    )
    stage_1 = dsa_kl_monitor.DSAMonitor(
        attention_mass={"enabled": True, "topk_values": [128, 4096]},
    )

    stage_1.fit_start(stage_1_state, _Logger())

    assert stage_1.attention_mass_topk_values == (128, 4096)
    assert stage_1_layer.attention_mass_monitor_topk_values == (128, 4096)

    stage_2_state, stage_2_layer = _state_with_dsa_layer(
        indexer_topk=2048,
        dsa_stage=2,
    )
    stage_2 = dsa_kl_monitor.DSAMonitor(
        attention_mass={"enabled": True, "topk_values": [128]},
    )

    with pytest.raises(ValueError, match="only for DSA stage 1"):
        stage_2.fit_start(stage_2_state, _Logger())
    assert not hasattr(stage_2_layer, "attention_mass_monitor_topk_values")


def test_kl_monitor_only_touches_and_logs_loss():
    state, layer = _state_with_dsa_layer()
    callback = dsa_kl_monitor.DSAMonitor(
        kl_loss={"enabled": True, "log_interval": 1},
    )
    logger = _Logger()

    callback.fit_start(state, logger)
    callback.batch_start(state, logger)
    callback.batch_end(state, logger)

    assert layer._compute_kl_value
    assert not hasattr(layer, "_compute_attention_mass_value")
    assert not hasattr(layer, "_compute_indexer_topk_value")
    assert set(logger.metrics) == {"dsa_kl/layer_3", "dsa_kl/mean"}


def test_legacy_kl_callback_configuration():
    callback = dsa_kl_monitor.DSAKLMonitor(batch_log_interval=7)

    assert callback.kl_loss.enabled
    assert callback.kl_loss.log_interval == 7
    assert not callback.attention_mass.enabled
    assert not callback.indexer_topk.enabled
