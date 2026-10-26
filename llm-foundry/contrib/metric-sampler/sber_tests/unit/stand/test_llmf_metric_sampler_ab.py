import importlib
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    # llm-foundry and Composer are external training-runtime dependencies.
    files = {
        "composer/__init__.py": "",
        "composer/core.py": "class State: pass\n",
        "composer/loggers.py": "class Logger: pass\n",
        "composer/utils/__init__.py": "from . import dist\n",
        "composer/utils/dist.py": "def get_global_rank(): return 2\n",
        "composer/utils/shared_state.py": (
            "from types import SimpleNamespace\nglobal_actlog_state = SimpleNamespace(activation_observer=None)\n"
        ),
        "llmfoundry/__init__.py": "",
        "llmfoundry/models/__init__.py": "",
        "llmfoundry/models/layers/__init__.py": "",
        "llmfoundry/models/layers/moe/__init__.py": "",
        "llmfoundry/models/layers/moe/routers.py": "class MoEGate: pass\n",
        "llmfoundry/callbacks/__init__.py": "",
        "llmfoundry/callbacks/metric_sampler.py": """
from pathlib import Path
from types import SimpleNamespace
from composer.utils.shared_state import global_actlog_state
from llmfoundry.models.layers.moe.routers import MoEGate

class MetricSampler:
    def __init__(self, output_dir, window_steps):
        self.output_dir = Path(output_dir)
        self.disabled = False
        self.values = []
        self.events = []
    def fit_start(self, state, logger):
        self.disabled = state.fail_setup
        gate = MoEGate()
        gate.routing_observer = lambda value: self.values.append(("routing", value))
        self._routed_gates = [(0, gate)]
        self.causal_lm = SimpleNamespace(loss_observer=lambda value: self.values.append(("loss", value)))
        self._pre_reduce_super_tensors = [SimpleNamespace(pre_reduce_observer=lambda value: self.values.append(("gradient", value)))]
        self._dispatcher_observers = [(SimpleNamespace(assignment_observer=None), lambda value: self.values.append(("dispatch", value)))]
        self._activation_observer = lambda value: self.values.append(("activation", value))
    def batch_start(self, state, logger):
        self.events.append("batch_start")
    def before_forward(self, state, logger):
        self.events.append("before_forward")
        self.disabled = state.fail_forward
        if not self.disabled:
            global_actlog_state.activation_observer = self._activation_observer
            for dispatcher, observer in self._dispatcher_observers:
                dispatcher.assignment_observer = observer
    def after_forward(self, state, logger):
        self.events.append("after_forward")
        global_actlog_state.activation_observer = None
        for dispatcher, _ in self._dispatcher_observers:
            dispatcher.assignment_observer = None
    def after_loss(self, state, logger): self.events.append("after_loss")
    def before_backward(self, state, logger): self.events.append("before_backward")
    def after_backward(self, state, logger): self.events.append("after_backward")
    def after_train_batch(self, state, logger): self.events.append("after_train_batch")
    def batch_end(self, state, logger): self.events.append("batch_end")
    def fit_end(self, state, logger): self.events.append("fit_end")
""",  # noqa: E501
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    original_modules = sys.modules.copy()
    namespace = vars(importlib.import_module("stand.llmf_metric_sampler_ab"))
    state = SimpleNamespace(
        timestamp=SimpleNamespace(batch=SimpleNamespace(value=0)), fail_setup=False, fail_forward=False
    )
    yield namespace, state, tmp_path
    for name in set(sys.modules) - original_modules.keys():
        if name.startswith(("composer", "llmfoundry", "stand.llmf_metric_sampler")):
            del sys.modules[name]


@pytest.mark.parametrize("arm, expected", [
    ("off", []),
    ("base", ["dispatch"]),
    ("tokens", ["dispatch", "routing", "loss"]),
    ("full", ["dispatch", "routing", "loss", "gradient"]),
    ("acts", ["dispatch", "routing", "loss", "gradient", "activation"]),
    ("detailed", ["dispatch", "routing", "loss", "gradient", "activation"]),
])
def test_arm_controls_real_forward_capture_across_microbatches(wrapper, arm, expected):
    namespace, state, output = wrapper
    callback = namespace["MetricSamplerAB"](str(output), block_batches=1)
    callback.fit_start(state, None)
    state.timestamp.batch.value = namespace["ARMS"].index(arm)
    callback.batch_start(state, None)
    global_state = namespace["global_actlog_state"]
    gate = callback._routed_gates[0][1]
    dispatcher = callback._dispatcher_observers[0][0]
    gradient = callback._pre_reduce_super_tensors[0]
    for mb in range(2):
        callback.before_forward(state, None)
        for observer in (dispatcher.assignment_observer, gate.routing_observer, global_state.activation_observer):
            if observer is not None:
                observer(10 + mb)
        callback.after_forward(state, None)
        assert global_state.activation_observer is None
        assert dispatcher.assignment_observer is None
        if callback.causal_lm.loss_observer is not None:
            callback.causal_lm.loss_observer(10 + mb)
        callback.after_loss(state, None)
        callback.before_backward(state, None)
        if gradient.pre_reduce_observer is not None:
            gradient.pre_reduce_observer(10 + mb)
        callback.after_backward(state, None)
    callback.after_train_batch(state, None)
    callback.batch_end(state, None)
    assert sorted(callback.values) == sorted((family, 10 + mb) for family in expected for mb in range(2))
    expected_events = {} if arm == "off" else {
        "batch_start": 1, "before_forward": 2, "after_forward": 2, "after_loss": 2,
        "before_backward": 2, "after_backward": 2, "after_train_batch": 1, "batch_end": 1,
    }
    assert dict(Counter(callback.events)) == expected_events
    if arm != "off":
        assert callback.distribution_interval == (1 if arm == "detailed" else 100)
    callback.fit_end(state, None)
    line = (output / "ab_batch_starts/rank0002.tsv").read_text().strip().split("\t")
    assert line[:2] == [str(state.timestamp.batch.value), arm]
    assert float(line[2]) > 0


def test_leaving_activation_arm_detaches_every_observer_in_off_arm(wrapper):
    namespace, state, output = wrapper
    callback = namespace["MetricSamplerAB"](str(output), block_batches=1)
    callback.fit_start(state, None)
    state.timestamp.batch.value = 5
    callback.batch_start(state, None)
    callback.before_forward(state, None)
    state.timestamp.batch.value = 6
    callback.batch_start(state, None)
    assert callback.arm == "off"
    assert namespace["global_actlog_state"].activation_observer is None
    assert callback._dispatcher_observers[0][0].assignment_observer is None
    assert callback._routed_gates[0][1].routing_observer is None
    assert callback.causal_lm.loss_observer is None
    assert callback._pre_reduce_super_tensors[0].pre_reduce_observer is None


@pytest.mark.parametrize("block_batches", [0, -1])
def test_rejects_nonpositive_block_size(wrapper, block_batches):
    namespace, _, output = wrapper
    with pytest.raises(ValueError, match="block_batches must be positive"):
        namespace["MetricSamplerAB"](str(output), block_batches=block_batches)


def test_failed_setup_cannot_produce_a_successful_benchmark(wrapper):
    namespace, state, output = wrapper
    callback = namespace["MetricSamplerAB"](str(output))
    state.fail_setup = True
    with pytest.raises(RuntimeError, match="setup failed"):
        callback.fit_start(state, None)


def test_failed_forward_does_not_reattach_activation_observer(wrapper):
    namespace, state, output = wrapper
    callback = namespace["MetricSamplerAB"](str(output), block_batches=1)
    callback.fit_start(state, None)
    state.timestamp.batch.value = 4
    callback.batch_start(state, None)
    state.fail_forward = True
    callback.before_forward(state, None)
    assert namespace["global_actlog_state"].activation_observer is None
    with pytest.raises(RuntimeError, match="capture failed"):
        callback.batch_start(state, None)


def test_legacy_import_resolves_to_canonical_callback(wrapper):
    namespace, _, _ = wrapper
    module = importlib.import_module("stand.llmf_metric_sampler_callback")
    assert module.MetricSampler is namespace["MetricSampler"]
