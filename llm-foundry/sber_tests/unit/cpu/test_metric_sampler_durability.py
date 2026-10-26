import types
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

import torch  # noqa: E402
from composer.core import State  # noqa: E402
from composer.devices import DeviceCPU  # noqa: E402

from llmfoundry.callbacks.metric_sampler import ActivationDumpConfigError, LOSS_METRICS, MetricSampler  # noqa: E402
from llmfoundry.callbacks.metric_sampler_logger import MetricSamplerLogger  # noqa: E402
from metric_sampler.capture.layout import Layout  # noqa: E402
from metric_sampler.capture.recorder import CaptureCapacityError, RankRecorder  # noqa: E402
from metric_sampler.capture.ring import ShmRing  # noqa: E402
from metric_sampler.logged import LoggedWriter  # noqa: E402


def test_mla_latent_losses_have_distinct_microbatch_slots_and_record_the_model_values(tmp_path):
    callback = MetricSampler(str(tmp_path))
    callback.microbatches = 2
    requests = callback._layout_requests(
        experts=2, routed_layers={}, mtp_layers=[], streams=1, pre_reduce_keys=set(),
        activation_layers={point: [] for point in (
            "residual", "attn_out", "mlp_out", "shared_expert_out", "attn_residual_contribution",
            "mlp_residual_contribution", "swiglu_in", "down_in",
        )}, dumps_enabled=False,
    )
    layout = Layout(requests)
    losses = ("hidden_z_loss_mla_kv_latent", "hidden_z_loss_mtp0_mla_kv_latent")
    for key in losses:
        assert LOSS_METRICS[key] == f"loss/{key}"
        assert [(slot.layer, slot.microbatch, slot.width) for slot in layout.slots
                if slot.metric.name == LOSS_METRICS[key]] == [(-1, 0, 1), (-1, 1, 1)]
    ring = ShmRing.create(f"ms_test_{uuid.uuid4().hex}", layout, slot_count=1)
    callback.recorder = recorder = RankRecorder(layout, ring, torch.device("cpu"))
    state = State(model=torch.nn.Linear(1, 1), rank_zero_seed=1, run_name="unit", device=DeviceCPU())
    try:
        callback.recorder.begin_step(False)
        for microbatch, values in enumerate(((1.25, 2.5), (3.75, 5.0))):
            callback.microbatch = microbatch
            state.loss = {key: torch.tensor(value) for key, value in zip(losses, values)}
            callback.after_loss(state, None)
        assert callback.unknown_loss_keys == set()
        assert callback.disabled is False
        for microbatch, values in enumerate(((1.25, 2.5), (3.75, 5.0))):
            for key, expected in zip(losses, values):
                assert callback.recorder.accumulator(LOSS_METRICS[key], -1, microbatch).tolist() == [expected]
    finally:
        callback._shutdown()
        recorder.close()
        ring.unlink()


@pytest.mark.parametrize("error_type", [CaptureCapacityError, ActivationDumpConfigError])
def test_capture_failure_reaches_trainer_without_disabling_capture(tmp_path, error_type):
    callback = MetricSampler(str(tmp_path))
    error = error_type("capture cannot continue")

    def fail():
        raise error

    with pytest.raises(error_type) as raised:
        callback._guard("batch_end", fail)

    assert raised.value is error
    assert callback.disabled is False
    assert callback.output_dir == Path(tmp_path)


def test_recoverable_capture_error_still_disables_sampler(tmp_path):
    callback = MetricSampler(str(tmp_path))

    def fail():
        raise ValueError("unsupported observation")

    callback._guard("activation_observer", fail)

    assert callback.disabled is True


def test_exited_node_writer_stops_training_before_the_next_step(tmp_path):
    callback = MetricSampler(str(tmp_path))
    callback.rank_capture = types.SimpleNamespace(node_running=lambda: False)

    with pytest.raises(CaptureCapacityError, match="node writer exited"):
        callback.batch_start(None, None)

    assert callback.disabled is False
    assert callback.step_started == 0.0


@pytest.mark.parametrize("operation", ["add", "close"])
def test_logger_reports_unwritten_data_to_trainer(tmp_path, operation):
    logger = MetricSamplerLogger(str(tmp_path))
    writer = LoggedWriter(tmp_path, window_steps=10)
    logger.writer = writer
    if operation == "add":
        writer.close()
    else:
        (tmp_path / "logged").write_text("blocks the output directory")
        writer.add(12, "loss", 2.5)

    with pytest.raises(BufferError):
        if operation == "add":
            logger.log_metrics({"loss": 2.5}, step=12)
        else:
            logger.close(None, None)

    assert logger.writer is writer
    if operation == "close":
        (tmp_path / "logged").unlink()
        logger.close(None, None)
        assert logger.writer is None
        assert (tmp_path / "logged" / "w00000012-00000012" / "manifest.json").is_file()


@pytest.mark.parametrize("callback_type", [MetricSampler, MetricSamplerLogger])
def test_delivery_window_defaults_to_ten_and_accepts_override(tmp_path, callback_type):
    callback = callback_type(str(tmp_path))
    assert callback.window_steps == 10
    assert callback_type(str(tmp_path), window_steps=3).window_steps == 3


def test_shutdown_closes_the_rank_capture_once(tmp_path):
    callback = MetricSampler(str(tmp_path))
    closed = []
    callback.rank_capture = types.SimpleNamespace(close=lambda: closed.append(True))

    callback._shutdown()
    callback._shutdown()

    assert closed == [True]
    assert callback.rank_capture is None
    assert callback.recorder is None


def test_a_failed_rank_capture_close_reaches_the_trainer_and_is_not_retried(tmp_path):
    callback = MetricSampler(str(tmp_path))
    calls = []

    def close():
        calls.append(True)
        raise BufferError("metric sampler node writer exited with code 7")

    callback.rank_capture = types.SimpleNamespace(close=close)

    with pytest.raises(BufferError, match="exited with code 7"):
        callback._shutdown()
    callback._shutdown()

    assert calls == [True]
    assert callback.rank_capture is None
