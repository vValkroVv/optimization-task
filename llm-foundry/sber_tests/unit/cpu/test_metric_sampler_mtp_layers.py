from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("composer", reason="Requires the CPU training runtime")

from llmfoundry.callbacks.metric_sampler import MetricSampler  # noqa: E402


def _sampler(use_mtp, mtp_predictor_num, mtp_layers):
    sampler = MetricSampler.__new__(MetricSampler)
    sampler.causal_lm = SimpleNamespace(
        config=SimpleNamespace(use_mtp=use_mtp, mtp_predictor_num=mtp_predictor_num),
        model=SimpleNamespace(mtp_block=SimpleNamespace(mtp_layers=mtp_layers)),
    )
    return sampler


def test_disabled_mtp_allows_empty_module_tree():
    sampler = _sampler(use_mtp=False, mtp_predictor_num=1, mtp_layers=[])

    assert sampler._mtp_decoder_layers(num_layers=8) == []


def test_enabled_mtp_requires_configured_number_of_modules():
    sampler = _sampler(
        use_mtp=True,
        mtp_predictor_num=2,
        mtp_layers=[SimpleNamespace(decoder_layer=object())],
    )

    with pytest.raises(AssertionError, match="expected 2 MTP layers.*found 1"):
        sampler._mtp_decoder_layers(num_layers=8)


def test_enabled_mtp_numbers_modules_after_main_decoder_layers():
    first = object()
    second = object()
    sampler = _sampler(
        use_mtp=True,
        mtp_predictor_num=2,
        mtp_layers=[
            SimpleNamespace(decoder_layer=first),
            SimpleNamespace(decoder_layer=second),
        ],
    )

    assert sampler._mtp_decoder_layers(num_layers=8) == [(8, first), (9, second)]
