"""Conformer in-encoder length bucketing.

Runs a 2-layer, d16 conformer on CPU with ``attention_backend="sdpa"`` and
dropout 0. The model is bf16 because ``ConformerLayer.forward`` casts its
output to bf16 unconditionally. bf16 matmuls on CPU pick different blockings
for different batch shapes, so the unbucketed path itself moves by ~1 ulp when
the microbatch max length changes; the bucketed-vs-unbucketed checks therefore
compare valid positions at that tolerance. The stem pad mask that makes the
per-group split exact is checked in fp64 in test_conv_subsampling_pad_mask.py.
"""

import logging

import pytest
import torch
import torch.nn.functional as F

from sber_tests.test_utils.conformer_cpu import import_conformer_modules

modeling, configuration = import_conformer_modules()

pytestmark = [pytest.mark.unit, pytest.mark.fast]

CFG = dict(
    feat_in=8,
    n_layers=2,
    d_model=16,
    subsampling="striding_conv1d",
    subsampling_factor=4,
    subsampling_conv_channels=16,
    ff_expansion_factor=2,
    n_heads=2,
    pos_emb_max_len=64,
    conv_kernel_size=5,
    dropout=0.0,
    dropout_emb=0.0,
    dropout_att=0.0,
    attn_chunk_size=8,
    max_position_seq_len=256,
    attention_backend="sdpa",
)
BF16_TOL = dict(atol=2**-6, rtol=2**-6)
THREE_GROUP_LENGTHS = [[61, 7, 5], [30, 29, 3]]
WORST_MIN1S_LENGTHS = [
    22755, 1659, 1259, 1211, 883, 755, 747, 747, 675, 451, 411, 347, 283, 275, 219, 211,
]


def build(dtype=torch.bfloat16, train=False, **overrides):
    torch.manual_seed(0)
    config = configuration.ConformerConfig(**{**CFG, **overrides})
    model = modeling.ConformerForAudioLanguageModeling(config)
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    model.train(train)
    return model.to(dtype)


def bundles(lengths_per_bundle, dtype=torch.bfloat16):
    torch.manual_seed(1)
    return [
        {
            "spectrograms": torch.randn(sum(lengths), CFG["feat_in"]).to(dtype),
            "spectrogram_lengths": torch.tensor(lengths),
        }
        for lengths in lengths_per_bundle
    ]


def valid_rows(outputs):
    rows = []
    for spec, lengths in zip(outputs[::2], outputs[1::2]):
        rows.extend(spec[row, :n].float() for row, n in enumerate(lengths.tolist()))
    return rows


def assert_same_valid_rows(got, expected, **tol):
    got_rows, expected_rows = valid_rows(got), valid_rows(expected)
    assert [r.shape for r in got_rows] == [r.shape for r in expected_rows]
    for g, e in zip(got_rows, expected_rows):
        torch.testing.assert_close(g, e, **tol)


def run_on_off(lengths_per_bundle, train=False):
    inputs = bundles(lengths_per_bundle)
    with torch.no_grad():
        on = build(train=train, length_bucketing=True)(inputs)
        off = build(train=train, length_bucketing=False)(inputs)
    return inputs, on, off


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([50, 50, 50], [[0, 1, 2]]),
        ([100, 50], [[0, 1]]),
        ([100, 10], [[0], [1]]),
        ([10, 100, 40, 7, 95], [[1, 4, 2], [0, 3]]),
        ([0, 5], [[1], [0]]),
        (WORST_MIN1S_LENGTHS, [[0], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12], [13, 14, 15]]),
    ],
)
def test_length_bucketed_groups(lengths, expected):
    groups = modeling._length_bucketed_groups(torch.tensor(lengths))
    assert groups == expected
    assert sorted(i for group in groups for i in group) == list(range(len(lengths)))


def test_length_bucketed_groups_respects_pad_waste_cap():
    lengths = torch.tensor([100, 60])
    assert modeling._length_bucketed_groups(lengths, pad_waste_cap=0.5) == [[0, 1]]
    assert modeling._length_bucketed_groups(lengths, pad_waste_cap=0.1) == [[0], [1]]


@pytest.mark.parametrize("train", [False, True])
def test_bucketed_matches_unbucketed_three_groups(train):
    inputs, on, off = run_on_off(THREE_GROUP_LENGTHS, train=train)
    all_lengths = torch.cat([b["spectrogram_lengths"] for b in inputs])
    assert len(modeling._length_bucketed_groups(all_lengths)) == 3
    assert [t.shape for t in on] == [t.shape for t in off]
    for on_lens, off_lens in zip(on[1::2], off[1::2]):
        assert torch.equal(on_lens, off_lens)
    assert_same_valid_rows(on, off, **BF16_TOL)


@pytest.mark.parametrize("lengths", [[40, 1, 9], [40, 0, 9]])
def test_bucketed_matches_unbucketed_with_one_or_zero_frame_clip(lengths):
    inputs, on, off = run_on_off([lengths])
    all_lengths = inputs[0]["spectrogram_lengths"]
    assert len(modeling._length_bucketed_groups(all_lengths)) == 3
    assert on[1].tolist() == off[1].tolist()
    assert_same_valid_rows(on, off, **BF16_TOL)


def test_backward_through_groups_matches_single_stack():
    grads = {}
    for flag in (True, False):
        model = build(train=True, length_bucketing=flag)
        inputs = bundles(THREE_GROUP_LENGTHS)
        for bundle in inputs:
            bundle["spectrograms"].requires_grad_()
        out = model(inputs)
        sum(row.sum() for row in valid_rows(out)).backward()
        grads[flag] = [b["spectrograms"].grad.float() for b in inputs] + [
            model.pre_encode.conv[0].weight.grad.float(),
            model.layers[0].feed_forward1.linear1.weight.grad.float(),
            model.layers[-1].conv.depthwise_conv.weight.grad.float(),
        ]
    for on_grad, off_grad in zip(grads[True], grads[False]):
        assert torch.isfinite(on_grad).all()
        torch.testing.assert_close(on_grad, off_grad, atol=2**-5, rtol=2**-5)


@pytest.mark.parametrize("lengths_per_bundle", [[[37]], [[24, 24], [24]], [[24, 24, 24]]])
def test_fast_path_is_untouched_for_single_clip_and_homogeneous_batch(
    monkeypatch, lengths_per_bundle
):
    calls = []
    monkeypatch.setattr(
        modeling.ConformerEncoder,
        "_forward_groups",
        lambda self, *args: calls.append(args),
    )
    inputs, on, off = run_on_off(lengths_per_bundle)
    assert calls == []
    assert all(torch.equal(a, b) for a, b in zip(on, off))


def test_length_bucketing_false_never_enters_the_group_path(monkeypatch):
    calls = []
    original = modeling.ConformerEncoder._forward_groups

    def spy(self, *args):
        calls.append(self.length_bucketing)
        return original(self, *args)

    monkeypatch.setattr(modeling.ConformerEncoder, "_forward_groups", spy)
    with torch.no_grad():
        build(length_bucketing=False)(bundles(THREE_GROUP_LENGTHS))
        assert calls == []
        build(length_bucketing=True)(bundles(THREE_GROUP_LENGTHS))
    assert calls == [True]


def test_encoder_forward_on_padded_batch_matches_single_stack():
    # ConformerEncoder alone, the inference entry point: a padded [bs, T_max, n_mel]
    # batch takes the bucketed path and matches the single stack on valid rows.
    lengths = [61, 7, 5, 30, 29, 3]
    torch.manual_seed(3)
    clips = torch.split(torch.randn(sum(lengths), CFG["feat_in"], dtype=torch.bfloat16), lengths)
    padded = torch.stack([F.pad(clip, (0, 0, 0, 61 - clip.size(0))) for clip in clips])
    model = build()
    assert type(model).forward is not modeling.ConformerEncoder.forward
    assert "__init__" not in vars(modeling.ConformerForAudioLanguageModeling)
    with torch.no_grad():
        on, on_lens = modeling.ConformerEncoder.forward(model, padded, torch.tensor(lengths))
        off, off_lens = model._forward_stack(padded, torch.tensor(lengths))
    assert torch.equal(on_lens, off_lens)
    for row, n in enumerate(on_lens.tolist()):
        torch.testing.assert_close(on[row, :n], off[row, :n], **BF16_TOL)


def test_unwrap_tuple_output_with_bundles_spanning_groups():
    inputs, on, off = run_on_off(THREE_GROUP_LENGTHS)
    model = build()
    unwrapped_on = model.unwrap_tuple_output(inputs, on)
    unwrapped_off = model.unwrap_tuple_output(inputs, off)
    assert len(unwrapped_on) == len(inputs) == 2
    for got, expected, bundle in zip(unwrapped_on, unwrapped_off, inputs):
        assert got["spectrograms"].shape == expected["spectrograms"].shape
        assert got["spectrograms"].size(0) == len(bundle["spectrogram_lengths"])
        assert torch.equal(got["spectrogram_lengths"], expected["spectrogram_lengths"])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"use_per_layer_wrapping": True}, "single gigafsdp wrap unit"),
        ({"subsampling": "striding"}, "striding_conv1d"),
        ({"subsampling": "stacking"}, "striding_conv1d"),
        ({"conv_norm_type": "batch_norm"}, "layer_norm"),
    ],
)
def test_unsupported_configs_are_rejected_at_construction(overrides, match):
    with pytest.raises(ValueError, match=match):
        build(**overrides)
    model = build(length_bucketing=False, **overrides)
    assert model.length_bucketing is False


def test_eval_mode_buckets_like_train_mode(monkeypatch):
    calls = []
    original = modeling.ConformerEncoder._forward_groups

    def spy(self, *args):
        calls.append(len(args[2]))
        return original(self, *args)

    monkeypatch.setattr(modeling.ConformerEncoder, "_forward_groups", spy)
    inputs = bundles(THREE_GROUP_LENGTHS)
    with torch.no_grad():
        eval_out = build(train=False)(inputs)
        train_out = build(train=True)(inputs)
    assert calls == [3, 3]
    assert all(torch.equal(a, b) for a, b in zip(eval_out, train_out))


def test_config_round_trip_and_default():
    assert configuration.ConformerConfig().length_bucketing is True
    off = configuration.ConformerConfig(**CFG, length_bucketing=False)
    assert off.to_dict()["length_bucketing"] is False
    assert build(length_bucketing=False).length_bucketing is False
    legacy = {k: v for k, v in off.to_dict().items() if k != "length_bucketing"}
    assert configuration.ConformerConfig.from_dict(legacy).length_bucketing is True


def engagement_messages(caplog):
    return [r.message for r in caplog.records if "conformer length bucketing" in r.message]


def test_engagement_log_is_throttled(caplog):
    model = build()
    inputs = bundles(THREE_GROUP_LENGTHS)
    with caplog.at_level(logging.INFO), torch.no_grad():
        for _ in range(7):
            model(inputs)
    messages = engagement_messages(caplog)
    assert len(messages) == 5
    assert "6 clips -> 3 groups" in messages[0]
    assert model._bucketing_log_budget == 0


def test_engagement_log_is_rank0_only(monkeypatch, caplog):
    monkeypatch.setattr(modeling, "_is_rank_zero", lambda: False)
    model = build()
    with caplog.at_level(logging.INFO), torch.no_grad():
        model(bundles(THREE_GROUP_LENGTHS))
    assert engagement_messages(caplog) == []
    assert model._bucketing_log_budget == 5
