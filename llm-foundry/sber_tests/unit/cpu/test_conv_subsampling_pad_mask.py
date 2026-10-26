"""ConvSubsampling (striding_conv1d) per-stage pad mask.

The stem is checked exactly in fp64: every row of a padded batch must equal
the same clip run alone, and pad positions must come out as zeros.
"""

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


def build_stem():
    torch.manual_seed(0)
    model = modeling.ConformerForAudioLanguageModeling(configuration.ConformerConfig(**CFG))
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    return model.eval().to(torch.float64).pre_encode


def test_stem_pad_mask_makes_rows_independent_of_batch_padding():
    stem = build_stem()
    lengths = [61, 7, 5, 30, 29, 3, 1]
    torch.manual_seed(2)
    clips = torch.split(torch.randn(sum(lengths), CFG["feat_in"], dtype=torch.float64), lengths)
    full = torch.stack([F.pad(clip, (0, 0, 0, 61 - clip.size(0))) for clip in clips])
    with torch.no_grad():
        full_out, full_lens = stem(full, torch.tensor(lengths))
        for row, clip in enumerate(clips):
            own_out, own_lens = stem(clip.unsqueeze(0), torch.tensor(lengths[row : row + 1]))
            n = int(own_lens[0])
            assert int(full_lens[row]) == n
            torch.testing.assert_close(full_out[row, :n], own_out[0, :n], atol=1e-12, rtol=0)
            assert torch.all(full_out[row, n:] == 0)


def test_stem_output_lengths_match_single_pass_formula():
    stem = build_stem()
    lengths = torch.tensor([61, 7, 5, 30, 29, 3, 1])
    with torch.no_grad():
        _, out_lens = stem(torch.zeros(len(lengths), 61, CFG["feat_in"], dtype=torch.float64), lengths)
    expected = stem.calc_length(
        lengths,
        all_paddings=stem._left_padding + stem._right_padding,
        kernel_size=stem._kernel_size,
        stride=stem._stride,
        ceil_mode=stem._ceil_mode,
        repeat_num=stem._sampling_num,
    )
    assert torch.equal(out_lens, expected)


def test_stem_pad_mask_is_differentiable():
    # The mask is applied out of place: an in-place masked_fill_ would bump the
    # version counter of the tensor the inplace ReLU saved and break backward.
    stem = build_stem().train()
    x = torch.randn(2, 20, CFG["feat_in"], dtype=torch.float64, requires_grad=True)
    out, _ = stem(x, torch.tensor([20, 9]))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[1, :9].abs().sum() > 0
