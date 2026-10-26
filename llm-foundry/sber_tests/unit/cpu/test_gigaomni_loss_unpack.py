"""ComposerGigaOmniForCausalLM.loss reads GigaMixForCausalLM's output tuple from
the end; the draft-loss pairs are appended after everything else, so they have
to come off first or every older entry is read two slots early.

The wrapper is built with `__new__` around a stub tree that carries exactly
the attributes `loss` touches; the method under test is production code.
"""

import types

import pytest
import torch
from torch import nn

from llmfoundry.models.gigaomni.composer_gigaomni import ComposerGigaOmniForCausalLM

MTP_NUM = 3


def _wrapper(draft_loss_kinds, mtp_predictor_num=MTP_NUM):
    language_model = nn.Module()
    language_model.config = types.SimpleNamespace(return_dict=False)
    language_model.draft_loss_kinds = list(draft_loss_kinds)
    language_model.DRAFT_DIAGNOSTIC_KINDS = ("adaptive_weight_lambda",)

    model = nn.Module()
    model.language_model = language_model
    model.config = types.SimpleNamespace(
        language_model={"use_mtp": True, "mtp_predictor_num": mtp_predictor_num}
    )

    wrapper = ComposerGigaOmniForCausalLM.__new__(ComposerGigaOmniForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    return wrapper


def _scalar(v):
    return torch.tensor(float(v))


def _heads(base):
    return torch.tensor([base + 0.1 * i for i in range(1, MTP_NUM + 1)])


@pytest.mark.unit
@pytest.mark.fast
class TestDraftLossPairsComeOffFirst:
    def test_one_kind(self):
        total, logits = _scalar(100.0), torch.zeros(1, 4, 8)
        loss_ce, loss_mtp, per_head = _scalar(1.0), _scalar(2.0), _heads(3.0)
        lk_loss, lk_per_head = _scalar(4.0), _heads(5.0)
        # Append order of GigaMixForCausalLM.forward with z/hidden-z losses off.
        outputs = (total, logits, loss_ce, loss_mtp, per_head, lk_loss, lk_per_head)

        d = _wrapper(["lk"]).loss(outputs, batch={})

        assert torch.equal(d["total"], total)
        assert torch.equal(d["loss_ce"], loss_ce)
        assert torch.equal(d["mtp_loss"], loss_mtp)
        for i in range(MTP_NUM):
            assert torch.equal(d[f"loss_mtp_{i + 1}"], per_head[i])
        assert torch.equal(d["lk_loss"], lk_loss)
        for i in range(MTP_NUM):
            assert torch.equal(d[f"lk_loss{i + 1}"], lk_per_head[i])

    def test_two_kinds_and_a_diagnostic(self):
        loss_ce, loss_mtp, per_head = _scalar(1.0), _scalar(2.0), _heads(3.0)
        tv, tv_h = _scalar(6.0), _heads(7.0)
        lam_h = _heads(8.0)  # diagnostic: logged per head only, no scalar
        outputs = (
            _scalar(0.0), torch.zeros(1, 4, 8),
            loss_ce, loss_mtp, per_head,
            tv, tv_h,
            None, lam_h,
        )

        d = _wrapper(["tv", "adaptive_weight_lambda"]).loss(outputs, batch={})

        assert torch.equal(d["loss_ce"], loss_ce)
        assert torch.equal(d["mtp_loss"], loss_mtp)
        assert torch.equal(d["tv_loss"], tv)
        assert "adaptive_weight_lambda_loss" not in d
        assert "adaptive_weight_lambda" not in d  # None scalar -> skipped
        for i in range(MTP_NUM):
            assert torch.equal(d[f"tv_loss{i + 1}"], tv_h[i])

    def test_no_kinds_is_the_old_layout(self):
        loss_ce, loss_mtp, per_head = _scalar(1.0), _scalar(2.0), _heads(3.0)
        outputs = (_scalar(0.0), torch.zeros(1, 4, 8), loss_ce, loss_mtp, per_head)

        d = _wrapper([]).loss(outputs, batch={})

        assert torch.equal(d["loss_ce"], loss_ce)
        assert torch.equal(d["mtp_loss"], loss_mtp)
        assert d["loss_mtp_3"] == per_head[2]
