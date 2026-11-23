"""Exercise the exact folded CE bridge without loading a CUDA library."""

import ctypes
from types import SimpleNamespace

import pytest
import torch

from torchtitan.ops import exact_folded_ce


def test_exact_folded_ce_rejects_unverified_library(tmp_path):
    library = tmp_path / "centered.so"
    library.write_bytes(b"unverified")
    with pytest.raises(ValueError, match="SHA256"):
        exact_folded_ce.ExactFoldedCrossEntropy(str(library), num_chunks=2)


@pytest.mark.parametrize("mtp_weight", [0.3, 0.8])
@pytest.mark.parametrize("rows", [8192, 2048, 512])
def test_exact_folded_ce_bridge_checks_weighted_backward(monkeypatch, mtp_weight, rows):
    vocab = 3
    expected_grad = torch.arange(rows * vocab, dtype=torch.bfloat16).view(rows, vocab)
    denominator = 17.0
    scales = torch.tensor([1.0, 0.3]) / denominator
    predicted = torch.zeros_like(expected_grad, requires_grad=True)
    labels = torch.zeros(rows, dtype=torch.int64)

    def stub_launcher(
        logit_ptr, label_ptr, loss_ptr, scale_ptr, num_rows, stream, mode
    ):
        assert (num_rows, stream, mode) == (rows, 0, 13)
        assert label_ptr == labels.data_ptr()
        assert scale_ptr == scales.data_ptr()
        destination = (ctypes.c_int16 * expected_grad.numel()).from_address(logit_ptr)
        destination[:] = expected_grad.view(torch.int16).flatten().tolist()
        row_losses = (ctypes.c_float * rows).from_address(loss_ptr)
        row_losses[:] = [2.0] * (rows // 2) + [5.0] * (rows // 2)
        return 0

    monkeypatch.setattr(exact_folded_ce, "_load_launcher", lambda _: stub_launcher)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=0)
    )
    loss = SimpleNamespace(
        library_path="stub",
        compiled=exact_folded_ce._consume,
        scales_checked=False,
    )
    main, mtp = exact_folded_ce._PrecomputedGradient.apply(
        predicted, labels, scales, loss
    )
    torch.testing.assert_close(main.detach(), torch.tensor(float(rows)))
    torch.testing.assert_close(mtp.detach(), torch.tensor(5.0 * (rows // 2)))
    torch.testing.assert_close(predicted, expected_grad)
    total = main / denominator + (mtp / denominator) * mtp_weight
    if mtp_weight == 0.3:
        total.backward()
        assert loss.scales_checked
        torch.testing.assert_close(predicted.grad, expected_grad)
    else:
        with pytest.raises(RuntimeError, match="scales differ"):
            total.backward()
        assert not loss.scales_checked
