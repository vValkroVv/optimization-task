"""CPU boundary tests; native compiled FP8/SAC execution is a separate GPU gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "component_fixtures.py"
spec = importlib.util.spec_from_file_location("capture", PATH)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class Layer(torch.nn.Module):
    def forward(self, x, rope_cache=None, attention_masks=None, positions=None, token_mask=None):
        return x.sin() * 2


class MTP(torch.nn.Module):
    def forward(self, trunk_hidden, input_embeds, rope_cache=None, attention_masks=None, positions=None, token_mask=None):
        shifted = torch.cat([input_embeds[:, 1:], torch.zeros_like(input_embeds[:, :1])], 1)
        return (trunk_hidden + shifted).unsqueeze(0)


class Trainer:
    def __init__(self, compiled=False, checkpointed=False):
        self.checkpointed = checkpointed
        root = torch.nn.Module()
        root.tok_embeddings = torch.nn.Embedding(16, 3)
        root.layers = torch.nn.ModuleDict({"0": Layer()})
        if compiled:
            root.layers["0"].compile(backend="eager", fullgraph=True)
        root.mtp = MTP()
        self.model_parts = [root]
        self.config = SimpleNamespace(training=SimpleNamespace(steps=1), to_dict=lambda: {})
        self.gradient_accumulation_steps = 32
        self.ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int64)
        self.positions = torch.arange(4).expand(2, -1)
        self.token_mask = self.ids != 8

    def forward_backward_step(self, *, omit_mtp=False, duplicate=False):
        root = self.model_parts[0]
        emb = root.tok_embeddings(self.ids)
        def block(x):
            return root.layers["0"](x, None, positions=self.positions, token_mask=self.token_mask)
        if self.checkpointed:
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(block, emb, use_reentrant=False)
        else:
            h = block(emb)
        if duplicate:
            root.layers["0"](emb, None)
        out = h if omit_mtp else root.mtp(h, emb, positions=self.positions, token_mask=self.token_mask)
        out.sum().backward()
        return out.detach()


@pytest.mark.parametrize("compiled,checkpointed", [(False, False), (True, False), (False, True), (True, True)])
def test_capture_replay_preserves_outputs_gradients_and_boundaries(tmp_path, monkeypatch, compiled, checkpointed):
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    original_check_output = capture.subprocess.check_output
    monkeypatch.setattr(capture.subprocess, "check_output", lambda cmd, **kw: "test-head\n" if cmd == ["git", "rev-parse", "HEAD"] else original_check_output(cmd, **kw))
    trainer = Trainer(compiled, checkpointed)
    root = trainer.model_parts[0]
    original = root.layers["0"]._compiled_call_impl if compiled else root.layers["0"].forward
    torch.manual_seed(2)
    expected = trainer.forward_backward_step()
    expected_grad = root.tok_embeddings.weight.grad.clone()
    root.zero_grad()
    output = tmp_path / "runs/production-data-abcd/capture"
    with capture.NativeCapture(trainer, output, 0, 8) as active:
        actual = trainer.forward_backward_step()
        assert active.complete
        assert not active.restorations and not active.hooks and not active.records
        # Remaining accumulation calls have no instrumentation.
        assert "forward_backward_step" not in vars(trainer)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_grad, root.tok_embeddings.weight.grad)
    assert (root.layers["0"]._compiled_call_impl if compiled else root.layers["0"].forward) == original
    records, receipt = capture.load_fixture(output, 0, production=False)
    assert torch.equal(records["input_ids"], trainer.ids)
    assert torch.equal(records["mtp"]["embedding"], root.tok_embeddings(trainer.ids))
    assert torch.equal(records["layer0"]["upstream"], torch.ones_like(records["layer0"]["hidden"]))
    assert torch.equal(records["mtp"]["token_mask"], trainer.token_mask)
    assert receipt["world_size"] == 8
    with pytest.raises(ValueError, match="B2 S4096"):
        capture.load_fixture(output, 0)
    with (output / "rank0/tensors.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        capture.load_fixture(output, 0, production=False)


@pytest.mark.parametrize("kwargs,match", [({"omit_mtp": True}, "missing native"), ({"duplicate": True}, "duplicate native")])
def test_missing_duplicate_fail_and_restore(tmp_path, monkeypatch, kwargs, match):
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    trainer = Trainer()
    with pytest.raises(RuntimeError, match=match):
        with capture.NativeCapture(trainer, tmp_path / "runs/production-data-abcd/failure", 0, 8):
            trainer.forward_backward_step(**kwargs)
    assert "forward_backward_step" not in vars(trainer)
    assert "forward" not in vars(trainer.model_parts[0].mtp)
    assert "forward" not in vars(trainer.model_parts[0].layers["0"])


def test_reject_unsupported_masks():
    with pytest.raises(TypeError, match="do not silently drop masks"):
        capture.map_tensors(object(), lambda tensor: tensor)


def test_varlen_mask_metadata_roundtrip_preserves_dtypes_and_offsets(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parents[2] / "gift"))
    from torchtitan.models.common.attention.varlen_meta import VarlenMetadata

    offsets = torch.tensor([0, 2, 5, 8], dtype=torch.int32)
    metadata = VarlenMetadata(offsets, offsets.clone(), 3, 3)
    captured = capture.map_tensors({"varlen": metadata}, lambda t: t.detach().cpu().clone())
    restored = capture.restore(captured, "cpu")["varlen"]
    assert type(restored) is VarlenMetadata
    assert restored.max_q == restored.max_k == 3
    assert restored.cu_seq_q.dtype == restored.cu_seq_k.dtype == torch.int32
    assert torch.equal(restored.cu_seq_q, offsets)
    assert restored.cu_seq_q.data_ptr() != offsets.data_ptr()
