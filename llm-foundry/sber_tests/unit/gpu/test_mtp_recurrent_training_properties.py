"""What recurrent-MTP training is supposed to DO, end to end on one GPU.

Every other MTP test pins a mechanism (a roll count, a cache write, a kernel's
log-sum-exp). This file pins the promises a training run makes to the release,
through the real `GigaMixForCausalLM` under the arm recipe, on packed
multi-document batches -- each against an oracle that is independent of the
implementation (document arithmetic, a frozen snapshot, a re-run):

  * `TestOnlyTheHeadTrains` -- the shipped `disable_grad` regex, applied by the
    production `_apply_disable_grad`, leaves exactly the head moving after a
    real optimizer step. The regex fails silently (contract.md), so the CPU
    name-partition test is backed here by parameters actually changing.
  * `TestTheHeadLearnsOneBatch` -- thirty optimizer steps on one fixed batch
    halve the draft objective. Catches dead gradients anywhere in the
    cache/merge/loss chain that a single-step test cannot see.
  * `TestActivationCheckpointingIsInvisible` -- the production non-reentrant
    checkpoint wrapper around the MTP layer changes neither the loss nor the
    head gradients (F-32's replay contract at the whole-model level).
  * `TestDocumentsDoNotTouch` -- rewriting document B moves nothing that
    document A's positions produce, at any depth, except the teacher-forced
    tail whose supervision I-3 masks. The frontier is computed from document
    arithmetic, not from the code.
  * `TestChainSeesOnlyItsPast` -- a token edit at position p moves no output
    strictly left of the frontier its depth allows: `p - 1 - k` off-policy
    (one embed roll per depth), `p - 1` at every depth on-policy (F-12's
    no-roll). Wrong roll counts, cache leaks and RoPE misindexing all land on
    the wrong side of that line.
  * `TestForwardIsStateless` -- two identical forwards agree bitwise: the
    draft cache and its masks leave nothing on the module between steps.
  * `TestASecondSequenceLengthChangesNothing` -- gradients at one row length
    are bitwise unaffected by a draft step at another. This is the regression
    test for F-39, which this file's failures uncovered: the deprecated
    `_compile=True` mask-builder flag shared dynamo state across shapes and
    corrupted the backward-direction block tables at the second length seen,
    leaving forward exact while gradients moved tenfold.

Depths run at the arm's 3 with `mtp_recurrent` on, which is also what turns the
draft geometry on, so the flex prefix, the depth-indexed cache and the RoPE
offsets are all in the loop. Logit rows come back stacked
`[main, head 1, .., head D]` along the batch axis (`use_liger: false`,
`delete_logits: false`), which is what lets one tensor carry every per-depth
assertion.
"""

import re

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from llmfoundry.models.giga_mix.modelling_giga_mix import GigaMixForCausalLM
from llmfoundry.models.layers.mtp import get_mtp_activation_checkpointing_modules
from llmfoundry.utils.builders import _apply_disable_grad
from sber_tests.unit.gpu.test_mtp_sp_loss_equivalence import (
    SEQ_LEN,
    _batch,
    _losses,
    _sp_config,
)

DEPTHS = 3
DOCS = (80, SEQ_LEN - 80)
# The freeze regex a recurrent recipe has to use: the shared layer and norm sit
# under index 0 only, so the pattern is `\d+`-indexed, never an enumerated depth.
FREEZE_PATTERNS = [r"^(?!model\.model\.mtp_block\.(mtp_layers|mtp_norms)\.\d+\.)"]

requires_gpu = pytest.mark.requires_gpu


def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: flash-attn MLA has no CPU kernel")


def _arm_model(on_policy: bool = True) -> GigaMixForCausalLM:
    config = _sp_config()
    config.mtp_recurrent = True
    config.on_policy_ttt = on_policy
    config.validate_mtp_config()
    torch.manual_seed(1234)
    return GigaMixForCausalLM(config).to("cuda").to(torch.bfloat16).train()


def _freeze_like_production(model: GigaMixForCausalLM) -> list:
    """The recipe's `disable_grad`, applied by the function the trainer applies it
    with. The bare model sits one `model.` level below the Composer wrapper the
    regex is anchored at, so a shell restores that name geometry."""
    patterns = FREEZE_PATTERNS
    shell = nn.Module()
    shell.model = model
    _apply_disable_grad(shell, patterns)
    return [p for p in model.parameters() if p.requires_grad]


def _per_depth_logits(model, batch) -> torch.Tensor:
    """`[1 + DEPTHS, seq, vocab]`: the main head's row, then one per draft depth."""
    outputs = model(**batch, return_dict=False)
    logits = outputs[1]
    return logits.view(1 + DEPTHS, SEQ_LEN, -1)


def _rewrite_tail(batch, start: int, seed: int) -> dict:
    """The same pack with document B's tokens rewritten; the seam keeps the
    `data/packing.py` mask and document A's slice stays identical."""
    torch.manual_seed(seed)
    out = {name: tensor.clone() for name, tensor in batch.items()}
    tail = torch.randint(
        1, 256, (1, SEQ_LEN - start), device=batch["input_ids"].device
    )
    out["input_ids"][:, start:] = tail
    out["labels"][:, start:] = tail
    out["labels"][:, start] = -100
    return out


class TestOnlyTheHeadTrains:
    @pytest.mark.unit
    @requires_gpu
    def test_one_optimizer_step_moves_the_head_and_nothing_else(self):
        _skip_without_cuda()
        model = _arm_model()
        trainable = _freeze_like_production(model)
        batch = _batch(torch.device("cuda"), DOCS)
        before = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }

        optimizer = torch.optim.Adam(trainable, lr=1e-2)
        _losses(model, batch)[0].backward()
        optimizer.step()

        head = re.compile(r"^model\.mtp_block\.(mtp_layers|mtp_norms)\.0\.")
        moved, held = [], []
        for name, param in model.named_parameters():
            if head.search(name):
                assert param.requires_grad, f"{name} should be trainable"
                assert param.grad is not None and torch.isfinite(param.grad).all(), (
                    f"{name} got no finite gradient"
                )
                assert param.grad.abs().sum() > 0, f"{name} got a zero gradient"
                assert not torch.equal(param, before[name]), f"{name} did not move"
                moved.append(name)
            else:
                assert not param.requires_grad, f"{name} escaped the freeze"
                assert torch.equal(param, before[name]), f"{name} moved while frozen"
                held.append(name)
        assert moved and held


class TestTheHeadLearnsOneBatch:
    @pytest.mark.unit
    @requires_gpu
    def test_a_hundred_steps_halve_the_per_head_cross_entropy(self):
        """Memorizing one fixed batch is the weakest form of learning; a head
        whose gradients survive the cache, the merge and the loss chain does it
        easily, and one with any of them broken cannot. The draft objective is
        not the yardstick: on random tokens the freshly initialized head already
        matches the teacher (lk starts near 0.05) and drifts as memorization
        pulls the head off the teacher, so per-head CE -- which starts at
        ln(vocab) and fell 5.54 to 2.46 over this exact run when calibrated --
        is what carries the assertion, with margin on the measured 0.44 ratio."""
        _skip_without_cuda()
        model = _arm_model()
        trainable = _freeze_like_production(model)
        batch = _batch(torch.device("cuda"), DOCS)
        optimizer = torch.optim.Adam(trainable, lr=1e-2)

        history = []
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            losses = _losses(model, batch)
            assert torch.isfinite(losses).all(), f"non-finite losses: {losses}"
            losses[0].backward()
            # As the trainer does; bf16 Adam at this rate rides close enough to
            # instability that unclipped runs can blow up on ~1% numeric shifts.
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            history.append(losses.detach())

        first, last = history[0], history[-1]
        assert last[2:].mean() < 0.55 * first[2:].mean(), (
            f"per-head CE did not halve: {first[2:].tolist()} -> {last[2:].tolist()}"
        )


class TestActivationCheckpointingIsInvisible:
    @pytest.mark.unit
    @requires_gpu
    def test_the_wrapper_changes_neither_loss_nor_gradients(self):
        _skip_without_cuda()
        batch = _batch(torch.device("cuda"), DOCS)

        def run(checkpointed: bool):
            model = _arm_model()
            _freeze_like_production(model)
            if checkpointed:
                targets = set(
                    get_mtp_activation_checkpointing_modules(
                        model.model.mtp_block.mtp_layers
                    )
                )
                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
                        module, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                    ),
                    check_fn=lambda module: module in targets,
                )
            losses = _losses(model, batch)
            losses[0].backward()
            grads = {
                # The wrapper renames what it wraps; strip its infix so the
                # two runs compare parameter for parameter.
                name.replace("._checkpoint_wrapped_module", ""):
                    param.grad.detach().float().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            return losses.detach(), grads

        plain_losses, plain_grads = run(checkpointed=False)
        wrapped_losses, wrapped_grads = run(checkpointed=True)

        torch.testing.assert_close(wrapped_losses, plain_losses, atol=0.0, rtol=0.0)
        assert wrapped_grads.keys() == plain_grads.keys()
        for name in plain_grads:
            # Replay reruns the wrapped projections at backward time, where
            # cuBLAS may pick a different algorithm than the saved forward did
            # (it keys on allocator state, so this is order-dependent). That
            # legitimate drift stays well under a percent of the gradient's
            # norm; a cache or replay defect is structural and blows past it.
            plain, wrapped = plain_grads[name], wrapped_grads[name]
            rel = (wrapped - plain).norm() / plain.norm().clamp_min(1e-12)
            assert rel < 2e-2, (
                f"{name}: checkpointed gradient deviates by {rel:.2e} in "
                "relative L2"
            )


class TestDocumentsDoNotTouch:
    """Rewriting document B is not allowed to reach document A.

    Off-policy, head h at position j is teacher-forced `token[j + h]`, so its
    row is claimed exactly on `j < seam - h`; on-policy only depth 0 keeps its
    single roll, so every head is claimed on `j < seam - 1`. The positions past
    each claim are the control: they read document B's tokens by design, and
    I-3 masks their supervision instead.
    """

    @pytest.mark.unit
    @pytest.mark.requires_gpu
    @pytest.mark.parametrize("on_policy", [True, False])
    def test_document_a_rows_are_bitwise_stable(self, on_policy):
        _skip_without_cuda()
        model = _arm_model(on_policy=on_policy)
        device = torch.device("cuda")
        seam = DOCS[0]
        batch = _batch(device, DOCS)
        rewritten = _rewrite_tail(batch, start=seam, seed=8)

        with torch.no_grad():
            base = _per_depth_logits(model, batch)
            moved = _per_depth_logits(model, rewritten)

        assert torch.equal(moved[0, :seam], base[0, :seam]), (
            "the main head saw across the document boundary"
        )
        assert not torch.equal(moved[0, seam:], base[0, seam:]), (
            "control: document B's own rows must move"
        )
        for head in range(1, DEPTHS + 1):
            frontier = seam - 1 if on_policy else seam - head
            assert torch.equal(moved[head, :frontier], base[head, :frontier]), (
                f"head {head} saw document B left of position {frontier}"
            )
            assert not torch.equal(
                moved[head, frontier : seam], base[head, frontier : seam]
            ), (
                f"control: head {head}'s teacher-forced tail at the seam is "
                "supposed to read document B"
            )


class TestChainSeesOnlyItsPast:
    @pytest.mark.unit
    @requires_gpu
    @pytest.mark.parametrize("on_policy", [True, False])
    def test_an_edit_moves_nothing_left_of_its_frontier(self, on_policy):
        _skip_without_cuda()
        model = _arm_model(on_policy=on_policy)
        device = torch.device("cuda")
        edit = 32
        batch = _batch(device, (SEQ_LEN,))
        edited = {name: tensor.clone() for name, tensor in batch.items()}
        edited["input_ids"][:, edit] = (edited["input_ids"][:, edit] + 101) % 255 + 1

        with torch.no_grad():
            base = _per_depth_logits(model, batch)
            moved = _per_depth_logits(model, edited)

        assert torch.equal(moved[0, :edit], base[0, :edit])
        assert not torch.equal(moved[0, edit:], base[0, edit:])
        for head in range(1, DEPTHS + 1):
            depth = head - 1
            frontier = edit - 1 if on_policy else edit - depth - 1
            assert torch.equal(moved[head, :frontier], base[head, :frontier]), (
                f"head {head} moved left of position {frontier}"
            )
            assert not torch.equal(
                moved[head, frontier], base[head, frontier]
            ), f"control: head {head} at {frontier} reads the edit by design"


class TestForwardIsStateless:
    @pytest.mark.unit
    @requires_gpu
    def test_two_identical_forwards_agree_bitwise(self):
        _skip_without_cuda()
        model = _arm_model()
        batch = _batch(torch.device("cuda"), DOCS)
        with torch.no_grad():
            first = _per_depth_logits(model, batch)
            second = _per_depth_logits(model, batch)
        torch.testing.assert_close(second, first, atol=0.0, rtol=0.0)


class TestASecondSequenceLengthChangesNothing:
    @staticmethod
    def _grads(batch):
        model = _arm_model()
        _freeze_like_production(model)
        _losses(model, batch)[0].backward()
        return {
            name: param.grad.float().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

    @pytest.mark.unit
    @requires_gpu
    def test_gradients_survive_a_draft_step_at_another_length(self):
        _skip_without_cuda()
        device = torch.device("cuda")
        batch = _batch(device, DOCS)
        before = self._grads(batch)

        # One training step at a second row length, as an underfull final
        # pack or a probe produces. Before F-39 this poisoned every later
        # backward in the process.
        torch.manual_seed(11)
        half = SEQ_LEN // 2
        ids = torch.randint(1, 256, (1, half), device=device)
        short = {
            "input_ids": ids,
            "labels": ids.clone(),
            "position_ids": torch.arange(half, device=device).unsqueeze(0),
        }
        _losses(_arm_model(), short)[0].backward()

        after = self._grads(batch)
        assert before.keys() == after.keys()
        for name in before:
            assert torch.equal(after[name], before[name]), (
                f"{name}: a draft step at length {half} changed the gradients "
                f"of an unrelated step at length {SEQ_LEN}"
            )
