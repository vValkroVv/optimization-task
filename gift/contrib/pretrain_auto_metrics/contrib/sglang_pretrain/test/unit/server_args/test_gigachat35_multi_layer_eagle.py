"""Gigachat35 multi-layer EAGLE auto-enable and step-count validation.

Regression tests for the dead-code bug where the auto-enable lived in the
Step3p5 `elif` of `_handle_model_specific_adjustments`, which Gigachat35 could
never reach (it matches the DeepSeek-family `if` earlier in the same chain),
so plain EAGLE silently chained MTP head 0 for every draft step.

All tests drive handle_speculative_decoding, the public entry point: the
auto-enable runs inside _handle_eagle_family, after the algorithm is
normalized (NEXTN -> EAGLE) and after adaptive compatibility is settled.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs

_GIGACHAT35_ARCH = "Gigachat35ForCausalLM"
_DEEPSEEK_ARCHES = ("DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM")


def _patch_draft_config():
    # handle_speculative_decoding loads the draft checkpoint's config while
    # resolving the NEXTN alias whenever a draft path is set; stub it out.
    return patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=SimpleNamespace(architectures=["LlamaForCausalLM"]),
    )


def _hf_config(num_nextn_predict_layers=3, arch=_GIGACHAT35_ARCH):
    cfg = SimpleNamespace(
        architectures=[arch],
        get_text_config=lambda: SimpleNamespace(),
    )
    if num_nextn_predict_layers is not None:
        cfg.num_nextn_predict_layers = num_nextn_predict_layers
    return cfg


def _make_args(num_nextn_predict_layers=3, arch=_GIGACHAT35_ARCH, steps=3, **overrides):
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = "EAGLE"
    args.device = "cuda"
    args.page_size = 1
    args.speculative_num_steps = steps
    args.speculative_eagle_topk = 1 if steps is not None else None
    args.speculative_num_draft_tokens = steps + 1 if steps is not None else None
    for key, value in overrides.items():
        setattr(args, key, value)
    cfg = _hf_config(num_nextn_predict_layers, arch)
    args.get_model_config = lambda: SimpleNamespace(hf_config=cfg, hf_text_config=cfg)
    return args


class TestGigachat35MultiLayerEagleAutoEnable(unittest.TestCase):
    def test_enables_for_multi_head_checkpoint_with_unset_steps(self):
        args = _make_args(steps=None)
        handle_speculative_decoding(args)
        self.assertTrue(args.enable_multi_layer_eagle)
        self.assertTrue(args._multi_layer_eagle_auto_enabled)

    def test_enables_for_unnormalized_algorithm_spellings(self):
        # The auto-enable runs after the hook upper-cases the algorithm and
        # resolves the NEXTN alias, so every documented spelling works.
        for algorithm in ("NEXTN", "nextn", "eagle"):
            args = _make_args(speculative_algorithm=algorithm)
            handle_speculative_decoding(args)
            self.assertTrue(args.enable_multi_layer_eagle, f"algorithm={algorithm}")
            self.assertEqual(args.speculative_algorithm, "EAGLE")

    def test_active_adaptive_wins_over_auto_enable(self):
        args = _make_args(steps=None, speculative_adaptive=True)
        handle_speculative_decoding(args)
        self.assertTrue(args.speculative_adaptive)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_skips_single_head_checkpoint(self):
        args = _make_args(num_nextn_predict_layers=1)
        handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_skips_checkpoint_without_mtp_heads(self):
        args = _make_args(num_nextn_predict_layers=None)
        handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_enables_for_deepseek_arches(self):
        for arch in _DEEPSEEK_ARCHES:
            args = _make_args(arch=arch)
            handle_speculative_decoding(args)
            self.assertTrue(args.enable_multi_layer_eagle, f"arch={arch}")

    def test_skips_unsupported_arch(self):
        args = _make_args(arch="LlamaForCausalLM")
        handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_skips_eagle3_algorithm(self):
        args = _make_args(speculative_algorithm="EAGLE3")
        handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_skips_non_eagle_algorithm(self):
        args = _make_args(speculative_algorithm=None)
        handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_skips_distinct_draft_model_path(self):
        args = _make_args(speculative_draft_model_path="/elsewhere/eagle-draft")
        with _patch_draft_config():
            handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_explicit_flag_is_not_marked_auto_enabled(self):
        args = _make_args(enable_multi_layer_eagle=True)
        handle_speculative_decoding(args)
        self.assertTrue(args.enable_multi_layer_eagle)
        self.assertFalse(getattr(args, "_multi_layer_eagle_auto_enabled", False))


class TestMultiLayerEagleStepsValidation(unittest.TestCase):
    def test_explicit_flag_with_too_many_steps_raises(self):
        args = _make_args(steps=4, enable_multi_layer_eagle=True)
        with self.assertRaises(ValueError) as ctx:
            handle_speculative_decoding(args)
        self.assertIn("num_nextn_predict_layers", str(ctx.exception))

    def test_auto_enabled_flag_downgrades_instead_of_raising(self):
        args = _make_args(steps=4)
        handle_speculative_decoding(args)
        self.assertTrue(getattr(args, "_multi_layer_eagle_auto_enabled", False))
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_explicit_flag_within_head_count_passes(self):
        args = _make_args(steps=3, enable_multi_layer_eagle=True)
        handle_speculative_decoding(args)
        self.assertTrue(args.enable_multi_layer_eagle)

    def test_explicit_flag_with_tree_topk_raises(self):
        args = _make_args(steps=3, enable_multi_layer_eagle=True)
        args.speculative_eagle_topk = 4
        args.speculative_num_draft_tokens = 4
        with self.assertRaises(ValueError) as ctx:
            handle_speculative_decoding(args)
        self.assertIn("speculative_eagle_topk", str(ctx.exception))

    def test_auto_enabled_flag_with_tree_topk_downgrades(self):
        args = _make_args(steps=3)
        args.speculative_eagle_topk = 4
        args.speculative_num_draft_tokens = 4
        handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)

    def test_wrapper_arch_reads_heads_from_text_config(self):
        arch = "Step3p7ForConditionalGeneration"
        text_cfg = SimpleNamespace(num_nextn_predict_layers=3)
        top_cfg = SimpleNamespace(
            architectures=[arch], get_text_config=lambda: text_cfg
        )
        args = _make_args(steps=3, enable_multi_layer_eagle=True)
        args.get_model_config = lambda: SimpleNamespace(
            hf_config=top_cfg, hf_text_config=text_cfg
        )
        handle_speculative_decoding(args)
        self.assertTrue(args.enable_multi_layer_eagle)

    def test_explicit_flag_without_mtp_heads_raises(self):
        for heads in (None, 0):
            args = _make_args(
                num_nextn_predict_layers=heads, enable_multi_layer_eagle=True
            )
            with self.assertRaises(ValueError, msg=f"heads={heads}") as ctx:
                handle_speculative_decoding(args)
            self.assertIn("num_nextn_predict_layers", str(ctx.exception))

    def test_explicit_flag_with_distinct_draft_skips_target_heads_check(self):
        # The target's num_nextn_predict_layers does not bound the draft depth
        # when the draft weights come from a different checkpoint.
        args = _make_args(
            num_nextn_predict_layers=2,
            steps=3,
            enable_multi_layer_eagle=True,
            speculative_draft_model_path="/elsewhere/multi-head-draft",
        )
        with _patch_draft_config():
            handle_speculative_decoding(args)
        self.assertTrue(args.enable_multi_layer_eagle)

    def test_auto_enabled_flag_with_rejection_sampling_downgrades(self):
        args = _make_args(steps=3)
        args.speculative_use_rejection_sampling = True
        handle_speculative_decoding(args)
        self.assertFalse(args.enable_multi_layer_eagle)


class TestWorkerChoiceIsLogged(unittest.TestCase):
    """The resolved worker is stated at startup, not left to be inferred.

    Which worker runs is decided by the draft checkpoint's head count, not by
    anything in the serving config, and the two give the draft chain different
    attention geometry. A single-head export is the quiet case: nothing is
    auto-enabled, no warning fires, and the serving YAML says EAGLE either way.
    """

    _LOGGER = "sglang.srt.arg_groups.speculative_hook"

    def _worker_line(self, args):
        with self.assertLogs(self._LOGGER, level="INFO") as captured:
            handle_speculative_decoding(args)
        lines = [m for m in captured.output if "Speculative worker:" in m]
        self.assertEqual(len(lines), 1, captured.output)
        return lines[0]

    def test_single_head_export_logs_plain_eagle(self):
        line = self._worker_line(_make_args(num_nextn_predict_layers=1))
        self.assertIn("Speculative worker: EAGLEWorkerV2", line)
        self.assertIn("num_nextn_predict_layers=1", line)
        self.assertIn("enable_multi_layer_eagle=False", line)
        self.assertIn("auto_enabled=False", line)

    def test_multi_head_export_logs_the_multi_layer_worker(self):
        line = self._worker_line(_make_args(num_nextn_predict_layers=3))
        self.assertIn("Speculative worker: MultiLayerEagleWorkerV2", line)
        self.assertIn("auto_enabled=True", line)

    def test_downgraded_run_logs_the_worker_it_ended_up_with(self):
        # steps > heads: auto-enabled, then downgraded. The log runs after.
        line = self._worker_line(_make_args(steps=4))
        self.assertIn("Speculative worker: EAGLEWorkerV2", line)
        self.assertIn("auto_enabled=True", line)


if __name__ == "__main__":
    unittest.main()
