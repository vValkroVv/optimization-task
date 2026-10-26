# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for FLOPs-based pipeline stage balancing.

Unit costs are pinned against ``get_blockwise_nparams_and_flops`` (the MFU
FLOPs source of truth) on real debug configs, and ``partition_minimax`` is
checked against a brute-force optimal splitter on random costs.
"""

import itertools
import random
import unittest
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import torch

from tests.unit_tests.config_helpers import load_model_spec
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.pipeline_balance import (
    flops_balanced_module_fqns,
    model_pipeline_units,
    partition_minimax,
    PipelineUnit,
)
from torchtitan.distributed.pipeline_parallel import _partition_segments, pipeline_llm
from torchtitan.tools.pipeline import _format_modules_with_ranges, log_pipeline_layout

_SEQ_LEN = 512


# Debug presets with the SDPA kernel and the single-process all-to-all
# dispatcher, so the models build on CPU.
_SDPA = {"attention": {"inner_attention": {"type": "sdpa"}}}
_ALL_TO_ALL = {"moe": {"experts": {"token_dispatcher": {"type": "all_to_all"}}}}
_MLA_BLOCKS = {"mla_dense": _SDPA, "mla_moe": {**_SDPA, **_ALL_TO_ALL}}
_DEBUG_PRESETS = {
    "llama3": ("llama3/debugmodel.yaml", {"config": {"blocks": {"dense": _SDPA}}}),
    "deepseek_v3": (
        "deepseek_v3/debugmodel.yaml",
        {"config": {"blocks": _MLA_BLOCKS}},
    ),
    "gift_moe": (
        "gift_moe/debugmodel.yaml",
        {
            "config": {
                "blocks": _MLA_BLOCKS,
                # gift_moe's MTP heads take its MoE block.
                "mtp": {"layers": [{"block": "mla_moe", "repeat": 2}]},
            }
        },
    ),
}


def _build(name: str):
    """Build a (config, model) pair for a debugmodel of the named model."""
    preset, model_overrides = _DEBUG_PRESETS[name]
    cfg = load_model_spec(preset, model_overrides).model
    return cfg, cfg.build()


def _ac(
    mode: Literal["selective", "full", "memory_budget", "none"],
) -> ActivationCheckpointConfig:
    return ActivationCheckpointConfig(mode=mode)


def _segments(model):
    stem_fqns, body_items, tail_fqns, _ = _partition_segments(model)
    return stem_fqns, body_items, tail_fqns


def _block_fqns(model):
    return {s.fqn for s in model.named_segments() if s.is_block}


def _model_atoms(model, cfg, ac):
    return model_pipeline_units(
        model,
        cfg,
        _SEQ_LEN,
        ac,
        segments=_segments(model),
        block_fqns=_block_fqns(model),
    )


def _balanced(model, cfg, parallelism, ac, num_stages):
    stages, _ = flops_balanced_module_fqns(
        model,
        cfg,
        parallelism,
        _SEQ_LEN,
        ac,
        num_stages,
        segments=_segments(model),
        block_fqns=_block_fqns(model),
    )
    return stages


class TestDecoderPipelineUnits(unittest.TestCase):
    def _units_by_fqn(self, model, cfg, ac):
        atoms = _model_atoms(model, cfg, ac)
        return {u.fqns[0]: u for atom in atoms for u in atom}

    def test_costs_match_mfu_source_of_truth(self):
        """With no AC, cost = 3x fwd FLOPs, so summed unit costs must
        re-derive get_blockwise_nparams_and_flops."""
        for name in ["llama3", "deepseek_v3", "gift_moe"]:
            with self.subTest(model=name):
                cfg, model = _build(name)
                _, flops_per_token = cfg.owner().get_nparams_and_flops(
                    cfg, model, _SEQ_LEN
                )
                atoms = _model_atoms(model, cfg, _ac("none"))
                self.assertAlmostEqual(
                    sum(u.cost for atom in atoms for u in atom),
                    flops_per_token,
                    delta=flops_per_token * 1e-9,
                )

    def test_head_costing_definition_is_shared_with_mfu_counter(self):
        from torchtitan.models.utils import prediction_head_costing

        self.assertEqual(prediction_head_costing(_build("gift_moe")[0]), ("lm_head", 3))
        self.assertEqual(prediction_head_costing(_build("llama3")[0]), ("lm_head", 1))

    def test_atom_structure(self):
        """Stem and tail are single atoms; each body item is its own atom."""
        cfg, model = _build("gift_moe")
        atoms = _model_atoms(model, cfg, _ac("selective"))
        expected_layers = [f"layers.{i}" for i in range(len(cfg.layers))]
        expected_mtp = [f"mtp.layers.{i}" for i in range(len(cfg.mtp.layers))]
        flat = [fqn for atom in atoms for u in atom for fqn in u.fqns]
        self.assertEqual(
            flat,
            ["tok_embeddings", *expected_layers, "norm", "lm_head", *expected_mtp],
        )
        self.assertEqual([u.fqns for u in atoms[0]], [("tok_embeddings",)])
        self.assertEqual(
            [u.fqns[0] for u in atoms[-1]], ["norm", "lm_head", *expected_mtp]
        )
        for atom in atoms[1:-1]:
            self.assertEqual(len(atom), 1)

    def test_moe_layer_costs_more_than_dense(self):
        """gift_moe debugmodel: layer 0 is dense FFN, later trunk layers MoE."""
        cfg, model = _build("gift_moe")
        by_fqn = self._units_by_fqn(model, cfg, _ac("selective"))
        self.assertGreater(by_fqn["layers.1"].cost, by_fqn["layers.0"].cost)

    def test_ac_scales_blocks_but_not_head(self):
        """Full AC adds recompute to trunk/MTP blocks only; the head (runs
        inside the loss, unwrapped by AC) keeps the no-AC scale."""
        cfg, model = _build("gift_moe")
        none = self._units_by_fqn(model, cfg, _ac("none"))
        full = self._units_by_fqn(model, cfg, _ac("full"))
        self.assertEqual(full["lm_head"].cost, none["lm_head"].cost)
        self.assertEqual(full["tok_embeddings"].cost, none["tok_embeddings"].cost)
        for fqn in ["layers.1", "mtp.layers.0"]:
            self.assertAlmostEqual(full[fqn].cost / none[fqn].cost, 4 / 3)

    def test_lm_head_counts_one_pass_per_mtp_head(self):
        """The head unit is priced once per prediction head (main + MTPs)."""
        cfg, model = _build("gift_moe")
        assert cfg.mtp is not None
        num_heads = len(cfg.mtp.layers)
        self.assertGreater(num_heads, 0)
        units = self._units_by_fqn(model, cfg, _ac("none"))
        one_pass = 3 * 2 * cfg.lm_head.in_features * cfg.lm_head.out_features
        expected = one_pass * (1 + num_heads)
        self.assertAlmostEqual(units["lm_head"].cost, expected, delta=expected * 1e-9)


class TestPipelineLayout(unittest.TestCase):
    def test_pipeline_llm_prices_layer_count_and_explicit_splits(self):
        cfg, model = _build("gift_moe")
        self.assertEqual(cfg.tok_embeddings.fwd_flops_per_token(_SEQ_LEN), 0)
        atoms = _model_atoms(model, cfg, _ac("full"))
        costs = {u.fqns[0]: u.cost for atom in atoms for u in atom}
        explicit = [
            [f for atom in atoms[:-1] for u in atom for f in u.fqns],
            [f for u in atoms[-1] for f in u.fqns],
        ]
        interleaved = [
            [f for atom in atoms[:3] for u in atom for f in u.fqns],
            *[[f for u in atom for f in u.fqns] for atom in atoms[3:]],
        ]
        for schedule, split, indices in (
            ("1F1B", None, [(0,), (1,)]),
            ("1F1B", explicit, [(0,), (1,)]),
            ("Interleaved1F1B", interleaved, [(0, 2, 4), (1, 3, 5)]),
        ):
            with (
                self.subTest(schedule=schedule, explicit=split is not None),
                patch(
                    "torchtitan.distributed.pipeline_parallel._pipeline_module_split",
                    return_value=([], []),
                ),
                patch(
                    "torchtitan.distributed.pipeline_parallel._build_pipeline_schedule"
                ),
                patch(
                    "torchtitan.distributed.pipeline_parallel.log_pipeline_layout"
                ) as log,
            ):
                pipeline_llm(
                    model,
                    parallel_dims=SimpleNamespace(pp=2, get_mesh=lambda _: None),
                    training=TrainingConfig(local_batch_size=8, seq_len=_SEQ_LEN),
                    parallelism=ParallelismConfig(
                        pipeline_parallel_degree=2,
                        pipeline_parallel_schedule=schedule,
                        module_fqns_per_model_part=split,
                    ),
                    compile_config=CompileConfig(),
                    ac_config=_ac("full"),
                    dump_folder="unused",
                    device=torch.device("cpu"),
                    model_config=cfg,
                    parallelize_fn=None,
                    loss_fn=None,
                )
                layout = log.call_args.kwargs
                stages = layout["module_names_per_stage"]
                self.assertEqual(
                    layout["stage_costs"],
                    [sum(costs.get(f, 0.0) for f in stage) for stage in stages],
                )
                self.assertEqual(
                    layout["stage_blocks"],
                    [sum(f in _block_fqns(model) for f in stage) for stage in stages],
                )
                self.assertGreater(sum(layout["stage_costs"]), 0)
                self.assertFalse(layout["costs_are_measured_ms"])
                self.assertEqual(layout["stage_indices_per_rank"], indices)

    def test_layout_costs_and_activation_proxy(self):
        for schedule, expected in (
            ("1F1B", ["6", "6", "4", "1"]),
            ("GPipe", ["6", "6", "6", "3"]),
            ("Interleaved1F1B", ["-", "-", "-", "-"]),
        ):
            with self.subTest(schedule=schedule), self.assertLogs(level="INFO") as logs:
                log_pipeline_layout(
                    module_names_per_stage=[[f"blocks.{i}"] for i in range(4)],
                    stage_indices_per_rank=[(0, 2), (1, 3)],
                    pp_schedule=schedule,
                    num_layers=7,
                    stage_costs=[1.0, 2.0, 3.0, 4.0],
                    stage_blocks=[2, 2, 2, 1],
                    num_microbatches=3,
                    costs_are_measured_ms=True,
                )
            text = logs.output[0]
            rows = [
                [cell.strip() for cell in line.split("|")[1:-1]]
                for line in text.splitlines()
                if line.startswith("|") and "PP rank" not in line
            ]
            self.assertEqual(
                {int(row[1]): row[4] for row in rows}, dict(enumerate(expected))
            )
            self.assertIn("cost, ms", text)
            self.assertIn("Per-rank cost: rank 0: 4.0, rank 1: 6.0", text)
            if schedule == "1F1B":
                self.assertIn(
                    "Per-rank blocks x in-flight: rank 0: 10, rank 1: 7", text
                )
        with self.assertLogs(level="INFO") as logs:
            log_pipeline_layout(
                module_names_per_stage=[["embed"], ["head"]],
                stage_indices_per_rank=[(0,), (1,)],
                pp_schedule="1F1B",
                num_layers=0,
                stage_costs=[0.0, 0.0],
                stage_blocks=[0, 0],
                num_microbatches=1,
            )
        self.assertIn("Per-rank cost: rank 0: -, rank 1: -", logs.output[0])
        self.assertEqual(
            _format_modules_with_ranges(
                ["blocks.0", "blocks.1", "blocks.norm", "head"]
            ),
            "blocks.0-1, blocks.norm, head",
        )


def _atoms(costs, lead_glue=0, trail_glue=0):
    """Single-unit atoms, except the first ``lead_glue`` and last
    ``trail_glue`` units, which are glued into one atom each."""
    units = [PipelineUnit(fqns=(f"u{i}",), cost=c) for i, c in enumerate(costs)]
    n = len(units)
    atoms = [units[:lead_glue]] if lead_glue else []
    atoms += [[u] for u in units[lead_glue : n - trail_glue]]
    if trail_glue:
        atoms.append(units[n - trail_glue :])
    return atoms


def _partition_key(stage_costs):
    """The partitioner's objective: (bottleneck, sum of squared stage costs)."""
    return (max(stage_costs), sum(c * c for c in stage_costs))


def _brute_force_best_key(costs, num_stages):
    """Best key among splits whose last stage is the lightest; when none
    exists, among splits whose last stage is the single last atom (the
    partitioner's fallback)."""
    best_constrained = best_minimal_tail = (float("inf"), float("inf"))
    for cuts in itertools.combinations(range(1, len(costs)), num_stages - 1):
        bounds = [0, *cuts, len(costs)]
        stage_costs = [sum(costs[a:b]) for a, b in itertools.pairwise(bounds)]
        key = _partition_key(stage_costs)
        if stage_costs[-1] <= min(stage_costs):
            best_constrained = min(best_constrained, key)
        if num_stages == 1 or cuts[-1] == len(costs) - 1:
            best_minimal_tail = min(best_minimal_tail, key)
    if best_constrained[0] < float("inf"):
        return best_constrained
    return best_minimal_tail


class TestPartitionMinimax(unittest.TestCase):
    def _check_valid(self, stages, atoms, num_stages):
        self.assertEqual(len(stages), num_stages)
        self.assertTrue(all(stage for stage in stages))
        flat = [fqn for stage in stages for fqn in stage]
        self.assertEqual(flat, [f for atom in atoms for u in atom for f in u.fqns])

    def test_matches_brute_force_on_random_costs(self):
        rng = random.Random(0)
        for trial in range(200):
            n = rng.randint(2, 12)
            num_stages = rng.randint(1, min(n, 5))
            costs = [rng.randint(1, 20) for _ in range(n)]
            atoms = _atoms(costs)
            with self.subTest(trial=trial, costs=costs, num_stages=num_stages):
                stages = partition_minimax(atoms, num_stages)
                self._check_valid(stages, atoms, num_stages)
                cost = {f"u{i}": c for i, c in enumerate(costs)}
                key = _partition_key([sum(cost[f] for f in s) for s in stages])
                self.assertEqual(key, _brute_force_best_key(costs, num_stages))

    def test_uniform_costs_reproduce_even_split(self):
        stages = partition_minimax(_atoms([1] * 8), 4)
        self.assertEqual([len(s) for s in stages], [2, 2, 2, 2])

    def test_slack_goes_to_late_middle_stages_on_ties(self):
        # 5 equal units over 4 stages: the doubled stage is late-middle.
        stages = partition_minimax(_atoms([1] * 5), 4)
        self.assertEqual([len(s) for s in stages], [1, 1, 2, 1])
        # 7 over 4: the last stage is the last to receive slack.
        stages = partition_minimax(_atoms([1] * 7), 4)
        self.assertEqual([len(s) for s in stages], [2, 2, 2, 1])

    def test_last_stage_lightest_on_measured_100b_costs(self):
        # 512-GPU trace regression: the unconstrained optimum loads the last
        # stage (cheap head), stretching fill/drain; keep it the lightest.
        atoms = [[PipelineUnit(("tok_embeddings",), 0.5)]]
        atoms += [[PipelineUnit(("layers.0",), 30.0)]]
        atoms += [[PipelineUnit((f"layers.{i}",), 48.0)] for i in range(1, 40)]
        atoms.append(
            [
                PipelineUnit(("norm",), 0.2),
                PipelineUnit(("lm_head",), 100.0),
                PipelineUnit(("mtp",), 34.0),
            ]
        )
        stages = partition_minimax(atoms, 8)
        cost = {u.fqns[0]: u.cost for atom in atoms for u in atom}
        stage_costs = [sum(cost[f] for f in s) for s in stages]
        self.assertLessEqual(stage_costs[-1], min(stage_costs))
        layer_counts = [len([f for f in s if f.startswith("layers")]) for s in stages]
        self.assertEqual(layer_counts, [6, 5, 5, 5, 5, 6, 6, 2])

    def test_heavy_tail_falls_back_to_minimal_last_stage(self):
        # A tail heavier than any achievable stage makes the constraint
        # unsatisfiable; the last stage then holds just the tail atom.
        atoms = _atoms([1, 1, 1, 100], trail_glue=1)
        stages = partition_minimax(atoms, 3)
        self._check_valid(stages, atoms, 3)
        self.assertEqual(stages[-1], ["u3"])

    def test_glued_runs_stay_on_edge_stages(self):
        # A heavy trailing glued run must not be split off the last stage even
        # though splitting would lower the bottleneck.
        atoms = _atoms([1, 1, 1, 10, 10], trail_glue=2)
        stages = partition_minimax(atoms, 2)
        self._check_valid(stages, atoms, 2)
        self.assertEqual(stages[1][-2:], ["u3", "u4"])

        atoms = _atoms([10, 10, 1, 1, 1], lead_glue=2)
        stages = partition_minimax(atoms, 2)
        self.assertEqual(stages[0][:2], ["u0", "u1"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            partition_minimax(_atoms([1, 2]), 0)
        with self.assertRaises(ValueError):
            # 4 units glued into 2 atoms < 3 stages.
            partition_minimax(_atoms([1, 1, 1, 1], lead_glue=2, trail_glue=2), 3)

    def test_flops_balanced_module_fqns_end_to_end(self):
        """The balancer returns a valid FQN cover on a gift_moe config."""
        cfg, model = _build("gift_moe")
        parallelism = ParallelismConfig(pipeline_parallel_balancing="flops")
        stages = _balanced(model, cfg, parallelism, _ac("selective"), 4)
        self.assertEqual(len(stages), 4)
        flat = [fqn for stage in stages for fqn in stage]
        expected_mtp = [f"mtp.layers.{i}" for i in range(len(cfg.mtp.layers))]
        expected = (
            ["tok_embeddings"]
            + [f"layers.{i}" for i in range(len(cfg.layers))]
            + ["norm", "lm_head", *expected_mtp]
        )
        self.assertEqual(flat, expected)
        self.assertEqual(stages[0][0], "tok_embeddings")
        tail = ["norm", "lm_head", *expected_mtp]
        self.assertEqual(stages[-1][-len(tail) :], tail)

    def test_single_stage_gets_everything(self):
        atoms = _atoms([3, 1, 2], lead_glue=1, trail_glue=1)
        self.assertEqual(partition_minimax(atoms, 1), [["u0", "u1", "u2"]])


class TestCostOverrides(unittest.TestCase):
    """User-measured module costs (pipeline_parallel_module_costs_ms)."""

    def _parallelism(self, costs_ms, **kw):
        return ParallelismConfig(
            pipeline_parallel_balancing="flops",
            pipeline_parallel_module_costs_ms=costs_ms,
            **kw,
        )

    def _full_cover(self, **overrides):
        costs = {
            "tok_embeddings": 0.1,
            "layers": 6.0,
            "norm": 0.1,
            "lm_head": 2.0,
            "mtp": 6.0,
        }
        costs.update(overrides)
        return costs

    def test_measured_costs_drive_the_split(self):
        """A layer measured far heavier than its FLOPs suggest gets its
        own stage (the last body layer — isolable under the constraint)."""
        cfg, model = _build("gift_moe")
        num_layers = len(cfg.layers)
        heavy = f"layers.{num_layers - 1}"
        # Dict order matters: the specific key must precede the "layers" prefix.
        costs = {heavy: 1000.0}
        costs.update(self._full_cover())
        stages = _balanced(model, cfg, self._parallelism(costs), _ac("none"), 3)
        self.assertIn([heavy], stages)

    def test_only_ratios_matter(self):
        cfg, model = _build("gift_moe")
        ac = _ac("none")
        base = self._full_cover()
        scaled = {k: v * 1e3 for k, v in base.items()}
        self.assertEqual(
            _balanced(model, cfg, self._parallelism(base), ac, 3),
            _balanced(model, cfg, self._parallelism(scaled), ac, 3),
        )

    def test_incomplete_coverage_rejected(self):
        cfg, model = _build("gift_moe")
        costs = self._full_cover()
        del costs["lm_head"]
        with self.assertRaisesRegex(ValueError, "missing costs for.*lm_head"):
            _balanced(model, cfg, self._parallelism(costs), _ac("none"), 3)

    def test_unmatched_key_rejected(self):
        cfg, model = _build("gift_moe")
        costs = self._full_cover(lmhead=2.0)  # typo: no dot-prefix match
        with self.assertRaisesRegex(ValueError, "match no pipeline module.*lmhead"):
            _balanced(model, cfg, self._parallelism(costs), _ac("none"), 3)

    def test_negative_cost_rejected(self):
        cfg, model = _build("gift_moe")
        with self.assertRaisesRegex(ValueError, "negative"):
            _balanced(
                model,
                cfg,
                self._parallelism(self._full_cover(norm=-1.0)),
                _ac("none"),
                3,
            )

    def test_flops_rejects_layers_per_stage(self):
        with self.assertRaisesRegex(ValueError, "num_stages_per_rank"):
            ParallelismConfig(
                pipeline_parallel_balancing="flops",
                pipeline_parallel_layers_per_stage=10,
            )

    def test_num_stages_per_rank_and_layers_per_stage_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            ParallelismConfig(
                pipeline_parallel_num_stages_per_rank=2,
                pipeline_parallel_layers_per_stage=10,
            )

    def test_num_stages_per_rank_drives_metadata(self):
        from torchtitan.distributed.pipeline_parallel import _get_pipeline_metadata

        parallelism = ParallelismConfig(
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="Interleaved1F1B",
            pipeline_parallel_balancing="flops",
            pipeline_parallel_num_stages_per_rank=2,
        )
        dims = SimpleNamespace(pp=4)
        num_stages, _, _, _ = _get_pipeline_metadata(dims, parallelism, 40)
        self.assertEqual(num_stages, 8)

    def test_num_stages_per_rank_respects_schedule(self):
        from torchtitan.distributed.pipeline_parallel import _get_pipeline_metadata

        single = ParallelismConfig(
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="1F1B",
            pipeline_parallel_num_stages_per_rank=2,
        )
        with self.assertRaisesRegex(ValueError, "exactly 1 stage per rank"):
            _get_pipeline_metadata(SimpleNamespace(pp=4), single, 40)
        multi = ParallelismConfig(
            pipeline_parallel_degree=4,
            pipeline_parallel_schedule="Interleaved1F1B",
            pipeline_parallel_num_stages_per_rank=1,
        )
        with self.assertRaisesRegex(ValueError, "at least 2 stages per rank"):
            _get_pipeline_metadata(SimpleNamespace(pp=4), multi, 40)


if __name__ == "__main__":
    unittest.main()
