"""Exercise the EP4 observer and actual untrained native launch parsers."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import ep4


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNNER = Path("runs/efsdp-optimizations/run.py")
OVERLAY = Path("runs/efsdp-optimizations/numerics/ep4-owner/native-ep4-owner.patch")
INHERITED = OVERLAY.parents[1] / "native-numerics.patch"
FEATURES = tuple(
    arg for flag in ep4.shared.OPTIMIZATIONS for arg in ("--optimization", flag)
)


def parser_sandbox(destination):
    for name in ("production-data-abcd", "flash-qla", "overnight-streaming-perf"):
        source = ROOT / "runs" / name
        target = destination / "runs" / name
        target.mkdir(parents=True)
        for path in source.iterdir():
            if path.name != "artifacts" and not path.name.startswith("."):
                if path.suffix == ".py":
                    shutil.copy2(path, target / path.name)
                else:
                    (target / path.name).symlink_to(path)
    (destination / "gift").symlink_to(ROOT / "gift")
    (destination / RUNNER).parent.mkdir()
    shutil.copy2(ROOT / RUNNER, destination / RUNNER)
    (destination / RUNNER.parent / "numerics").symlink_to(HERE.parent)
    (destination / RUNNER.parent / "direct-dw").symlink_to(
        ROOT / RUNNER.parent / "direct-dw"
    )
    for patch_path in (INHERITED, OVERLAY):
        subprocess.run(
            ["git", "apply", "--check", str(ROOT / patch_path)],
            cwd=destination,
            check=True,
        )
        subprocess.run(
            ["git", "apply", str(ROOT / patch_path)], cwd=destination, check=True
        )


class Checks(unittest.TestCase):
    def test_flat_owner_ranges_and_empty_optimizer_state(self):
        w13, w2 = ep4.shared.NAMES[:2]
        for rank in range(8):
            expected = (301989888, 0) if rank < 4 else (100663296, 201326592)
            self.assertEqual(
                tuple(ep4.owner_numel(rank, name) for name in (w13, w2)), expected
            )
        self.assertIn(f"pre_reduce/00/{w2}", ep4.required_for_rank(0))
        self.assertNotIn(f"pre_clip/{w2}", ep4.required_for_rank(0))
        self.assertIn(f"pre_clip/{w2}", ep4.required_for_rank(4))

        with patch.dict("os.environ", {"RANK": "0"}):
            obs = ep4.Observation(
                "/unused", ep4.shared.OPTIMIZATIONS, "digest", ep4.COMPACT
            )
        params = [torch.nn.Parameter(torch.empty(0)) for _ in obs.empty_owner]
        optimizer = torch.optim.AdamW(params, fused=True)
        for name, parameter in zip(obs.empty_owner, params, strict=True):
            obs.owners[name] = optimizer, parameter
            obs.empty_owner[name]["parameter"] = "empty_fp32"
        params[0].grad = torch.empty(0)
        optimizer.step()
        with (
            patch.object(ep4.shared.Observation, "gradient_shards"),
            patch.object(ep4.shared.Observation, "post_optimizer"),
        ):
            obs.gradient_shards("pre_clip")
            obs.gradient_shards("post_clip")
            obs.post_optimizer(None)

        self.assertEqual(obs.empty_owner[w2]["adam"], "empty_moments")
        self.assertEqual(obs.samples[f"optimizer_step/{w2}"].item(), 1)
        self.assertEqual(
            next(r for n, r in obs.empty_owner.items() if n != w2)["adam"], "absent"
        )
        values = {field: None for field in ep4.required_for_rank(0)}
        values.update({k: v.numpy() for k, v in obs.samples.items()})
        ep4.validate_samples(0, obs.empty_owner, values)
        del values[f"optimizer_step/{w2}"]
        with self.assertRaisesRegex(RuntimeError, "Adam counter"):
            ep4.validate_samples(0, obs.empty_owner, values)

    def test_full_slot_sampling_preserves_fp32_and_exact_bf16_widening(self):
        produced = torch.arange(64, dtype=torch.float32).view(4, 16) / 128
        indices = torch.tensor([0, 19, 32, 63])
        expected = produced.view(-1).index_select(0, indices)
        for compact in (False, True):
            source = produced.to(torch.bfloat16) if compact else produced.clone()
            capture = ep4.sample_full(source, (4, 16), indices, compact)
            source.zero_()
            self.assertEqual(capture.dtype, torch.float32)
            self.assertTrue(
                torch.equal(capture.view(torch.int32), expected.view(torch.int32))
            )
        for source, shape, compact in (
            (produced, (4, 16), True),
            (produced, (2, 32), False),
            (produced.T, (16, 4), False),
        ):
            with self.assertRaisesRegex(RuntimeError, "shape, dtype or layout"):
                ep4.sample_full(source, shape, indices, compact)

    def test_ep4_packed_collective_verification_rejects_wrong_peer_offset(self):
        with patch.dict("os.environ", {"RANK": "4"}):
            obs = ep4.Observation(
                "/unused", ep4.shared.OPTIMIZATIONS, "digest", ep4.COMPACT
            )
        group = SimpleNamespace(rank=lambda: 1)
        slot = torch.zeros(16, dtype=torch.bfloat16)
        receive = torch.empty(4, dtype=torch.bfloat16)
        splits = [4, 0]
        sent = []

        def actual(output, input, **kwargs):
            self.assertIs(output, receive)
            sent.append(input.data_ptr())

        args = {
            "group": group,
            "input_split_sizes": splits,
            "output_split_sizes": splits,
            "async_op": False,
        }
        with patch.multiple(ep4, CHUNK=4, OWNER_ELEMENTS=8):
            history = []
            obs.expert_context.set((group, slot, history))
            with self.assertRaisesRegex(RuntimeError, "physical BF16"):
                obs.collective(ep4.COMPACT, actual, receive, slot[8:12], **args)
            for offset in (0, 4):
                obs.collective(
                    ep4.COMPACT, actual, receive, slot[offset : offset + 4], **args
                )
            self.assertEqual(history, [ep4.COMPACT, ep4.COMPACT])
            self.assertEqual(sent, [slot.data_ptr(), slot.data_ptr() + 8])

    def test_compare_requires_initial_input_and_counter_identity(self):
        import numpy as np

        arrays = {
            key: np.array(1, dtype=np.int64)
            for key in (
                "global_valid_tokens",
                "tokens_seen",
                *(f"optimizer_step/{n}" for n in ep4.shared.NAMES),
            )
        }
        arrays["loss"] = np.array(1.0, dtype=np.float32)
        receipt = {
            **dict.fromkeys(
                (
                    "recipe",
                    "input_provenance",
                    "helper",
                    "shared_helper_and_build",
                    "compiler_cache_policy",
                ),
                "same",
            ),
            "rank": 0,
            "initial_state": {"sha256": "same"},
            "final_state": {
                "data_sha256": "same",
                "scheduler": "same",
                "empty_owner": {},
            },
        }
        candidate = {**arrays, "loss": np.array(1.5, dtype=np.float32)}
        rows = ep4.compare([(receipt, arrays)], [(receipt, candidate)])
        self.assertEqual(next(r for r in rows if r["field"] == "loss")["max_abs"], 0.5)
        mismatch = copy.deepcopy(receipt)
        mismatch["final_state"]["data_sha256"] = "other"
        with self.assertRaisesRegex(RuntimeError, "input or scheduler"):
            ep4.compare([(receipt, arrays)], [(mismatch, arrays)])
        with self.assertRaisesRegex(RuntimeError, "exact counter"):
            ep4.compare(
                [(receipt, arrays)], [(receipt, {**arrays, "tokens_seen": np.array(2)})]
            )

    def test_real_launcher_and_native_rank_parsers_both_roles(self):
        with TemporaryDirectory(prefix="ep4-owner-plan-") as directory:
            temp = Path(directory)
            parser_sandbox(temp)
            for role in ep4.ROLES:
                with self.subTest(role=role):
                    command = [
                        sys.executable,
                        "-B",
                        str(temp / RUNNER),
                        "plan",
                        f"ep4-{role}",
                        "--ep",
                        "4",
                        "--numerical-ep4-owner",
                        role,
                        "--direct-dw-build",
                        str(temp / "unused-build"),
                        *FEATURES,
                    ]
                    plan = json.loads(subprocess.check_output(command, cwd=temp))
                    argv = plan["argv"]
                    rank = argv.index("rank")
                    self.assertEqual(argv[rank + 2], "4")
                    self.assertEqual(
                        argv[rank + 6 : rank + 10],
                        [
                            "--numerical-config-digest",
                            "",
                            "--numerical-ep4-owner",
                            role,
                        ],
                    )
                    self.assertEqual(argv[argv.index("--training.steps") + 1], "1")
                    self.assertEqual(
                        argv[argv.index("--debug.deterministic") + 1], "true"
                    )
                    self.assertEqual(
                        argv[argv.index("--parallelism.expert_parallel_degree") + 1],
                        "4",
                    )


if __name__ == "__main__":
    unittest.main()
