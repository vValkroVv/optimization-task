import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

import ep8


class CaptureChecks(unittest.TestCase):
    def test_local_tensor_respects_ragged_shard_to_local_contract(self):
        tensor = torch.ones(2, requires_grad=True)

        def to_local():
            if torch.is_grad_enabled():
                raise NotImplementedError("ragged shard local view requires no grad")
            return tensor

        with torch.enable_grad():
            local = ep8.local_tensor(SimpleNamespace(to_local=to_local))
            self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(local.requires_grad)
        self.assertEqual(local.data_ptr(), tensor.data_ptr())

    def test_noncontiguous_initial_digest_is_logical_and_bounded(self):
        tensor = torch.arange(30, dtype=torch.int64).view(5, 6).T
        expected = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
        with patch.object(ep8, "CHUNK_BYTES", 16):
            self.assertEqual(ep8.StateDigest().tensor(tensor)["sha256"], expected)
            self.assertTrue(all(chunk.numel() <= 2 for chunk in ep8.chunks(tensor)))
        with patch.object(ep8, "INITIAL_BYTES", 8):
            with self.assertRaisesRegex(RuntimeError, "read budget"):
                ep8.StateDigest().tensor(tensor)

    def test_before_reduce_copies_live_view_and_checks_alignment(self):
        sample_name = ep8.NAMES[0]
        arena = torch.arange(8, dtype=torch.float32)
        with patch.dict("os.environ", {"RANK": "0"}):
            observation = ep8.Observation("/unused", [], "example")
        observation.microbatch = 2
        observation.indices[sample_name] = object()
        sharded_parameter = object()
        observation.names_by_parameter[id(sharded_parameter)] = sample_name
        calls = {}

        def copy(key, value, indices):
            self.assertIs(value, arena)
            self.assertIs(indices, observation.indices[sample_name])
            calls[key] = value.clone()

        observation.sample = copy
        param = SimpleNamespace(
            sharded_param=sharded_parameter,
            _param_fqn="layers.1._checkpoint_wrapped_module._orig_mod.moe.experts.w13_E2FD",
            _unsharded_accumulated_grad=arena,
        )
        with self.assertRaisesRegex(RuntimeError, "eFSDP=1"):
            observation.before_reduce([param], SimpleNamespace(size=lambda: 2))
        observation.before_reduce([param], SimpleNamespace(size=lambda: 1))
        arena.zero_()
        self.assertEqual(list(calls), [f"pre_reduce/02/{sample_name}"])
        torch.testing.assert_close(
            next(iter(calls.values())), torch.arange(8, dtype=torch.float32)
        )

    def test_first_step_refuses_to_run_before_config_validation(self):
        with patch.dict("os.environ", {"RANK": "0"}):
            observation = ep8.Observation("/unused", [], "example")
        with self.assertRaisesRegex(RuntimeError, "must pass validation"):
            observation.start_update(None)

    def test_config_digest_only_ignores_output_folder(self):
        config = {"debug": {"deterministic": True}, "compile": {"enable": True}}
        expected = ep8.resolved_digest(config)
        self.assertEqual(
            expected, ep8.resolved_digest({**config, "dump_folder": "/other"})
        )
        self.assertNotEqual(
            expected, ep8.resolved_digest({**config, "compile": {"enable": False}})
        )

    def test_compare_requires_exact_initial_state_and_reports_float_delta(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(ep8, "PROCESSES", 1),
            patch.object(ep8, "SAMPLES", 2),
        ):
            a = self.make_run(directory, "a")
            a2 = self.make_run(directory, "a2")
            b = self.make_run(directory, "b", selected=ep8.OPTIMIZATIONS, loss=2.0)
            self.assertTrue(all(row["exact"] for row in ep8.compare(a, a2, True)))
            rows = ep8.compare(a, b, False)
            loss = next(row for row in rows if row["field"] == "loss")
            self.assertFalse(loss["exact"])
            self.assertEqual(loss["max_abs"], 1.0)
            wrong = self.make_run(directory, "wrong-initial", initial="other")
            with self.assertRaisesRegex(RuntimeError, "initial state differs"):
                ep8.compare(a, wrong, True)
            omitted = self.make_run(directory, "omitted", missing="loss")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                ep8.compare(a, omitted, True)

    def test_compare_validates_native_router_vectors_separately_from_weights(self):
        bias = "pre_clip/layers.1.moe.expert_bias_E"
        count = "post_optimizer/layers.1.moe.tokens_per_expert_E"
        router_arrays = {
            bias: np.zeros(128, dtype=np.float32),
            count: np.ones(128, dtype=np.float32),
        }
        with (
            TemporaryDirectory() as directory,
            patch.object(ep8, "PROCESSES", 1),
            patch.object(ep8, "SAMPLES", 2),
        ):
            a = self.make_run(directory, "a", overrides=router_arrays)
            a2 = self.make_run(directory, "a2", overrides=router_arrays)
            measured = {row["field"]: row["exact"] for row in ep8.compare(a, a2, True)}
            self.assertTrue(measured[bias] and measured[count])
            for index, (field, invalid) in enumerate(
                (
                    (bias, np.zeros(127, dtype=np.float32)),
                    (count, np.zeros(128, dtype=np.float64)),
                    (f"pre_clip/{ep8.NAMES[0]}", np.zeros(128, dtype=np.float32)),
                )
            ):
                with self.subTest(field=field):
                    malformed = self.make_run(
                        directory,
                        f"malformed-{index}",
                        overrides={**router_arrays, field: invalid},
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "invalid native sampled tensor"
                    ):
                        list(ep8.load_run(malformed))

    def make_run(
        self,
        directory,
        name,
        selected=(),
        loss=1.0,
        initial="same",
        missing=None,
        overrides=None,
    ):
        run = Path(directory) / name
        config_path = run / "native/config/resolved_config.yaml"
        config_path.parent.mkdir(parents=True)
        config = {
            "training": {"steps": 1},
            "parallelism": {},
            "dataloader": {"dataset": {"streams": [{"local": str(run / "data")}]}},
            "dump_folder": str(run / "native"),
        }
        config_path.write_text(yaml.safe_dump(config))
        if not selected:
            config["dataloader"]["dataset"]["streams"][0]["local"] = "a-data"
            config_path.write_text(yaml.safe_dump(config))
        source = {"tracked_and_untracked_sha256": "b" if selected else "a"}
        metadata = {
            "source_snapshot_before": source,
            "steps": 1,
            "warmup_updates": 0,
            "window_updates": 0,
            "source_head_before": ep8.ORIGINAL_REVISION,
            "config_hashes_before": {"expert_gfsdp_numerics_sha256": "harness"},
            "compiler_cache_policy": "fresh",
            "input_data": {field: "verified" for field in ep8.INPUT_FIELDS},
        }
        completion = {
            "status": "PASS",
            "resolved_config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
        }
        for filename, contents in (
            ("metadata", metadata),
            ("completion", completion),
            ("source-final", source),
        ):
            (run / f"{filename}.json").write_text(json.dumps(contents))
        stem = run / "measurement" / ep8.RECEIPT.format(rank=0)
        stem.parent.mkdir()
        arrays = {
            key: np.ones((2,), dtype=np.float32) for key in ep8.required_samples()
        }
        arrays["loss"] = np.array(loss, dtype=np.float32)
        arrays["loss_per_microbatch"] = np.ones((ep8.MICROBATCHES,), dtype=np.float32)
        if overrides is not None:
            arrays.update(overrides)
        if missing:
            arrays.pop(missing)
        np.savez(stem.with_suffix(".npz"), **arrays)
        raw = stem.with_suffix(".npz").read_bytes()
        receipt = {
            "version": 1,
            "rank": 0,
            "resolved_config_digest": ep8.resolved_digest(config),
            "microbatches": ep8.MICROBATCHES,
            "layers": ep8.LAYERS,
            "samples_per_weight": 2,
            "payload_bytes": len(raw),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "requested": list(selected),
            "selected_local_numel": {"example": 10},
            "initial_state": {"sha256": initial},
            "final_state": {"data_sha256": "data", "scheduler": {}},
        }
        stem.with_suffix(".json").write_text(json.dumps(receipt))
        return run


if __name__ == "__main__":
    unittest.main()
