# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

from torchtitan.config.decode import decode
from torchtitan.config.errors import ConfigFieldError, ConfigSchemaError
from torchtitan.config.schema.tools.profiler import ProfilerConfig
from torchtitan.tools.profiler import Profiler


class TestProfilerConfig(unittest.TestCase):
    def test_default_field_values(self):
        cfg = Profiler.Config()
        self.assertFalse(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "profiling/traces")
        self.assertEqual(cfg.profile_freq, 10)
        self.assertEqual(cfg.profiler_active, 1)
        self.assertEqual(cfg.profiler_warmup, 3)
        self.assertIsNone(cfg.profiler_repeat)
        self.assertIsNone(cfg.profiler_skip_first)
        self.assertIsNone(cfg.profiler_skip_first_wait)
        self.assertFalse(cfg.enable_memory_snapshot)
        self.assertEqual(cfg.save_memory_snapshot_folder, "profiling/memory_snapshot")
        self.assertIsNone(cfg.profile_ranks)
        self.assertTrue(cfg.record_shapes)

    def test_custom_field_values(self):
        cfg = Profiler.Config(
            enable_profiling=True,
            save_traces_folder="my_traces",
            profile_freq=50,
            profiler_repeat=2,
            profiler_skip_first=5,
            profiler_skip_first_wait=3,
            profile_ranks=[0, 2],
            record_shapes=False,
        )
        self.assertTrue(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "my_traces")
        self.assertEqual(cfg.profile_freq, 50)
        self.assertEqual(cfg.profiler_repeat, 2)
        self.assertEqual(cfg.profiler_skip_first, 5)
        self.assertEqual(cfg.profiler_skip_first_wait, 3)
        self.assertEqual(cfg.profile_ranks, [0, 2])
        self.assertFalse(cfg.record_shapes)

    def test_rank_controls_decode_and_reject_invalid_values(self):
        cfg = decode(
            ProfilerConfig,
            {"profile_ranks": [0, 2], "record_shapes": False},
            path="profiler",
        )
        self.assertEqual(cfg.profile_ranks, [0, 2])
        self.assertFalse(cfg.record_shapes)

        for ranks in ([], [-1], [0, 0], [True]):
            with self.subTest(profile_ranks=ranks):
                with self.assertRaises(ConfigSchemaError):
                    decode(ProfilerConfig, {"profile_ranks": ranks}, path="profiler")

        with self.assertRaises(ConfigFieldError):
            ProfilerConfig(profile_ranks=(0,))

        with self.assertRaisesRegex(ConfigSchemaError, r"profiler\.record_shape"):
            decode(
                ProfilerConfig,
                {"record_shape": False},
                path="profiler",
            )

    def test_build_returns_profiler_instance(self):
        """Profiler.Config.build() auto-wires to Profiler via Configurable."""
        cfg = Profiler.Config()
        profiler = cfg.build()
        self.assertIsInstance(profiler, Profiler)


class TestProfilerInit(unittest.TestCase):
    def test_default_runtime_attrs(self):
        """Profiler initializes runtime attrs to safe defaults."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)


class TestProfilerDisabledPaths(unittest.TestCase):
    """Tests for the no-op / disabled paths that require no GPU."""

    def test_build_torch_profiler_disabled_returns_none(self):
        """build_torch_profiler returns None when profiling is disabled."""
        profiler = Profiler(Profiler.Config(enable_profiling=False))
        result = profiler.build_torch_profiler(
            global_step=0, base_folder="/tmp", leaf_folder=""
        )
        self.assertIsNone(result)

    def test_build_memory_profiler_disabled_returns_none(self):
        """build_memory_profiler returns None when memory snapshot is disabled."""
        profiler = Profiler(Profiler.Config(enable_memory_snapshot=False))
        with mock.patch(
            "torchtitan.tools.profiler.device_module.memory._record_memory_history"
        ) as record_memory_history:
            result = profiler.build_memory_profiler(
                global_step=0, base_folder="/tmp", leaf_folder=""
            )
        self.assertIsNone(result)
        record_memory_history.assert_not_called()

    def test_runtime_args_stored_on_init(self):
        """Runtime kwargs passed to __init__ are stored on the instance."""
        profiler = Profiler(
            Profiler.Config(), global_step=42, base_folder="/data", leaf_folder="sub"
        )
        self.assertEqual(profiler._global_step, 42)
        self.assertEqual(profiler._base_folder, "/data")
        self.assertEqual(profiler._leaf_folder, "sub")

    def test_context_manager_step_is_noop(self):
        """With everything disabled, context manager and step() don't raise."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            self.assertIs(prof, profiler)
            self.assertIsNone(prof.torch_profiler)
            self.assertIsNone(prof.memory_profiler)
            prof.step()
            prof.step()

    def test_default_args_context_manager(self):
        """Profiler with default runtime args works as a context manager."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            prof.step()

    def test_step_noop_when_both_profilers_none(self):
        """step() is a no-op when torch_profiler and memory_profiler are both None."""
        profiler = Profiler(Profiler.Config())
        profiler.step()
        profiler.step()

    def test_exit_resets_profiler_attrs(self):
        """After __exit__, torch_profiler and memory_profiler are reset to None."""
        profiler = Profiler(Profiler.Config())
        with profiler:
            pass
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)

    def test_active_updates_runtime_args(self):
        """active() updates runtime args and returns self for context manager use."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")

        result = profiler.active(
            global_step=10, base_folder="/output", leaf_folder="replica_0"
        )
        self.assertIs(result, profiler)
        self.assertEqual(profiler._global_step, 10)
        self.assertEqual(profiler._base_folder, "/output")
        self.assertEqual(profiler._leaf_folder, "replica_0")

    def test_active_as_context_manager(self):
        """active() can be used as a context manager with 'with' statement."""
        profiler = Profiler(Profiler.Config())
        with profiler.active(global_step=5, base_folder="/tmp") as prof:
            self.assertIs(prof, profiler)
            self.assertEqual(prof._global_step, 5)
            prof.step()


class TestProfilerEnabledPaths(unittest.TestCase):
    """Tests for enabled profiler paths — uses mocked distributed rank."""

    def setUp(self):
        self.patcher_rank = mock.patch("torch.distributed.get_rank", return_value=0)
        self.patcher_rank.start()

    def tearDown(self):
        self.patcher_rank.stop()

    def test_build_torch_profiler_returns_active_handle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_freq=4,
                    profiler_warmup=1,
                    profiler_active=1,
                ),
                global_step=0,
                base_folder=tmpdir,
            )
            with profiler:
                self.assertIsNotNone(profiler.torch_profiler)

    def test_selected_rank_passes_profile_controls(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_ranks=[1],
                    record_shapes=False,
                    profile_freq=4,
                    profiler_warmup=1,
                    profiler_active=1,
                ),
                base_folder=tmpdir,
            )
            native_profiler = mock.MagicMock()
            with (
                mock.patch("torch.distributed.is_initialized", return_value=True),
                mock.patch("torch.distributed.get_rank", return_value=1),
                mock.patch("torch.distributed.get_world_size", return_value=2),
                mock.patch(
                    "torchtitan.tools.profiler.torch.profiler.profile",
                    return_value=native_profiler,
                ) as make_profiler,
            ):
                result = profiler.build_torch_profiler(
                    global_step=0, base_folder=tmpdir, leaf_folder=""
                )

        self.assertIs(result, native_profiler)
        self.assertTrue((native_profiler.__enter__).called)
        kwargs = make_profiler.call_args.kwargs
        self.assertFalse(kwargs["record_shapes"])
        self.assertFalse(kwargs["with_stack"])
        self.assertFalse(kwargs["profile_memory"])

    def test_default_rank_and_shape_settings_keep_all_rank_capture(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_freq=4,
                    profiler_warmup=1,
                    profiler_active=1,
                ),
                base_folder=tmpdir,
            )
            native_profiler = mock.MagicMock()
            with (
                mock.patch("torch.distributed.is_initialized", return_value=True),
                mock.patch("torch.distributed.get_rank", return_value=1),
                mock.patch("torch.distributed.get_world_size", return_value=2),
                mock.patch(
                    "torchtitan.tools.profiler.torch.profiler.profile",
                    return_value=native_profiler,
                ) as make_profiler,
            ):
                result = profiler.build_torch_profiler(
                    global_step=0, base_folder=tmpdir, leaf_folder=""
                )

        self.assertIs(result, native_profiler)
        self.assertTrue(make_profiler.called)
        self.assertTrue(make_profiler.call_args.kwargs["record_shapes"])

    def test_nonselected_rank_skips_trace_profiler_but_keeps_memory_snapshot(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_ranks=[1],
                    enable_memory_snapshot=True,
                    save_traces_folder="traces",
                    save_memory_snapshot_folder="snapshots",
                ),
                base_folder=tmpdir,
            )
            with (
                mock.patch("torch.distributed.is_initialized", return_value=True),
                mock.patch("torch.distributed.get_rank", return_value=0),
                mock.patch("torch.distributed.get_world_size", return_value=2),
                mock.patch(
                    "torchtitan.tools.profiler.torch.profiler.profile"
                ) as make_profiler,
                mock.patch(
                    "torchtitan.tools.profiler.device_module.memory._record_memory_history"
                ) as record_memory_history,
            ):
                with profiler:
                    self.assertIsNone(profiler.torch_profiler)
                    self.assertIsNotNone(profiler.memory_profiler)

            make_profiler.assert_not_called()
            record_memory_history.assert_called_once()
            self.assertFalse((Path(tmpdir) / "traces").exists())
            self.assertTrue((Path(tmpdir) / "snapshots").is_dir())

    def test_profile_rank_outside_world_fails_before_trace_setup(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(enable_profiling=True, profile_ranks=[2]),
                base_folder=tmpdir,
            )
            with (
                mock.patch("torch.distributed.is_initialized", return_value=True),
                mock.patch("torch.distributed.get_rank", return_value=0),
                mock.patch("torch.distributed.get_world_size", return_value=2),
                mock.patch(
                    "torchtitan.tools.profiler.torch.profiler.profile"
                ) as make_profiler,
            ):
                with self.assertRaisesRegex(ValueError, r"world size 2: \[2\]"):
                    profiler.build_torch_profiler(
                        global_step=0, base_folder=tmpdir, leaf_folder=""
                    )

            make_profiler.assert_not_called()
            self.assertFalse((Path(tmpdir) / "profiling/traces").exists())

    def test_native_cpu_schedule_exports_only_the_target_update(self):
        import json
        import tempfile
        from pathlib import Path

        import torch

        num_updates = 6
        profiler = Profiler(
            Profiler.Config(
                enable_profiling=True,
                profile_freq=2,
                profiler_warmup=1,
                profiler_active=1,
                profiler_repeat=1,
                profiler_skip_first=num_updates - 2,
                profiler_skip_first_wait=0,
                profile_ranks=[0],
                record_shapes=False,
                save_traces_folder="traces",
            ),
            global_step=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler.active(base_folder=tmpdir)
            with profiler:
                for update in range(1, num_updates + 1):
                    with torch.profiler.record_function(f"update_{update}"):
                        for microbatch in range(32):
                            with torch.profiler.record_function(
                                f"microbatch_{update}_{microbatch}"
                            ):
                                pass
                        with torch.profiler.record_function(f"optimizer_{update}"):
                            pass
                    profiler.step()

            traces = sorted(Path(tmpdir).rglob("*.json"))
            self.assertEqual(
                [path.relative_to(tmpdir).as_posix() for path in traces],
                [f"traces/iteration_{num_updates}/rank0_trace.json"],
            )
            events = json.loads(traces[0].read_text())["traceEvents"]
            names = [event.get("name") for event in events]
            marker_names = {
                name
                for name in names
                if isinstance(name, str)
                and name.startswith(("update_", "microbatch_", "optimizer_"))
            }
            expected_microbatches = {
                f"microbatch_{num_updates}_{microbatch}" for microbatch in range(32)
            }
            self.assertEqual(
                marker_names,
                {f"update_{num_updates}", f"optimizer_{num_updates}"}
                | expected_microbatches,
            )
            self.assertEqual(names.count(f"update_{num_updates}"), 1)
            self.assertEqual(names.count(f"optimizer_{num_updates}"), 1)


if __name__ == "__main__":
    unittest.main()
