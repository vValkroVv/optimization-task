import json
import os
from pathlib import Path

import pytest
import requests
from _pytest.monkeypatch import MonkeyPatch

from node_runner.src.config_model import (
    RouterConfig,
    SglangInferenceConfig,
)
from node_runner.src.inference.sglang.sglang_engine import SglangEngine  # isort:skip
from node_runner.src.inference.sglang.sglang_inference import (  # isort:skip
    InferenceStartError,
    SglangInference,
)
from node_runner.src.logger import Logger
from node_runner.src.trace import MetricsScraper
from node_runner.src.inference.sglang.sglang_router_manager import SglangRouter

pytestmark = pytest.mark.unit

MTP_SPECULATIVE_ARGS = {
    "--speculative-algorithm": "EAGLE",
    "--speculative-num-steps": "2",
    "--speculative-eagle-topk": "1",
    "--speculative-num-draft-tokens": "3",
}

TIMEOUT_ENVS = (
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT",
)


def _write_file(path: Path, content: str = "data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _collect_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.exists():
        return result
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            full_path = Path(dirpath) / filename
            result[str(full_path.relative_to(root))] = full_path.read_text()
    return result


def _inference(**kwargs) -> SglangInference:
    return SglangInference(
        **{
            "hf_checkpoint_path": Path("/m"),
            "inference_config": SglangInferenceConfig(tp=1),
            "port": 1,
            "url": "http://x",
            "nnodes": 1,
            "node_rank": 0,
            "dist_init_address": "x:2",
            **kwargs,
        }
    )


@pytest.fixture(autouse=True)
def _dg_cache_in_tmp(monkeypatch: MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("PAM_DG_CACHE_DIR", str(tmp_path / "dg_cache" / "nfs"))
    monkeypatch.setattr(
        SglangInference,
        "_get_dg_cache_dir_local",
        lambda self: tmp_path / "dg_cache" / "local",
    )


@pytest.fixture(autouse=True)
def _skip_router_subprocess(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(SglangRouter, "start_router", lambda self, worker_urls: None)


@pytest.fixture(autouse=True)
def _skip_metrics_scraper(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(MetricsScraper, "start", lambda self: None)
    monkeypatch.setattr(MetricsScraper, "stop", lambda self: None)


class TestSglangEngineSetup:
    def test_inference_setup_nnodes_2(self, monkeypatch: MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        inference_cfg = SglangInferenceConfig.model_validate(
            {
                "tp": 8,
                "sglang_communication_port": 52532,
                "dp": 1,
                "dtype": "auto",
                "enable-torch-compile": False,
                "enable-dp-attention": True,
                "mem-fraction-static": 0.7,
                "trust-remote-code": True,
            }
        )
        engine = SglangEngine(
            hf_checkpoint_path=Path("path/to/hf/model"),
            nodes={0: "127.0.0.1", 1: "127.0.0.2"},
            inference_config=inference_cfg,
            global_node_rank=0,
            gpus_per_node=4,
            router_config=RouterConfig(port=30000),
        )
        monkeypatch.setattr(SglangInference, "start_inference", lambda self: None)
        monkeypatch.setattr(
            SglangInference,
            "wait_for_inference_to_initialize",
            lambda self: None,
        )
        engine.start()

        inference_manager = engine.inference
        assert len(engine.unit.ranks) == 2
        assert engine.group_node_rank == 0
        assert inference_manager.hf_checkpoint_path == Path("path/to/hf/model")
        assert inference_manager.bash_cmd_list == [
            "/home/user/sglang/bin/python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "path/to/hf/model",
            "--host",
            "0.0.0.0",
            "--port",
            "30001",
            "--nnodes",
            "2",
            "--node-rank",
            "0",
            "--tp-size",
            "8",
            "--pp-size",
            "1",
            "--ep-size",
            "1",
            "--dp-size",
            "1",
            "--dtype",
            "auto",
            "--mem-fraction-static",
            "0.7",
            "--chunked-prefill-size",
            "8192",
            "--max-running-requests",
            "1024",
            "--max-queued-requests",
            "8192",
            "--trust-remote-code",
            "--enable-metrics",
            "--allow-auto-truncate",
            "--schedule-policy",
            "fcfs",
            "--enable-priority-scheduling",
            "--disable-priority-preemption",
            "--default-priority-value",
            "0",
            "--enable-dp-attention",
            "--dist-init-addr",
            "127.0.0.1:52532",
        ]

    def test_inference_setup_nnodes_1(self, monkeypatch: MonkeyPatch, tmp_path: Path):
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        inference_cfg = SglangInferenceConfig.model_validate(
            {
                "tp": 8,
                "sglang_communication_port": 52532,
                "dp": 1,
                "dtype": "auto",
                "enable-torch-compile": False,
                "enable-dp-attention": True,
                "mem-fraction-static": 0.7,
                "trust-remote-code": True,
            }
        )
        engine = SglangEngine(
            hf_checkpoint_path=Path("path/to/hf/model"),
            nodes={0: "127.0.0.1"},
            inference_config=inference_cfg,
            global_node_rank=0,
            gpus_per_node=8,
            router_config=RouterConfig(port=30000),
        )
        monkeypatch.setattr(SglangInference, "start_inference", lambda self: None)
        monkeypatch.setattr(
            SglangInference,
            "wait_for_inference_to_initialize",
            lambda self: None,
        )
        engine.start()

        inference_manager = engine.inference
        assert len(engine.unit.ranks) == 1
        assert engine.group_node_rank == 0
        assert inference_manager.hf_checkpoint_path == Path("path/to/hf/model")
        assert inference_manager.bash_cmd_list == [
            "/home/user/sglang/bin/python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "path/to/hf/model",
            "--host",
            "0.0.0.0",
            "--port",
            "30001",
            "--tp-size",
            "8",
            "--pp-size",
            "1",
            "--ep-size",
            "1",
            "--dp-size",
            "1",
            "--dtype",
            "auto",
            "--mem-fraction-static",
            "0.7",
            "--chunked-prefill-size",
            "8192",
            "--max-running-requests",
            "1024",
            "--max-queued-requests",
            "8192",
            "--trust-remote-code",
            "--enable-metrics",
            "--allow-auto-truncate",
            "--schedule-policy",
            "fcfs",
            "--enable-priority-scheduling",
            "--disable-priority-preemption",
            "--default-priority-value",
            "0",
            "--enable-dp-attention",
        ]

    def test_inference_setup_speculative_eagle(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        inference_cfg = SglangInferenceConfig.model_validate(
            {
                "tp": 8,
                "sglang_communication_port": 52532,
                "dp": 1,
                "dtype": "auto",
                "mem-fraction-static": 0.7,
                "trust-remote-code": True,
                "speculative-algorithm": "EAGLE",
                "speculative-num-steps": 3,
                "speculative-eagle-topk": 1,
                "speculative-num-draft-tokens": 4,
            }
        )
        engine = SglangEngine(
            hf_checkpoint_path=Path("path/to/hf/model"),
            nodes={0: "127.0.0.1"},
            inference_config=inference_cfg,
            global_node_rank=0,
            gpus_per_node=8,
            router_config=RouterConfig(port=30000),
        )
        monkeypatch.setattr(SglangInference, "start_inference", lambda self: None)
        monkeypatch.setattr(
            SglangInference,
            "wait_for_inference_to_initialize",
            lambda self: None,
        )
        engine.start()

        cmd = engine.inference.bash_cmd_list
        for pair in (
            ("--speculative-algorithm", "EAGLE"),
            ("--speculative-num-steps", "3"),
            ("--speculative-eagle-topk", "1"),
            ("--speculative-num-draft-tokens", "4"),
        ):
            i = cmd.index(pair[0])
            assert cmd[i + 1] == pair[1]

    def test_inference_setup_no_speculative_when_unset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        inference_cfg = SglangInferenceConfig.model_validate(
            {"tp": 1, "sglang_communication_port": 52532}
        )
        engine = SglangEngine(
            hf_checkpoint_path=Path("path/to/hf/model"),
            nodes={0: "127.0.0.1"},
            inference_config=inference_cfg,
            global_node_rank=0,
            gpus_per_node=1,
            router_config=RouterConfig(port=30000),
        )
        monkeypatch.setattr(SglangInference, "start_inference", lambda self: None)
        monkeypatch.setattr(
            SglangInference,
            "wait_for_inference_to_initialize",
            lambda self: None,
        )
        engine.start()

        cmd = engine.inference.bash_cmd_list
        assert not any(tok.startswith("--speculative-") for tok in cmd)

    def test_speculative_num_steps_rejects_zero(self):
        with pytest.raises(Exception):
            SglangInferenceConfig.model_validate({"speculative-num-steps": 0})


class TestSglangInferenceRuntime:
    @pytest.mark.parametrize(
        ("disaggregation_mode", "preset_env"),
        [
            (None, None),
            ("prefill", None),
            ("decode", None),
            ("decode", "60"),
        ],
    )
    def test_start_inference_calls_subprocess_manager(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        disaggregation_mode: str | None,
        preset_env: str | None,
    ):
        called = {}
        monkeypatch.delenv("SGLANG_FUSED_MOE_CHUNK_SIZE", raising=False)
        monkeypatch.delenv("SGLANG_TIMEOUT_KEEP_ALIVE", raising=False)
        for timeout_env in TIMEOUT_ENVS:
            if preset_env is not None:
                monkeypatch.setenv(timeout_env, preset_env)
            else:
                monkeypatch.delenv(timeout_env, raising=False)

        class FakeSP:
            def __init__(self, cmd, log, cwd, process_name=None, env=None):
                called["cmd"] = cmd
                called["log"] = log
                called["cwd"] = cwd
                called["process_name"] = process_name
                called["env"] = env

            def start(self):
                called["started"] = True

            def is_alive(self):
                called["alive"] = True
                return True

        monkeypatch.setattr(
            "node_runner.src.inference.sglang.sglang_inference.SubprocessManager",
            FakeSP,
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        im = _inference(
            inference_config=SglangInferenceConfig(
                tp=1, disaggregation_mode=disaggregation_mode
            ),
        )
        im.start_inference()
        assert called.get("started") is True and "--tp-size" in called["cmd"]
        for timeout_env in TIMEOUT_ENVS:
            assert called["env"].get(timeout_env) == preset_env
        assert called["env"]["SGLANG_FUSED_MOE_CHUNK_SIZE"] == "16384"
        assert called["env"]["SGLANG_TIMEOUT_KEEP_ALIVE"] == "75"

    def test_sglang_is_not_ready(self, monkeypatch: MonkeyPatch):
        class FakeResponse:
            def __init__(self, status_code: int):
                self.status_code = status_code

            def json(self):
                return {}

        seen_urls: list[str] = []

        def fake_get(url, timeout):
            seen_urls.append(url)
            return FakeResponse(status_code=400)

        monkeypatch.setattr(requests, "get", fake_get)
        assert SglangInference._check_sglang_is_ready("http://1.1.1.1:52532") is False
        assert seen_urls == ["http://1.1.1.1:52532/health_generate"]

        monkeypatch.setattr(
            requests, "get", lambda url, timeout: FakeResponse(status_code=200)
        )
        assert SglangInference._check_sglang_is_ready("http://1.1.1.1:52532") is True

        def raise_exception(*args, **kwargs):
            raise requests.exceptions.RequestException("Request failed")

        monkeypatch.setattr(
            requests,
            "get",
            raise_exception,
        )
        assert SglangInference._check_sglang_is_ready("http://1.1.1.1:52532") is False


class TestSglangInferenceAutoMtp:
    @staticmethod
    def _setup_runtime(
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        weight_names: list[str] | None,
        inference_config: SglangInferenceConfig,
    ) -> SglangInference:
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        checkpoint = tmp_path / "model"
        checkpoint.mkdir()
        if weight_names is not None:
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(weight_names, "shard")})
            )
        return _inference(
            hf_checkpoint_path=checkpoint, inference_config=inference_config
        )

    @classmethod
    def _speculative_args(
        cls,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        weight_names: list[str] | None,
        inference_config: SglangInferenceConfig,
    ) -> dict[str, str]:
        cmd = cls._setup_runtime(
            monkeypatch, tmp_path, weight_names, inference_config
        ).bash_cmd_list
        return {
            tok: cmd[i + 1]
            for i, tok in enumerate(cmd)
            if tok.startswith("--speculative-")
        }

    @pytest.mark.parametrize(
        "weight_names",
        [
            ["model.layers.0.mlp.weight", "model.layers.40.eh_proj.weight"],
            ["model.layers.0.mlp.weight", "mtp.0.hc_head_base"],
            ["model.layers.0.mlp.weight", "mtp.layers.0.self_attn.q_proj.weight"],
        ],
        ids=["nextn_eh_proj", "deepseek_v4_mtp", "qwen3_5_mtp"],
    )
    def test_enables_mtp_when_checkpoint_has_mtp_weights(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, weight_names: list[str]
    ):
        args = self._speculative_args(
            monkeypatch,
            tmp_path,
            weight_names,
            SglangInferenceConfig.model_validate({"tp": 1, "auto-mtp": True}),
        )

        assert args == MTP_SPECULATIVE_ARGS

    @pytest.mark.parametrize(
        ("weight_names", "inference_config"),
        [
            (
                ["model.layers.0.mlp.weight"],
                SglangInferenceConfig.model_validate({"tp": 1, "auto-mtp": True}),
            ),
            (None, SglangInferenceConfig.model_validate({"tp": 1, "auto-mtp": True})),
            (["model.layers.40.eh_proj.weight"], SglangInferenceConfig(tp=1)),
            (
                ["model.layers.40.eh_proj.weight"],
                SglangInferenceConfig.model_validate(
                    {"tp": 1, "pp": 2, "auto-mtp": True}
                ),
            ),
        ],
        ids=[
            "no_mtp_weights",
            "no_index",
            "auto_mtp_off_by_default",
            "pipeline_parallel",
        ],
    )
    def test_keeps_cmd_without_mtp(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        weight_names: list[str] | None,
        inference_config: SglangInferenceConfig,
    ):
        args = self._speculative_args(
            monkeypatch, tmp_path, weight_names, inference_config
        )

        assert args == {}

    def test_explicit_speculative_config_wins_over_mtp(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ):
        inference_config = SglangInferenceConfig.model_validate(
            {
                "tp": 1,
                "auto-mtp": True,
                "speculative-algorithm": "NGRAM",
                "speculative-num-draft-tokens": 8,
            }
        )

        args = self._speculative_args(
            monkeypatch, tmp_path, ["model.layers.40.eh_proj.weight"], inference_config
        )

        assert args == {
            "--speculative-algorithm": "NGRAM",
            "--speculative-num-draft-tokens": "8",
        }

    @staticmethod
    def _started_env(monkeypatch: MonkeyPatch, im: SglangInference) -> dict[str, str]:
        started = {}

        class FakeSubprocessManager:
            def __init__(self, cmd, log, cwd, process_name=None, env=None):
                started["env"] = env

            def start(self):
                pass

        monkeypatch.setattr(
            "node_runner.src.inference.sglang.sglang_inference.SubprocessManager",
            FakeSubprocessManager,
        )
        im.start_inference()
        return started["env"]

    @pytest.mark.parametrize(
        ("weight_names", "inference_params", "dispatch_tokens"),
        [
            (
                ["model.layers.40.eh_proj.weight"],
                {"moe-a2a-backend": "deepep", "auto-mtp": True},
                "384",
            ),
            (
                ["model.layers.0.mlp.weight"],
                {
                    "moe-a2a-backend": "deepep",
                    "speculative-algorithm": "EAGLE",
                    "speculative-num-draft-tokens": 4,
                },
                "512",
            ),
            (["model.layers.40.eh_proj.weight"], {"auto-mtp": True}, None),
            (["model.layers.0.mlp.weight"], {"moe-a2a-backend": "deepep"}, None),
        ],
        ids=[
            "auto_mtp_with_deepep",
            "explicit_speculative_with_deepep",
            "mtp_without_deepep",
            "deepep_without_speculative",
        ],
    )
    def test_scales_deepep_dispatch_tokens_by_draft_tokens(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        weight_names: list[str],
        inference_params: dict,
        dispatch_tokens: str | None,
    ):
        monkeypatch.delenv(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", raising=False
        )
        im = self._setup_runtime(
            monkeypatch,
            tmp_path,
            weight_names,
            SglangInferenceConfig.model_validate({"tp": 1, **inference_params}),
        )

        env = self._started_env(monkeypatch, im)

        assert env.get("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK") == (
            dispatch_tokens
        )

    def test_keeps_user_deepep_dispatch_tokens(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "256")
        im = self._setup_runtime(
            monkeypatch,
            tmp_path,
            ["model.layers.40.eh_proj.weight"],
            SglangInferenceConfig.model_validate(
                {"tp": 1, "moe-a2a-backend": "deepep", "auto-mtp": True}
            ),
        )

        env = self._started_env(monkeypatch, im)

        assert env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] == "256"


class TestSglangDeepGemmCache:
    def test_setup_deepgemm_cache_dir_copies_nfs_cache(
        self, tmp_path: Path, monkeypatch
    ):
        nfs_dir = tmp_path / "nfs"
        local_dir = tmp_path / "local"
        nfs_cache = nfs_dir / "cache"
        local_cache = local_dir / "cache"

        _write_file(nfs_cache / "a.txt", "a")
        _write_file(nfs_cache / "nested" / "b.txt", "b")
        _write_file(local_cache / "old.txt", "old")

        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        manager = _inference()
        manager._setup_deepgemm_cache_dir(nfs_dir, local_dir)

        assert _collect_files(local_cache) == _collect_files(nfs_cache)
        assert not (local_cache / "old.txt").exists()

    def test_setup_deepgemm_cache_dir_missing_nfs_cache(
        self, tmp_path: Path, monkeypatch
    ):
        nfs_dir = tmp_path / "nfs"
        local_dir = tmp_path / "local"
        local_cache = local_dir / "cache"

        _write_file(local_cache / "stale.txt", "stale")

        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        manager = _inference()
        manager._setup_deepgemm_cache_dir(nfs_dir, local_dir)

        assert local_cache.exists()
        assert _collect_files(local_cache) == {}

    def test_copy_dg_cache_to_nfs_after_successfull_inference(
        self, tmp_path: Path, monkeypatch
    ):
        nfs_dir = tmp_path / "nfs"
        local_dir = tmp_path / "local"
        local_cache = local_dir / "cache"
        nfs_cache = nfs_dir / "cache"
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: tmp_path / "nfs"
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: tmp_path / "local"
        )
        manager = _inference()

        _write_file(local_cache / "a.txt", "new")
        _write_file(local_cache / "nested" / "b.txt", "value")
        _write_file(nfs_cache / "old.txt", "old")
        _write_file(nfs_dir / "outside.txt", "keep")
        manager._copy_dg_cache_to_nfs_after_successfull_inference(nfs_dir, local_dir)

        nfs_files = _collect_files(nfs_cache)
        assert nfs_files.get("a.txt") == "new"
        assert nfs_files.get(str(Path("nested") / "b.txt")) == "value"
        assert nfs_files.get("old.txt") == "old"
        assert (nfs_dir / "outside.txt").read_text() == "keep"


class TestSglangInferenceLifecycle:
    def test_wait_for_inference_raises_if_subprocess_died(
        self, monkeypatch: MonkeyPatch
    ):
        manager = _inference()
        called = {"raise_for_exit_code": 0}

        class _SP:
            def is_alive(self):
                return False

            def raise_for_exit_code(self):
                called["raise_for_exit_code"] += 1
                raise RuntimeError("dead")

        manager.subprocess_manager = _SP()
        with pytest.raises(RuntimeError, match="dead"):
            SglangInference.wait_for_inference_to_initialize.__wrapped__(manager)
        assert called["raise_for_exit_code"] == 1

    def test_wait_for_inference_timeout_branch(self, monkeypatch: MonkeyPatch):
        manager = _inference()
        manager.subprocess_manager = type(
            "_SP",
            (),
            {"is_alive": lambda self: True, "raise_for_exit_code": lambda self: None},
        )()
        monkeypatch.setattr(
            SglangInference,
            "_check_sglang_is_ready",
            staticmethod(lambda _addr: False),
        )
        with pytest.raises(InferenceStartError, match="did not start"):
            SglangInference.wait_for_inference_to_initialize.__wrapped__(manager)

    def test_copy_dg_cache_to_nfs_is_explicit(self, monkeypatch: MonkeyPatch):
        manager = _inference()
        manager.subprocess_manager = type(
            "_SP",
            (),
            {"is_alive": lambda self: True, "raise_for_exit_code": lambda self: None},
        )()
        copied = {"called": 0}
        monkeypatch.setattr(
            SglangInference,
            "_check_sglang_is_ready",
            staticmethod(lambda _addr: True),
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir", lambda self: Path("/nfs")
        )
        monkeypatch.setattr(
            SglangInference, "_get_dg_cache_dir_local", lambda self: Path("/local")
        )
        monkeypatch.setattr(
            manager,
            "_copy_dg_cache_to_nfs_after_successfull_inference",
            lambda nfs, local: copied.__setitem__("called", copied["called"] + 1),
        )

        SglangInference.wait_for_inference_to_initialize.__wrapped__(manager)
        assert copied["called"] == 0

        manager.copy_dg_cache_to_nfs()
        assert copied["called"] == 1

    def test_wait_for_inference_ready_worker_does_not_copy_cache(
        self, monkeypatch: MonkeyPatch
    ):
        manager = _inference()
        manager.subprocess_manager = type(
            "_SP",
            (),
            {"is_alive": lambda self: True, "raise_for_exit_code": lambda self: None},
        )()
        copied = {"called": 0}
        monkeypatch.setattr(
            SglangInference,
            "_check_sglang_is_ready",
            staticmethod(lambda _addr: True),
        )
        monkeypatch.setattr(
            manager,
            "_copy_dg_cache_to_nfs_after_successfull_inference",
            lambda *_: copied.__setitem__("called", copied["called"] + 1),
        )

        SglangInference.wait_for_inference_to_initialize.__wrapped__(manager)
        assert copied["called"] == 0

    def test_atomic_copy_no_overwrite(self, tmp_path: Path):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst" / "dst.txt"
        src.write_text("payload")
        assert SglangInference._atomic_copy_no_overwrite(src, dst) is True
        assert dst.read_text() == "payload"
        assert SglangInference._atomic_copy_no_overwrite(src, dst) is False

    def test_stop_calls_graceful_stop(self):
        manager = _inference()
        called = {"stop": 0}
        manager.subprocess_manager = type(
            "_SP",
            (),
            {
                "graceful_stop": lambda self: called.__setitem__(
                    "stop", called["stop"] + 1
                )
            },
        )()
        manager.stop()
        assert called["stop"] == 1

    def test_stop_without_subprocess_is_noop(self):
        manager = _inference()
        manager.subprocess_manager = None
        assert manager.stop() is None

    def test_get_dg_cache_dirs_from_defaults_and_env(self, monkeypatch: MonkeyPatch):
        manager = _inference()
        monkeypatch.undo()
        monkeypatch.delenv("PAM_DG_CACHE_DIR", raising=False)
        assert manager._get_dg_cache_dir() == Path("/tmp/dg_cache")
        assert manager._get_dg_cache_dir_local() == Path("/home/user/dg_cache")

        monkeypatch.setenv("PAM_DG_CACHE_DIR", "/custom/cache")
        assert manager._get_dg_cache_dir() == Path("/custom/cache")

    def test_atomic_copy_no_overwrite_unexpected_error_cleanup(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst" / "dst.txt"
        src.write_text("payload")

        monkeypatch.setattr(
            "node_runner.src.inference.sglang.sglang_inference.os.link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("link failed")
            ),
        )

        with pytest.raises(RuntimeError, match="link failed"):
            SglangInference._atomic_copy_no_overwrite(src, dst)

        tmp_candidates = list((tmp_path / "dst").glob(".*.tmp-*"))
        assert tmp_candidates == []

    def test_atomic_copy_no_overwrite_file_exists_branch(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst" / "dst.txt"
        src.write_text("payload")

        monkeypatch.setattr(
            "node_runner.src.inference.sglang.sglang_inference.os.link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(FileExistsError("exists")),
        )

        assert SglangInference._atomic_copy_no_overwrite(src, dst) is False

    def test_atomic_copy_no_overwrite_cleanup_unlink_fails(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ):
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst" / "dst.txt"
        src.write_text("payload")

        monkeypatch.setattr(
            "node_runner.src.inference.sglang.sglang_inference.os.link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("link failed")
            ),
        )

        orig_unlink = Path.unlink

        def _broken_unlink(self, missing_ok=False):
            if ".tmp-" in self.name:
                raise OSError("unlink failed")
            return orig_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _broken_unlink)

        with pytest.raises(RuntimeError, match="link failed"):
            SglangInference._atomic_copy_no_overwrite(src, dst)

    def test_copy_dg_cache_to_nfs_when_local_cache_missing(self, tmp_path: Path):
        manager = _inference()
        nfs_dir = tmp_path / "nfs"
        local_dir = tmp_path / "local"
        manager._copy_dg_cache_to_nfs_after_successfull_inference(nfs_dir, local_dir)
        assert not (nfs_dir / "cache").exists()

    def test_copy_dg_cache_to_nfs_when_no_new_files(self, tmp_path: Path):
        manager = _inference()
        nfs_dir = tmp_path / "nfs"
        local_dir = tmp_path / "local"
        local_cache = local_dir / "cache"
        nfs_cache = nfs_dir / "cache"
        _write_file(local_cache / "a.txt", "same")
        _write_file(nfs_cache / "a.txt", "same")

        manager._copy_dg_cache_to_nfs_after_successfull_inference(nfs_dir, local_dir)
        assert (nfs_cache / "a.txt").read_text() == "same"


class TestSglangInferenceLogProfiles:
    def test_inference_log_file_follows_profile(self, tmp_path: Path):
        Logger.node_log_dir = tmp_path
        manager = _inference(log_profile="judge")

        assert manager.inference_log_file == tmp_path / "judge_inference.log"
