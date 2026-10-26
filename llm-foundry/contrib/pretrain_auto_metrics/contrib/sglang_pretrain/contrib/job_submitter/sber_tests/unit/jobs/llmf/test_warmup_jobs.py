from pathlib import Path

import pytest
import yaml
from job_submitter.asserts.warmup import WarmupJobAssertor
from job_submitter.jobs.llmf.warmup import WarmupJobLLMF
from sber_tests.dry_run_helpers import assert_dry_run_submit_contract


def _make_warmup_config(tmp_path: Path, warmup_type: str = "both") -> Path:
    """Write a minimal warmup YAML config and return its path."""
    config = {}
    if warmup_type in ("triton", "both"):
        config["triton"] = {
            "cache_dir": str(tmp_path / "triton_cache"),
            "kernel_groups": ["blocks_quantization", "row_quantization"],
            "dense_groups": ["dense_block_quantize"],
            "num_groups": 6,
            "hid_dim": 7168,
            "int_dim": 2048,
        }
    if warmup_type in ("deep_gemm", "both"):
        config["deep_gemm"] = {
            "cache_dir": str(tmp_path / "dg_cache"),
            "gemm_groups": ["w1_m_grouped", "w2_m_grouped", "w2_k_grouped"],
            "hid_dim": 7168,
            "int_dim": 2048,
            "num_groups": 16,
            "min_size": 128,
            "max_size": 64000,
        }
    cfg_path = tmp_path / "warmup_config.yaml"
    cfg_path.write_text(yaml.dump(config))
    return cfg_path


@pytest.mark.unit
class TestWarmupJobLLMFScript:
    def test_script_builds_triton_command(self, tmp_path: Path) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            kernel_group="blocks_quantization",
        )
        script = job._script()
        assert "triton_warmup.py" in script
        assert "--config /tmp/warmup.yaml" in script
        assert "--device 0" in script
        assert "--kernel-group blocks_quantization" in script

    def test_script_builds_deep_gemm_command(self, tmp_path: Path) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="deep_gemm",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            kernel_group="w1_m_grouped",
        )
        script = job._script()
        assert "deep_gemm_warmup.py" in script
        assert "--config /tmp/warmup.yaml" in script
        assert "--device 0" in script
        assert "--gemm w1_m_grouped" in script

    def test_script_deep_gemm_with_stages_overrides(self) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="deep_gemm",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            kernel_group="w1_k_grouped",
            max_size_override=128000,
            max_buffer_tokens=256000,
        )
        script = job._script()
        assert "--max-size-override 128000" in script
        assert "--max-buffer-tokens 256000" in script

    def test_script_all_groups_no_flag_triton(self) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            kernel_group=None,
        )
        script = job._script()
        assert "--kernel-group" not in script

    def test_script_all_groups_no_flag_deep_gemm(self) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="deep_gemm",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            kernel_group=None,
        )
        script = job._script()
        assert "--gemm" not in script


@pytest.mark.unit
class TestWarmupJobLLMFEnv:
    def test_env_vars_triton(self, monkeypatch) -> None:
        monkeypatch.setattr(
            WarmupJobLLMF, "_get_current_region", lambda self: ("SR008", "cloud")
        )
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/triton_cache",
        )
        env = job._build_env()
        assert env["TRITON_CONFIGS_DIR"] == "/tmp/triton_cache"
        assert env["TRITON_MODE"] == "warmup"
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert "DG_JIT_CACHE_DIR" not in env

    def test_env_vars_deep_gemm(self, monkeypatch) -> None:
        monkeypatch.setattr(
            WarmupJobLLMF, "_get_current_region", lambda self: ("SR008", "cloud")
        )
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="deep_gemm",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/dg_cache",
        )
        env = job._build_env()
        assert env["DG_JIT_CACHE_DIR"] == "/tmp/dg_cache"
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert "TRITON_CONFIGS_DIR" not in env
        assert "TRITON_MODE" not in env


@pytest.mark.unit
class TestWarmupJobLLMFMisc:
    def test_external_code_libs_no_flash_linear_attention(self) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
        )
        libs = job._external_code_libraries()
        for lib in libs:
            assert "flash-linear-attention" not in lib
        assert any("composer" in lib for lib in libs)
        assert any("streaming" in lib for lib in libs)
        assert any("gigafsdp" in lib for lib in libs)

    def test_job_desc_with_group(self) -> None:
        job = WarmupJobLLMF(
            user_login="user1",
            exp_name="warmup_ultra",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            kernel_group="blocks_quantization",
        )
        assert job._resolve_job_desc() == (
            "user1 | warmup_ultra | triton_blocks_quantization"
        )

    def test_job_desc_all_groups(self) -> None:
        job = WarmupJobLLMF(
            user_login="user1",
            exp_name="warmup_ultra",
            submission_type="cloud",
            region="SR008",
            warmup_type="deep_gemm",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            kernel_group=None,
        )
        assert job._resolve_job_desc() == ("user1 | warmup_ultra | deep_gemm_all")

    def test_submit_dry_run(self, monkeypatch) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            dry_run=True,
        )
        assert_dry_run_submit_contract(monkeypatch=monkeypatch, job=job)

    def test_before_submit_deep_gemm_creates_dirs(self, tmp_path: Path) -> None:
        cache = tmp_path / "dg_cache"
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="deep_gemm",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir=str(cache),
        )
        job._before_submit()
        assert (cache / "tmp").is_dir()
        assert (cache / "cache").is_dir()

    def test_before_submit_triton_noop(self, tmp_path: Path) -> None:
        cache = tmp_path / "triton_cache"
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir=str(cache),
        )
        job._before_submit()
        assert not cache.exists()


@pytest.mark.unit
class TestBuildWarmupJobs:
    def test_triton_one_job_per_group(self, tmp_path: Path) -> None:
        cfg_path = _make_warmup_config(tmp_path, "triton")
        jobs = WarmupJobLLMF.build_warmup_jobs(
            user_login="u",
            exp_name="e",
            warmup_type="triton",
            warmup_config_path=cfg_path,
            submission_type="cloud",
            region="SR008",
        )
        # 2 kernel_groups + 1 dense_group = 3
        assert len(jobs) == 3
        groups = [j.kernel_group for j in jobs]
        assert "blocks_quantization" in groups
        assert "row_quantization" in groups
        assert "dense_block_quantize" in groups

    def test_deep_gemm_with_stages_coverage(self, tmp_path: Path) -> None:
        """max_size=64000 < 128000 threshold, so stages-coverage jobs are added."""
        cfg_path = _make_warmup_config(tmp_path, "deep_gemm")
        jobs = WarmupJobLLMF.build_warmup_jobs(
            user_login="u",
            exp_name="e",
            warmup_type="deep_gemm",
            warmup_config_path=cfg_path,
            submission_type="cloud",
            region="SR008",
        )
        # 3 base groups + 3 stages-coverage groups = 6
        assert len(jobs) == 6
        stages_jobs = [j for j in jobs if j.max_size_override is not None]
        assert len(stages_jobs) == 3
        for j in stages_jobs:
            assert j.max_size_override == 128000

    def test_deep_gemm_no_stages_when_max_size_above_threshold(
        self, tmp_path: Path
    ) -> None:
        config = {
            "deep_gemm": {
                "cache_dir": str(tmp_path / "dg_cache"),
                "gemm_groups": ["w1_m_grouped"],
                "hid_dim": 7168,
                "int_dim": 2048,
                "num_groups": 16,
                "min_size": 128,
                "max_size": 200000,  # above threshold
            }
        }
        cfg_path = tmp_path / "warmup_config.yaml"
        cfg_path.write_text(yaml.dump(config))
        jobs = WarmupJobLLMF.build_warmup_jobs(
            user_login="u",
            exp_name="e",
            warmup_type="deep_gemm",
            warmup_config_path=cfg_path,
            submission_type="cloud",
            region="SR008",
        )
        # 1 base group, no stages-coverage
        assert len(jobs) == 1
        assert jobs[0].max_size_override is None

    def test_all_jobs_have_correct_cache_dir(self, tmp_path: Path) -> None:
        cfg_path = _make_warmup_config(tmp_path, "triton")
        jobs = WarmupJobLLMF.build_warmup_jobs(
            user_login="u",
            exp_name="e",
            warmup_type="triton",
            warmup_config_path=cfg_path,
            submission_type="cloud",
            region="SR008",
        )
        expected_cache = str(tmp_path / "triton_cache")
        for job in jobs:
            assert job.cache_dir == expected_cache

    def test_base_kwargs_passed_through(self, tmp_path: Path) -> None:
        cfg_path = _make_warmup_config(tmp_path, "triton")
        jobs = WarmupJobLLMF.build_warmup_jobs(
            user_login="u",
            exp_name="e",
            warmup_type="triton",
            warmup_config_path=cfg_path,
            submission_type="slurm",
            region="o1",
            queue_name="shared",
        )
        for job in jobs:
            assert job.submission_type == "slurm"
            assert job.region == "o1"
            assert job.queue_name == "shared"

    def test_deep_gemm_stages_coverage_fits_in_memory(self, tmp_path: Path) -> None:
        """When peak memory fits in budget, max_buffer_tokens is None."""
        config = {
            "deep_gemm": {
                "cache_dir": str(tmp_path / "dg_cache"),
                "gemm_groups": ["w1_k_grouped"],
                "hid_dim": 1536,
                "int_dim": 1280,
                "num_groups": 1,  # tiny: peak_est << 70 GiB
                "min_size": 128,
                "max_size": 1024,
            }
        }
        cfg_path = tmp_path / "warmup_config.yaml"
        cfg_path.write_text(yaml.dump(config))
        jobs = WarmupJobLLMF.build_warmup_jobs(
            user_login="u",
            exp_name="e",
            warmup_type="deep_gemm",
            warmup_config_path=cfg_path,
            submission_type="cloud",
            region="SR008",
        )
        stages_jobs = [j for j in jobs if j.max_size_override is not None]
        assert len(stages_jobs) == 1
        assert stages_jobs[0].max_size_override == 128000
        assert stages_jobs[0].max_buffer_tokens is None

    def test_deep_gemm_stages_coverage_with_buffer_cap(self, tmp_path: Path) -> None:
        """When peak memory exceeds budget, max_buffer_tokens is set."""
        config = {
            "deep_gemm": {
                "cache_dir": str(tmp_path / "dg_cache"),
                "gemm_groups": ["w1_k_grouped"],
                "hid_dim": 7168,
                "int_dim": 7168,
                "num_groups": 256,
                "min_size": 128,
                "max_size": 1024,  # well below threshold
            }
        }
        cfg_path = tmp_path / "warmup_config.yaml"
        cfg_path.write_text(yaml.dump(config))
        jobs = WarmupJobLLMF.build_warmup_jobs(
            user_login="u",
            exp_name="e",
            warmup_type="deep_gemm",
            warmup_config_path=cfg_path,
            submission_type="cloud",
            region="SR008",
        )
        stages_jobs = [j for j in jobs if j.max_size_override is not None]
        assert len(stages_jobs) == 1
        assert stages_jobs[0].max_size_override == 128000
        assert stages_jobs[0].max_buffer_tokens is not None
        assert stages_jobs[0].max_buffer_tokens % 128 == 0


@pytest.mark.unit
class TestWarmupJobAssertor:
    def test_assert_invalid_warmup_type(self) -> None:
        assertor = WarmupJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_warmup_type("invalid")

    def test_assert_valid_warmup_types(self) -> None:
        assertor = WarmupJobAssertor()
        assertor._assert_warmup_type("triton")
        assertor._assert_warmup_type("deep_gemm")

    def test_assert_missing_config(self) -> None:
        assertor = WarmupJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_warmup_config("/nonexistent/path.yaml")

    def test_assert_valid_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("triton: {}")
        assertor = WarmupJobAssertor()
        assertor._assert_warmup_config(str(cfg))

    def test_assert_empty_cache_dir(self) -> None:
        assertor = WarmupJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_cache_dir("")

    def test_assert_valid_cache_dir(self) -> None:
        assertor = WarmupJobAssertor()
        assertor._assert_cache_dir("/some/cache/dir")

    def test_assert_single_gpu_violation_n_gpus(self) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            n_gpus=8,
        )
        assertor = WarmupJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_single_gpu(job)

    def test_assert_single_gpu_violation_n_workers(self) -> None:
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path="/tmp/warmup.yaml",
            cache_dir="/tmp/cache",
            n_workers=4,
        )
        assertor = WarmupJobAssertor()
        with pytest.raises(AssertionError):
            assertor._assert_single_gpu(job)

    def test_assert_job_calls_all_checks(self, tmp_path: Path, monkeypatch) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("triton: {}")
        job = WarmupJobLLMF(
            user_login="u",
            exp_name="e",
            submission_type="cloud",
            region="SR008",
            warmup_type="triton",
            warmup_config_path=str(cfg),
            cache_dir="/tmp/cache",
        )
        monkeypatch.setattr(
            "job_submitter.asserts.base.BaseJobAssertor.assert_job",
            lambda self, j, t: None,
        )
        assertor = WarmupJobAssertor()
        assertor.assert_job(job, object())  # should not raise
