from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

from job_submitter.asserts.warmup import WarmupJobAssertor
from job_submitter.jobs.base import BaseJob
from job_submitter.jobs.image_registry import LLMF_BASE_IMAGES

# DeepGEMM k_grouped wgrad: the num_stages template parameter changes at this
# threshold (SM90, FP8 1D1D block config 128x152x128, bf16 output).
# max_k <= 127 744 -> num_stages = 5; max_k > 127 744 -> num_stages = 4.
# Warmup must cross this boundary so both kernel variants are compiled.
_SM90_STAGES_THRESHOLD = 128_000
_GPU_MEM_BUDGET = 70 * (1 << 30)  # conservative 70 GiB for 80 GB GPU


@dataclass(slots=True, kw_only=True)
class WarmupJobLLMF(BaseJob, WarmupJobAssertor):
    warmup_type: str  # "triton" | "deep_gemm"
    warmup_config_path: Union[str, Path]
    cache_dir: str
    kernel_group: Optional[str] = None  # specific group, or None = run all
    max_size_override: Optional[int] = None  # stages-coverage
    max_buffer_tokens: Optional[int] = None  # stages-coverage OOM guard

    # Warmup is always single-GPU, single-worker
    n_gpus: int = 1
    n_workers: int = 1
    processes_per_worker: int = 1
    script_type: str = "binary"

    BASE_IMAGES: ClassVar[dict[str, str]] = LLMF_BASE_IMAGES

    @property
    def _job_label(self) -> str:
        if self.kernel_group:
            return f"{self.warmup_type}_{self.kernel_group}"
        return f"{self.warmup_type}_all"

    def _resolve_job_desc(self) -> str:
        return f"{self.user_login} | {self.exp_name} | {self._job_label}"

    def _script(self) -> str:
        warmup_dir = self.llm_f_root / "llmfoundry" / "models" / "ops" / "warmup"

        if self.warmup_type == "triton":
            cmd = (
                f"python {warmup_dir}/triton_warmup.py"
                f" --config {self.warmup_config_path}"
                f" --device 0"
            )
            if self.kernel_group:
                cmd += f" --kernel-group {self.kernel_group}"
        else:
            cmd = (
                f"python {warmup_dir}/deep_gemm_warmup.py"
                f" --config {self.warmup_config_path}"
                f" --device 0"
            )
            if self.kernel_group:
                cmd += f" --gemm {self.kernel_group}"
            if self.max_size_override is not None:
                cmd += f" --max-size-override {self.max_size_override}"
            if self.max_buffer_tokens is not None:
                cmd += f" --max-buffer-tokens {self.max_buffer_tokens}"

        return cmd

    def _external_code_libraries(self) -> list[str]:
        return [
            str(self.llm_f_root),
            str(self.llm_f_root / "contrib" / "composer"),
            str(self.llm_f_root / "contrib" / "streaming"),
            str(self.llm_f_root / "contrib" / "gigafsdp"),
        ]

    def _build_env(self) -> Dict[str, str]:
        env = BaseJob._build_env(self)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        if self.warmup_type == "triton":
            env["TRITON_CONFIGS_DIR"] = self.cache_dir
            env["TRITON_MODE"] = "warmup"
        else:
            env["DG_JIT_CACHE_DIR"] = self.cache_dir

        return env

    def _before_submit(self) -> None:
        if self.warmup_type == "deep_gemm":
            os.makedirs(os.path.join(self.cache_dir, "tmp"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "cache"), exist_ok=True)

    @staticmethod
    def _compute_stages_coverage(
        warmup_config: dict,
    ) -> List[tuple]:
        """Compute extra (max_size_override, max_buffer_tokens) pairs for
        stages-coverage passes when the config's max_size is below the SM90
        num_stages threshold.

        Returns a list of (max_size_override, max_buffer_tokens) tuples.
        Empty list means no extra passes are needed.
        """
        max_size = warmup_config.get("max_size", 8192)
        num_groups = warmup_config.get("num_groups", 1)
        hid_dim = warmup_config.get("hid_dim", 7168)
        int_dim = warmup_config.get("int_dim", 2048)

        if max_size >= _SM90_STAGES_THRESHOLD:
            return []

        largest_dim = max(hid_dim, 2 * int_dim)
        buf_tokens = _SM90_STAGES_THRESHOLD * num_groups
        peak_est = buf_tokens * largest_dim * 2 * 3

        if peak_est < _GPU_MEM_BUDGET:
            return [(_SM90_STAGES_THRESHOLD, None)]
        else:
            stages_buf = min(
                _SM90_STAGES_THRESHOLD * 2,
                _GPU_MEM_BUDGET // (largest_dim * 2 * 3),
            )
            stages_buf = (stages_buf // 128) * 128
            return [(_SM90_STAGES_THRESHOLD, stages_buf)]

    @staticmethod
    def build_warmup_jobs(
        user_login: str,
        exp_name: str,
        warmup_type: str,
        warmup_config_path: Union[str, Path],
        **base_kwargs,
    ) -> List["WarmupJobLLMF"]:
        """Build a list of WarmupJobLLMF instances from a warmup YAML config.

        One job per kernel group is created so they can run in parallel
        on the cluster.

        Args:
            user_login: User login / telegram nickname.
            exp_name: Short experiment label.
            warmup_type: "triton" or "deep_gemm".
            warmup_config_path: Absolute path to the warmup YAML config.
            **base_kwargs: Passed through to WarmupJobLLMF
                (region, submission_type, base_image, queue_name, etc.).

        Returns:
            List of WarmupJobLLMF instances ready to be submitted.
        """
        warmup_config_path = Path(warmup_config_path)
        full_config = WarmupJobAssertor._load_config(warmup_config_path)
        warmup_config_path = str(warmup_config_path)

        assert warmup_type in full_config, (
            f"Warmup config does not contain a '{warmup_type}' section"
        )
        warmup_config = full_config[warmup_type]

        cache_dir = warmup_config.get("cache_dir", "")
        assert cache_dir, f"cache_dir must be set in the '{warmup_type}' section"

        # Collect kernel groups
        if warmup_type == "triton":
            kernel_groups = list(warmup_config.get("kernel_groups") or [])
            kernel_groups.extend(warmup_config.get("dense_groups") or [])
        else:
            kernel_groups = list(warmup_config.get("gemm_groups") or [])

        assert kernel_groups, (
            f"Warmup config must contain at least one kernel group for '{warmup_type}'"
        )

        common_kwargs = dict(
            user_login=user_login,
            exp_name=exp_name,
            warmup_type=warmup_type,
            warmup_config_path=warmup_config_path,
            cache_dir=cache_dir,
            **base_kwargs,
        )

        jobs: List[WarmupJobLLMF] = []

        for group in kernel_groups:
            jobs.append(WarmupJobLLMF(kernel_group=group, **common_kwargs))

        # DeepGEMM stages-coverage: extra passes to cross the num_stages
        # threshold boundary.
        if warmup_type == "deep_gemm":
            stages_params = WarmupJobLLMF._compute_stages_coverage(warmup_config)
            for max_size_override, max_buffer_tokens in stages_params:
                for group in kernel_groups:
                    jobs.append(
                        WarmupJobLLMF(
                            kernel_group=group,
                            max_size_override=max_size_override,
                            max_buffer_tokens=max_buffer_tokens,
                            **common_kwargs,
                        )
                    )

        return jobs
