import json
import shutil
from pathlib import Path

import pytest
from node_runner.src.subprocess_manager import SubprocessManager
from sber_tests.test_support import assert_metrics_against_goldens

pytestmark = pytest.mark.e2e_430b

MODEL_PATH = "/workspace-SR008.nfs2/models/430b_reasoning_test_no_delete"
GOLDEN_MODEL_DIRNAME = "430b_reasoning_test_no_delete"


class TestE2E430B:
    @property
    def pam_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    def test_430b_quantized_dp_attention_e2e(self):
        model_path = Path(MODEL_PATH)
        config_path = self.pam_root / "sber_tests/test_data/configs/430b_test.yaml"
        artifacts_path = model_path / "artifacts"

        shutil.rmtree(artifacts_path / "metrics", ignore_errors=True)
        shutil.rmtree(artifacts_path / "hf_checkpoint_fp8", ignore_errors=True)

        cmd = [
            "python3",
            "node_runner.py",
            "--checkpoint",
            str(model_path),
            "--config",
            str(config_path),
            "--mode",
            "autometrics",
        ]
        log_file = artifacts_path / "logs" / "node_runner.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.unlink(missing_ok=True)
        sm = SubprocessManager(
            cmd,
            log_file=log_file,
            cwd=self.pam_root / "node_runner",
        )
        sm.start()
        sm.wait_for_completion()
        sm.raise_for_exit_code()

        quantized_config = artifacts_path / "hf_checkpoint_fp8" / "config.json"
        assert quantized_config.is_file(), (
            f"Quantization produced no FP8 checkpoint at {quantized_config.parent}"
        )
        quant_method = (
            json.loads(quantized_config.read_text())
            .get("quantization_config", {})
            .get("quant_method")
        )
        assert quant_method == "fp8", (
            f"Expected an FP8 checkpoint, got quant_method={quant_method!r} "
            f"in {quantized_config}"
        )

        assert_metrics_against_goldens(
            model_path=str(model_path),
            golden_root=self.pam_root
            / "sber_tests"
            / "test_data"
            / "goldens"
            / GOLDEN_MODEL_DIRNAME,
            tolerance=0.05,
            tolerance_growth_coeff=3,
        )
