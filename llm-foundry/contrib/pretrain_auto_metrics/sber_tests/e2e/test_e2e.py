import os
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from node_runner.src.inference.generations.server import MISS_MARKER
from node_runner.src.subprocess_manager import SubprocessManager
from sber_tests.test_support import assert_metrics_against_goldens

pytestmark = pytest.mark.e2e

PAM_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = "/home/jovyan/models/10b_v4_1_no_delete"


class TestE2E:
    def test_10b_e2e(self, tmp_path: Path):
        root = Path(os.getenv("PAM_E2E_CHECKPOINT", MODEL_PATH))
        for name in ("metrics", "hf_checkpoint", "logs", "generations"):
            shutil.rmtree(root / "artifacts" / name, ignore_errors=True)

        configs = [PAM_ROOT / "sber_tests/test_data/configs/10b_test.yaml"]
        if os.getenv("PAM_E2E_OVERLAY"):
            configs.append(PAM_ROOT / os.environ["PAM_E2E_OVERLAY"])
        config = tmp_path / "pam_config.yaml"
        OmegaConf.save(OmegaConf.merge(*map(OmegaConf.load, configs)), config)

        log_file = root / "artifacts" / "logs" / "node_runner.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        sm = SubprocessManager(
            ["python3", "node_runner.py", "--checkpoint", str(root)]
            + ["--config", str(config), "--mode", "autometrics"],
            log_file=log_file,
            cwd=PAM_ROOT / "node_runner",
        )
        sm.start()
        sm.wait_for_completion()
        sm.raise_for_exit_code()

        misses = [
            line
            for path in sorted(root.glob("artifacts/logs/**/*.log"))
            for line in path.read_text(errors="replace").splitlines()
            if MISS_MARKER in line
        ]
        assert not misses, (
            f"{len(misses)} requests are not in the recording; record again. "
            f"First: {misses[0]}"
        )

        assert_metrics_against_goldens(
            model_path=str(root),
            golden_root=PAM_ROOT / "sber_tests/test_data/goldens/10b_v4_1_no_delete",
            tolerance=0.05,
            tolerance_growth_coeff=3,
        )
