import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "merge_safetensors_ranks.py"


def test_cli_lists_model_families():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    assert "gigachat3_5_dense" in result.stdout
