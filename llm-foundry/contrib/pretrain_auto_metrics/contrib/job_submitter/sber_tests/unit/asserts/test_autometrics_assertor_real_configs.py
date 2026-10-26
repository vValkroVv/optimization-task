from pathlib import Path

import pytest
from job_submitter.asserts.autometrics import AutometricsJobAssertor


class _FakeAutoJob:
    def __init__(
        self,
        submission_type: str = "cloud",
        mode: str = "autometrics",
        n_workers: int = 2,
        n_gpus: int = 8,
    ):
        self.submission_type = submission_type
        self.mode = mode
        self.n_workers = n_workers
        self.n_gpus = n_gpus
        self.checkpoint_dir = "."
        self.path_to_config = None


TEST_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "test_data" / "pam_configs"


def _load_pam_config(name: str) -> dict:
    assertor = AutometricsJobAssertor()
    return assertor._load_config(TEST_CONFIGS_DIR / name)


def _materialize_obm_proxy_template_if_needed(cfg: dict, tmp_path: Path) -> None:
    obm_proxy = cfg.get("autometrics", {}).get("obm_proxy", {})
    if not isinstance(obm_proxy, dict):
        return

    chat_template = obm_proxy.get("chat-template", obm_proxy.get("chat_template"))
    if not isinstance(chat_template, str) or not chat_template:
        return

    if Path(chat_template).exists():
        return

    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")

    if "chat-template" in obm_proxy:
        obm_proxy["chat-template"] = str(template)
    else:
        obm_proxy["chat_template"] = str(template)


@pytest.mark.unit
def test_qwen_config_from_pretrain_auto_metrics_passes_asserts(tmp_path: Path) -> None:
    cfg = _load_pam_config("qwen3-coder.yaml")
    _materialize_obm_proxy_template_if_needed(cfg, tmp_path)
    assertor = AutometricsJobAssertor()
    assertor._assert_pam_config(cfg, _FakeAutoJob())


@pytest.mark.unit
def test_deepseek_config_from_pretrain_auto_metrics_passes_asserts(
    tmp_path: Path,
) -> None:
    cfg = _load_pam_config("deepseek-r1.yaml")
    _materialize_obm_proxy_template_if_needed(cfg, tmp_path)
    assertor = AutometricsJobAssertor()
    assertor._assert_pam_config(cfg, _FakeAutoJob())


@pytest.mark.unit
def test_10b_config_from_pretrain_auto_metrics_passes_asserts(tmp_path: Path) -> None:
    cfg = _load_pam_config("10b_MoE.yaml")
    _materialize_obm_proxy_template_if_needed(cfg, tmp_path)
    assertor = AutometricsJobAssertor()
    assertor._assert_pam_config(cfg, _FakeAutoJob())
