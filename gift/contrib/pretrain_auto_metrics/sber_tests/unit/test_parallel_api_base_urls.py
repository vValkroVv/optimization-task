import json

import pytest

from node_runner.src.config_model import (
    ForgeEvaluatorConfig,
    Tau2EvaluatorConfig,
)
from sber_tests.test_support import judge_endpoints

pytestmark = pytest.mark.unit

BASE_URL = "http://127.0.0.1:29999"
SECONDARY_BASE_URL = "http://10.0.0.5:30000"


def _tau2_config() -> Tau2EvaluatorConfig:
    cfg = Tau2EvaluatorConfig(domains=["airline"], user_model_name="user-model")
    cfg.judge_endpoints = judge_endpoints({"user-model": SECONDARY_BASE_URL})
    cfg.apply_defaults(
        inference_instances=1,
        base_url=BASE_URL,
    )
    return cfg


def test_tau2_apply_defaults_appends_v1():
    cfg = _tau2_config()
    assert cfg.agent_api_base == f"{BASE_URL}/v1"
    assert cfg.user_api_base == f"{SECONDARY_BASE_URL}/v1"


def test_tau2_apply_defaults_scales_max_concurrency():
    cfg = Tau2EvaluatorConfig(domains=["airline"], user_model_name="user-model")
    cfg.judge_endpoints = judge_endpoints({"user-model": SECONDARY_BASE_URL})
    cfg.apply_defaults(
        inference_instances=4,
        base_url=BASE_URL,
    )
    assert (
        cfg.max_concurrency == 128
    )  # 4 * TAU2_DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE


def test_tau2_apply_defaults_keeps_explicit_max_concurrency():
    cfg = Tau2EvaluatorConfig(
        domains=["airline"], user_model_name="user-model", **{"max-concurrency": 9}
    )
    cfg.judge_endpoints = judge_endpoints({"user-model": SECONDARY_BASE_URL})
    cfg.apply_defaults(
        inference_instances=4,
        base_url=BASE_URL,
    )
    assert cfg.max_concurrency == 9


def test_tau2_build_cli_run_uses_v1_api_bases():
    cli = _tau2_config().build_cli_run(domain="airline", save_to="run")
    agent_args = json.loads(cli[cli.index("--agent-llm-args") + 1])
    user_args = json.loads(cli[cli.index("--user-llm-args") + 1])
    nl_args = json.loads(cli[cli.index("--nl-assertions-llm-args") + 1])
    assert agent_args["api_base"] == f"{BASE_URL}/v1"
    assert user_args["api_base"] == f"{SECONDARY_BASE_URL}/v1"
    assert nl_args["api_base"] == f"{SECONDARY_BASE_URL}/v1"
    assert agent_args["timeout"] == 21600


def test_forge_meta_override_uses_v1_api_bases():
    cfg = ForgeEvaluatorConfig(judge="judge-model")
    cfg.judge_endpoints = judge_endpoints({"judge-model": SECONDARY_BASE_URL})
    cfg.apply_defaults(
        inference_instances=1,
        base_url=BASE_URL,
    )
    override = cfg.build_meta_override("some-recipe")
    assert override["model"]["base_url"] == f"{BASE_URL}/v1"
    assert override["judge"]["base_url"] == f"{SECONDARY_BASE_URL}/v1"


def test_forge_apply_defaults_scales_workers():
    cfg = ForgeEvaluatorConfig(judge="judge-model")
    cfg.judge_endpoints = judge_endpoints({"judge-model": SECONDARY_BASE_URL})
    cfg.apply_defaults(
        inference_instances=4,
        base_url=BASE_URL,
    )
    assert cfg.workers == 512  # 4 * DEFAULT_PARALLEL_PER_INFERENCE_INSTANCE


def test_forge_apply_defaults_keeps_explicit_workers():
    cfg = ForgeEvaluatorConfig(judge="judge-model", workers=7)
    cfg.judge_endpoints = judge_endpoints({"judge-model": SECONDARY_BASE_URL})
    cfg.apply_defaults(
        inference_instances=4,
        base_url=BASE_URL,
    )
    assert cfg.workers == 7
